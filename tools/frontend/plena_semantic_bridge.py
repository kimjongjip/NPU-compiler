#!/usr/bin/env python3
"""Graph-certified bridge from official torch-mlir import to PLENA ISA.

This module is deliberately stricter than a configuration template frontend.
It accepts a *captured* torch.export package and the corresponding official
torch-mlir Linalg package, proves that their graph/state ABI is one of the
dense decoder semantics implemented by the ETRI target, and only then emits
the existing verified ETRI semantic IR.  A config file supplies constants and
static specialization choices, but cannot authorize a graph whose operators,
parameter roles, shapes, ordering, or output contract disagree.

The bridge is a model-level legalization for the currently supported static
dense causal-decoder feature set.  Architecture names are provenance only:
the imported ATen graph, tensor shapes, parameter roles, and target capability
report authorize compilation.  It is not a generic lowering of arbitrary
Linalg operations and fails closed for any unknown computation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Iterable, Mapping, Sequence


BRIDGE_SCHEMA_VERSION = 1
BRIDGE_KIND = "plena_graph_certified_dense_decoder"
RECOGNIZER = "plena-dense-decoder-v1"
OFFICIAL_PACKAGE_KIND = "plena_torch_export_frontend"
OFFICIAL_IMPORTER = "torch_mlir.fx.export_and_import(ExportedProgram)"


class SemanticBridgeError(RuntimeError):
    """The imported graph cannot be proven equivalent to a supported target graph."""


@dataclass(frozen=True)
class SemanticCertificate:
    document: Mapping[str, Any]
    model_spec: Any
    capability_report: Any


@dataclass(frozen=True)
class BridgePackageResult:
    output_dir: Path
    files: Mapping[str, Path]
    manifest: Mapping[str, Any]


_ALLOWED_CALL_FUNCTIONS = frozenset({
    "aten::_assert_tensor_metadata",
    "aten::_to_copy",
    "aten::_unsafe_view",
    "aten::add.Tensor",
    "aten::alias",
    "aten::arange",
    "aten::arange.start",
    "aten::cat",
    "aten::clone",
    "aten::cos",
    "aten::embedding",
    "aten::expand",
    "aten::index.Tensor",
    "aten::le.Tensor",
    "aten::linear",
    "aten::logical_and",
    "aten::matmul",
    "aten::mean.dim",
    "aten::mul.Tensor",
    "aten::neg",
    "aten::new_ones",
    "aten::pow.Tensor_Scalar",
    "aten::rsqrt",
    "aten::silu",
    "aten::sin",
    "aten::slice.Tensor",
    "aten::softmax.int",
    "aten::transpose.int",
    "aten::unsqueeze",
    "aten::view",
    "aten::where.ScalarOther",
})


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SemanticBridgeError(f"cannot read JSON object {path}: {error}") from error
    if not isinstance(value, dict):
        raise SemanticBridgeError(f"JSON root is not an object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_bytes(_json_bytes(value))


def _verify_package(
    directory: Path, *, require_output_type: str | None = None
) -> dict[str, Any]:
    if not directory.is_dir():
        raise SemanticBridgeError(f"graph package directory does not exist: {directory}")
    package_path = directory / "package.json"
    package = _read_json(package_path)
    if package.get("package_kind") != OFFICIAL_PACKAGE_KIND:
        raise SemanticBridgeError(
            f"{package_path} is not an official torch.export frontend package"
        )
    if package.get("fallback") is not None:
        raise SemanticBridgeError("official graph package records a fallback")
    artifacts = package.get("artifacts")
    if not isinstance(artifacts, list):
        raise SemanticBridgeError("official graph package artifacts must be an array")
    by_file: dict[str, Mapping[str, Any]] = {}
    for record in artifacts:
        if not isinstance(record, dict) or not isinstance(record.get("file"), str):
            raise SemanticBridgeError("malformed official graph artifact record")
        name = record["file"]
        if Path(name).name != name or name in by_file:
            raise SemanticBridgeError(f"invalid or duplicate artifact name {name!r}")
        path = directory / name
        if not path.is_file():
            raise SemanticBridgeError(f"official graph artifact is absent: {path}")
        if record.get("bytes") != path.stat().st_size:
            raise SemanticBridgeError(f"official graph artifact size mismatch: {path}")
        if record.get("sha256") != _sha256(path):
            raise SemanticBridgeError(f"official graph artifact hash mismatch: {path}")
        by_file[name] = record
    for required in ("graph_metadata.json", "parameter_mapping.json"):
        if required not in by_file:
            raise SemanticBridgeError(f"official graph package lacks {required}")
    if require_output_type is not None:
        required_file = {
            "torch": "model.torch.mlir",
            "tosa": "model.tosa.mlir",
            "linalg": "model.linalg.mlir",
        }.get(require_output_type)
        if required_file is None:
            raise SemanticBridgeError(
                f"unknown required MLIR output type {require_output_type!r}"
            )
        if required_file not in by_file:
            raise SemanticBridgeError(
                f"import package has no official {require_output_type} MLIR artifact"
            )
        imports = package.get("mlir_imports")
        if not isinstance(imports, list):
            raise SemanticBridgeError("import package has no MLIR importer records")
        selected = [
            item for item in imports
            if isinstance(item, dict)
            and item.get("output_type") == require_output_type
        ]
        if len(selected) != 1 or selected[0].get("importer_api") != OFFICIAL_IMPORTER:
            raise SemanticBridgeError(
                f"{require_output_type} MLIR was not produced by the official "
                "torch-mlir FX importer"
            )
    package["_artifact_records_by_file"] = by_file
    return package


def _graph_core(metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: metadata.get(key)
        for key in (
            "capture_format",
            "graph_signature",
            "range_constraints",
            "node_count",
            "operator_counts",
            "nodes",
        )
    }


def _replace_node_names(value: Any, indices: Mapping[str, int]) -> Any:
    if isinstance(value, list):
        return [_replace_node_names(item, indices) for item in value]
    if isinstance(value, dict):
        if set(value) == {"node"} and isinstance(value.get("node"), str):
            name = value["node"]
            if name not in indices:
                raise SemanticBridgeError(f"graph argument refers to unknown node {name!r}")
            return {"node_index": indices[name]}
        return {
            key: _replace_node_names(value[key], indices)
            for key in sorted(value)
        }
    return value


def _graph_equivalence_core(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Canonicalize process/version-local FX node names to graph indices."""

    core = _graph_core(metadata)
    nodes = core.get("nodes")
    if not isinstance(nodes, list):
        raise SemanticBridgeError("graph metadata nodes must be an array")
    indices: dict[str, int] = {}
    for ordinal, node in enumerate(nodes):
        if not isinstance(node, dict) or not isinstance(node.get("name"), str):
            raise SemanticBridgeError("graph metadata contains a malformed node")
        index = node.get("index")
        if index != ordinal:
            raise SemanticBridgeError("graph metadata node indices are not contiguous")
        indices[node["name"]] = ordinal
    canonical_nodes: list[dict[str, Any]] = []
    for ordinal, node in enumerate(nodes):
        canonical = {
            key: node[key]
            for key in sorted(node)
            if key not in {"name", "inputs", "arguments", "keyword_arguments"}
            and not (key == "value" and node.get("op") == "output")
        }
        inputs = node.get("inputs", [])
        if not isinstance(inputs, list) or any(name not in indices for name in inputs):
            raise SemanticBridgeError(f"node {ordinal} has malformed dependencies")
        canonical["input_indices"] = [indices[name] for name in inputs]
        canonical["arguments"] = _replace_node_names(node.get("arguments"), indices)
        canonical["keyword_arguments"] = _replace_node_names(
            node.get("keyword_arguments"), indices
        )
        canonical_nodes.append(canonical)
    return {
        key: core.get(key)
        for key in (
            "capture_format",
            "graph_signature",
            "range_constraints",
            "node_count",
            "operator_counts",
        )
    } | {"nodes": canonical_nodes}


def _load_legacy_frontend() -> Any:
    """Load the physically vendored static capability/model normalizer."""

    path = Path(__file__).with_name("plena_static_frontend.py")
    spec = importlib.util.spec_from_file_location("plena_static_frontend_for_bridge", path)
    if spec is None or spec.loader is None:
        raise SemanticBridgeError(f"cannot load PLENA semantic frontend module {path}")
    module = importlib.util.module_from_spec(spec)
    # Dataclasses consult sys.modules while decorating classes.
    import sys

    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as error:
        raise SemanticBridgeError(
            f"cannot initialize PLENA semantic frontend module: {type(error).__name__}: {error}"
        ) from error
    return module


def _shape(binding: Mapping[str, Any]) -> list[int] | None:
    tensor = binding.get("tensor")
    if not isinstance(tensor, dict):
        return None
    value = tensor.get("shape")
    if not isinstance(value, list) or any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        return None
    return value


def _expected_parameters(spec: Any) -> list[tuple[str, list[int]]]:
    h = spec.hidden_size
    qh = spec.query_hidden_size
    kvh = spec.kv_hidden_size
    i = spec.intermediate_size
    d = spec.head_dim
    result: list[tuple[str, list[int]]] = [
        ("model.embed_tokens.weight", [spec.vocab_size, h]),
    ]
    for layer in range(spec.num_layers):
        prefix = f"model.layers.{layer}"
        result.extend([
            (f"{prefix}.input_layernorm.weight", [h]),
            (f"{prefix}.post_attention_layernorm.weight", [h]),
        ])
        if spec.qk_head_norm:
            result.extend([
                (f"{prefix}.self_attn.q_norm.weight", [d]),
                (f"{prefix}.self_attn.k_norm.weight", [d]),
            ])
        result.extend([
            (f"{prefix}.self_attn.q_proj.weight", [qh, h]),
            (f"{prefix}.self_attn.k_proj.weight", [kvh, h]),
            (f"{prefix}.self_attn.v_proj.weight", [kvh, h]),
            (f"{prefix}.self_attn.o_proj.weight", [h, qh]),
            (f"{prefix}.mlp.gate_proj.weight", [i, h]),
            (f"{prefix}.mlp.up_proj.weight", [i, h]),
            (f"{prefix}.mlp.down_proj.weight", [h, i]),
        ])
    result.append(("model.norm.weight", [h]))
    if not spec.tie_word_embeddings:
        result.append(("lm_head.weight", [spec.vocab_size, h]))
    return result


def _expected_linear_sources(spec: Any) -> list[str]:
    result: list[str] = []
    for layer in range(spec.num_layers):
        prefix = f"model.layers.{layer}"
        result.extend([
            f"{prefix}.self_attn.q_proj.weight",
            f"{prefix}.self_attn.k_proj.weight",
            f"{prefix}.self_attn.v_proj.weight",
            f"{prefix}.self_attn.o_proj.weight",
            f"{prefix}.mlp.gate_proj.weight",
            f"{prefix}.mlp.up_proj.weight",
            f"{prefix}.mlp.down_proj.weight",
        ])
    result.append("model.embed_tokens.weight" if spec.tie_word_embeddings else "lm_head.weight")
    return result


def _node_map(metadata: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    nodes = metadata.get("nodes")
    if not isinstance(nodes, list):
        raise SemanticBridgeError("graph metadata nodes must be an array")
    result: dict[str, Mapping[str, Any]] = {}
    for node in nodes:
        if not isinstance(node, dict) or not isinstance(node.get("name"), str):
            raise SemanticBridgeError("graph metadata contains a malformed node")
        name = node["name"]
        if name in result:
            raise SemanticBridgeError(f"graph metadata repeats node {name!r}")
        if not isinstance(node.get("arguments"), dict):
            raise SemanticBridgeError(
                "graph metadata lacks semantic argument trees; recapture with the current frontend"
            )
        result[name] = node
    return result


def _tuple_arguments(node: Mapping[str, Any]) -> list[Any]:
    arguments = node.get("arguments")
    if not isinstance(arguments, dict) or not isinstance(arguments.get("tuple"), list):
        raise SemanticBridgeError(f"node {node.get('name')!r} has no tuple argument tree")
    return arguments["tuple"]


def _node_reference(value: Any) -> str | None:
    return value.get("node") if isinstance(value, dict) and isinstance(value.get("node"), str) else None


def _collect_node_references(value: Any) -> list[str]:
    reference = _node_reference(value)
    if reference is not None:
        return [reference]
    if isinstance(value, list):
        result: list[str] = []
        for item in value:
            result.extend(_collect_node_references(item))
        return result
    if isinstance(value, dict):
        result = []
        for item in value.values():
            result.extend(_collect_node_references(item))
        return result
    return []


def _validate_operator_contract(metadata: Mapping[str, Any], spec: Any) -> dict[str, Any]:
    nodes = _node_map(metadata)
    call_functions = [node for node in nodes.values() if node.get("op") == "call_function"]
    unknown = sorted({str(node.get("target")) for node in call_functions} - _ALLOWED_CALL_FUNCTIONS)
    if unknown:
        raise SemanticBridgeError(
            "official graph contains operations outside the dense-decoder legality set: "
            + ", ".join(unknown)
        )
    counts = metadata.get("operator_counts")
    if not isinstance(counts, dict):
        raise SemanticBridgeError("graph metadata operator_counts must be an object")
    norm_count = (4 if spec.qk_head_norm else 2) * spec.num_layers + 1
    expected_counts = {
        "aten::embedding": 1,
        "aten::linear": 7 * spec.num_layers + 1,
        "aten::silu": spec.num_layers,
        "aten::softmax.int": spec.num_layers,
        "aten::matmul": 2 * spec.num_layers + 1,
        "aten::mean.dim": norm_count,
        "aten::pow.Tensor_Scalar": norm_count,
        "aten::rsqrt": norm_count,
        "aten::cos": 1,
        "aten::sin": 1,
    }
    mismatches = {
        name: {"actual": counts.get(name, 0), "expected": expected}
        for name, expected in expected_counts.items()
        if counts.get(name, 0) != expected
    }
    if mismatches:
        raise SemanticBridgeError(
            "official graph does not have the required dense-decoder operator multiplicities: "
            + json.dumps(mismatches, sort_keys=True)
        )

    for node in call_functions:
        target = node.get("target")
        arguments = _tuple_arguments(node)
        if target == "aten::linear":
            if len(arguments) < 2 or (len(arguments) >= 3 and arguments[2] is not None):
                raise SemanticBridgeError(f"linear node {node['name']} has a bias or malformed ABI")
        elif target == "aten::softmax.int":
            if len(arguments) < 2 or arguments[1] != -1:
                raise SemanticBridgeError(f"softmax node {node['name']} does not reduce the last axis")
        elif target == "aten::pow.Tensor_Scalar":
            if len(arguments) < 2 or float(arguments[1]) != 2.0:
                raise SemanticBridgeError(f"RMS square node {node['name']} does not use exponent 2")
        elif target == "aten::mean.dim":
            if len(arguments) < 3 or arguments[1] not in ({"list": [-1]}, {"tuple": [-1]}) or arguments[2] is not True:
                raise SemanticBridgeError(f"RMS mean node {node['name']} is not keepdim over axis -1")

    mean_nodes = [node for node in call_functions if node.get("target") == "aten::mean.dim"]
    add_nodes = [node for node in call_functions if node.get("target") == "aten::add.Tensor"]
    rsqrt_nodes = [node for node in call_functions if node.get("target") == "aten::rsqrt"]
    epsilon_values: list[float] = []
    for mean in mean_nodes:
        matches = []
        for add in add_nodes:
            arguments = _tuple_arguments(add)
            if (
                len(arguments) >= 2
                and _node_reference(arguments[0]) == mean["name"]
                and isinstance(arguments[1], (int, float))
                and not isinstance(arguments[1], bool)
            ):
                matches.append(add)
        if len(matches) != 1:
            raise SemanticBridgeError(
                f"RMS mean node {mean['name']} does not feed one scalar epsilon add"
            )
        add = matches[0]
        epsilon = float(_tuple_arguments(add)[1])
        if not math.isclose(epsilon, spec.rms_norm_epsilon, rel_tol=0.0, abs_tol=1.0e-12):
            raise SemanticBridgeError(
                f"RMS epsilon in graph is {epsilon}, config requires {spec.rms_norm_epsilon}"
            )
        epsilon_values.append(epsilon)
        if sum(
            1
            for rsqrt in rsqrt_nodes
            if _node_reference(_tuple_arguments(rsqrt)[0]) == add["name"]
        ) != 1:
            raise SemanticBridgeError(f"RMS epsilon add {add['name']} does not feed one rsqrt")

    attention_scale = 1.0 / math.sqrt(spec.head_dim)
    scale_nodes = []
    for node in call_functions:
        if node.get("target") != "aten::mul.Tensor":
            continue
        arguments = _tuple_arguments(node)
        scalars = [
            float(item)
            for item in arguments
            if isinstance(item, (int, float)) and not isinstance(item, bool)
        ]
        if any(math.isclose(item, attention_scale, rel_tol=0.0, abs_tol=1.0e-12) for item in scalars):
            shape = node.get("value", {}).get("shape")
            if shape == [1, spec.query_heads, spec.sequence_length, spec.sequence_length]:
                scale_nodes.append(node)
    if len(scale_nodes) != spec.num_layers:
        raise SemanticBridgeError(
            "official graph does not apply exactly one 1/sqrt(head_dim) attention scale per layer"
        )

    output_nodes = [node for node in nodes.values() if node.get("op") == "output"]
    if len(output_nodes) != 1:
        raise SemanticBridgeError("official graph must have exactly one output node")
    output_references = _collect_node_references(output_nodes[0].get("arguments"))
    if len(output_references) != 1 or output_references[0] not in nodes:
        raise SemanticBridgeError("official graph output does not reference one tensor value")
    output_shape = nodes[output_references[0]].get("value", {}).get("shape")
    expected_output = [1, spec.sequence_length, spec.vocab_size]
    if output_shape != expected_output:
        raise SemanticBridgeError(
            f"official graph output shape is {output_shape!r}, expected {expected_output!r}"
        )
    return {
        "allowed_call_function_count": len(call_functions),
        "operator_multiplicities": expected_counts,
        "rms_epsilon": spec.rms_norm_epsilon,
        "attention_scale": attention_scale,
        "output_shape": expected_output,
        "decoder_dataflow": {
            "norm_placement": spec.norm_placement,
            "qk_head_norm": bool(spec.qk_head_norm),
            "attention_input": (
                "layer_hidden" if spec.norm_placement == "post"
                else "rms_norm(layer_hidden)"
            ),
            "attention_residual_rhs": (
                "rms_norm(attention_output)"
                if spec.norm_placement == "post" else "attention_output"
            ),
            "ffn_input": (
                "attention_residual" if spec.norm_placement == "post"
                else "rms_norm(attention_residual)"
            ),
            "ffn_residual_rhs": (
                "rms_norm(ffn_output)"
                if spec.norm_placement == "post" else "ffn_output"
            ),
        },
    }


def _validate_parameter_contract(
    metadata: Mapping[str, Any], mapping: Mapping[str, Any], spec: Any
) -> dict[str, Any]:
    if mapping.get("binding_contract") != "functional_call_external_state_v1":
        raise SemanticBridgeError("parameter mapping does not use the external-state ABI")
    if mapping.get("exported_program_state_dict_empty") is not True:
        raise SemanticBridgeError("ExportedProgram state_dict is not empty")
    bindings = mapping.get("bindings")
    if not isinstance(bindings, list):
        raise SemanticBridgeError("parameter mapping bindings must be an array")
    external = [
        binding
        for binding in bindings
        if isinstance(binding, dict) and str(binding.get("kind", "")).startswith("external_")
    ]
    by_source: dict[str, Mapping[str, Any]] = {}
    by_input: dict[str, Mapping[str, Any]] = {}
    for binding in external:
        source = binding.get("source_name")
        graph_input = binding.get("graph_input")
        if not isinstance(source, str) or not isinstance(graph_input, str):
            raise SemanticBridgeError("external state binding lacks source_name or graph_input")
        if source in by_source or graph_input in by_input:
            raise SemanticBridgeError(f"duplicate external binding for {source!r}")
        by_source[source] = binding
        by_input[graph_input] = binding

    external_parameters = [
        binding for binding in external
        if binding.get("kind") == "external_parameter"
    ]
    parameter_dtypes = {
        str(binding.get("tensor", {}).get("dtype"))
        for binding in external_parameters
        if isinstance(binding.get("tensor"), dict)
    }
    if len(parameter_dtypes) != 1 or not parameter_dtypes.issubset(
        {"float16", "bfloat16", "float32"}
    ):
        raise SemanticBridgeError(
            f"external parameters do not have one uniform source dtype: {sorted(parameter_dtypes)}"
        )

    nodes = _node_map(metadata)
    embedding_nodes = [node for node in nodes.values() if node.get("target") == "aten::embedding"]
    if len(embedding_nodes) != 1:
        raise SemanticBridgeError("dense decoder requires one embedding operation")
    embedding_input = _node_reference(_tuple_arguments(embedding_nodes[0])[0])
    embedding = by_input.get(str(embedding_input))
    if embedding is None or embedding.get("kind") != "external_parameter" or \
            _shape(embedding) != [spec.vocab_size, spec.hidden_size]:
        raise SemanticBridgeError(
            "embedding operation is not bound to one [vocab, hidden] parameter"
        )

    linear_nodes = sorted(
        (node for node in nodes.values() if node.get("target") == "aten::linear"),
        key=lambda node: int(node["index"]),
    )
    if len(linear_nodes) != 7 * spec.num_layers + 1:
        raise SemanticBridgeError("dense decoder linear count is not 7*layers+1")
    linear_bindings: list[Mapping[str, Any]] = []
    for node in linear_nodes:
        arguments = _tuple_arguments(node)
        weight_input = _node_reference(arguments[1]) if len(arguments) >= 2 else None
        binding = by_input.get(str(weight_input))
        if binding is None or binding.get("kind") != "external_parameter":
            raise SemanticBridgeError(f"linear node {node['name']} has an unbound weight")
        linear_bindings.append(binding)

    checkpoint_bindings: dict[str, str] = {}
    used_parameter_inputs = {str(embedding["graph_input"])}

    def bind_role(logical: str, binding: Mapping[str, Any], shape: list[int]) -> None:
        if _shape(binding) != shape:
            raise SemanticBridgeError(
                f"graph-derived parameter role {logical} has shape "
                f"{_shape(binding)!r}, expected {shape!r}"
            )
        source = binding.get("source_name")
        graph_input = binding.get("graph_input")
        if not isinstance(source, str) or not isinstance(graph_input, str):
            raise SemanticBridgeError(f"graph-derived role {logical} lacks source metadata")
        checkpoint_bindings[logical] = source
        used_parameter_inputs.add(graph_input)

    h, qh, kvh, intermediate = (
        spec.hidden_size, spec.query_hidden_size, spec.kv_hidden_size,
        spec.intermediate_size,
    )
    for layer in range(spec.num_layers):
        base = layer * 7
        roles = (
            ("q_weight", [qh, h]),
            ("k_weight", [kvh, h]),
            ("v_weight", [kvh, h]),
            ("o_weight", [h, qh]),
            ("gate_weight", [intermediate, h]),
            ("up_weight", [intermediate, h]),
            ("down_weight", [h, intermediate]),
        )
        for offset, (role, shape) in enumerate(roles):
            bind_role(
                f"layer_{layer:02d}_{role}",
                linear_bindings[base + offset], shape,
            )

    lm_head = linear_bindings[-1]
    bind_role(
        "token_embedding_lm_head_weight",
        embedding if spec.tie_word_embeddings else lm_head,
        [spec.vocab_size, h],
    )
    if spec.tie_word_embeddings:
        if lm_head.get("graph_input") != embedding.get("graph_input"):
            raise SemanticBridgeError(
                "tied model graph does not reuse the embedding parameter for LM head"
            )
    elif lm_head.get("graph_input") == embedding.get("graph_input"):
        raise SemanticBridgeError("untied model graph aliases embedding and LM head")

    direct_uses: dict[str, list[Mapping[str, Any]]] = {name: [] for name in by_input}
    for node in nodes.values():
        for input_name in node.get("inputs", []):
            if input_name in direct_uses:
                direct_uses[input_name].append(node)
    gamma_candidates: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    for binding in external_parameters:
        graph_input = str(binding["graph_input"])
        if graph_input in used_parameter_inputs:
            continue
        uses = direct_uses[graph_input]
        if len(uses) == 1 and uses[0].get("target") == "aten::mul.Tensor":
            gamma_candidates.append((binding, uses[0]))

    consumed_gamma_inputs: set[str] = set()

    def depends_on(node: Mapping[str, Any], ancestor_name: str) -> bool:
        pending = list(node.get("inputs", []))
        seen: set[str] = set()
        while pending:
            name = pending.pop()
            if name == ancestor_name:
                return True
            if name in seen:
                continue
            seen.add(name)
            upstream = nodes.get(name)
            if upstream is not None:
                pending.extend(upstream.get("inputs", []))
        return False

    def gamma_between(
        logical: str, lower: int, upper: int, shape: list[int],
        *, select: str = "only",
    ) -> Mapping[str, Any]:
        matches = sorted([
            (binding, int(use["index"]))
            for binding, use in gamma_candidates
            if lower < int(use["index"]) < upper and _shape(binding) == shape
            and str(binding["graph_input"]) not in consumed_gamma_inputs
        ], key=lambda item: item[1])
        if not matches or (select == "only" and len(matches) != 1):
            raise SemanticBridgeError(
                f"graph topology cannot identify one {logical} gamma between "
                f"operation indices {lower} and {upper}"
            )
        if select not in {"only", "first", "last"}:
            raise SemanticBridgeError(f"invalid gamma selection policy {select!r}")
        binding = matches[-1 if select == "last" else 0][0]
        consumed_gamma_inputs.add(str(binding["graph_input"]))
        bind_role(logical, binding, shape)
        return binding

    def gamma_from_linear(
        logical: str, linear: Mapping[str, Any], upper: int, shape: list[int],
        *, select: str = "only",
    ) -> Mapping[str, Any]:
        matches = sorted([
            (binding, int(use["index"]))
            for binding, use in gamma_candidates
            if int(linear["index"]) < int(use["index"]) < upper
            and _shape(binding) == shape
            and str(binding["graph_input"]) not in consumed_gamma_inputs
            and depends_on(use, str(linear["name"]))
        ], key=lambda item: item[1])
        if not matches or (select == "only" and len(matches) != 1):
            raise SemanticBridgeError(
                f"graph dataflow cannot identify one {logical} gamma from "
                f"linear {linear['name']}"
            )
        if select not in {"only", "first", "last"}:
            raise SemanticBridgeError(f"invalid gamma selection policy {select!r}")
        binding = matches[-1 if select == "last" else 0][0]
        consumed_gamma_inputs.add(str(binding["graph_input"]))
        bind_role(logical, binding, shape)
        return binding

    previous_down = -1
    for layer in range(spec.num_layers):
        q, k, v, o, gate, _up, down = linear_nodes[layer * 7:(layer + 1) * 7]
        if spec.norm_placement == "pre":
            gamma_between(
                f"layer_{layer:02d}_input_norm_gamma",
                previous_down, int(q["index"]), [h],
            )
        if spec.qk_head_norm:
            gamma_from_linear(
                f"layer_{layer:02d}_q_norm_gamma",
                q, int(o["index"]), [spec.head_dim],
            )
            gamma_from_linear(
                f"layer_{layer:02d}_k_norm_gamma",
                k, int(o["index"]), [spec.head_dim],
            )
        if spec.norm_placement == "post":
            gamma_from_linear(
                f"layer_{layer:02d}_input_norm_gamma",
                o, int(gate["index"]), [h],
            )
        else:
            gamma_between(
                f"layer_{layer:02d}_post_attention_norm_gamma",
                int(o["index"]), int(gate["index"]), [h],
            )
        if spec.norm_placement == "post":
            next_q_or_lm = (
                int(linear_nodes[(layer + 1) * 7]["index"])
                if layer + 1 < spec.num_layers
                else int(linear_nodes[-1]["index"])
            )
            gamma_from_linear(
                f"layer_{layer:02d}_post_attention_norm_gamma",
                down, next_q_or_lm, [h],
                select="first",
            )
        previous_down = int(down["index"])
    gamma_between("final_norm_gamma", previous_down,
                  int(linear_nodes[-1]["index"]), [h])

    if len(consumed_gamma_inputs) != len(gamma_candidates):
        unexpected = sorted(
            str(binding["source_name"])
            for binding, _use in gamma_candidates
            if str(binding["graph_input"]) not in consumed_gamma_inputs
        )
        raise SemanticBridgeError(
            f"external normalization parameters are not part of the supported topology: {unexpected}"
        )
    if used_parameter_inputs != {
        str(binding["graph_input"]) for binding in external_parameters
    }:
        raise SemanticBridgeError(
            "external parameter set contains values not assigned a graph-derived role"
        )

    rotary_candidates = [
        binding for binding in external
        if binding.get("kind") == "external_buffer"
        and _shape(binding) == [spec.head_dim // 2]
    ]
    if len(rotary_candidates) != 1:
        raise SemanticBridgeError("rotary inv_freq buffer ABI is absent or has the wrong shape")
    rotary = rotary_candidates[0]

    return {
        "external_parameter_count": len(external_parameters),
        "external_state_count": len(external),
        "external_parameter_dtype": next(iter(parameter_dtypes)),
        "linear_role_sequence_sha256": _sha256_bytes(
            _json_bytes([binding["source_name"] for binding in linear_bindings])
        ),
        "tied_embedding_lm_head": bool(spec.tie_word_embeddings),
        "rotary_buffer": rotary["source_name"],
        "checkpoint_bindings": checkpoint_bindings,
    }


def _infer_decoder_norm_features(
    metadata: Mapping[str, Any], mapping: Mapping[str, Any], spec: Any
) -> tuple[str, bool]:
    """Recover norm placement/QK head norm from graph use, not family names."""

    nodes = _node_map(metadata)
    linears = sorted(
        (node for node in nodes.values() if node.get("target") == "aten::linear"),
        key=lambda node: int(node["index"]),
    )
    if len(linears) < 3:
        raise SemanticBridgeError("decoder graph has fewer than Q/K/V linear operations")
    bindings = mapping.get("bindings")
    if not isinstance(bindings, list):
        raise SemanticBridgeError("parameter mapping bindings must be an array")
    parameter_by_input = {
        str(binding["graph_input"]): binding
        for binding in bindings
        if isinstance(binding, dict)
        and binding.get("kind") == "external_parameter"
        and isinstance(binding.get("graph_input"), str)
    }
    direct_uses: dict[str, list[Mapping[str, Any]]] = {
        graph_input: [] for graph_input in parameter_by_input
    }
    for node in nodes.values():
        for graph_input in node.get("inputs", []):
            if graph_input in direct_uses:
                direct_uses[graph_input].append(node)
    gamma_uses: list[tuple[Mapping[str, Any], int]] = []
    for graph_input, binding in parameter_by_input.items():
        uses = direct_uses[graph_input]
        if len(uses) == 1 and uses[0].get("target") == "aten::mul.Tensor":
            gamma_uses.append((binding, int(uses[0]["index"])))

    first_q = int(linears[0]["index"])
    hidden_pre_norms = [
        binding for binding, use_index in gamma_uses
        if _shape(binding) == [spec.hidden_size] and use_index < first_q
    ]
    if len(hidden_pre_norms) > 1:
        raise SemanticBridgeError(
            "graph has multiple hidden-width normalization parameters before first Q"
        )
    placement = "pre" if hidden_pre_norms else "post"

    first_o = int(linears[3]["index"])
    qk_gamma_count = sum(
        1 for binding, use_index in gamma_uses
        if _shape(binding) == [spec.head_dim] and first_q < use_index < first_o
    )
    if qk_gamma_count not in {0, 2}:
        raise SemanticBridgeError(
            "graph must contain either zero or two Q/K head-normalization gammas"
        )
    return placement, qk_gamma_count == 2


def certify_dense_decoder(
    capture_package: str | os.PathLike[str],
    import_package: str | os.PathLike[str],
    *,
    sequence_length: int,
    max_context: int | None = None,
    decode_buckets: Sequence[int] | None = None,
) -> SemanticCertificate:
    capture_dir = Path(capture_package).resolve()
    import_dir = Path(import_package).resolve()
    _verify_package(capture_dir)
    import_manifest = _verify_package(
        import_dir, require_output_type="torch"
    )
    capture_metadata = _read_json(capture_dir / "graph_metadata.json")
    import_metadata = _read_json(import_dir / "graph_metadata.json")
    capture_mapping = _read_json(capture_dir / "parameter_mapping.json")
    import_mapping = _read_json(import_dir / "parameter_mapping.json")
    capture_core = _graph_core(capture_metadata)
    import_core = _graph_core(import_metadata)
    capture_equivalence = _graph_equivalence_core(capture_metadata)
    import_equivalence = _graph_equivalence_core(import_metadata)
    if capture_equivalence != import_equivalence:
        raise SemanticBridgeError("capture and official import packages describe different graphs")
    if capture_mapping != import_mapping:
        raise SemanticBridgeError("capture and official import packages have different state ABIs")
    config_path = capture_dir / "huggingface_config.json"
    if not config_path.is_file():
        raise SemanticBridgeError("capture package lacks huggingface_config.json")

    legacy = _load_legacy_frontend()
    try:
        model_spec = legacy.load_hf_config(
            config_path,
            sequence_length=sequence_length,
            max_context=max_context,
        )
        norm_placement, qk_head_norm = _infer_decoder_norm_features(
            capture_core, capture_mapping, model_spec
        )
        model_spec = replace(
            model_spec,
            norm_placement=norm_placement,
            qk_head_norm=qk_head_norm,
        )
        capability_report = legacy.analyze_capabilities(model_spec, decode_buckets)
    except Exception as error:
        raise SemanticBridgeError(f"cannot specialize captured model: {error}") from error
    if not capability_report.supported:
        details = "; ".join(issue.message for issue in capability_report.errors)
        raise SemanticBridgeError(f"captured graph is outside target capabilities: {details}")

    source = _read_json(capture_dir / "source.json")
    if source.get("kind") != "huggingface_local_model":
        raise SemanticBridgeError("semantic bridge only accepts an actual local Hugging Face capture")
    if source.get("batch_size") != 1 or source.get("sequence_length") != sequence_length:
        raise SemanticBridgeError(
            "captured static batch/sequence does not match the requested specialization"
        )
    if source.get("model_type") != model_spec.source_model_type:
        raise SemanticBridgeError("captured model_type disagrees with the source configuration")
    if sorted(source.get("architectures", [])) != sorted(model_spec.architectures):
        raise SemanticBridgeError("captured architectures disagree with the source configuration")

    operator_evidence = _validate_operator_contract(capture_core, model_spec)
    parameter_evidence = _validate_parameter_contract(
        capture_core, capture_mapping, model_spec
    )
    torch_mlir_path = import_dir / "model.torch.mlir"
    torch_mlir_text = torch_mlir_path.read_text(encoding="utf-8")
    if (
        "official torch-mlir FX importer" not in torch_mlir_text
        or "torch." not in torch_mlir_text
    ):
        raise SemanticBridgeError(
            "official Torch MLIR artifact lacks importer provenance or Torch ops"
        )

    graph_sha = _sha256_bytes(_json_bytes(capture_equivalence))
    mapping_sha = _sha256_bytes(_json_bytes(capture_mapping))
    certificate: dict[str, Any] = {
        "schema_version": BRIDGE_SCHEMA_VERSION,
        "certificate_kind": BRIDGE_KIND,
        "recognizer": RECOGNIZER,
        "official_frontend": {
            "capture_format": "torch.export.ExportedProgram",
            "importer": OFFICIAL_IMPORTER,
            "fallback": None,
            "graph_core_sha256": graph_sha,
            "parameter_mapping_sha256": mapping_sha,
            "canonical_mlir": "torch",
            "torch_mlir_sha256": _sha256(torch_mlir_path),
            "torch_mlir_bytes": torch_mlir_path.stat().st_size,
        },
        "specialization": {
            "batch_size": 1,
            "sequence_length": model_spec.sequence_length,
            "max_context": model_spec.max_context,
            "decode_buckets": list(capability_report.decode_buckets),
            "model_type": model_spec.source_model_type,
            "architectures": list(model_spec.architectures),
        },
        "operator_evidence": operator_evidence,
        "parameter_evidence": parameter_evidence,
        "serving_boundary": {
            "embedding": "host_gather_to_hidden_state",
            "attention_mask": "external_additive_mask",
            "kv_cache": "materialized_target_state",
            "output": "last_token_logits",
        },
        "capability_report_sha256": _sha256_bytes(
            _json_bytes(capability_report.to_dict())
        ),
        "model_spec_sha256": _sha256_bytes(_json_bytes(model_spec.to_dict())),
    }
    # Ensure the import manifest itself cannot be silently replaced after the
    # artifact hashes above were verified.
    certificate["official_frontend"]["import_package_sha256"] = _sha256_bytes(
        _json_bytes({key: value for key, value in import_manifest.items() if not key.startswith("_")})
    )
    return SemanticCertificate(certificate, model_spec, capability_report)


def _run(command: Sequence[str], *, cwd: Path) -> None:
    result = subprocess.run(
        list(command),
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise SemanticBridgeError(
            f"command failed ({result.returncode}): {' '.join(command)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )


def emit_bridge_package(
    capture_package: str | os.PathLike[str],
    import_package: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    sequence_length: int,
    max_context: int | None = None,
    decode_buckets: Sequence[int] | None = None,
    etrinpu_opt: str | os.PathLike[str],
    etrinpu_compile: str | os.PathLike[str],
    compile_programs: bool = True,
    emit_debug_ir: bool = False,
) -> BridgePackageResult:
    certificate = certify_dense_decoder(
        capture_package,
        import_package,
        sequence_length=sequence_length,
        max_context=max_context,
        decode_buckets=decode_buckets,
    )
    legacy = _load_legacy_frontend()
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise SemanticBridgeError(f"output directory already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    files: dict[str, Path] = {}
    try:
        if emit_debug_ir:
            for source_name, destination_name in (
                ("model.torch.mlir", "official_model.torch.mlir"),
                ("model.tosa.mlir", "official_model.tosa.mlir"),
                ("model.linalg.mlir", "official_model.linalg.mlir"),
                ("graph_metadata.json", "official_graph_metadata.json"),
                ("parameter_mapping.json", "official_parameter_mapping.json"),
                ("huggingface_config.json", "huggingface_config.json"),
            ):
                source_root = (
                    Path(capture_package)
                    if source_name == "huggingface_config.json"
                    else Path(import_package)
                )
                source = source_root / source_name
                if not source.is_file():
                    continue
                target = staging / destination_name
                shutil.copyfile(source, target)
                files[destination_name.removesuffix(".json").removesuffix(".mlir")] = target

        certificate_path = staging / "semantic_certificate.json"
        _write_json(certificate_path, certificate.document)
        files["semantic_certificate"] = certificate_path
        model_spec_path = staging / "model_spec.json"
        report_path = staging / "capability_report.json"
        _write_json(model_spec_path, certificate.model_spec.to_dict())
        _write_json(report_path, certificate.capability_report.to_dict())
        files["model_spec"] = model_spec_path
        files["capability_report"] = report_path

        prefill_mlir = staging / "prefill.mlir"
        prefill_config = staging / "prefill_config.json"
        prefill_mlir.write_text(
            legacy.render_prefill_mlir(certificate.model_spec), encoding="utf-8", newline="\n"
        )
        checkpoint_bindings = dict(
            certificate.document["parameter_evidence"]["checkpoint_bindings"]
        )
        prefill_configuration = legacy.make_compiler_config(
            certificate.model_spec, "prefill"
        )
        prefill_configuration["checkpoint_bindings"] = checkpoint_bindings
        _write_json(prefill_config, prefill_configuration)
        files["prefill_mlir"] = prefill_mlir
        files["prefill_config"] = prefill_config
        for bucket in certificate.capability_report.decode_buckets:
            decode_mlir = staging / f"decode_b{bucket}.mlir"
            decode_config = staging / f"decode_b{bucket}_config.json"
            decode_mlir.write_text(
                legacy.render_decode_mlir(certificate.model_spec, bucket),
                encoding="utf-8",
                newline="\n",
            )
            decode_configuration = legacy.make_compiler_config(
                certificate.model_spec, "decode", bucket
            )
            decode_configuration["checkpoint_bindings"] = checkpoint_bindings
            _write_json(decode_config, decode_configuration)
            files[f"decode_b{bucket}_mlir"] = decode_mlir
            files[f"decode_b{bucket}_config"] = decode_config

        emit_chunked = (
            certificate.model_spec.max_context >
            certificate.model_spec.sequence_length
        )
        if emit_chunked:
            chunked_mlir = staging / "chunked_prefill.mlir"
            chunked_config = staging / "chunked_prefill_config.json"
            chunked_mlir.write_text(
                legacy.render_chunked_prefill_mlir(certificate.model_spec),
                encoding="utf-8", newline="\n",
            )
            configuration = legacy.make_compiler_config(
                certificate.model_spec, "chunked_prefill"
            )
            configuration["checkpoint_bindings"] = checkpoint_bindings
            _write_json(chunked_config, configuration)
            files["chunked_prefill_mlir"] = chunked_mlir
            files["chunked_prefill_config"] = chunked_config
            for bucket in certificate.capability_report.decode_buckets:
                shared_mlir = staging / f"chunked_decode_b{bucket}.mlir"
                shared_config = staging / f"chunked_decode_b{bucket}_config.json"
                shared_mlir.write_text(
                    legacy.render_shared_layout_decode_mlir(
                        certificate.model_spec, bucket
                    ),
                    encoding="utf-8", newline="\n",
                )
                configuration = legacy.make_compiler_config(
                    certificate.model_spec, "shared_layout_decode", bucket
                )
                configuration["checkpoint_bindings"] = checkpoint_bindings
                _write_json(shared_config, configuration)
                files[f"chunked_decode_b{bucket}_mlir"] = shared_mlir
                files[f"chunked_decode_b{bucket}_config"] = shared_config

        opt = Path(etrinpu_opt).resolve()
        compiler = Path(etrinpu_compile).resolve()
        if not opt.is_file():
            raise SemanticBridgeError(f"etrinpu-opt does not exist: {opt}")
        if not compiler.is_file():
            raise SemanticBridgeError(f"etrinpu-compile does not exist: {compiler}")
        _run([str(opt), str(prefill_mlir), "-o", os.devnull], cwd=staging)
        for bucket in certificate.capability_report.decode_buckets:
            _run(
                [str(opt), str(staging / f"decode_b{bucket}.mlir"),
                 "-o", os.devnull],
                cwd=staging,
            )
        if emit_chunked:
            _run([str(opt), str(staging / "chunked_prefill.mlir"),
                  "-o", os.devnull], cwd=staging)
            for bucket in certificate.capability_report.decode_buckets:
                _run(
                    [str(opt), str(staging / f"chunked_decode_b{bucket}.mlir"),
                     "-o", os.devnull], cwd=staging,
                )

        compilation_records: list[dict[str, Any]] = []
        if compile_programs:
            prefill_artifacts = staging / "prefill_artifacts"
            _run(
                [
                    str(compiler),
                    str(prefill_mlir),
                    "--config",
                    str(prefill_config),
                    "--output-dir",
                    str(prefill_artifacts),
                ],
                cwd=staging,
            )
            program = prefill_artifacts / "program_memory.bin"
            compilation_records.append({
                "mode": "prefill",
                "artifact_directory": prefill_artifacts.name,
                "program_bytes": program.stat().st_size,
                "program_sha256": _sha256(program),
            })
            files["prefill_artifacts"] = prefill_artifacts
            for bucket in certificate.capability_report.decode_buckets:
                artifact_dir = staging / f"decode_b{bucket}_artifacts"
                _run(
                    [
                        str(compiler),
                        str(staging / f"decode_b{bucket}.mlir"),
                        "--config",
                        str(staging / f"decode_b{bucket}_config.json"),
                        "--output-dir",
                        str(artifact_dir),
                    ],
                    cwd=staging,
                )
                program = artifact_dir / "program_memory.bin"
                compilation_records.append({
                    "mode": "decode",
                    "bucket": bucket,
                    "artifact_directory": artifact_dir.name,
                    "program_bytes": program.stat().st_size,
                    "program_sha256": _sha256(program),
                })
                files[f"decode_b{bucket}_artifacts"] = artifact_dir
            if emit_chunked:
                chunked_artifacts = staging / "chunked_prefill_artifacts"
                _run(
                    [str(compiler), str(staging / "chunked_prefill.mlir"),
                     "--config", str(staging / "chunked_prefill_config.json"),
                     "--output-dir", str(chunked_artifacts)], cwd=staging,
                )
                program = chunked_artifacts / "program_memory.bin"
                compilation_records.append({
                    "mode": "chunked_prefill",
                    "artifact_directory": chunked_artifacts.name,
                    "program_bytes": program.stat().st_size,
                    "program_sha256": _sha256(program),
                })
                files["chunked_prefill_artifacts"] = chunked_artifacts
                for bucket in certificate.capability_report.decode_buckets:
                    artifact_dir = staging / f"chunked_decode_b{bucket}_artifacts"
                    _run(
                        [str(compiler),
                         str(staging / f"chunked_decode_b{bucket}.mlir"),
                         "--config",
                         str(staging / f"chunked_decode_b{bucket}_config.json"),
                         "--output-dir", str(artifact_dir)], cwd=staging,
                    )
                    program = artifact_dir / "program_memory.bin"
                    compilation_records.append({
                        "mode": "chunked_layout_decode",
                        "bucket": bucket,
                        "artifact_directory": artifact_dir.name,
                        "program_bytes": program.stat().st_size,
                        "program_sha256": _sha256(program),
                    })
                    files[f"chunked_decode_b{bucket}_artifacts"] = artifact_dir

        artifact_records: list[dict[str, Any]] = []
        for logical_name, path in sorted(files.items()):
            if path.is_file():
                artifact_records.append({
                    "logical_name": logical_name,
                    "file": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                })
        manifest: dict[str, Any] = {
            "schema_version": BRIDGE_SCHEMA_VERSION,
            "package_kind": BRIDGE_KIND,
            "recognizer": RECOGNIZER,
            "official_graph_core_sha256": certificate.document["official_frontend"]["graph_core_sha256"],
            "semantic_certificate": certificate_path.name,
            "fallback": None,
            "lowering": "graph-certified semantic specialization",
            "compiled": compile_programs,
            "compilations": compilation_records,
            "artifacts": artifact_records,
        }
        manifest_path = staging / "package.json"
        _write_json(manifest_path, manifest)
        files["package"] = manifest_path
        os.replace(staging, destination)
        return BridgePackageResult(
            destination,
            {name: destination / path.name for name, path in files.items()},
            manifest,
        )
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def parse_decode_buckets(value: str | None) -> tuple[int, ...] | None:
    if value is None:
        return None
    if value.strip().lower() in {"", "none"}:
        return ()
    try:
        return tuple(int(item.strip(), 10) for item in value.split(","))
    except ValueError as error:
        raise SemanticBridgeError(f"invalid decode bucket list {value!r}") from error
