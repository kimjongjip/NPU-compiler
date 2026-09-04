#!/usr/bin/env python3
"""Official PyTorch graph frontend vendored for the PLENA compiler.

This module deliberately keeps the graph-import path separate from the legacy
Hugging Face ``config.json`` template generator.  It first captures an actual
``torch.nn.Module.forward`` with :func:`torch.export.export`, then hands that
exact :class:`torch.export.ExportedProgram` to torch-mlir's FX importer.

The production wrapper runs capture and torch-mlir import in one pinned Python
environment. Capture-only remains useful for inspecting or archiving a real
ExportedProgram; it never substitutes configuration-derived MLIR.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile
from typing import Any, Callable, Iterable, Mapping, Sequence
import warnings


SCHEMA_VERSION = 1
PACKAGE_KIND = "plena_torch_export_frontend"
_OUTPUT_TYPES = ("torch", "tosa", "linalg")
_TORCH_MLIR_OUTPUT_NAMES = {
    "torch": "torch",
    "tosa": "tosa",
    "linalg": "linalg-on-tensors",
}
_OUTPUT_FILES = {
    "torch": "model.torch.mlir",
    "tosa": "model.tosa.mlir",
    "linalg": "model.linalg.mlir",
}
_NORMALIZATION_POLICIES = ("plena-decoder-v1", "none")


class GraphFrontendError(RuntimeError):
    """Base class for graph-frontend failures."""


class DependencyUnavailableError(GraphFrontendError):
    """A required official frontend dependency is absent or incompatible."""


class CaptureError(GraphFrontendError):
    """PyTorch could not export the requested module."""


class TorchMlirImportError(GraphFrontendError):
    """torch-mlir could not import or lower the ExportedProgram."""


@dataclass(frozen=True)
class LoadedModel:
    """A module plus deterministic example inputs and source information."""

    module: Any
    args: tuple[Any, ...]
    kwargs: Mapping[str, Any]
    source: Mapping[str, Any]
    parameter_source_names: Mapping[str, str]
    source_artifacts: Mapping[str, Any]
    capture_context: Any | None = None


@dataclass(frozen=True)
class _ExternalState:
    module_name: str
    source_name: str
    kind: str
    aliases: tuple[str, ...]
    value: Any


@dataclass(frozen=True)
class CaptureResult:
    """The official PyTorch capture and its stable, value-free descriptions."""

    exported_program: Any
    graph_metadata: Mapping[str, Any]
    parameter_mapping: Mapping[str, Any]
    source: Mapping[str, Any]
    source_artifacts: Mapping[str, Any]
    pre_normalization_graph_metadata: Mapping[str, Any]


@dataclass(frozen=True)
class MlirImportResult:
    """One torch-mlir result and the API path used to produce it."""

    output_type: str
    text: str
    importer_api: str
    torch_mlir_output_type: str


@dataclass(frozen=True)
class PackageResult:
    """Paths and metadata for one emitted graph-frontend package."""

    output_dir: Path
    files: Mapping[str, Path]
    manifest: Mapping[str, Any]


@dataclass(frozen=True)
class _TorchMlirAdapter:
    version: str
    module: Any
    public_import: Callable[..., Any] | None
    public_signature: str | None
    low_level_importer: type[Any] | None


@contextmanager
def _frontend_warning_scope() -> Iterable[None]:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"`isinstance\(treespec, LeafSpec\)` is deprecated.*",
            category=FutureWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r"While compiling, we found certain side effects happened in the model\.forward.*",
            category=UserWarning,
        )
        yield


def _require_torch() -> Any:
    try:
        torch = importlib.import_module("torch")
    except Exception as error:  # ImportError does not cover shared-library errors.
        raise DependencyUnavailableError(
            "PyTorch with torch.export is required for graph capture; "
            f"importing 'torch' failed: {type(error).__name__}: {error}"
        ) from error
    if not hasattr(torch, "export") or not callable(getattr(torch.export, "export", None)):
        raise DependencyUnavailableError(
            f"installed PyTorch {getattr(torch, '__version__', 'unknown')} has no "
            "torch.export.export API"
        )
    return torch


def _package_version(distribution: str, module: Any) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return str(getattr(module, "__version__", "unknown"))


def _discover_torch_mlir() -> _TorchMlirAdapter:
    """Feature-detect the current torch-mlir FX APIs without guessing a version."""

    try:
        torch_mlir = importlib.import_module("torch_mlir")
    except Exception as error:
        raise DependencyUnavailableError(
            "torch-mlir is required to emit MLIR, but importing 'torch_mlir' "
            f"failed: {type(error).__name__}: {error}. Use --capture-only to "
            "emit an ExportedProgram, or run this frontend in the pinned "
            "torch-mlir toolchain environment. No config-template fallback was used."
        ) from error

    public_import: Callable[..., Any] | None = None
    public_signature: str | None = None
    try:
        fx_module = importlib.import_module("torch_mlir.fx")
        candidate = getattr(fx_module, "export_and_import", None)
        if callable(candidate):
            public_import = candidate
            try:
                public_signature = str(inspect.signature(candidate))
            except (TypeError, ValueError):
                public_signature = "unknown"
    except Exception:
        # A low-level FxImporter is still an official FX import path for raw
        # Torch dialect IR, so probe it below and diagnose only if both are gone.
        pass

    low_level_importer: type[Any] | None = None
    try:
        extras = importlib.import_module("torch_mlir.extras.fx_importer")
        candidate = getattr(extras, "FxImporter", None)
        if isinstance(candidate, type) and (
            callable(getattr(candidate, "import_program", None))
            or callable(getattr(candidate, "import_frozen_program", None))
        ):
            low_level_importer = candidate
    except Exception:
        pass

    if public_import is None and low_level_importer is None:
        raise DependencyUnavailableError(
            "the installed torch-mlir package exposes neither "
            "torch_mlir.fx.export_and_import nor extras.fx_importer.FxImporter; "
            "install a torch-mlir build with the official FX importer"
        )

    return _TorchMlirAdapter(
        version=_package_version("torch-mlir", torch_mlir),
        module=torch_mlir,
        public_import=public_import,
        public_signature=public_signature,
        low_level_importer=low_level_importer,
    )


def detect_toolchain(*, probe_torch_mlir: bool = True) -> dict[str, Any]:
    """Return a machine-readable dependency report.

    Missing torch-mlir is represented as data here.  Operations that request
    MLIR use :func:`import_exported_program` and raise a hard error instead.
    """

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "python": platform.python_version(),
        "torch": {"available": False},
        "torch_mlir": {"available": False},
    }
    try:
        torch = _require_torch()
        report["torch"] = {
            "available": True,
            "version": str(getattr(torch, "__version__", "unknown")),
            "exported_program": hasattr(torch.export, "ExportedProgram"),
        }
    except DependencyUnavailableError as error:
        report["torch"]["diagnostic"] = str(error)
        return report

    if not probe_torch_mlir:
        report["torch_mlir"] = {
            "available": None,
            "probed": False,
            "diagnostic": (
                "not imported because capture-only mode stops before the "
                "official torch-mlir import"
            ),
        }
        return report

    try:
        adapter = _discover_torch_mlir()
        report["torch_mlir"] = {
            "available": True,
            "version": adapter.version,
            "fx_export_and_import": adapter.public_import is not None,
            "fx_export_and_import_signature": adapter.public_signature,
            "fx_importer": adapter.low_level_importer is not None,
            "requested_output_types": list(_OUTPUT_TYPES),
        }
    except DependencyUnavailableError as error:
        report["torch_mlir"]["diagnostic"] = str(error)
    return report


def _enum_name(value: Any) -> str:
    name = getattr(value, "name", None)
    if isinstance(name, str):
        return name.lower()
    text = str(value)
    return text.rsplit(".", 1)[-1].lower()


def _symbolic_text(value: Any) -> str | int:
    try:
        return int(value)
    except (TypeError, ValueError, RuntimeError):
        return str(value)


def _target_name(target: Any) -> str:
    if isinstance(target, str):
        return target
    schema = getattr(target, "_schema", None)
    if schema is not None:
        name = getattr(schema, "name", None)
        overload = getattr(schema, "overload_name", None)
        if isinstance(name, str):
            return f"{name}.{overload}" if overload else name
    module = getattr(target, "__module__", None)
    qualname = getattr(target, "__qualname__", None) or getattr(target, "__name__", None)
    if isinstance(qualname, str):
        return f"{module}.{qualname}" if isinstance(module, str) else qualname
    # Avoid repr(), which frequently contains a process-specific address.
    return f"{type(target).__module__}.{type(target).__qualname__}"


def _tensor_descriptor(value: Any, torch: Any) -> dict[str, Any] | None:
    if not isinstance(value, torch.Tensor):
        return None
    descriptor: dict[str, Any] = {
        "shape": [_symbolic_text(dimension) for dimension in value.shape],
        "dtype": str(value.dtype).removeprefix("torch."),
        "device": str(value.device),
        "requires_grad": bool(value.requires_grad),
    }
    try:
        descriptor["stride"] = [_symbolic_text(item) for item in value.stride()]
    except (RuntimeError, TypeError):
        pass
    return descriptor


def _value_descriptor(value: Any, torch: Any) -> Any:
    tensor = _tensor_descriptor(value, torch)
    if tensor is not None:
        return {"kind": "tensor", **tensor}
    if isinstance(value, (tuple, list)):
        return {
            "kind": "tuple" if isinstance(value, tuple) else "list",
            "items": [_value_descriptor(item, torch) for item in value],
        }
    if value is None or isinstance(value, (bool, int, float, str)):
        return {"kind": type(value).__name__.lower(), "value": value}
    if type(value).__name__ in {"SymInt", "SymFloat", "SymBool"}:
        return {"kind": type(value).__name__.lower(), "expression": str(value)}
    return {"kind": f"{type(value).__module__}.{type(value).__qualname__}"}


def _argument_descriptor(argument: Any) -> dict[str, Any]:
    result = {"type": type(argument).__name__}
    name = getattr(argument, "name", None)
    if isinstance(name, str):
        result["name"] = name
    if hasattr(argument, "value"):
        value = argument.value
        if value is None or isinstance(value, (bool, int, float, str)):
            result["value"] = value
        else:
            result["value_type"] = f"{type(value).__module__}.{type(value).__qualname__}"
    return result


def _signature_spec(spec: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "kind": _enum_name(spec.kind),
        "argument": _argument_descriptor(spec.arg),
    }
    target = getattr(spec, "target", None)
    if target is not None:
        result["target"] = str(target)
    persistent = getattr(spec, "persistent", None)
    if persistent is not None:
        result["persistent"] = bool(persistent)
    return result


def _graph_signature_metadata(exported_program: Any) -> dict[str, Any]:
    signature = exported_program.graph_signature
    result: dict[str, Any] = {
        "inputs": [_signature_spec(spec) for spec in signature.input_specs],
        "outputs": [_signature_spec(spec) for spec in signature.output_specs],
    }
    for attribute in (
        "inputs_to_parameters",
        "inputs_to_buffers",
        "inputs_to_lifted_tensor_constants",
        "buffers_to_mutate",
        "user_inputs_to_mutate",
    ):
        value = getattr(signature, attribute, None)
        if isinstance(value, Mapping):
            result[attribute] = {str(key): str(value[key]) for key in sorted(value, key=str)}
    for attribute in ("user_inputs", "user_outputs"):
        value = getattr(signature, attribute, None)
        if value is not None:
            result[attribute] = [str(item) for item in value]
    return result


def _node_dependencies(value: Any, torch: Any) -> list[str]:
    node_type = torch.fx.Node
    if isinstance(value, node_type):
        return [value.name]
    if isinstance(value, (tuple, list)):
        result: list[str] = []
        for item in value:
            result.extend(_node_dependencies(item, torch))
        return result
    if isinstance(value, Mapping):
        result = []
        for key in sorted(value, key=str):
            result.extend(_node_dependencies(value[key], torch))
        return result
    return []


def _graph_argument_tree(value: Any, torch: Any) -> Any:
    """Serialize FX arguments without tensor payloads.

    Dependencies alone are sufficient for graph visualization but not for a
    semantic legalization boundary: a compiler must also distinguish, for
    example, ``softmax(dim=-1)`` from another reduction axis and record the
    epsilon used by RMSNorm.  Keep node references symbolic and retain only
    scalar/container attributes; tensor values remain excluded.
    """

    node_type = torch.fx.Node
    if isinstance(value, node_type):
        return {"node": value.name}
    if isinstance(value, tuple):
        return {"tuple": [_graph_argument_tree(item, torch) for item in value]}
    if isinstance(value, list):
        return {"list": [_graph_argument_tree(item, torch) for item in value]}
    if isinstance(value, Mapping):
        return {
            "dict": [
                {
                    "key": str(key),
                    "value": _graph_argument_tree(value[key], torch),
                }
                for key in sorted(value, key=str)
            ]
        }
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if type(value).__name__ in {"SymInt", "SymFloat", "SymBool"}:
        return {"symbolic": str(value), "type": type(value).__name__}
    tensor = _tensor_descriptor(value, torch)
    if tensor is not None:
        return {"tensor_descriptor": tensor}
    return {"type": f"{type(value).__module__}.{type(value).__qualname__}"}


def graph_metadata(exported_program: Any) -> dict[str, Any]:
    """Create a deterministic, tensor-value-free description of an export."""

    torch = _require_torch()
    nodes: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for index, node in enumerate(exported_program.graph.nodes):
        target = _target_name(node.target)
        counts[target] += 1
        node_info: dict[str, Any] = {
            "index": index,
            "name": node.name,
            "op": node.op,
            "target": target,
            "inputs": _node_dependencies((node.args, node.kwargs), torch),
            "arguments": _graph_argument_tree(node.args, torch),
            "keyword_arguments": _graph_argument_tree(node.kwargs, torch),
        }
        if "val" in node.meta:
            node_info["value"] = _value_descriptor(node.meta["val"], torch)
        nodes.append(node_info)

    constraints = []
    for expression, value_range in sorted(
        getattr(exported_program, "range_constraints", {}).items(), key=lambda item: str(item[0])
    ):
        constraints.append({"expression": str(expression), "range": str(value_range)})

    return {
        "schema_version": SCHEMA_VERSION,
        "capture_format": "torch.export.ExportedProgram",
        "graph_signature": _graph_signature_metadata(exported_program),
        "range_constraints": constraints,
        "node_count": len(nodes),
        "operator_counts": {key: counts[key] for key in sorted(counts)},
        "nodes": nodes,
    }


def normalize_exported_program(
    exported_program: Any,
    policy: str = "plena-decoder-v1",
) -> tuple[Any, dict[str, Any]]:
    """Apply a named, auditable PyTorch decomposition policy.

    Hugging Face causal-mask construction commonly exports
    ``aten.__and__.Tensor``.  It is semantically boolean logical-and for that
    graph, but torch-mlir's TOSA/Linalg paths do not legalize the generic
    bitwise spelling.  The public ExportedProgram decomposition API rewrites it
    to ``aten.logical_and`` before any MLIR import.  This policy is deliberately
    small and recorded in every artifact; ``none`` is an explicit opt-out.
    """

    if policy not in _NORMALIZATION_POLICIES:
        raise GraphFrontendError(
            f"unknown normalization policy {policy!r}; expected "
            f"{', '.join(_NORMALIZATION_POLICIES)}"
        )
    before_metadata = graph_metadata(exported_program)
    before_abi = _user_input_abi(exported_program)
    record: dict[str, Any] = {
        "policy": policy,
        "api": "torch.export.ExportedProgram.run_decompositions",
        "decompositions": [],
        "input_abi": {
            "verified_unchanged": True,
            "before": before_abi,
            "after": before_abi,
        },
        "graphs": {
            "before": {
                "node_count": before_metadata["node_count"],
                "metadata_sha256": _metadata_sha256(before_metadata),
            },
        },
    }
    if policy == "none":
        record["reapplied_noop"] = False
        record["graphs"]["after"] = dict(record["graphs"]["before"])
        return exported_program, record

    torch = _require_torch()
    source_names = {"aten::__and__.Tensor", "aten::__and__.Tensor.Tensor"}
    source_nodes = [
        node
        for node in exported_program.graph.nodes
        if _target_name(node.target) in source_names
    ]
    source_count_before = len(source_nodes)
    occurrence_dtypes: list[dict[str, Any]] = []
    for node in source_nodes:
        tensor_arguments = [
            argument
            for argument in node.args
            if isinstance(argument, torch.fx.Node)
            and isinstance(argument.meta.get("val"), torch.Tensor)
        ]
        operand_dtypes = [str(argument.meta["val"].dtype).removeprefix("torch.") for argument in tensor_arguments]
        output_value = node.meta.get("val")
        output_dtype = (
            str(output_value.dtype).removeprefix("torch.")
            if isinstance(output_value, torch.Tensor)
            else None
        )
        occurrence_dtypes.append(
            {
                "node": node.name,
                "operand_dtypes": operand_dtypes,
                "output_dtype": output_dtype,
            }
        )
        non_boolean = [
            argument.name
            for argument in tensor_arguments
            if argument.meta["val"].dtype is not torch.bool
        ]
        if len(tensor_arguments) != 2 or non_boolean or output_dtype != "bool":
            raise CaptureError(
                "normalization refuses to rewrite non-boolean aten.__and__.Tensor "
                f"at node {node.name!r}; operand_dtypes={operand_dtypes}, "
                f"output_dtype={output_dtype!r}"
            )
    try:
        source_op = torch.ops.aten.__and__.Tensor

        def tensor_and_to_logical_and(left: Any, right: Any) -> Any:
            return torch.logical_and(left, right)

        # Avoid re-running PyTorch's surrounding core decomposition machinery
        # when a serialized package is already normalized.
        if source_count_before:
            with _frontend_warning_scope():
                exported_program = exported_program.run_decompositions(
                    {source_op: tensor_and_to_logical_and}
                )
    except Exception as error:
        raise CaptureError(
            "normalization policy plena-decoder-v1 failed while rewriting "
            f"aten.__and__.Tensor to aten.logical_and: {type(error).__name__}: {error}"
        ) from error
    source_count_after = sum(
        1
        for node in exported_program.graph.nodes
        if _target_name(node.target) in source_names
    )
    record["reapplied_noop"] = source_count_before == 0
    record["decompositions"].append(
        {
            "name": "causal-mask-tensor-and-to-logical-and",
            "source": "aten.__and__.Tensor",
            "replacement": "aten.logical_and.default",
            "api": "torch.export.ExportedProgram.run_decompositions",
            "reason": "torch-mlir TOSA/Linalg boolean-mask legalization",
            "source_op_count_before": source_count_before,
            "source_op_count_after": source_count_after,
            "applied": source_count_before > 0,
            "dtype_guard": "all tensor operands are torch.bool",
            "occurrences": occurrence_dtypes,
        }
    )
    if source_count_before and source_count_after:
        raise CaptureError(
            "normalization did not eliminate every aten.__and__.Tensor operation"
        )
    after_metadata = graph_metadata(exported_program)
    after_abi = _user_input_abi(exported_program)
    record["input_abi"]["after"] = after_abi
    if before_abi != after_abi:
        record["input_abi"]["verified_unchanged"] = False
        raise CaptureError(
            "normalization changed the ordered USER_INPUT kind/shape/dtype ABI; "
            "refusing to reuse the external parameter mapping"
        )
    record["graphs"]["after"] = {
        "node_count": after_metadata["node_count"],
        "metadata_sha256": _metadata_sha256(after_metadata),
    }
    return exported_program, record


def _metadata_sha256(metadata: Mapping[str, Any]) -> str:
    return hashlib.sha256(_json_text(metadata).encode("utf-8")).hexdigest()


def _user_input_abi(exported_program: Any) -> list[dict[str, Any]]:
    """Snapshot the ordered USER_INPUT contract around graph normalization."""

    torch = _require_torch()
    placeholders = {
        node.name: node
        for node in exported_program.graph.nodes
        if node.op == "placeholder"
    }
    result: list[dict[str, Any]] = []
    for ordinal, spec in enumerate(exported_program.graph_signature.input_specs):
        if _enum_name(spec.kind) != "user_input":
            continue
        name = getattr(spec.arg, "name", None)
        value = placeholders.get(name).meta.get("val") if name in placeholders else None
        descriptor = _tensor_descriptor(value, torch)
        contract: dict[str, Any] = {"kind": "user_input"}
        if descriptor is not None:
            contract["shape"] = descriptor["shape"]
            contract["dtype"] = descriptor["dtype"]
        else:
            contract["value"] = _value_descriptor(value, torch)
        result.append(
            {
                "ordinal": ordinal,
                "graph_input": str(name),
                "contract": contract,
            }
        )
    return result


def parameter_mapping(
    exported_program: Any,
    parameter_source_names: Mapping[str, str] | None = None,
    external_state: Sequence[_ExternalState] = (),
) -> dict[str, Any]:
    """Describe every lifted parameter/buffer and its external source name."""

    torch = _require_torch()
    source_names = dict(parameter_source_names or {})
    state_dict = exported_program.state_dict
    constants = getattr(exported_program, "constants", {})
    bindings: list[dict[str, Any]] = []
    targets_to_inputs: dict[str, list[str]] = {}

    # Parameters externalized with torch.func.functional_call intentionally
    # appear as USER_INPUT, not PARAMETER.  Their order is the first flattened
    # tuple argument supplied to torch.export.  Keep that ABI explicit.
    user_input_specs = [
        spec
        for spec in exported_program.graph_signature.input_specs
        if _enum_name(spec.kind) == "user_input"
    ]
    if len(user_input_specs) < len(external_state):
        raise CaptureError(
            "torch.export graph signature has fewer USER_INPUT entries than the "
            "external parameter tuple"
        )
    for argument_index, (state, spec) in enumerate(zip(external_state, user_input_specs)):
        graph_input = getattr(spec.arg, "name", None)
        if not isinstance(graph_input, str):
            raise CaptureError(
                f"external state argument {argument_index} has no tensor graph-input name"
            )
        tensor = _tensor_descriptor(state.value, torch)
        binding: dict[str, Any] = {
            "graph_input": graph_input,
            "kind": f"external_{state.kind}",
            "argument_index": argument_index,
            "module_name": state.module_name,
            "source_name": state.source_name,
            "aliases": list(state.aliases),
        }
        if tensor is not None:
            binding["tensor"] = tensor
        bindings.append(binding)

    for spec in exported_program.graph_signature.input_specs:
        kind = _enum_name(spec.kind)
        if kind not in {"parameter", "buffer", "constant_tensor"}:
            continue
        graph_input = getattr(spec.arg, "name", None)
        target = getattr(spec, "target", None)
        if not isinstance(graph_input, str) or target is None:
            continue
        target = str(target)
        value = state_dict.get(target)
        if value is None and isinstance(constants, Mapping):
            value = constants.get(target)
        binding: dict[str, Any] = {
            "graph_input": graph_input,
            "kind": kind,
            "program_target": target,
            "source_name": source_names.get(target, target),
        }
        persistent = getattr(spec, "persistent", None)
        if persistent is not None:
            binding["persistent"] = bool(persistent)
        tensor = _tensor_descriptor(value, torch)
        if tensor is not None:
            binding["tensor"] = tensor
        bindings.append(binding)
        targets_to_inputs.setdefault(target, []).append(graph_input)

    state_entries = []
    for state in external_state:
        entry: dict[str, Any] = {
            "program_target": state.module_name,
            "source_name": state.source_name,
            "aliases": list(state.aliases),
            "graph_inputs": [
                binding["graph_input"]
                for binding in bindings
                if binding.get("module_name") == state.module_name
            ],
            "kind": state.kind,
        }
        tensor = _tensor_descriptor(state.value, torch)
        if tensor is not None:
            entry["tensor"] = tensor
        state_entries.append(entry)
    for target in sorted(state_dict):
        value = state_dict[target]
        entry: dict[str, Any] = {
            "program_target": target,
            "source_name": source_names.get(target, target),
            "graph_inputs": sorted(targets_to_inputs.get(target, [])),
        }
        tensor = _tensor_descriptor(value, torch)
        if tensor is not None:
            entry["tensor"] = tensor
        state_entries.append(entry)

    return {
        "schema_version": SCHEMA_VERSION,
        "binding_contract": (
            "functional_call_external_state_v1"
            if external_state
            else "torch_export_graph_signature"
        ),
        "parameters_externalized": bool(external_state),
        "exported_program_state_dict_empty": len(state_dict) == 0,
        "bindings": bindings,
        "state_entries": state_entries,
        "binding_count": len(bindings),
        "state_entry_count": len(state_entries),
    }


def capture_program(
    module: Any,
    args: Sequence[Any],
    kwargs: Mapping[str, Any] | None = None,
    *,
    dynamic_shapes: Any = None,
    strict: bool = True,
    source: Mapping[str, Any] | None = None,
    parameter_source_names: Mapping[str, str] | None = None,
    source_artifacts: Mapping[str, Any] | None = None,
    externalize_state: bool = True,
    normalization_policy: str = "plena-decoder-v1",
) -> CaptureResult:
    """Capture ``module.forward`` using the public ``torch.export`` API."""

    torch = _require_torch()
    if not isinstance(module, torch.nn.Module):
        raise CaptureError(
            f"capture_program requires torch.nn.Module, got {type(module).__qualname__}"
        )
    positional = tuple(args)
    keyword = {key: (kwargs or {})[key] for key in sorted(kwargs or {})}
    export_module = module
    external_state: tuple[_ExternalState, ...] = ()
    if externalize_state:
        export_module, positional, keyword, external_state = _externalize_module_state(
            module,
            positional,
            keyword,
            parameter_source_names or {},
        )
    export_signature = inspect.signature(torch.export.export)
    export_kwargs: dict[str, Any] = {}
    if "strict" in export_signature.parameters:
        export_kwargs["strict"] = strict
    elif not strict:
        raise CaptureError("installed torch.export.export cannot request strict=False")
    if dynamic_shapes is not None:
        if "dynamic_shapes" not in export_signature.parameters:
            raise CaptureError("installed torch.export.export has no dynamic_shapes support")
        export_kwargs["dynamic_shapes"] = dynamic_shapes

    was_training = bool(module.training)
    module.eval()
    export_module.eval()
    try:
        with _frontend_warning_scope():
            exported_program = torch.export.export(
                export_module, positional, keyword, **export_kwargs
            )
    except Exception as error:
        raise CaptureError(
            "torch.export.export failed while capturing the actual forward graph: "
            f"{type(error).__name__}: {error}"
        ) from error
    finally:
        module.train(was_training)

    exported_type = getattr(torch.export, "ExportedProgram", None)
    if exported_type is not None and not isinstance(exported_program, exported_type):
        raise CaptureError(
            "torch.export.export returned an unexpected object instead of ExportedProgram"
        )

    pre_normalization_metadata = graph_metadata(exported_program)
    exported_program, normalization = normalize_exported_program(
        exported_program, normalization_policy
    )
    metadata = graph_metadata(exported_program)
    metadata["normalization"] = normalization
    return CaptureResult(
        exported_program=exported_program,
        graph_metadata=metadata,
        parameter_mapping=parameter_mapping(
            exported_program, parameter_source_names, external_state
        ),
        source=dict(source or {"kind": "python_module", "module_type": type(module).__qualname__}),
        source_artifacts=dict(source_artifacts or {}),
        pre_normalization_graph_metadata=pre_normalization_metadata,
    )


def _externalize_module_state(
    module: Any,
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
    parameter_source_names: Mapping[str, str],
) -> tuple[Any, tuple[Any, ...], Mapping[str, Any], tuple[_ExternalState, ...]]:
    """Turn module state into ordered USER_INPUT tensors via functional_call.

    This is the critical scalability contract: torch-mlir's normal frozen
    import embeds parameters as tensor literals.  A 4B checkpoint cannot be
    represented that way.  Keeping the base module unregistered and invoking
    it functionally makes every weight/buffer an ordinary graph argument while
    preserving the exact captured operation graph.
    """

    torch = _require_torch()
    state: list[_ExternalState] = []

    def collect(kind: str, iterator: Iterable[tuple[str, Any]], aliases: Iterable[tuple[str, Any]]) -> None:
        alias_map: dict[int, list[str]] = {}
        for alias_name, value in aliases:
            alias_map.setdefault(id(value), []).append(alias_name)
        for name, value in sorted(iterator, key=lambda item: item[0]):
            all_aliases = tuple(
                parameter_source_names.get(alias, alias)
                for alias in sorted(alias_map.get(id(value), [name]))
            )
            state.append(
                _ExternalState(
                    module_name=name,
                    source_name=parameter_source_names.get(name, name),
                    kind=kind,
                    aliases=all_aliases,
                    value=value,
                )
            )

    collect(
        "parameter",
        module.named_parameters(remove_duplicate=True),
        module.named_parameters(remove_duplicate=False),
    )
    collect(
        "buffer",
        module.named_buffers(remove_duplicate=True),
        module.named_buffers(remove_duplicate=False),
    )
    external_state = tuple(state)
    state_names = tuple(item.module_name for item in external_state)
    state_values = tuple(item.value for item in external_state)

    class FunctionalStateWrapper(torch.nn.Module):
        def __init__(self, base: Any) -> None:
            super().__init__()
            # nn.Module.__setattr__ would register ``base`` and torch.export
            # would classify its tensors as PARAMETER again.
            object.__setattr__(self, "_base_unregistered", base)

        def forward(self, flat_state: Any, *user_args: Any, **user_kwargs: Any) -> Any:
            replacements = {
                name: value for name, value in zip(state_names, flat_state)
            }
            return torch.func.functional_call(
                self._base_unregistered,
                replacements,
                user_args,
                user_kwargs,
                tie_weights=True,
                strict=False,
            )

    wrapper = FunctionalStateWrapper(module)
    return wrapper, (state_values, *args), dict(kwargs), external_state


def load_exported_program(
    path: str | os.PathLike[str],
    *,
    parameter_mapping_path: str | os.PathLike[str] | None = None,
    normalization_policy: str = "plena-decoder-v1",
) -> CaptureResult:
    """Load a previously captured official ``.pt2`` artifact."""

    torch = _require_torch()
    source_path = Path(path)
    if not source_path.is_file():
        raise CaptureError(f"ExportedProgram does not exist: {source_path}")
    loader = getattr(torch.export, "load", None)
    if not callable(loader):
        raise DependencyUnavailableError("installed PyTorch has no torch.export.load API")
    embedded_files = {
        "plena/parameter_mapping.json": "",
        "plena/normalization.json": "",
        "plena/source.json": "",
    }
    try:
        with _frontend_warning_scope():
            exported_program = loader(source_path, extra_files=embedded_files)
    except Exception as error:
        raise CaptureError(
            f"could not load ExportedProgram {source_path}: {type(error).__name__}: {error}"
        ) from error
    embedded: dict[str, Any] = {}
    for name, contents in embedded_files.items():
        if not contents:
            continue
        try:
            if isinstance(contents, bytes):
                contents = contents.decode("utf-8")
            embedded[name] = json.loads(contents)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise CaptureError(f"invalid embedded PT2 metadata {name}: {error}") from error

    pre_normalization_metadata = graph_metadata(exported_program)
    exported_program, normalization = normalize_exported_program(
        exported_program, normalization_policy
    )
    metadata = graph_metadata(exported_program)
    metadata["normalization"] = normalization
    embedded_mapping = embedded.get("plena/parameter_mapping.json")
    mapping = parameter_mapping(exported_program)
    if embedded_mapping is not None:
        if not isinstance(embedded_mapping, dict):
            raise CaptureError("embedded PT2 parameter mapping is not a JSON object")
        mapping = embedded_mapping
    if parameter_mapping_path is not None:
        mapping_path = Path(parameter_mapping_path)
        try:
            loaded_mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise CaptureError(
                f"could not read parameter mapping {mapping_path}: {error}"
            ) from error
        if not isinstance(loaded_mapping, dict):
            raise CaptureError(f"parameter mapping {mapping_path} is not a JSON object")
        if loaded_mapping.get("schema_version") != SCHEMA_VERSION:
            raise CaptureError(
                f"parameter mapping {mapping_path} has unsupported schema_version "
                f"{loaded_mapping.get('schema_version')!r}"
            )
        if embedded_mapping is not None and loaded_mapping != embedded_mapping:
            raise CaptureError(
                "external parameter mapping does not exactly match the mapping "
                "embedded in the PT2 archive"
            )
        mapping = loaded_mapping
    elif embedded_mapping is None:
        raise CaptureError(
            "ExportedProgram has no embedded PLENA parameter mapping; pass "
            "--parameter-mapping from its capture package"
        )
    _validate_parameter_mapping(exported_program, mapping)
    embedded_source = embedded.get("plena/source.json")
    source = (
        embedded_source
        if isinstance(embedded_source, dict)
        else {"kind": "exported_program", "file_name": source_path.name}
    )
    prior_normalization = embedded.get("plena/normalization.json")
    if isinstance(prior_normalization, dict):
        normalization["prior_embedded_record"] = prior_normalization
    return CaptureResult(
        exported_program=exported_program,
        graph_metadata=metadata,
        parameter_mapping=mapping,
        source=source,
        source_artifacts={},
        pre_normalization_graph_metadata=pre_normalization_metadata,
    )


def _validate_parameter_mapping(exported_program: Any, mapping: Mapping[str, Any]) -> None:
    if mapping.get("schema_version") != SCHEMA_VERSION:
        raise CaptureError(
            "parameter mapping has unsupported schema_version "
            f"{mapping.get('schema_version')!r}"
        )
    bindings = mapping.get("bindings")
    if not isinstance(bindings, list):
        raise CaptureError("parameter mapping bindings must be an array")
    external = [
        binding
        for binding in bindings
        if isinstance(binding, dict)
        and str(binding.get("kind", "")).startswith("external_")
    ]
    if not external:
        return
    if exported_program.state_dict:
        raise CaptureError(
            "external-state parameter mapping requires an empty ExportedProgram state_dict"
        )
    user_inputs = _user_input_abi(exported_program)
    if len(user_inputs) < len(external):
        raise CaptureError(
            "parameter mapping contains more external bindings than graph USER_INPUT values"
        )
    for index, binding in enumerate(external):
        expected = user_inputs[index]
        if binding.get("argument_index") != index:
            raise CaptureError(
                f"external parameter binding {index} has invalid argument_index "
                f"{binding.get('argument_index')!r}"
            )
        if binding.get("graph_input") != expected["graph_input"]:
            raise CaptureError(
                f"external parameter binding {index} names graph input "
                f"{binding.get('graph_input')!r}, expected {expected['graph_input']!r}"
            )
        tensor = binding.get("tensor")
        contract = expected["contract"]
        if isinstance(tensor, dict) and (
            tensor.get("shape") != contract.get("shape")
            or tensor.get("dtype") != contract.get("dtype")
        ):
            raise CaptureError(
                f"external parameter binding {index} shape/dtype does not match "
                "the ExportedProgram USER_INPUT contract"
            )


def _make_low_level_importer(adapter: _TorchMlirAdapter) -> Any:
    assert adapter.low_level_importer is not None
    try:
        ir = importlib.import_module("torch_mlir.ir")
        torch_dialect = importlib.import_module("torch_mlir.dialects.torch")
        context = ir.Context()
        register = getattr(torch_dialect, "register_dialect", None)
        if callable(register):
            register(context)
        return adapter.low_level_importer(context=context)
    except Exception as error:
        raise TorchMlirImportError(
            "torch-mlir FxImporter was found but could not be initialized: "
            f"{type(error).__name__}: {error}"
        ) from error


def _mlir_text(module: Any) -> str:
    operation = getattr(module, "operation", None)
    if operation is not None:
        get_asm = getattr(operation, "get_asm", None)
        if callable(get_asm):
            try:
                text = get_asm(enable_debug_info=False)
            except TypeError:
                text = get_asm()
        else:
            text = str(module)
    else:
        text = str(module)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if not text.strip():
        raise TorchMlirImportError("torch-mlir returned an empty MLIR module")
    return text.rstrip() + "\n"


def import_exported_program(
    exported_program: Any,
    output_type: str = "torch",
    *,
    function_name: str = "main",
) -> MlirImportResult:
    """Import one ExportedProgram with a feature-detected official FX API."""

    if output_type not in _OUTPUT_TYPES:
        raise GraphFrontendError(
            f"unknown MLIR output type {output_type!r}; expected one of {', '.join(_OUTPUT_TYPES)}"
        )
    adapter = _discover_torch_mlir()
    torch_mlir_output = _TORCH_MLIR_OUTPUT_NAMES[output_type]

    if adapter.public_import is not None:
        kwargs: dict[str, Any] = {"output_type": torch_mlir_output}
        try:
            signature = inspect.signature(adapter.public_import)
            accepts_var_kwargs = any(
                parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in signature.parameters.values()
            )
            if "func_name" in signature.parameters or accepts_var_kwargs:
                kwargs["func_name"] = function_name
        except (TypeError, ValueError):
            pass
        try:
            with _frontend_warning_scope():
                module = adapter.public_import(exported_program, **kwargs)
        except Exception as error:
            raise TorchMlirImportError(
                "torch_mlir.fx.export_and_import failed for the captured "
                f"ExportedProgram (output_type={torch_mlir_output!r}, "
                f"torch-mlir={adapter.version}): {type(error).__name__}: {error}. "
                "The frontend did not generate configuration-derived fallback MLIR."
            ) from error
        return MlirImportResult(
            output_type=output_type,
            text=_mlir_text(module),
            importer_api="torch_mlir.fx.export_and_import(ExportedProgram)",
            torch_mlir_output_type=torch_mlir_output,
        )

    # The low-level class is the documented integrator API, but it only gives
    # us raw Torch dialect IR.  Backend lowering choices require the public
    # export_and_import driver and are rejected rather than emulated.
    if output_type != "torch":
        raise DependencyUnavailableError(
            f"installed torch-mlir {adapter.version} only exposes the low-level "
            f"FxImporter; requested {output_type!r} lowering requires "
            "torch_mlir.fx.export_and_import"
        )
    importer = _make_low_level_importer(adapter)
    method = getattr(importer, "import_program", None)
    method_name = "import_program"
    if not callable(method):
        method = getattr(importer, "import_frozen_program", None)
        method_name = "import_frozen_program"
    if not callable(method):
        raise DependencyUnavailableError("torch-mlir FxImporter has no ExportedProgram entry point")
    try:
        parameters = inspect.signature(method).parameters
        kwargs = {"func_name": function_name} if "func_name" in parameters else {}
        method(exported_program, **kwargs)
    except Exception as error:
        raise TorchMlirImportError(
            f"torch-mlir FxImporter.{method_name} failed: {type(error).__name__}: {error}"
        ) from error
    return MlirImportResult(
        output_type=output_type,
        text=_mlir_text(importer.module),
        importer_api=f"torch_mlir.extras.fx_importer.FxImporter.{method_name}",
        torch_mlir_output_type="raw-torch",
    )


def _normalise_output_types(values: Iterable[str]) -> tuple[str, ...]:
    requested: set[str] = set()
    for value in values:
        for item in value.split(","):
            item = item.strip().lower()
            if item:
                if item not in _OUTPUT_TYPES:
                    raise GraphFrontendError(
                        f"unknown output type {item!r}; expected {', '.join(_OUTPUT_TYPES)}"
                    )
                requested.add(item)
    # Torch dialect is the primary handoff contract and is always retained.
    requested.add("torch")
    return tuple(item for item in _OUTPUT_TYPES if item in requested)


def _json_text(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", newline="\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save_exported_program_without_examples(capture: CaptureResult, path: Path) -> None:
    """Serialize graph/signature without duplicating external tensor payloads.

    PyTorch's PT2 archive includes ``ExportedProgram.example_inputs``.  Once
    weights are explicit USER_INPUT tensors this would copy the whole model
    into ``data/sample_inputs/model.pt``, defeating externalization.  Current
    public ``torch.export.save`` has no include-example-inputs switch, while
    the public ExportedProgram constructor explicitly permits
    ``example_inputs=None``.  Serialize a shallow graph/program clone so the
    live captured object is never mutated.
    """

    torch = _require_torch()
    exported_program = capture.exported_program
    saver = getattr(torch.export, "save", None)
    if not callable(saver):
        raise DependencyUnavailableError("installed PyTorch has no torch.export.save API")
    try:
        constructor = torch.export.ExportedProgram
        serializable_program = constructor(
            root=exported_program.graph_module,
            graph=exported_program.graph,
            graph_signature=exported_program.graph_signature,
            state_dict=exported_program.state_dict,
            range_constraints=exported_program.range_constraints,
            module_call_graph=exported_program.module_call_graph,
            example_inputs=None,
            constants=getattr(exported_program, "constants", None),
            verifiers=getattr(exported_program, "verifiers", None),
        )
        extra_files = {
            "plena/parameter_mapping.json": _json_text(capture.parameter_mapping),
            "plena/normalization.json": _json_text(
                capture.graph_metadata.get("normalization", {})
            ),
            "plena/source.json": _json_text(capture.source),
        }
        with _frontend_warning_scope():
            saver(serializable_program, path, extra_files=extra_files)
    except Exception as error:
        raise CaptureError(
            f"torch.export.save failed: {type(error).__name__}: {error}"
        ) from error


def emit_package(
    capture: CaptureResult,
    output_dir: str | os.PathLike[str],
    *,
    output_types: Sequence[str] = ("torch",),
    capture_only: bool = False,
    function_name: str = "main",
) -> PackageResult:
    """Atomically emit an ExportedProgram, metadata, and requested MLIR."""

    torch = _require_torch()
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise GraphFrontendError(
            f"output directory already exists: {destination}; choose a new path "
            "so a failed import cannot mix old and new artifacts"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    files: dict[str, Path] = {}
    try:
        exported_path = staging / "exported_program.pt2"
        _save_exported_program_without_examples(capture, exported_path)
        files["exported_program"] = exported_path

        for logical_name, file_name, value in (
            ("source", "source.json", capture.source),
            ("graph_metadata", "graph_metadata.json", capture.graph_metadata),
            ("parameter_mapping", "parameter_mapping.json", capture.parameter_mapping),
        ):
            path = staging / file_name
            _write_text(path, _json_text(value))
            files[logical_name] = path

        signature_path = staging / "graph_signature.json"
        _write_text(
            signature_path,
            _json_text(capture.graph_metadata["graph_signature"]),
        )
        files["graph_signature"] = signature_path

        pre_metadata_path = staging / "graph_metadata.pre_normalization.json"
        _write_text(
            pre_metadata_path,
            _json_text(capture.pre_normalization_graph_metadata),
        )
        files["graph_metadata:pre_normalization"] = pre_metadata_path

        for file_name in sorted(capture.source_artifacts):
            if Path(file_name).name != file_name or file_name in {path.name for path in files.values()}:
                raise GraphFrontendError(f"invalid or duplicate source artifact name: {file_name!r}")
            path = staging / file_name
            value = capture.source_artifacts[file_name]
            if isinstance(value, str):
                _write_text(path, value.rstrip() + "\n")
            else:
                _write_text(path, _json_text(value))
            files[f"source_artifact:{file_name}"] = path

        requested = () if capture_only else _normalise_output_types(output_types)
        imports: list[dict[str, Any]] = []
        for output_type in requested:
            result = import_exported_program(
                capture.exported_program, output_type, function_name=function_name
            )
            path = staging / _OUTPUT_FILES[output_type]
            header = (
                "// Generated from torch.export.ExportedProgram by the official "
                "torch-mlir FX importer.\n"
                "// No Hugging Face config-template fallback was used.\n"
            )
            _write_text(path, header + result.text)
            files[f"mlir:{output_type}"] = path
            imports.append(
                {
                    "output_type": result.output_type,
                    "torch_mlir_output_type": result.torch_mlir_output_type,
                    "importer_api": result.importer_api,
                    "file": path.name,
                }
            )

        toolchain = detect_toolchain(probe_torch_mlir=not capture_only)
        toolchain_path = staging / "toolchain.json"
        _write_text(toolchain_path, _json_text(toolchain))
        files["toolchain"] = toolchain_path

        artifact_records = []
        for logical_name, path in sorted(files.items(), key=lambda item: item[1].name):
            artifact_records.append(
                {
                    "logical_name": logical_name,
                    "file": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "package_kind": PACKAGE_KIND,
            "capture_format": "torch.export.ExportedProgram",
            "example_inputs_serialized": False,
            "state_binding_contract": capture.parameter_mapping.get("binding_contract"),
            "normalization": capture.graph_metadata.get("normalization"),
            "normalizations": capture.graph_metadata.get("normalization", {}).get(
                "decompositions", []
            ),
            "capture_only": capture_only,
            "mlir_imports": imports,
            "artifacts": artifact_records,
            "fallback": None,
        }
        manifest_path = staging / "package.json"
        _write_text(manifest_path, _json_text(manifest))
        files["package"] = manifest_path

        os.replace(staging, destination)
        resolved_files = {name: destination / path.name for name, path in files.items()}
        return PackageResult(destination, resolved_files, manifest)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def make_tiny_decoder(
    *,
    batch_size: int = 1,
    sequence_length: int = 8,
    seed: int = 0,
) -> LoadedModel:
    """Create a deterministic, small decoder-only PyTorch fixture."""

    torch = _require_torch()
    if batch_size <= 0 or sequence_length <= 0:
        raise GraphFrontendError("batch size and sequence length must be positive")

    class TinyDecoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.vocab_size = 128
            self.hidden_size = 32
            self.heads = 4
            self.head_dim = 8
            self.embedding = torch.nn.Embedding(self.vocab_size, self.hidden_size)
            self.input_norm = torch.nn.RMSNorm(self.hidden_size, eps=1.0e-5)
            self.q_proj = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=False)
            self.k_proj = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=False)
            self.v_proj = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=False)
            self.o_proj = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=False)
            self.post_norm = torch.nn.RMSNorm(self.hidden_size, eps=1.0e-5)
            self.gate_proj = torch.nn.Linear(self.hidden_size, 64, bias=False)
            self.up_proj = torch.nn.Linear(self.hidden_size, 64, bias=False)
            self.down_proj = torch.nn.Linear(64, self.hidden_size, bias=False)
            self.lm_head = torch.nn.Linear(self.hidden_size, self.vocab_size, bias=False)

        def forward(self, input_ids: Any, attention_mask: Any) -> Any:
            hidden = self.embedding(input_ids)
            normed = self.input_norm(hidden)
            batch, sequence, _ = normed.shape
            q = self.q_proj(normed).view(
                batch, sequence, self.heads, self.head_dim
            ).transpose(1, 2)
            k = self.k_proj(normed).view(
                batch, sequence, self.heads, self.head_dim
            ).transpose(1, 2)
            v = self.v_proj(normed).view(
                batch, sequence, self.heads, self.head_dim
            ).transpose(1, 2)
            scores = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim ** 0.5)
            causal = torch.ones(
                (sequence, sequence), dtype=torch.bool, device=scores.device
            ).triu(diagonal=1)
            scores = scores.masked_fill(causal, -3.4028234663852886e38)
            scores = scores.masked_fill(attention_mask[:, None, None, :] == 0, -3.4028234663852886e38)
            context = torch.matmul(torch.softmax(scores, dim=-1), v)
            context = context.transpose(1, 2).contiguous().view(
                batch, sequence, self.hidden_size
            )
            hidden = hidden + self.o_proj(context)
            normed = self.post_norm(hidden)
            hidden = hidden + self.down_proj(
                torch.nn.functional.silu(self.gate_proj(normed)) * self.up_proj(normed)
            )
            return self.lm_head(hidden)

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        module = TinyDecoder()
    module.eval()
    input_ids = (
        torch.arange(batch_size * sequence_length, dtype=torch.int64)
        .reshape(batch_size, sequence_length)
        .remainder(module.vocab_size)
    )
    attention_mask = torch.ones((batch_size, sequence_length), dtype=torch.int64)
    source = {
        "kind": "builtin_fixture",
        "name": "tiny_decoder",
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "seed": seed,
    }
    fixture_config = {
        "architecture": "decoder_only_causal_lm",
        "vocab_size": module.vocab_size,
        "hidden_size": module.hidden_size,
        "intermediate_size": 64,
        "num_attention_heads": module.heads,
        "head_dim": module.head_dim,
        "num_hidden_layers": 1,
    }
    return LoadedModel(
        module=module,
        args=(input_ids, attention_mask),
        kwargs={},
        source=source,
        parameter_source_names={},
        source_artifacts={"fixture_config.json": fixture_config},
        capture_context=None,
    )


def load_huggingface_model(
    model_directory: str | os.PathLike[str],
    *,
    batch_size: int = 1,
    sequence_length: int = 8,
    load_mode: str = "config-cpu",
    seed: int = 0,
) -> LoadedModel:
    """Instantiate a local HF causal LM and provide deterministic examples.

    ``config-cpu`` initializes deterministic random weights from the local
    configuration. ``config-fake`` constructs CPU FakeTensors backed by meta
    storage, avoiding checkpoint-sized allocation while retaining a valid CPU
    dispatch device for Hugging Face autocast. ``pretrained-cpu`` reads local
    checkpoint tensors. Network model identifiers and remote code are rejected.
    """

    torch = _require_torch()
    directory = Path(model_directory).resolve()
    if not directory.is_dir() or not (directory / "config.json").is_file():
        raise GraphFrontendError(
            f"--hf-model must name a local directory containing config.json: {directory}"
        )
    if load_mode == "config-meta":
        raise GraphFrontendError(
            "config-meta is not a valid Hugging Face execution device: current "
            "Transformers autocast rejects device_type='meta'; use config-fake"
        )
    if load_mode not in {"config-cpu", "config-fake", "pretrained-cpu"}:
        raise GraphFrontendError(f"unknown Hugging Face load mode: {load_mode}")
    if batch_size <= 0 or sequence_length <= 0:
        raise GraphFrontendError("batch size and sequence length must be positive")
    try:
        transformers = importlib.import_module("transformers")
    except Exception as error:
        raise DependencyUnavailableError(
            "Transformers is required for --hf-model: "
            f"{type(error).__name__}: {error}"
        ) from error

    try:
        config = transformers.AutoConfig.from_pretrained(
            directory, local_files_only=True, trust_remote_code=False
        )
        # Preserve the exact on-disk Hugging Face contract before applying
        # capture-only overrides.  Recent Transformers versions canonicalize
        # rope_theta/rope_scaling into an internal rope_parameters object in
        # Config.to_dict(); feeding that rewritten object to the independent
        # AOT frontend would silently select default RoPE values.  The local
        # config.json is the checkpoint/runtime source of truth.
        source_config_dict = json.loads(
            (directory / "config.json").read_text(encoding="utf-8")
        )
        if not isinstance(source_config_dict, dict):
            raise ValueError("local Hugging Face config.json root is not an object")
        config.use_cache = False
        # Eager attention yields an explicit graph instead of selecting a
        # host-specific fused SDPA kernel during capture.
        if hasattr(config, "_attn_implementation"):
            config._attn_implementation = "eager"
        if load_mode == "pretrained-cpu":
            model = transformers.AutoModelForCausalLM.from_pretrained(
                directory,
                config=config,
                local_files_only=True,
                trust_remote_code=False,
            )
        elif load_mode == "config-fake":
            try:
                fake_tensor = importlib.import_module("torch._subclasses.fake_tensor")
                fake_mode = fake_tensor.FakeTensorMode(allow_non_fake_inputs=False)
            except Exception as error:
                raise DependencyUnavailableError(
                    "this PyTorch build does not provide FakeTensorMode required "
                    f"by config-fake: {type(error).__name__}: {error}"
                ) from error
            with fake_mode:
                model = transformers.AutoModelForCausalLM.from_config(
                    config, trust_remote_code=False
                )
        else:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                model = transformers.AutoModelForCausalLM.from_config(
                    config, trust_remote_code=False
                )
    except Exception as error:
        raise CaptureError(
            "could not construct the local Hugging Face causal LM: "
            f"{type(error).__name__}: {error}"
        ) from error
    model.eval()

    class HuggingFaceLogits(torch.nn.Module):
        def __init__(self, wrapped_model: Any) -> None:
            super().__init__()
            self.wrapped_model = wrapped_model

        def forward(self, input_ids: Any, attention_mask: Any) -> Any:
            outputs = self.wrapped_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=False,
            )
            return outputs[0]

    wrapper = HuggingFaceLogits(model)
    vocab_size = int(config.vocab_size)
    input_context = fake_mode if load_mode == "config-fake" else _NullContext()
    with input_context:
        input_ids = (
            torch.arange(batch_size * sequence_length, dtype=torch.int64, device="cpu")
            .reshape(batch_size, sequence_length)
            .remainder(vocab_size)
        )
        attention_mask = torch.ones(
            (batch_size, sequence_length), dtype=torch.int64, device="cpu"
        )
    source_names: dict[str, str] = {}
    source_state_names = set(model.state_dict())
    source_state_names.update(name for name, _ in model.named_parameters(remove_duplicate=False))
    source_state_names.update(name for name, _ in model.named_buffers(remove_duplicate=False))
    for name in sorted(source_state_names):
        source_names[f"wrapped_model.{name}"] = name
    source = {
        "kind": "huggingface_local_model",
        "model_type": str(getattr(config, "model_type", "unknown")),
        "architectures": list(getattr(config, "architectures", None) or []),
        "load_mode": load_mode,
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "seed": seed if load_mode == "config-cpu" else None,
    }
    return LoadedModel(
        module=wrapper,
        args=(input_ids, attention_mask),
        kwargs={},
        source=source,
        parameter_source_names=source_names,
        source_artifacts={"huggingface_config.json": source_config_dict},
        capture_context=fake_mode if load_mode == "config-fake" else None,
    )


def capture_loaded_model(
    model: LoadedModel,
    *,
    strict: bool = True,
    normalization_policy: str = "plena-decoder-v1",
) -> CaptureResult:
    context = model.capture_context or _NullContext()
    with context:
        return capture_program(
            model.module,
            model.args,
            model.kwargs,
            strict=strict,
            source=model.source,
            parameter_source_names=model.parameter_source_names,
            source_artifacts=model.source_artifacts,
            normalization_policy=normalization_policy,
        )


class _NullContext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        return False


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Capture an actual PyTorch forward graph with torch.export and import "
            "the ExportedProgram through torch-mlir's official FX frontend"
        )
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--fixture", choices=("tiny-decoder",), help="use the built-in deterministic fixture"
    )
    source.add_argument("--hf-model", type=Path, help="local Hugging Face model directory")
    source.add_argument(
        "--exported-program", type=Path, help="import an existing torch.export .pt2 file"
    )
    parser.add_argument(
        "--parameter-mapping",
        type=Path,
        help="mapping JSON retained from the capture that produced --exported-program",
    )
    parser.add_argument("--output-dir", required=False, type=Path)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--hf-load-mode",
        choices=("config-cpu", "config-fake", "config-meta", "pretrained-cpu"),
        default="config-cpu",
    )
    parser.add_argument(
        "--output-type",
        action="append",
        default=[],
        metavar="TYPE[,TYPE]",
        help="torch (always emitted), tosa, or linalg; may be repeated",
    )
    parser.add_argument(
        "--capture-only",
        action="store_true",
        help="emit ExportedProgram and metadata without requiring torch-mlir",
    )
    parser.add_argument(
        "--no-strict", action="store_true", help="pass strict=False to torch.export.export"
    )
    parser.add_argument(
        "--normalization-policy",
        choices=_NORMALIZATION_POLICIES,
        default="plena-decoder-v1",
        help="named torch.export decomposition policy recorded in package metadata",
    )
    parser.add_argument(
        "--check-toolchain",
        action="store_true",
        help="print dependency capabilities as JSON and do not capture a model",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.check_toolchain:
        report = detect_toolchain()
        print(_json_text(report), end="")
        return 0 if report["torch"]["available"] and report["torch_mlir"]["available"] else 3
    if args.output_dir is None:
        parser.error("--output-dir is required unless --check-toolchain is used")
    selected = sum(
        item is not None for item in (args.fixture, args.hf_model, args.exported_program)
    )
    if selected == 0:
        parser.error("one of --fixture, --hf-model, or --exported-program is required")

    try:
        if args.exported_program is not None:
            capture = load_exported_program(
                args.exported_program,
                parameter_mapping_path=args.parameter_mapping,
                normalization_policy=args.normalization_policy,
            )
        else:
            if args.parameter_mapping is not None:
                parser.error("--parameter-mapping requires --exported-program")
            if args.hf_model is not None:
                loaded = load_huggingface_model(
                    args.hf_model,
                    batch_size=args.batch_size,
                    sequence_length=args.sequence_length,
                    load_mode=args.hf_load_mode,
                    seed=args.seed,
                )
            else:
                loaded = make_tiny_decoder(
                    batch_size=args.batch_size,
                    sequence_length=args.sequence_length,
                    seed=args.seed,
                )
            capture = capture_loaded_model(
                loaded,
                strict=not args.no_strict,
                normalization_policy=args.normalization_policy,
            )
        result = emit_package(
            capture,
            args.output_dir,
            output_types=args.output_type or ("torch",),
            capture_only=args.capture_only,
        )
    except DependencyUnavailableError as error:
        print(f"plena-torch-frontend: dependency unavailable: {error}", file=sys.stderr)
        return 3
    except CaptureError as error:
        print(f"plena-torch-frontend: capture failed: {error}", file=sys.stderr)
        return 4
    except TorchMlirImportError as error:
        print(f"plena-torch-frontend: torch-mlir import failed: {error}", file=sys.stderr)
        return 5
    except GraphFrontendError as error:
        print(f"plena-torch-frontend: {error}", file=sys.stderr)
        return 1

    print(f"package: {result.output_dir}")
    print(f"capture: {result.files['exported_program']}")
    print(f"nodes: {capture.graph_metadata['node_count']}")
    if args.capture_only:
        print("mlir: not requested (capture-only)")
    else:
        emitted = [item["output_type"] for item in result.manifest["mlir_imports"]]
        print(f"mlir: {','.join(emitted)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
