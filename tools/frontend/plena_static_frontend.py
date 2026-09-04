#!/usr/bin/env python3
"""Normalize a supported Hugging Face decoder config for the PLENA target.

This frontend deliberately has no Transformers, PyTorch, ONNX, or NumPy
dependency.  It consumes the stable architecture fields in ``config.json``
and emits the already-bufferized MLIR boundary understood by the current
backend.  Capability analysis is strict: producing syntactically plausible IR
for an operation or shape that the ISA backend cannot execute is an error.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = 1
# PLENA's configurable baseline uses a 32x32 spatial array.  Sequence rows are
# allowed to be tails; only dimensions that are tiled as fixed-width feature
# axes by the transitional full-model backend must be aligned.
TILE = 32
VECTOR_CAPACITY = 4096
ADDRESS_BITS = 32
SCALAR_ADDRESS_BITS = 16
UINT32_MAX = (1 << ADDRESS_BITS) - 1
UINT16_MAX = (1 << 16) - 1
FLOAT32_MIN_SUBNORMAL = 2.0 ** -149
FLOAT32_MAX = float.fromhex("0x1.fffffep+127")
FLOAT16_MIN_SUBNORMAL = 2.0 ** -24
FLOAT16_MAX = 65504.0
_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")
_FLOAT_DTYPES = frozenset({"float16", "bfloat16", "float32", "fp16", "bf16", "fp32"})


class FrontendError(ValueError):
    """A deterministic malformed-input or package-generation failure."""


class UnsupportedModelError(FrontendError):
    """The input is valid JSON but outside the current NPU capability set."""

    def __init__(self, report: "CapabilityReport") -> None:
        self.report = report
        summary = "; ".join(issue.message for issue in report.errors[:4])
        if len(report.errors) > 4:
            summary += f"; and {len(report.errors) - 4} more"
        super().__init__(summary or "model is unsupported")


@dataclass(frozen=True)
class RopeSpec:
    kind: str
    theta: float
    factor: float | None = None
    low_frequency_factor: float | None = None
    high_frequency_factor: float | None = None
    original_context: int | None = None

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {"rope_type": self.kind, "theta": self.theta}
        if self.kind == "llama3":
            result.update({
                "factor": self.factor,
                "low_freq_factor": self.low_frequency_factor,
                "high_freq_factor": self.high_frequency_factor,
                "original_max_position_embeddings": self.original_context,
            })
        return result


@dataclass(frozen=True)
class ModelSpec:
    """Normalized, versioned description of a dense causal decoder."""

    model_name: str
    source_model_type: str
    architectures: tuple[str, ...]
    source_dtype: str
    source_max_position_embeddings: int
    sequence_length: int
    max_context: int
    hidden_size: int
    intermediate_size: int
    query_heads: int
    kv_heads: int
    head_dim: int
    num_layers: int
    vocab_size: int
    rms_norm_epsilon: float
    hidden_activation: str
    attention_bias: bool
    mlp_bias: bool
    tie_word_embeddings: bool
    rope: RopeSpec
    use_cache: bool
    attention_dropout: float
    pretraining_tp: int
    sliding_window: int | None
    num_local_experts: int | None
    num_experts_per_token: int | None
    quantized: bool
    qk_head_norm: bool
    norm_placement: str

    @property
    def kv_hidden_size(self) -> int:
        return self.kv_heads * self.head_dim

    @property
    def query_hidden_size(self) -> int:
        return self.query_heads * self.head_dim

    @property
    def attention_kind(self) -> str:
        if self.query_heads == self.kv_heads:
            return "mha"
        if self.kv_heads == 1:
            return "mqa"
        return "gqa"

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "source_format": "huggingface_config",
            "model_name": self.model_name,
            "architecture": "dense_decoder_causal_lm",
            "source": {
                "model_type": self.source_model_type,
                "architectures": list(self.architectures),
                "dtype": self.source_dtype,
                "max_position_embeddings": self.source_max_position_embeddings,
            },
            "execution": {
                "element_type": "fp16",
                "sequence_length": self.sequence_length,
                "max_context": self.max_context,
                "static_shapes": True,
            },
            "model": {
                "hidden_size": self.hidden_size,
                "intermediate_size": self.intermediate_size,
                "query_heads": self.query_heads,
                "kv_heads": self.kv_heads,
                "head_dim": self.head_dim,
                "query_hidden_size": self.query_hidden_size,
                "kv_hidden_size": self.kv_hidden_size,
                "num_layers": self.num_layers,
                "vocab_size": self.vocab_size,
                "tie_word_embeddings": self.tie_word_embeddings,
            },
            "operators": {
                "normalization": "rms_norm",
                "rms_norm_epsilon": self.rms_norm_epsilon,
                "activation": self.hidden_activation,
                "mlp": "swiglu",
                "attention": self.attention_kind,
                "attention_bias": self.attention_bias,
                "mlp_bias": self.mlp_bias,
                "attention_dropout": self.attention_dropout,
                "qk_head_norm": self.qk_head_norm,
                "norm_placement": self.norm_placement,
            },
            "rope": self.rope.to_dict(),
            "cache": {"enabled": self.use_cache, "layout": "token_kv_head_head_dim"},
            "checkpoint": {
                "format": "huggingface_safetensors",
                "pretraining_tensor_parallel": self.pretraining_tp,
                "quantized": self.quantized,
            },
        }
        # Keep the normalized dimensions available at the root as a stable,
        # low-friction API for package consumers.  The grouped objects above
        # remain the semantic source of truth and make the schema readable.
        result.update({
            "sequence_length": self.sequence_length,
            "max_context": self.max_context,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "query_heads": self.query_heads,
            "kv_heads": self.kv_heads,
            "head_dim": self.head_dim,
            "query_hidden_size": self.query_hidden_size,
            "num_layers": self.num_layers,
            "vocab_size": self.vocab_size,
            "tie_word_embeddings": self.tie_word_embeddings,
            "rms_norm_epsilon": self.rms_norm_epsilon,
            "rope_scaling": self.rope.to_dict(),
            "qk_head_norm": self.qk_head_norm,
        })
        return result


@dataclass(frozen=True)
class CapabilityIssue:
    code: str
    path: str
    message: str
    actual: object | None = None
    limit: object | None = None

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "code": self.code,
            "path": self.path,
            "message": self.message,
        }
        if self.actual is not None:
            result["actual"] = self.actual
        if self.limit is not None:
            result["limit"] = self.limit
        return result


@dataclass(frozen=True)
class CapabilityReport:
    model_name: str
    supported: bool
    features: Mapping[str, Mapping[str, object]]
    limits: Mapping[str, object]
    memory: Mapping[str, object]
    decode_buckets: tuple[int, ...]
    errors: tuple[CapabilityIssue, ...]
    warnings: tuple[CapabilityIssue, ...]

    def to_dict(self) -> dict[str, object]:
        error_values = [issue.to_dict() for issue in self.errors]
        warning_values = [issue.to_dict() for issue in self.warnings]
        return {
            "schema_version": SCHEMA_VERSION,
            "target": "etrinpu_npu64_fp16",
            "model_name": self.model_name,
            "supported": self.supported,
            "features": dict(self.features),
            "limits": dict(self.limits),
            "memory": dict(self.memory),
            "estimated_total_elements": self.memory["total_elements"],
            "estimated_total_bytes": self.memory["total_bytes"],
            "decode_buckets": list(self.decode_buckets),
            "errors": error_values,
            "warnings": warning_values,
            "diagnostics": error_values + warning_values,
        }


@dataclass(frozen=True)
class PackageResult:
    output_directory: Path
    model_spec: ModelSpec
    capability_report: CapabilityReport
    files: Mapping[str, Path]


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise FrontendError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _read_json(path: Path) -> dict[str, object]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise FrontendError(f"cannot read Hugging Face config {path}: {error}") from error
    try:
        value = json.loads(text, object_pairs_hook=_unique_object)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise FrontendError(f"invalid JSON in {path}: {error}") from error
    if not isinstance(value, dict):
        raise FrontendError("Hugging Face config root must be an object")
    return value


def _config_path(path: str | os.PathLike[str]) -> Path:
    result = Path(path)
    if result.is_dir():
        result = result / "config.json"
    if result.name != "config.json" and result.suffix.lower() != ".json":
        raise FrontendError("input must be a model directory or JSON configuration")
    return result


def _integer(root: Mapping[str, object], key: str, *, default: int | None = None,
             minimum: int = 1) -> int:
    value = root.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise FrontendError(f"{key} must be an integer >= {minimum}")
    return value


def _number(root: Mapping[str, object], key: str, *, default: float | None = None,
            positive: bool = False) -> float:
    value = root.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FrontendError(f"{key} must be a number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "positive and finite" if positive else "finite"
        raise FrontendError(f"{key} must be {qualifier}")
    return result


def _boolean(root: Mapping[str, object], key: str, *, default: bool) -> bool:
    value = root.get(key, default)
    if not isinstance(value, bool):
        raise FrontendError(f"{key} must be a boolean")
    return value


def _optional_integer(root: Mapping[str, object], key: str) -> int | None:
    value = root.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise FrontendError(f"{key} must be null or a positive integer")
    return value


def _safe_model_name(value: str) -> str:
    sanitized = _SAFE_NAME.sub("_", value.strip()).strip("_.-")
    return sanitized or "huggingface_model"


def _parse_rope(root: Mapping[str, object]) -> RopeSpec:
    theta = _number(root, "rope_theta", default=10000.0, positive=True)
    raw = root.get("rope_scaling")
    if raw is None:
        return RopeSpec("none", theta)
    if not isinstance(raw, dict):
        raise FrontendError("rope_scaling must be null or an object")
    type_value = raw.get("rope_type", raw.get("type", "none"))
    if not isinstance(type_value, str):
        raise FrontendError("rope_scaling.rope_type must be a string")
    kind = type_value.lower()
    if kind in ("none", "plain", "default"):
        return RopeSpec("none", theta)
    if kind != "llama3":
        # Preserve the unsupported kind so capability analysis can emit a
        # structured, user-visible rejection instead of losing provenance.
        return RopeSpec(kind, theta)
    factor = _number(raw, "factor", positive=True)
    low = _number(raw, "low_freq_factor", positive=True)
    high = _number(raw, "high_freq_factor", positive=True)
    original = _integer(raw, "original_max_position_embeddings")
    return RopeSpec("llama3", theta, factor, low, high, original)


def load_hf_config(path: str | os.PathLike[str], sequence_length: int = 128,
                   max_context: int | None = None,
                   model_name: str | None = None) -> ModelSpec:
    """Load and normalize a Hugging Face ``config.json``.

    Shape fields must already be concrete integers.  The frontend intentionally
    does not execute remote Transformers code or infer architecture from model
    Python classes.
    """

    config_path = _config_path(path)
    root = _read_json(config_path)
    if isinstance(sequence_length, bool) or not isinstance(sequence_length, int) or sequence_length <= 0:
        raise FrontendError("sequence_length must be a positive integer")
    if max_context is None:
        max_context = sequence_length
    if isinstance(max_context, bool) or not isinstance(max_context, int) or max_context <= 0:
        raise FrontendError("max_context must be a positive integer")

    hidden = _integer(root, "hidden_size")
    heads = _integer(root, "num_attention_heads")
    head_dim_value = root.get("head_dim")
    if head_dim_value is None:
        if hidden % heads:
            raise FrontendError(
                "head_dim is absent and hidden_size is not divisible by num_attention_heads"
            )
        head_dim = hidden // heads
    else:
        head_dim = _integer(root, "head_dim")
    kv_heads = _integer(root, "num_key_value_heads", default=heads)

    raw_architectures = root.get("architectures", [])
    if raw_architectures is None:
        raw_architectures = []
    if not isinstance(raw_architectures, list) or not all(
        isinstance(item, str) and item for item in raw_architectures
    ):
        raise FrontendError("architectures must be an array of non-empty strings")
    model_type = root.get("model_type", "")
    if not isinstance(model_type, str):
        raise FrontendError("model_type must be a string")
    dtype = root.get("torch_dtype", "float16")
    if not isinstance(dtype, str):
        raise FrontendError("torch_dtype must be a string")
    activation = root.get("hidden_act", "silu")
    if not isinstance(activation, str):
        raise FrontendError("hidden_act must be a string")

    fallback_name = root.get("_name_or_path")
    if not isinstance(fallback_name, str) or not fallback_name.strip():
        fallback_name = config_path.parent.name
    name = _safe_model_name(model_name if model_name is not None else fallback_name)
    quantization = root.get("quantization_config")

    # Family adapters may infer missing semantic config fields, but model
    # names never authorize compilation.  Capability analysis below accepts
    # or rejects the normalized operation contract only.
    inferred_qk_head_norm = model_type in {"qwen3", "exaone4"}
    qk_head_norm = _boolean(
        root, "qk_head_norm", default=inferred_qk_head_norm
    )
    raw_norm_placement = root.get(
        "etrinpu_norm_placement",
        "post" if model_type == "exaone4" else "pre",
    )
    if raw_norm_placement not in {"pre", "post"}:
        raise FrontendError("etrinpu_norm_placement must be 'pre' or 'post'")

    return ModelSpec(
        model_name=name,
        source_model_type=model_type,
        architectures=tuple(raw_architectures),
        source_dtype=dtype.lower(),
        source_max_position_embeddings=_integer(root, "max_position_embeddings"),
        sequence_length=sequence_length,
        max_context=max_context,
        hidden_size=hidden,
        intermediate_size=_integer(root, "intermediate_size"),
        query_heads=heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        num_layers=_integer(root, "num_hidden_layers"),
        vocab_size=_integer(root, "vocab_size"),
        rms_norm_epsilon=_number(root, "rms_norm_eps", positive=True),
        hidden_activation=activation.lower(),
        attention_bias=_boolean(root, "attention_bias", default=False),
        mlp_bias=_boolean(root, "mlp_bias", default=False),
        tie_word_embeddings=_boolean(root, "tie_word_embeddings", default=False),
        rope=_parse_rope(root),
        use_cache=_boolean(root, "use_cache", default=True),
        attention_dropout=_number(root, "attention_dropout", default=0.0),
        pretraining_tp=_integer(root, "pretraining_tp", default=1),
        sliding_window=_optional_integer(root, "sliding_window"),
        num_local_experts=_optional_integer(root, "num_local_experts"),
        num_experts_per_token=_optional_integer(root, "num_experts_per_tok"),
        quantized=quantization is not None,
        qk_head_norm=qk_head_norm,
        norm_placement=raw_norm_placement,
    )


def estimate_full_model_elements(spec: ModelSpec) -> tuple[int, dict[str, int]]:
    """Mirror the backend's no-gap full-model MemoryPlan element counts."""

    s, c, h, i = (spec.sequence_length, spec.max_context, spec.hidden_size,
                  spec.intermediate_size)
    q, d, k = spec.query_heads, spec.head_dim, spec.kv_hidden_size
    qh = spec.query_hidden_size
    scalars = 14 + int(spec.qk_head_norm)
    runtime = s * h + s * d + s * s
    per_layer_weights = (2 * h + 2 * qh * h + 2 * k * h + 3 * i * h
                         + (2 * d if spec.qk_head_norm else 0))
    all_weights = spec.num_layers * per_layer_weights + h + spec.vocab_size * h
    caches = spec.num_layers * 2 * c * k
    scratch = (
        2 * h + s * h + s * qh + s * k + d + s * qh
        + 3 * q * s * s + 3 * s
        + s * qh + s * h + s * h
        + 2 * h + s * h + 7 * s * i + s * h + spec.vocab_size
    )
    parts = {
        "scalars": scalars,
        "runtime_inputs": runtime,
        "weights": all_weights,
        "kv_cache": caches,
        "activation_workspace_and_logits": scratch,
    }
    return sum(parts.values()), parts


def _default_decode_buckets(max_context: int) -> tuple[int, ...]:
    result: list[int] = []
    value = TILE
    while value <= max_context:
        result.append(value)
        value *= 2
    return tuple(result)


def analyze_capabilities(spec: ModelSpec,
                         decode_buckets: Sequence[int] | None = None) -> CapabilityReport:
    """Return an exhaustive capability report without mutating the model."""

    buckets = tuple(_default_decode_buckets(spec.max_context)
                    if decode_buckets is None else decode_buckets)
    errors: list[CapabilityIssue] = []
    warnings: list[CapabilityIssue] = []

    def reject(code: str, path: str, message: str, actual: object | None = None,
               limit: object | None = None) -> None:
        errors.append(CapabilityIssue(code, path, message, actual, limit))

    if spec.num_local_experts not in (None, 1) or spec.num_experts_per_token not in (None, 1):
        reject("mixture_of_experts", "model.num_local_experts",
               "mixture-of-experts routing is not supported",
               {"num_local_experts": spec.num_local_experts,
                "num_experts_per_token": spec.num_experts_per_token})
    if spec.hidden_activation not in ("silu", "swish"):
        reject("unsupported_activation", "hidden_act",
               "the target dense MLP requires SiLU/SwiGLU", spec.hidden_activation)
    if spec.attention_bias:
        reject("attention_bias", "attention_bias",
               "attention projection bias is not representable by the current schedule", True)
    if spec.mlp_bias:
        reject("mlp_bias", "mlp_bias",
               "MLP projection bias is not representable by the current schedule", True)
    if spec.sliding_window is not None:
        reject("sliding_window", "sliding_window",
               "the current attention schedule is dense causal attention", spec.sliding_window)
    if not spec.use_cache:
        reject("kv_cache_disabled", "use_cache",
               "prefill/decode compilation requires persistent K/V cache state", False)
    if spec.pretraining_tp != 1:
        reject("tensor_parallel_checkpoint", "pretraining_tp",
               "pretraining tensor-parallel checkpoint reconstruction is unsupported",
               spec.pretraining_tp, 1)
    if spec.quantized:
        reject("quantized_checkpoint", "quantization_config",
               "the current data packer accepts floating-point safetensors only")
    if spec.source_dtype not in _FLOAT_DTYPES:
        reject("unsupported_dtype", "torch_dtype",
               "checkpoint dtype must be float16, bfloat16, or float32",
               spec.source_dtype)
    if spec.rope.kind not in ("none", "llama3"):
        reject("unsupported_rope", "rope_scaling.rope_type",
               "only plain RoPE and Llama-3 frequency scaling are supported",
               spec.rope.kind)
    if spec.rope.kind == "llama3":
        assert spec.rope.factor is not None
        assert spec.rope.low_frequency_factor is not None
        assert spec.rope.high_frequency_factor is not None
        assert spec.rope.original_context is not None
        if spec.rope.high_frequency_factor <= spec.rope.low_frequency_factor:
            reject("invalid_rope_scaling", "rope_scaling.high_freq_factor",
                   "high_freq_factor must exceed low_freq_factor",
                   spec.rope.high_frequency_factor, spec.rope.low_frequency_factor)
    # Runtime RoPE tables are intentionally materialized with IEEE float32
    # arithmetic, while epsilon is stored in one FP16 scalar region.  Reject
    # values that would become zero or infinity at those actual boundaries
    # instead of accepting them merely because Python's binary64 can hold them.
    rope_numbers = {"rope_theta": spec.rope.theta}
    if spec.rope.kind == "llama3":
        rope_numbers.update({
            "rope_scaling.factor": spec.rope.factor,
            "rope_scaling.low_freq_factor": spec.rope.low_frequency_factor,
            "rope_scaling.high_freq_factor": spec.rope.high_frequency_factor,
        })
    for path, value in rope_numbers.items():
        assert value is not None
        if value < FLOAT32_MIN_SUBNORMAL or value > FLOAT32_MAX:
            reject("float32_parameter_range", path,
                   f"{path} must remain finite and nonzero after float32 conversion",
                   value, {"min": FLOAT32_MIN_SUBNORMAL, "max": FLOAT32_MAX})
    if not (FLOAT16_MIN_SUBNORMAL <= spec.rms_norm_epsilon <= FLOAT16_MAX):
        reject("fp16_epsilon_range", "rms_norm_eps",
               "RMSNorm epsilon must remain finite and nonzero in its FP16 scalar region",
               spec.rms_norm_epsilon,
               {"min": FLOAT16_MIN_SUBNORMAL, "max": FLOAT16_MAX})
    if spec.max_context < spec.sequence_length:
        reject("context_shorter_than_prefill", "execution.max_context",
               "max context cannot be shorter than prefill sequence length",
               spec.max_context, spec.sequence_length)
    if spec.max_context > spec.source_max_position_embeddings:
        reject("context_exceeds_model", "execution.max_context",
               "requested context exceeds the model configuration",
               spec.max_context, spec.source_max_position_embeddings)
    if spec.query_heads % spec.kv_heads:
        reject("unsupported_head_grouping", "model.kv_heads",
               "query head count must be divisible by KV head count",
               {"query_heads": spec.query_heads, "kv_heads": spec.kv_heads})
    if spec.head_dim % 2:
        reject("odd_head_dimension", "model.head_dim",
               "split-half RoPE requires an even head dimension", spec.head_dim)

    tiled = {
        "hidden_size": spec.hidden_size,
        "intermediate_size": spec.intermediate_size,
        "head_dim": spec.head_dim,
        "query_hidden_size": spec.query_hidden_size,
        "kv_hidden_size": spec.kv_hidden_size,
        "vocab_size": spec.vocab_size,
    }
    for name, value in tiled.items():
        if value % TILE:
            reject("partial_tile_unsupported", f"model.{name}",
                   f"{name} must be divisible by the fixed {TILE}-element tile",
                   value, TILE)
    if spec.hidden_size > VECTOR_CAPACITY:
        reject("vector_capacity", "model.hidden_size",
               "RMSNorm row exceeds vector capacity", spec.hidden_size,
               VECTOR_CAPACITY)
    if spec.qk_head_norm and spec.head_dim > VECTOR_CAPACITY:
        reject("vector_capacity", "model.head_dim",
               "Q/K head RMSNorm row exceeds vector capacity", spec.head_dim,
               VECTOR_CAPACITY)
    for name, value in {
        "sequence_length": spec.sequence_length,
        "max_context": spec.max_context,
        "hidden_size": spec.hidden_size,
        "intermediate_size": spec.intermediate_size,
        "query_hidden_size": spec.query_hidden_size,
    }.items():
        if value > UINT16_MAX:
            reject("isa_extent_overflow", f"model.{name}",
                   f"{name} exceeds the 16-bit main-matrix ISA extent",
                   value, UINT16_MAX)

    seen_buckets: set[int] = set()
    for index, bucket in enumerate(buckets):
        if isinstance(bucket, bool) or not isinstance(bucket, int):
            reject("invalid_decode_bucket", f"decode_buckets[{index}]",
                   "decode bucket must be an integer", bucket)
            continue
        if bucket in seen_buckets:
            reject("duplicate_decode_bucket", f"decode_buckets[{index}]",
                   "decode buckets must be unique", bucket)
        seen_buckets.add(bucket)
        if bucket <= 0 or bucket % TILE:
            reject("invalid_decode_bucket", f"decode_buckets[{index}]",
                   f"decode bucket must be a positive multiple of {TILE}", bucket, TILE)
        if bucket > spec.max_context:
            reject("decode_bucket_exceeds_context", f"decode_buckets[{index}]",
                   "decode bucket exceeds max context", bucket, spec.max_context)
        if bucket > spec.sequence_length * spec.sequence_length:
            reject("decode_mask_workspace", f"decode_buckets[{index}]",
                   "prefill mask arena cannot hold the decode mask prefix",
                   bucket, spec.sequence_length * spec.sequence_length)
    if buckets and TILE * spec.kv_hidden_size > (
        spec.query_heads * spec.sequence_length * spec.sequence_length
    ):
        reject("decode_v_workspace", "model.kv_hidden_size",
               f"attention score arena cannot hold the physical {TILE}-row V projection scratch",
               TILE * spec.kv_hidden_size,
               spec.query_heads * spec.sequence_length * spec.sequence_length)

    total, parts = estimate_full_model_elements(spec)
    individual_regions = {
        "lm_head_weight": spec.vocab_size * spec.hidden_size,
        "q_weight": spec.query_hidden_size * spec.hidden_size,
        "o_weight": spec.hidden_size * spec.query_hidden_size,
        "gate_weight": spec.intermediate_size * spec.hidden_size,
        "kv_cache_per_layer": spec.max_context * spec.kv_hidden_size,
        "attention_scores": spec.query_heads * spec.sequence_length * spec.sequence_length,
    }
    for name, value in individual_regions.items():
        if value > UINT32_MAX:
            reject("region_address_overflow", f"memory.{name}",
                   "one physical region exceeds the uint32 element limit",
                   value, UINT32_MAX)
    if total > UINT32_MAX:
        reject("gbuffer_address_overflow", "memory.total_elements",
               "resident weights, K/V state, and workspace exceed the uint32 FP16-element address space",
               total, UINT32_MAX)
    if spec.attention_dropout != 0.0:
        warnings.append(CapabilityIssue(
            "inference_dropout_ignored", "attention_dropout",
            "attention dropout is disabled for inference", spec.attention_dropout
        ))

    features: dict[str, dict[str, object]] = {
        "dense_decoder": {
            "supported": spec.num_local_experts in (None, 1),
            "implementation": "feature-gated dense causal decoder",
            "source_model_type": spec.source_model_type,
        },
        "rms_norm": {"supported": True, "epsilon": spec.rms_norm_epsilon},
        "norm_placement": {
            "supported": spec.norm_placement in {"pre", "post"},
            "value": spec.norm_placement,
        },
        "qk_head_norm": {"supported": True, "enabled": spec.qk_head_norm},
        "silu_swiglu": {"supported": spec.hidden_activation in ("silu", "swish")},
        "projection_bias": {"supported": not spec.attention_bias and not spec.mlp_bias,
                            "required": "no bias"},
        "attention": {"supported": spec.query_heads % spec.kv_heads == 0,
                      "kind": spec.attention_kind,
                      "query_heads": spec.query_heads, "kv_heads": spec.kv_heads},
        "rope": {"supported": spec.rope.kind in ("none", "llama3"),
                 "rope_type": spec.rope.kind, "tables": "host_materialized"},
        "embedding_and_lm_head": {"supported": True,
                                  "tied": spec.tie_word_embeddings,
                                  "embedding": "cpu", "lm_head": "npu"},
        "static_shapes": {"supported": True, "dynamic_shapes": False},
        "fp16_execution": {"supported": spec.source_dtype in _FLOAT_DTYPES,
                           "checkpoint_dtype": spec.source_dtype},
    }
    memory = {
        "address_unit": "fp16_element",
        "total_elements": total,
        "total_bytes": total * 2,
        "uint32_limit_elements": UINT32_MAX,
        "fits_uint32": total <= UINT32_MAX,
        "breakdown_elements": parts,
    }
    limits = {
        "matrix_tile": [TILE, TILE, TILE],
        "vector_capacity_elements": VECTOR_CAPACITY,
        "address_bits": ADDRESS_BITS,
        "scalar_address_bits": SCALAR_ADDRESS_BITS,
        "main_matrix_extent_bits": 16,
        "partial_tiles": False,
        "element_type": "fp16",
    }
    return CapabilityReport(
        model_name=spec.model_name,
        supported=not errors,
        features=features,
        limits=limits,
        memory=memory,
        decode_buckets=buckets,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


def _arg(name: str, ty: str) -> tuple[str, str]:
    return name, ty


def _prefill_abi(spec: ModelSpec) -> dict[str, list[tuple[str, str]]]:
    s, h, k, i = (spec.sequence_length, spec.hidden_size,
                  spec.kv_hidden_size, spec.intermediate_size)
    q, qh, hh = spec.query_heads, spec.query_hidden_size, spec.head_dim // 2
    scalar_constants = [
            _arg("sqrt_hidden_size_scalar", "memref<1xf16>"),
    ]
    if spec.qk_head_norm:
        scalar_constants.append(_arg("sqrt_head_dim_scalar", "memref<1xf16>"))
    scalar_constants += [
            _arg("epsilon_scalar", "memref<1xf16>"),
            _arg("attention_scale", "memref<1xf16>"),
            _arg("silu_one_scalar", "memref<1xf16>"),
    ]
    weights = [
            _arg("input_norm_gamma", f"memref<{h}xf16>"),
            _arg("post_attention_norm_gamma", f"memref<{h}xf16>"),
    ]
    if spec.qk_head_norm:
        weights += [
            _arg("q_norm_gamma", f"memref<{spec.head_dim}xf16>"),
            _arg("k_norm_gamma", f"memref<{spec.head_dim}xf16>"),
        ]
    weights += [
            _arg("q_weight", f"memref<{qh}x{h}xf16>"),
            _arg("k_weight", f"memref<{k}x{h}xf16>"),
            _arg("v_weight", f"memref<{k}x{h}xf16>"),
            _arg("o_weight", f"memref<{h}x{qh}xf16>"),
            _arg("gate_weight", f"memref<{i}x{h}xf16>"),
            _arg("up_weight", f"memref<{i}x{h}xf16>"),
            _arg("down_weight", f"memref<{h}x{i}xf16>"),
    ]
    return {
        "scalar_constants": scalar_constants,
        "runtime": [
            _arg("rope_cosine", f"memref<{s}x{hh}xf16>"),
            _arg("rope_sine", f"memref<{s}x{hh}xf16>"),
            _arg("additive_causal_mask", f"memref<{s}x{s}xf16>"),
        ],
        "weights": weights,
        "caches": [
            _arg("value_cache", f"memref<{s}x{k}xf16>"),
            _arg("key_cache", f"memref<{s}x{k}xf16>"),
        ],
        "scalar_scratch": [
            _arg(name, "memref<1xf16>") for name in (
                "input_norm_sum_squares", "input_norm_mean_square", "input_norm_rms",
                "post_norm_sum_squares", "post_norm_mean_square", "post_norm_rms",
                "softmax_chunk_max", "softmax_global_max", "softmax_chunk_sum",
                "softmax_global_sum",
            )
        ],
        "activation_scratch": [
            _arg("input_norm_squares", f"memref<{h}xf16>"),
            _arg("input_norm_normalized", f"memref<{h}xf16>"),
            _arg("input_norm_output", f"memref<{s}x{h}xf16>"),
            _arg("query", f"memref<{s}x{qh}xf16>"),
            _arg("key", f"memref<{s}x{k}xf16>"),
            _arg("rope_temp0", f"memref<{hh}xf16>"),
            _arg("rope_temp1", f"memref<{hh}xf16>"),
            _arg("query_rope", f"memref<{s}x{qh}xf16>"),
            _arg("attention_scores", f"memref<{q}x{s}x{s}xf16>"),
            _arg("attention_scaled_temp", f"memref<{s}xf16>"),
            _arg("masked_scores", f"memref<{q}x{s}x{s}xf16>"),
            _arg("softmax_centered", f"memref<{s}xf16>"),
            _arg("softmax_exponentials", f"memref<{s}xf16>"),
            _arg("probabilities", f"memref<{q}x{s}x{s}xf16>"),
            _arg("attention_output", f"memref<{s}x{qh}xf16>"),
            _arg("projected_attention", f"memref<{s}x{h}xf16>"),
            _arg("attention_residual", f"memref<{s}x{h}xf16>"),
            _arg("post_norm_squares", f"memref<{h}xf16>"),
            _arg("post_norm_normalized", f"memref<{h}xf16>"),
            _arg("post_norm_output", f"memref<{s}x{h}xf16>"),
            _arg("gate", f"memref<{s}x{i}xf16>"),
            _arg("silu_negative", f"memref<{s}x{i}xf16>"),
            _arg("silu_exponential", f"memref<{s}x{i}xf16>"),
            _arg("silu_denominator", f"memref<{s}x{i}xf16>"),
            _arg("silu_gate", f"memref<{s}x{i}xf16>"),
            _arg("up", f"memref<{s}x{i}xf16>"),
            _arg("swiglu", f"memref<{s}x{i}xf16>"),
            _arg("down", f"memref<{s}x{h}xf16>"),
        ],
    }


def _helper_args(spec: ModelSpec) -> list[tuple[str, str]]:
    abi = _prefill_abi(spec)
    return ([_arg("hidden", f"memref<{spec.sequence_length}x{spec.hidden_size}xf16>")]
            + abi["scalar_constants"] + abi["runtime"] + abi["weights"]
            + abi["caches"] + abi["scalar_scratch"] + abi["activation_scratch"])


def _fmt_signature(items: Sequence[tuple[str, str]], indent: str = "      ") -> str:
    rendered = [f"%{name}: {ty}" for name, ty in items]
    lines: list[str] = []
    current = ""
    for item in rendered:
        candidate = item if not current else current + ", " + item
        if len(indent) + len(candidate) > 104 and current:
            lines.append(indent + current + ",")
            current = item
        else:
            current = candidate
    lines.append(indent + current)
    return "\n".join(lines)


def _types(items: Sequence[tuple[str, str]]) -> str:
    return ", ".join(ty for _, ty in items)


def _alloc(name: str, ty: str, role: str, extra: str = "") -> str:
    attrs = f'npu.name = "{name}", npu.role = "{role}"'
    if extra:
        attrs += ", " + extra
    return f'    %{name} = "memref.alloc"() {{{attrs}}} : () -> {ty}'


def _layer_region(layer: int, local: str) -> str:
    return f"layer_{layer:02d}_{local}"


def _mlir_float(value: float) -> str:
    return format(value, ".6e")


def _emit_prefill_helper(spec: ModelSpec) -> str:
    s, h, k, i = (spec.sequence_length, spec.hidden_size,
                  spec.kv_hidden_size, spec.intermediate_size)
    q, kv, d, hh = (spec.query_heads, spec.kv_heads, spec.head_dim,
                    spec.head_dim // 2)
    qh = spec.query_hidden_size
    eps = _mlir_float(spec.rms_norm_epsilon)
    signature = _fmt_signature(_helper_args(spec))
    q_head_norm = ""
    k_head_norm = ""
    if spec.qk_head_norm:
        q_head_norm = f'''
    "etrinpu.rms_norm"(%query, %q_norm_gamma, %sqrt_head_dim_scalar,
        %epsilon_scalar, %input_norm_squares, %input_norm_sum_squares,
        %input_norm_mean_square, %input_norm_rms, %input_norm_normalized,
        %query) {{epsilon = {eps} : f32, hidden_size = {d} : i64}} :
        (memref<{s}x{qh}xf16>, memref<{d}xf16>, memref<1xf16>, memref<1xf16>,
         memref<{h}xf16>, memref<1xf16>, memref<1xf16>, memref<1xf16>,
         memref<{h}xf16>, memref<{s}x{qh}xf16>) -> ()
'''
        k_head_norm = f'''
    "etrinpu.rms_norm"(%key, %k_norm_gamma, %sqrt_head_dim_scalar,
        %epsilon_scalar, %input_norm_squares, %input_norm_sum_squares,
        %input_norm_mean_square, %input_norm_rms, %input_norm_normalized,
        %key) {{epsilon = {eps} : f32, hidden_size = {d} : i64}} :
        (memref<{s}x{k}xf16>, memref<{d}xf16>, memref<1xf16>, memref<1xf16>,
         memref<{h}xf16>, memref<1xf16>, memref<1xf16>, memref<1xf16>,
         memref<{h}xf16>, memref<{s}x{k}xf16>) -> ()
'''
    if spec.norm_placement == "pre":
        input_norm = f'''
    "etrinpu.rms_norm"(%hidden, %input_norm_gamma, %sqrt_hidden_size_scalar,
        %epsilon_scalar, %input_norm_squares, %input_norm_sum_squares,
        %input_norm_mean_square, %input_norm_rms, %input_norm_normalized,
        %input_norm_output) {{epsilon = {eps} : f32,
        hidden_size = {h} : i64}} :
        (memref<{s}x{h}xf16>, memref<{h}xf16>, memref<1xf16>, memref<1xf16>,
         memref<{h}xf16>, memref<1xf16>, memref<1xf16>, memref<1xf16>,
         memref<{h}xf16>, memref<{s}x{h}xf16>) -> ()
'''
        attention_input = "%input_norm_output"
        attention_transition = f'''
    "etrinpu.elementwise"(%hidden, %projected_attention,
        %attention_residual) {{kind = "add"}} :
        (memref<{s}x{h}xf16>, memref<{s}x{h}xf16>,
         memref<{s}x{h}xf16>) -> ()
    "etrinpu.rms_norm"(%attention_residual, %post_attention_norm_gamma,
        %sqrt_hidden_size_scalar, %epsilon_scalar, %post_norm_squares,
        %post_norm_sum_squares, %post_norm_mean_square, %post_norm_rms,
        %post_norm_normalized, %post_norm_output) {{
        epsilon = {eps} : f32, hidden_size = {h} : i64}} :
        (memref<{s}x{h}xf16>, memref<{h}xf16>, memref<1xf16>, memref<1xf16>,
         memref<{h}xf16>, memref<1xf16>, memref<1xf16>, memref<1xf16>,
         memref<{h}xf16>, memref<{s}x{h}xf16>) -> ()
'''
        ffn_input = "%post_norm_output"
        ffn_transition = f'''
    "etrinpu.elementwise"(%attention_residual, %down, %hidden) {{kind = "add"}} :
        (memref<{s}x{h}xf16>, memref<{s}x{h}xf16>,
         memref<{s}x{h}xf16>) -> ()
'''
    else:
        input_norm = ""
        attention_input = "%hidden"
        attention_transition = f'''
    "etrinpu.rms_norm"(%projected_attention, %input_norm_gamma,
        %sqrt_hidden_size_scalar, %epsilon_scalar, %input_norm_squares,
        %input_norm_sum_squares, %input_norm_mean_square, %input_norm_rms,
        %input_norm_normalized, %input_norm_output) {{
        epsilon = {eps} : f32, hidden_size = {h} : i64}} :
        (memref<{s}x{h}xf16>, memref<{h}xf16>, memref<1xf16>, memref<1xf16>,
         memref<{h}xf16>, memref<1xf16>, memref<1xf16>, memref<1xf16>,
         memref<{h}xf16>, memref<{s}x{h}xf16>) -> ()
    "etrinpu.elementwise"(%hidden, %input_norm_output,
        %attention_residual) {{kind = "add"}} :
        (memref<{s}x{h}xf16>, memref<{s}x{h}xf16>,
         memref<{s}x{h}xf16>) -> ()
'''
        ffn_input = "%attention_residual"
        ffn_transition = f'''
    "etrinpu.rms_norm"(%down, %post_attention_norm_gamma,
        %sqrt_hidden_size_scalar, %epsilon_scalar, %post_norm_squares,
        %post_norm_sum_squares, %post_norm_mean_square, %post_norm_rms,
        %post_norm_normalized, %post_norm_output) {{
        epsilon = {eps} : f32, hidden_size = {h} : i64}} :
        (memref<{s}x{h}xf16>, memref<{h}xf16>, memref<1xf16>, memref<1xf16>,
         memref<{h}xf16>, memref<1xf16>, memref<1xf16>, memref<1xf16>,
         memref<{h}xf16>, memref<{s}x{h}xf16>) -> ()
    "etrinpu.elementwise"(%attention_residual, %post_norm_output,
        %hidden) {{kind = "add"}} :
        (memref<{s}x{h}xf16>, memref<{s}x{h}xf16>,
         memref<{s}x{h}xf16>) -> ()
'''
    return f'''  func.func private @decoder_layer(
{signature}) -> memref<{s}x{h}xf16> attributes {{etrinpu.layer_template}} {{
    %zero = arith.constant 0.000000e+00 : f16
{input_norm.rstrip()}

    linalg.fill ins(%zero : f16) outs(%query : memref<{s}x{qh}xf16>)
    linalg.matmul indexing_maps = [#lhs, #rhs_transposed, #result]
        ins({attention_input}, %q_weight : memref<{s}x{h}xf16>,
            memref<{qh}x{h}xf16>) outs(%query : memref<{s}x{qh}xf16>)
    linalg.fill ins(%zero : f16) outs(%key : memref<{s}x{k}xf16>)
    linalg.matmul indexing_maps = [#lhs, #rhs_transposed, #result]
        ins({attention_input}, %k_weight : memref<{s}x{h}xf16>,
            memref<{k}x{h}xf16>) outs(%key : memref<{s}x{k}xf16>)
{q_head_norm.rstrip()}
{k_head_norm.rstrip()}
    linalg.fill ins(%zero : f16) outs(%value_cache : memref<{s}x{k}xf16>)
    linalg.matmul indexing_maps = [#lhs, #rhs_transposed, #result]
        ins({attention_input}, %v_weight : memref<{s}x{h}xf16>,
            memref<{k}x{h}xf16>) outs(%value_cache : memref<{s}x{k}xf16>)

    "etrinpu.rope"(%query, %rope_cosine, %rope_sine, %rope_temp0,
        %rope_temp1, %query_rope) {{heads = {q} : i64, head_dim = {d} : i64}} :
        (memref<{s}x{qh}xf16>, memref<{s}x{hh}xf16>, memref<{s}x{hh}xf16>,
         memref<{hh}xf16>, memref<{hh}xf16>, memref<{s}x{qh}xf16>) -> ()
    "etrinpu.rope"(%key, %rope_cosine, %rope_sine, %rope_temp0,
        %rope_temp1, %key_cache) {{heads = {kv} : i64, head_dim = {d} : i64}} :
        (memref<{s}x{k}xf16>, memref<{s}x{hh}xf16>, memref<{s}x{hh}xf16>,
         memref<{hh}xf16>, memref<{hh}xf16>, memref<{s}x{k}xf16>) -> ()
    "etrinpu.gqa_qk"(%query_rope, %key_cache, %attention_scores) {{
        head_dim = {d} : i64, kv_heads = {kv} : i64, query_heads = {q} : i64}} :
        (memref<{s}x{qh}xf16>, memref<{s}x{k}xf16>,
         memref<{q}x{s}x{s}xf16>) -> ()
    "etrinpu.scale_mask"(%attention_scores, %attention_scale,
        %additive_causal_mask, %attention_scaled_temp, %masked_scores) :
        (memref<{q}x{s}x{s}xf16>, memref<1xf16>, memref<{s}x{s}xf16>,
         memref<{s}xf16>, memref<{q}x{s}x{s}xf16>) -> ()
    "etrinpu.softmax"(%masked_scores, %softmax_chunk_max,
        %softmax_global_max, %softmax_centered, %softmax_exponentials,
        %softmax_chunk_sum, %softmax_global_sum, %probabilities) :
        (memref<{q}x{s}x{s}xf16>, memref<1xf16>, memref<1xf16>,
         memref<{s}xf16>, memref<{s}xf16>, memref<1xf16>, memref<1xf16>,
         memref<{q}x{s}x{s}xf16>) -> ()
    "etrinpu.gqa_pv"(%probabilities, %value_cache, %attention_output) {{
        head_dim = {d} : i64, kv_heads = {kv} : i64, query_heads = {q} : i64}} :
        (memref<{q}x{s}x{s}xf16>, memref<{s}x{k}xf16>,
         memref<{s}x{qh}xf16>) -> ()

    linalg.fill ins(%zero : f16)
        outs(%projected_attention : memref<{s}x{h}xf16>)
    linalg.matmul indexing_maps = [#lhs, #rhs_transposed, #result]
        ins(%attention_output, %o_weight : memref<{s}x{qh}xf16>,
            memref<{h}x{qh}xf16>)
        outs(%projected_attention : memref<{s}x{h}xf16>)
{attention_transition.rstrip()}

    linalg.fill ins(%zero : f16) outs(%gate : memref<{s}x{i}xf16>)
    linalg.matmul indexing_maps = [#lhs, #rhs_transposed, #result]
        ins({ffn_input}, %gate_weight : memref<{s}x{h}xf16>,
            memref<{i}x{h}xf16>) outs(%gate : memref<{s}x{i}xf16>)
    "etrinpu.silu"(%gate, %silu_one_scalar, %silu_negative,
        %silu_exponential, %silu_denominator, %silu_gate) :
        (memref<{s}x{i}xf16>, memref<1xf16>, memref<{s}x{i}xf16>,
         memref<{s}x{i}xf16>, memref<{s}x{i}xf16>,
         memref<{s}x{i}xf16>) -> ()
    linalg.fill ins(%zero : f16) outs(%up : memref<{s}x{i}xf16>)
    linalg.matmul indexing_maps = [#lhs, #rhs_transposed, #result]
        ins({ffn_input}, %up_weight : memref<{s}x{h}xf16>,
            memref<{i}x{h}xf16>) outs(%up : memref<{s}x{i}xf16>)
    "etrinpu.elementwise"(%silu_gate, %up, %swiglu) {{kind = "mul"}} :
        (memref<{s}x{i}xf16>, memref<{s}x{i}xf16>,
         memref<{s}x{i}xf16>) -> ()
    linalg.fill ins(%zero : f16) outs(%down : memref<{s}x{h}xf16>)
    linalg.matmul indexing_maps = [#lhs, #rhs_transposed, #result]
        ins(%swiglu, %down_weight : memref<{s}x{i}xf16>,
            memref<{h}x{i}xf16>) outs(%down : memref<{s}x{h}xf16>)
{ffn_transition.rstrip()}
    return %hidden : memref<{s}x{h}xf16>
  }}
'''


def _emit_prefill_entry(spec: ModelSpec) -> str:
    s, h, vocab = spec.sequence_length, spec.hidden_size, spec.vocab_size
    abi = _prefill_abi(spec)
    args = _helper_args(spec)
    lines = ['  func.func @prefill_full() attributes {etrinpu.entry_point = "prefill"} {']
    for name, ty in abi["scalar_constants"]:
        lines.append(_alloc(name, ty, "scalar_constant"))
    for name, ty in abi["scalar_scratch"]:
        lines.append(_alloc(name, ty, "scalar_scratch"))
    lines.append(_alloc("hidden_state", f"memref<{s}x{h}xf16>", "runtime_input"))
    for name, ty in abi["runtime"]:
        lines.append(_alloc(name, ty, "runtime_input"))
    for layer in range(spec.num_layers):
        for local, ty in abi["weights"]:
            lines.append(_alloc(_layer_region(layer, local), ty, "constant"))
    for layer in range(spec.num_layers):
        for local, ty in abi["caches"]:
            kind = "value" if local.startswith("value") else "key"
            extra = ('npu.capacity_axis = 0 : i64, '
                     'npu.capacity_from_config = "max_context", '
                     f'npu.kv_kind = "{kind}"')
            lines.append(_alloc(_layer_region(layer, local), ty, "state", extra))
    lines.append(_alloc("final_norm_gamma", f"memref<{h}xf16>", "constant"))
    lines.append(_alloc("token_embedding_lm_head_weight",
                        f"memref<{vocab}x{h}xf16>", "constant"))
    for name, ty in abi["activation_scratch"]:
        lines.append(_alloc(name, ty, "scratch"))
    lines.append(_alloc("logits", f"memref<1x{vocab}xf16>", "runtime_output"))
    lines.append("")

    prefix = [name for name, _ in abi["scalar_constants"] + abi["runtime"]]
    scratch = [name for name, _ in abi["scalar_scratch"] + abi["activation_scratch"]]
    helper_types = _types(args)
    previous = "%hidden_state"
    for layer in range(spec.num_layers):
        operands = [previous]
        operands += [f"%{name}" for name in prefix]
        operands += [f"%{_layer_region(layer, name)}" for name, _ in abi["weights"]]
        operands += [f"%{_layer_region(layer, name)}" for name, _ in abi["caches"]]
        operands += [f"%{name}" for name in scratch]
        result = f"%hidden_{layer + 1:02d}"
        lines.append(
            f'    {result} = "func.call"({", ".join(operands)}) '
            f'{{callee = @decoder_layer, npu.layer_index = {layer} : i64}} : '
            f'({helper_types}) -> memref<{s}x{h}xf16>'
        )
        previous = result

    eps = _mlir_float(spec.rms_norm_epsilon)
    last_row, offset = s - 1, (s - 1) * h
    lines += [
        "",
        f"    // Final norm overwrites hidden_state after decoder layer {spec.num_layers - 1}.",
        f'    "etrinpu.rms_norm"({previous}, %final_norm_gamma,',
        "        %sqrt_hidden_size_scalar, %epsilon_scalar, %input_norm_squares,",
        "        %input_norm_sum_squares, %input_norm_mean_square, %input_norm_rms,",
        "        %input_norm_normalized, %hidden_state) {",
        f"        epsilon = {eps} : f32, hidden_size = {h} : i64,",
        '        npu.stage = "final_norm"} :',
        f"        (memref<{s}x{h}xf16>, memref<{h}xf16>, memref<1xf16>,",
        f"         memref<1xf16>, memref<{h}xf16>, memref<1xf16>, memref<1xf16>,",
        f"         memref<1xf16>, memref<{h}xf16>, memref<{s}x{h}xf16>) -> ()",
        f"    %last_hidden = memref.subview %hidden_state[{last_row}, 0] [1, {h}] [1, 1] :",
        f"        memref<{s}x{h}xf16> to",
        f"        memref<1x{h}xf16, strided<[{h}, 1], offset: {offset}>>",
        "    %zero = arith.constant 0.000000e+00 : f16",
        f"    linalg.fill ins(%zero : f16) outs(%logits : memref<1x{vocab}xf16>)",
        "    linalg.matmul indexing_maps = [#lhs, #rhs_transposed, #result]",
        "        ins(%last_hidden, %token_embedding_lm_head_weight :",
        f"            memref<1x{h}xf16, strided<[{h}, 1], offset: {offset}>>,",
        f"            memref<{vocab}x{h}xf16>)",
        f"        outs(%logits : memref<1x{vocab}xf16>)",
        "    return",
        "  }",
    ]
    return "\n".join(lines) + "\n"


def render_prefill_mlir(spec: ModelSpec) -> str:
    """Render the connected full-model prefill buffer IR."""

    name = json.dumps(spec.model_name)
    header = f'''// Generated by etrinpu-frontend. Do not hand-edit.
// CPU supplies the pre-embedded hidden state and runtime RoPE/mask inputs.

#lhs = affine_map<(m, n, k) -> (m, k)>
#rhs_transposed = affine_map<(m, n, k) -> (n, k)>
#result = affine_map<(m, n, k) -> (m, n)>

module attributes {{
  etrinpu.model = {name},
  etrinpu.program = "full_prefill",
  etrinpu.sequence_length = {spec.sequence_length} : i64,
  etrinpu.layer_count = {spec.num_layers} : i64,
  etrinpu.vocab_size = {spec.vocab_size} : i64,
  etrinpu.norm_placement = "{spec.norm_placement}",
  etrinpu.cpu_preembedded_input
}} {{
'''
    text = header + _emit_prefill_helper(spec) + "\n" + _emit_prefill_entry(spec) + "}\n"
    if text.count('"func.call"') != spec.num_layers:
        raise FrontendError("internal error: prefill layer call count drift")
    return text


_DECODE_GENERATOR: Any | None = None


def _decode_generator() -> Any:
    global _DECODE_GENERATOR
    if _DECODE_GENERATOR is not None:
        return _DECODE_GENERATOR
    path = Path(__file__).with_name("plena_decode_generator.py")
    module_spec = importlib.util.spec_from_file_location("plena_dynamic_decode_generator", path)
    if module_spec is None or module_spec.loader is None:
        raise FrontendError(f"cannot load decode MLIR generator {path}")
    module = importlib.util.module_from_spec(module_spec)
    # dataclasses resolves postponed annotations through sys.modules.
    sys.modules[module_spec.name] = module
    module_spec.loader.exec_module(module)
    _DECODE_GENERATOR = module
    return module


def render_decode_mlir(spec: ModelSpec, bucket: int) -> str:
    """Render a one-token, fixed-bucket decode graph for ``spec``."""

    generator = _decode_generator()
    profile = generator.Profile(
        spec.model_name, spec.num_layers, spec.sequence_length, spec.max_context,
        spec.hidden_size, spec.kv_hidden_size, spec.intermediate_size,
        spec.vocab_size, spec.query_heads, spec.kv_heads, spec.head_dim,
        q_hidden=spec.query_hidden_size, qk_head_norm=spec.qk_head_norm,
        norm_placement=spec.norm_placement,
    )
    header = f'''// Generated by etrinpu-frontend. Do not hand-edit.
// Row zero is the logical token; the host commits staged K/V after the fence.

#lhs = affine_map<(m, n, k) -> (m, k)>
#rhs_transposed = affine_map<(m, n, k) -> (n, k)>
#result = affine_map<(m, n, k) -> (m, n)>

module attributes {{
  etrinpu.model = {json.dumps(spec.model_name)},
  etrinpu.program = "full_decode",
  etrinpu.sequence_length = {spec.sequence_length} : i64,
  etrinpu.layer_count = {spec.num_layers} : i64,
  etrinpu.vocab_size = {spec.vocab_size} : i64,
  etrinpu.decode.logical_query_length = 1 : i64,
  etrinpu.decode.physical_query_rows = 64 : i64,
  etrinpu.decode.kv_bucket = {bucket} : i64,
  etrinpu.decode.staging_slot = {bucket - 1} : i64,
  etrinpu.decode.runtime_rope_rows = 1 : i64,
  etrinpu.decode.runtime_mask_elements = {bucket} : i64,
  etrinpu.cpu_preembedded_input
}} {{
'''
    text = header + generator.emit_helper(profile, bucket) + "\n"
    text += generator.emit_entry(profile, bucket) + "}\n"
    if spec.rms_norm_epsilon != 1.0e-5:
        text = text.replace("1.000000e-05", _mlir_float(spec.rms_norm_epsilon))
    if text.count('"func.call"') != spec.num_layers:
        raise FrontendError("internal error: decode layer call count drift")
    return text


def _render_fixed_query_module(
    spec: ModelSpec, *, bucket: int, logical_rows: int, physical_rows: int,
    layout_query_rows: int, layout_kv_length: int,
) -> str:
    generator = _decode_generator()
    profile = generator.Profile(
        spec.model_name, spec.num_layers, spec.sequence_length, spec.max_context,
        spec.hidden_size, spec.kv_hidden_size, spec.intermediate_size,
        spec.vocab_size, spec.query_heads, spec.kv_heads, spec.head_dim,
        q_hidden=spec.query_hidden_size, qk_head_norm=spec.qk_head_norm,
        norm_placement=spec.norm_placement,
    )
    program = "full_chunked_prefill" if logical_rows > 1 else "full_decode"
    header = f'''// Generated by etrinpu-frontend. Do not hand-edit.

#lhs = affine_map<(m, n, k) -> (m, k)>
#rhs_transposed = affine_map<(m, n, k) -> (n, k)>
#result = affine_map<(m, n, k) -> (m, n)>

module attributes {{
  etrinpu.model = {json.dumps(spec.model_name)},
  etrinpu.program = "{program}",
  etrinpu.sequence_length = {spec.sequence_length} : i64,
  etrinpu.layer_count = {spec.num_layers} : i64,
  etrinpu.vocab_size = {spec.vocab_size} : i64,
  etrinpu.norm_placement = "{spec.norm_placement}",
  etrinpu.decode.logical_query_length = {logical_rows} : i64,
  etrinpu.decode.physical_query_rows = {physical_rows} : i64,
  etrinpu.decode.kv_bucket = {bucket} : i64,
  etrinpu.decode.staging_slot = {bucket - logical_rows} : i64,
  etrinpu.decode.runtime_rope_rows = {logical_rows} : i64,
  etrinpu.decode.runtime_mask_elements = {logical_rows * bucket} : i64,
  etrinpu.decode.layout_query_rows = {layout_query_rows} : i64,
  etrinpu.decode.layout_kv_length = {layout_kv_length} : i64,
  etrinpu.cpu_preembedded_input
}} {{
'''
    text = (header + generator.emit_helper(
                profile, bucket, logical_rows, physical_rows,
                layout_query_rows, layout_kv_length) + "\n" +
            generator.emit_entry(
                profile, bucket, logical_rows, physical_rows,
                layout_query_rows, layout_kv_length) + "}\n")
    if spec.rms_norm_epsilon != 1.0e-5:
        text = text.replace("1.000000e-05", _mlir_float(spec.rms_norm_epsilon))
    return text


def render_chunked_prefill_mlir(spec: ModelSpec) -> str:
    """Render one reusable query-chunk x max-context prefill graph."""

    if spec.sequence_length % TILE:
        raise FrontendError("chunked prefill query length must be tile aligned")
    return _render_fixed_query_module(
        spec, bucket=spec.max_context, logical_rows=spec.sequence_length,
        physical_rows=spec.sequence_length,
        layout_query_rows=spec.sequence_length,
        layout_kv_length=spec.max_context,
    )


def render_shared_layout_decode_mlir(spec: ModelSpec, bucket: int) -> str:
    """Render decode whose allocation layout matches chunked prefill."""
    return _render_fixed_query_module(
        spec, bucket=bucket, logical_rows=1, physical_rows=TILE,
        layout_query_rows=spec.sequence_length,
        layout_kv_length=spec.max_context,
    )


def _compiler_model(spec: ModelSpec) -> dict[str, object]:
    if spec.rope.kind == "llama3":
        rope_scaling: dict[str, object] = {
            "rope_type": "llama3",
            "factor": spec.rope.factor,
            "low_freq_factor": spec.rope.low_frequency_factor,
            "high_freq_factor": spec.rope.high_frequency_factor,
            "original_max_position_embeddings": spec.rope.original_context,
        }
    else:
        rope_scaling = {"rope_type": "none"}
    return {
        "sequence_length": spec.sequence_length,
        "max_context": spec.max_context,
        "hidden_size": spec.hidden_size,
        "intermediate_size": spec.intermediate_size,
        "query_heads": spec.query_heads,
        "kv_heads": spec.kv_heads,
        "head_dim": spec.head_dim,
        "qk_head_norm": spec.qk_head_norm,
        "norm_placement": spec.norm_placement,
        "num_layers": spec.num_layers,
        "vocab_size": spec.vocab_size,
        "rms_norm_epsilon": spec.rms_norm_epsilon,
        "tie_word_embeddings": spec.tie_word_embeddings,
        "rope_theta": spec.rope.theta,
        "rope_scaling": rope_scaling,
    }


def make_compiler_config(spec: ModelSpec, kind: str,
                         bucket: int | None = None) -> dict[str, object]:
    """Build the schema-v2 configuration consumed by ``etrinpu-compile``."""

    if kind not in ("prefill", "decode", "chunked_prefill",
                    "shared_layout_decode"):
        raise FrontendError("unsupported compiler config kind")
    if kind in ("decode", "shared_layout_decode") and bucket is None:
        raise FrontendError("decode compiler config requires a bucket")
    if kind in ("prefill", "chunked_prefill") and bucket is not None:
        raise FrontendError("prefill compiler config cannot carry a decode bucket")
    total, _ = estimate_full_model_elements(spec)
    root: dict[str, object] = {
        "schema_version": 2,
        "name": f"npu64-{spec.model_name}-full-{kind}",
        "program_kind": (
            "full_prefill" if kind == "prefill" else
            "full_chunked_prefill" if kind == "chunked_prefill" else
            "full_decode"
        ),
        "hardware": {
            "pe_rows": TILE, "pe_columns": TILE,
            "matrix_tile_m": TILE, "matrix_tile_n": TILE, "matrix_tile_k": TILE,
            "vector_capacity": VECTOR_CAPACITY,
            "address_bits": ADDRESS_BITS, "scalar_address_bits": SCALAR_ADDRESS_BITS,
            "address_unit": "fp16_element",
            "require_full_matrix_tiles": True, "pad_partial_tiles": True,
        },
        "model": _compiler_model(spec),
        "memory": {
            "allocation_policy": "lexical_no_gap",
            "expand_kv_capacity_to_max_context": True,
            "reuse_activation_workspace": True,
            "inplace_hidden_state": True,
            "expected_total_elements": total,
            "expected_total_bytes": total * 2,
        },
        "runtime": {"embedding": "cpu", "argmax": "cpu"},
    }
    if kind == "prefill":
        root["runtime_inputs"] = {
            "hidden_state": "cpu", "rope_cosine": "cpu",
            "rope_sine": "cpu", "additive_causal_mask": "cpu",
        }
    elif kind == "chunked_prefill":
        root["decode"] = {
            "logical_query_length": spec.sequence_length,
            "physical_query_rows": spec.sequence_length,
            "kv_bucket": spec.max_context,
            "staging_slot": spec.max_context - spec.sequence_length,
            "layout_query_rows": spec.sequence_length,
            "layout_kv_length": spec.max_context,
            "per_head_attention_workspace": True,
            "host_append_after_step": True,
        }
        memory = root["memory"]
        assert isinstance(memory, dict)
        memory.pop("expected_total_elements", None)
        memory.pop("expected_total_bytes", None)
        memory["fixed_query_per_head_workspace"] = True
    else:
        assert bucket is not None
        root["decode"] = {
            "logical_query_length": 1, "physical_query_rows": TILE,
            "kv_bucket": bucket, "staging_slot": bucket - 1,
            "host_append_after_step": True,
        }
        memory = root["memory"]
        assert isinstance(memory, dict)
        memory["reuse_full_prefill_abi"] = True
        if kind == "shared_layout_decode":
            memory.pop("expected_total_elements", None)
            memory.pop("expected_total_bytes", None)
            memory["fixed_query_per_head_workspace"] = True
            decode = root["decode"]
            assert isinstance(decode, dict)
            decode["layout_query_rows"] = spec.sequence_length
            decode["layout_kv_length"] = spec.max_context
            decode["per_head_attention_workspace"] = True
        runtime = root["runtime"]
        assert isinstance(runtime, dict)
        runtime["kv_append"] = "cpu_after_full_step"
        root["runtime_inputs"] = {
            "hidden_state_row_0": "cpu", "rope_cosine_row_0": "cpu",
            "rope_sine_row_0": "cpu", "additive_attention_mask_prefix": "cpu",
        }
    return root


def _json_text(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _atomic_write(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(contents)
        temporary.replace(path)
    except OSError as error:
        try:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise FrontendError(f"cannot write {path}: {error}") from error


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1 << 20)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def emit_package(config_path: str | os.PathLike[str],
                 output_directory: str | os.PathLike[str],
                 sequence_length: int = 128,
                 max_context: int | None = None,
                 decode_buckets: Sequence[int] | None = None,
                 model_name: str | None = None,
                 analyze_only: bool = False) -> PackageResult:
    """Normalize, analyze, and emit a complete frontend package.

    ``model_spec.json`` and ``capability_report.json`` are written even when a
    valid model is outside target capabilities.  Normal compilation then
    raises :class:`UnsupportedModelError`; ``analyze_only=True`` returns the
    unsupported report without emitting MLIR.
    """

    spec = load_hf_config(config_path, sequence_length, max_context, model_name)
    report = analyze_capabilities(spec, decode_buckets)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    files: dict[str, Path] = {
        "model_spec": output / "model_spec.json",
        "capability_report": output / "capability_report.json",
    }
    _atomic_write(files["model_spec"], _json_text(spec.to_dict()))
    _atomic_write(files["capability_report"], _json_text(report.to_dict()))
    if analyze_only:
        return PackageResult(output, spec, report, files)
    if not report.supported:
        raise UnsupportedModelError(report)

    files.update({
        "prefill_mlir": output / "prefill.mlir",
        "prefill_config": output / "prefill_config.json",
    })
    _atomic_write(files["prefill_mlir"], render_prefill_mlir(spec))
    _atomic_write(files["prefill_config"],
                  _json_text(make_compiler_config(spec, "prefill")))
    for bucket in report.decode_buckets:
        mlir_key, config_key = f"decode_b{bucket}_mlir", f"decode_b{bucket}_config"
        files[mlir_key] = output / f"decode_b{bucket}.mlir"
        files[config_key] = output / f"decode_b{bucket}_config.json"
        _atomic_write(files[mlir_key], render_decode_mlir(spec, bucket))
        _atomic_write(files[config_key],
                      _json_text(make_compiler_config(spec, "decode", bucket)))

    package_path = output / "package.json"
    artifacts: dict[str, object] = {}
    for key, path in sorted(files.items()):
        artifacts[key] = {
            "file": path.name,
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    package = {
        "schema_version": SCHEMA_VERSION,
        "package_kind": "etrinpu_frontend",
        "requires_backend_contract": "etrinpu-static-dense-decoder-v3",
        "model_name": spec.model_name,
        "supported": True,
        "artifacts": artifacts,
        "compile_commands": {
            "prefill": "etrinpu-compile prefill.mlir --config prefill_config.json --output-dir <dir>",
            "decode": "etrinpu-compile decode_b<B>.mlir --config decode_b<B>_config.json --output-dir <dir>",
        },
    }
    _atomic_write(package_path, _json_text(package))
    files["package"] = package_path
    return PackageResult(output, spec, report, files)


def _parse_buckets(text: str | None) -> tuple[int, ...] | None:
    if text is None:
        return None
    if text.strip().lower() in ("", "none"):
        return ()
    result: list[int] = []
    for item in text.split(","):
        try:
            result.append(int(item.strip(), 10))
        except ValueError as error:
            raise FrontendError(f"invalid decode bucket {item!r}") from error
    return tuple(result)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Normalize a supported Hugging Face decoder config and emit ETRI NPU MLIR"
    )
    parser.add_argument("config", type=Path,
                        help="Hugging Face config.json or model directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=128,
                        help="static prefill sequence length (default: 128)")
    parser.add_argument("--max-context", type=int,
                        help="resident KV capacity (default: sequence length)")
    parser.add_argument("--decode-buckets",
                        help="comma-separated fixed buckets; default is powers of two through max context; use 'none' to omit decode")
    parser.add_argument("--model-name", help="safe package/model name override")
    parser.add_argument("--analyze-only", action="store_true",
                        help="write normalized spec/report but no MLIR; unsupported is not a command failure")
    args = parser.parse_args(argv)
    try:
        buckets = _parse_buckets(args.decode_buckets)
        result = emit_package(
            args.config, args.output_dir, args.sequence_length, args.max_context,
            buckets, args.model_name, args.analyze_only,
        )
    except UnsupportedModelError as error:
        print("etrinpu-frontend: model is unsupported:", file=sys.stderr)
        for issue in error.report.errors:
            print(f"  - [{issue.code}] {issue.path}: {issue.message}", file=sys.stderr)
        return 2
    except FrontendError as error:
        print(f"etrinpu-frontend: {error}", file=sys.stderr)
        return 1
    print(f"model_spec: {args.output_dir / 'model_spec.json'}")
    print(f"capability_report: {args.output_dir / 'capability_report.json'}")
    print(f"supported: {'yes' if result.capability_report.supported else 'no'}")
    if not args.analyze_only:
        print(f"package: {args.output_dir / 'package.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
