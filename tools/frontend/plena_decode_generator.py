#!/usr/bin/env python3
"""Emit deterministic fixed-query decode or chunked-prefill MLIR.

Unlike the prefill fixture, this source describes the active decode domains
directly.  Persistent K/V arguments have max-context capacity, attention reads
the [0, B) cache prefix, and the new K/V row is written only to staging slot
B-1.  Logical tensor work uses row zero; the target may execute a padded
64-row matrix tile, but those padding rows are not part of the source result.
With ``--query-rows 128`` the same graph becomes reusable chunked prefill:
attention spans the full KV bucket and the final 128 cache rows are staging.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import pathlib
import sys


@dataclass(frozen=True)
class Profile:
    name: str
    layers: int
    sequence: int
    max_context: int
    hidden: int
    kv_hidden: int
    intermediate: int
    vocab: int
    query_heads: int
    kv_heads: int
    head_dim: int
    q_hidden: int | None = None
    qk_head_norm: bool = False
    norm_placement: str = "pre"

    @property
    def half_head(self) -> int:
        return self.head_dim // 2

    @property
    def query_hidden(self) -> int:
        return self.q_hidden if self.q_hidden is not None else self.query_heads * self.head_dim


PROFILES = {
    "llama3_2_3b": Profile(
        "llama3_2_3b", 28, 128, 4096, 3072, 1024, 8192, 128256, 24, 8, 128
    ),
    # Matches the default pass configuration so etrinpu-opt can exercise the
    # schedule/lower boundary without a driver-owned JSON configuration.
    "llama3_2_3b_context128": Profile(
        "llama3_2_3b", 28, 128, 128, 3072, 1024, 8192, 128256, 24, 8, 128
    ),
    "tiny": Profile("tiny", 2, 64, 128, 64, 64, 64, 128, 1, 1, 64),
    "tiny_chunked": Profile(
        "tiny_chunked", 2, 128, 384, 64, 64, 64, 128, 1, 1, 64
    ),
    # Exercises Qwen3's expanded query projection and per-head Q/K RMSNorm
    # without changing the legacy tiny/Llama fixtures.
    "tiny_qwen3": Profile(
        "tiny_qwen3", 2, 64, 128, 64, 64, 128, 128, 2, 1, 64,
        q_hidden=128, qk_head_norm=True,
    ),
}


def arg(name: str, ty: str) -> tuple[str, str]:
    return name, ty


def abi(profile: Profile, bucket: int | None = None, logical_rows: int = 1,
        physical_rows: int = 64, layout_query_rows: int | None = None,
        layout_kv_length: int | None = None
        ) -> dict[str, list[tuple[str, str]]]:
    s, c = profile.sequence, profile.max_context
    h, qh, k, i = (profile.hidden, profile.query_hidden, profile.kv_hidden,
                    profile.intermediate)
    q, hh = profile.query_heads, profile.half_head
    fixed_query = (logical_rows > 1 or layout_query_rows is not None or
                   layout_kv_length is not None)
    layout_rows = layout_query_rows or (logical_rows if fixed_query else s)
    attention_kv = layout_kv_length or (
        bucket if fixed_query and bucket is not None else s
    )
    score_elements = max(layout_rows * attention_kv, layout_rows * k)
    score_type = (
        f"memref<{score_elements}xf16>" if fixed_query
        else f"memref<{q}x{s}x{s}xf16>"
    )
    attention_type = (
        f"memref<{layout_rows}x{attention_kv}xf16>" if fixed_query
        else f"memref<{q}x{s}x{s}xf16>"
    )
    return {
        "scalar_constants": [
            arg("sqrt_hidden_size_scalar", "memref<1xf16>"),
            *([arg("sqrt_head_dim_scalar", "memref<1xf16>")]
              if profile.qk_head_norm else []),
            arg("epsilon_scalar", "memref<1xf16>"),
            arg("attention_scale", "memref<1xf16>"),
            arg("silu_one_scalar", "memref<1xf16>"),
        ],
        "runtime": [
            arg("rope_cosine", f"memref<{s}x{hh}xf16>"),
            arg("rope_sine", f"memref<{s}x{hh}xf16>"),
            arg("additive_causal_mask",
                f"memref<{layout_rows}x{attention_kv}xf16>"
                if fixed_query else f"memref<{s}x{s}xf16>"),
        ],
        "weights": [
            arg("input_norm_gamma", f"memref<{h}xf16>"),
            arg("post_attention_norm_gamma", f"memref<{h}xf16>"),
            *([arg("q_norm_gamma", f"memref<{profile.head_dim}xf16>"),
               arg("k_norm_gamma", f"memref<{profile.head_dim}xf16>")]
              if profile.qk_head_norm else []),
            arg("q_weight", f"memref<{qh}x{h}xf16>"),
            arg("k_weight", f"memref<{k}x{h}xf16>"),
            arg("v_weight", f"memref<{k}x{h}xf16>"),
            arg("o_weight", f"memref<{h}x{qh}xf16>"),
            arg("gate_weight", f"memref<{i}x{h}xf16>"),
            arg("up_weight", f"memref<{i}x{h}xf16>"),
            arg("down_weight", f"memref<{h}x{i}xf16>"),
        ],
        "caches": [
            arg("value_cache", f"memref<{c}x{k}xf16>"),
            arg("key_cache", f"memref<{c}x{k}xf16>"),
        ],
        "scalar_scratch": [
            arg(name, "memref<1xf16>")
            for name in (
                "input_norm_sum_squares", "input_norm_mean_square",
                "input_norm_rms", "post_norm_sum_squares",
                "post_norm_mean_square", "post_norm_rms",
                "softmax_chunk_max", "softmax_global_max",
                "softmax_chunk_sum", "softmax_global_sum",
            )
        ],
        "activation_scratch": [
            arg("input_norm_squares", f"memref<{h}xf16>"),
            arg("input_norm_normalized", f"memref<{h}xf16>"),
            arg("input_norm_output", f"memref<{s}x{h}xf16>"),
            arg("query", f"memref<{s}x{qh}xf16>"),
            arg("key", f"memref<{s}x{k}xf16>"),
            arg("rope_temp0", f"memref<{hh}xf16>"),
            arg("rope_temp1", f"memref<{hh}xf16>"),
            arg("query_rope", f"memref<{s}x{qh}xf16>"),
            arg("attention_scores", score_type),
            arg("attention_scaled_temp", f"memref<{attention_kv}xf16>"),
            arg("masked_scores", attention_type),
            arg("softmax_centered", f"memref<{attention_kv}xf16>"),
            arg("softmax_exponentials", f"memref<{attention_kv}xf16>"),
            arg("probabilities", attention_type),
            arg("attention_output", f"memref<{s}x{qh}xf16>"),
            arg("projected_attention", f"memref<{s}x{h}xf16>"),
            arg("attention_residual", f"memref<{s}x{h}xf16>"),
            arg("post_norm_squares", f"memref<{h}xf16>"),
            arg("post_norm_normalized", f"memref<{h}xf16>"),
            arg("post_norm_output", f"memref<{s}x{h}xf16>"),
            arg("gate", f"memref<{s}x{i}xf16>"),
            arg("silu_negative", f"memref<{s}x{i}xf16>"),
            arg("silu_exponential", f"memref<{s}x{i}xf16>"),
            arg("silu_denominator", f"memref<{s}x{i}xf16>"),
            arg("silu_gate", f"memref<{s}x{i}xf16>"),
            arg("up", f"memref<{s}x{i}xf16>"),
            arg("swiglu", f"memref<{s}x{i}xf16>"),
            arg("down", f"memref<{s}x{h}xf16>"),
        ],
    }


def helper_args(profile: Profile, bucket: int | None = None,
                logical_rows: int = 1,
                physical_rows: int = 64,
                layout_query_rows: int | None = None,
                layout_kv_length: int | None = None) -> list[tuple[str, str]]:
    items = abi(profile, bucket, logical_rows, physical_rows,
                layout_query_rows, layout_kv_length)
    return (
        [arg("hidden", f"memref<{profile.sequence}x{profile.hidden}xf16>")]
        + items["scalar_constants"] + items["runtime"] + items["weights"]
        + items["caches"] + items["scalar_scratch"]
        + items["activation_scratch"]
    )


def fmt_signature(items: list[tuple[str, str]], indent: str = "      ") -> str:
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


def types(items: list[tuple[str, str]]) -> str:
    return ", ".join(ty for _, ty in items)


def alloc(name: str, ty: str, role: str, extra: str = "") -> str:
    attrs = f'npu.name = "{name}", npu.role = "{role}"'
    if extra:
        attrs += ", " + extra
    return f'    %{name} = "memref.alloc"() {{{attrs}}} : () -> {ty}'


def row_type(width: int, offset: int = 0, rows: int = 1) -> str:
    suffix = f", offset: {offset}" if offset else ""
    return f"memref<{rows}x{width}xf16, strided<[{width}, 1]{suffix}>>"


def emit_row_view(result: str, source: str, source_ty: str, width: int,
                  tag: str, row: int = 0, rows: int = 1) -> str:
    offset = row * width
    return (
        f"    %{result} = memref.subview %{source}[{row}, 0] [{rows}, {width}] "
        f"[1, 1] {{npu.decode_view = \"{tag}\"}} :\n"
        f"        {source_ty} to {row_type(width, offset, rows)}"
    )


def emit_helper(profile: Profile, bucket: int, logical_rows: int = 1,
                physical_rows: int = 64,
                layout_query_rows: int | None = None,
                layout_kv_length: int | None = None) -> str:
    p = profile
    s, c, h, qh, k, i = (p.sequence, p.max_context, p.hidden,
                          p.query_hidden, p.kv_hidden, p.intermediate)
    q, hh, slot = p.query_heads, p.half_head, bucket - logical_rows
    hidden_ty = f"memref<{s}x{h}xf16>"
    query_ty, key_ty = f"memref<{s}x{qh}xf16>", f"memref<{s}x{k}xf16>"
    cache_ty = f"memref<{c}x{k}xf16>"
    inter_ty = f"memref<{s}x{i}xf16>"
    row_h = row_type(h, rows=logical_rows)
    row_q = row_type(qh, rows=logical_rows)
    row_k = row_type(k, rows=logical_rows)
    row_i = row_type(i, rows=logical_rows)
    row_hh = row_type(hh, rows=logical_rows)
    signature = fmt_signature(
        helper_args(p, bucket, logical_rows, physical_rows,
                    layout_query_rows, layout_kv_length))
    abi_items = abi(p, bucket, logical_rows, physical_rows,
                    layout_query_rows, layout_kv_length)
    scratch_types = dict(abi_items["activation_scratch"])
    runtime_types = dict(abi_items["runtime"])
    attention_view_shape = (
        f"{bucket}" if logical_rows == 1
        else f"{logical_rows}x{bucket}"
    )
    attention_view_sizes = (
        f"[{bucket}]" if logical_rows == 1
        else f"[{logical_rows}, {bucket}]"
    )
    attention_view_strides = (
        "[1]" if logical_rows == 1 else f"[{bucket}, 1]"
    )
    attention_view_ty = f"memref<{attention_view_shape}xf16>"
    contract = (
        f"npu.decode.logical_query_length = {logical_rows} : i64, "
        f"npu.decode.physical_query_rows = {physical_rows} : i64, "
        f"npu.decode.kv_bucket = {bucket} : i64, "
        f"npu.decode.staging_slot = {slot} : i64"
    )
    views = [
        emit_row_view("hidden_row0", "hidden", hidden_ty, h, "hidden.row0",
                      rows=logical_rows),
        emit_row_view("input_norm_output_row0", "input_norm_output", hidden_ty,
                      h, "input_norm_output.row0", rows=logical_rows),
        emit_row_view("query_row0", "query", query_ty, qh, "query.row0",
                      rows=logical_rows),
        emit_row_view("key_row0", "key", key_ty, k, "key.row0",
                      rows=logical_rows),
        emit_row_view("rope_cosine_row0", "rope_cosine",
                      f"memref<{s}x{hh}xf16>", hh, "rope_cosine.row0",
                      rows=logical_rows),
        emit_row_view("rope_sine_row0", "rope_sine",
                      f"memref<{s}x{hh}xf16>", hh, "rope_sine.row0",
                      rows=logical_rows),
        emit_row_view("query_rope_row0", "query_rope", query_ty, qh,
                      "query_rope.row0", rows=logical_rows),
        (
            f"    %key_cache_prefix = memref.reinterpret_cast %key_cache to "
            f"offset: [0], sizes: [{bucket}, {p.kv_heads}, {p.head_dim}], "
            f"strides: [{k}, {p.head_dim}, 1] "
            f"{{npu.decode_view = \"key_cache.prefix\"}} :\n"
            f"        {cache_ty} to memref<{bucket}x{p.kv_heads}x{p.head_dim}xf16, "
            f"strided<[{k}, {p.head_dim}, 1]>>"
        ),
        (
            f"    %value_cache_prefix = memref.reinterpret_cast %value_cache to "
            f"offset: [0], sizes: [{bucket}, {p.kv_heads}, {p.head_dim}], "
            f"strides: [{k}, {p.head_dim}, 1] "
            f"{{npu.decode_view = \"value_cache.prefix\"}} :\n"
            f"        {cache_ty} to memref<{bucket}x{p.kv_heads}x{p.head_dim}xf16, "
            f"strided<[{k}, {p.head_dim}, 1]>>"
        ),
        emit_row_view("key_cache_staging", "key_cache", cache_ty, k,
                      "key_cache.staging", slot, logical_rows),
        emit_row_view("value_cache_staging", "value_cache", cache_ty, k,
                      "value_cache.staging", slot, logical_rows),
        (
            f"    %v_projection_scratch = memref.reinterpret_cast "
            f"%attention_scores to offset: [0], sizes: [{physical_rows}, {k}], "
            f"strides: [{k}, 1] "
            f"{{npu.decode_view = \"v_projection_scratch.physical\"}} :\n"
            f"        {scratch_types['attention_scores']} "
            f"to memref<{physical_rows}x{k}xf16>"
        ),
        emit_row_view("v_projection_row0", "v_projection_scratch",
                      f"memref<{physical_rows}x{k}xf16>", k,
                      "v_projection_scratch.row0", rows=logical_rows),
        (
            f"    %attention_mask_bucket = memref.reinterpret_cast "
            f"%additive_causal_mask to offset: [0], sizes: {attention_view_sizes}, "
            f"strides: {attention_view_strides} "
            f"{{npu.decode_view = \"attention_mask.bucket\"}} :\n"
            f"        {runtime_types['additive_causal_mask']} "
            f"to memref<{attention_view_shape}xf16>"
        ),
        (
            f"    %attention_scores_bucket = memref.reinterpret_cast "
            f"%attention_scores to offset: [0], sizes: {attention_view_sizes}, "
            f"strides: {attention_view_strides} "
            f"{{npu.decode_view = \"attention_scores.bucket\"}} :\n"
            f"        {scratch_types['attention_scores']} "
            f"to memref<{attention_view_shape}xf16>"
        ),
        (
            f"    %masked_scores_bucket = memref.reinterpret_cast "
            f"%masked_scores to offset: [0], sizes: {attention_view_sizes}, "
            f"strides: {attention_view_strides} "
            f"{{npu.decode_view = \"masked_scores.bucket\"}} :\n"
            f"        {scratch_types['masked_scores']} "
            f"to memref<{attention_view_shape}xf16>"
        ),
        (
            f"    %probabilities_bucket = memref.reinterpret_cast "
            f"%probabilities to offset: [0], sizes: {attention_view_sizes}, "
            f"strides: {attention_view_strides} "
            f"{{npu.decode_view = \"probabilities.bucket\"}} :\n"
            f"        {scratch_types['probabilities']} "
            f"to memref<{attention_view_shape}xf16>"
        ),
        emit_row_view("attention_output_row0", "attention_output", query_ty,
                      qh, "attention_output.row0", rows=logical_rows),
        emit_row_view("projected_attention_row0", "projected_attention",
                      hidden_ty, h, "projected_attention.row0",
                      rows=logical_rows),
        emit_row_view("attention_residual_row0", "attention_residual",
                      hidden_ty, h, "attention_residual.row0",
                      rows=logical_rows),
        emit_row_view("post_norm_output_row0", "post_norm_output", hidden_ty,
                      h, "post_norm_output.row0", rows=logical_rows),
        emit_row_view("gate_row0", "gate", inter_ty, i, "gate.row0",
                      rows=logical_rows),
        emit_row_view("silu_negative_row0", "silu_negative", inter_ty, i,
                      "silu_negative.row0", rows=logical_rows),
        emit_row_view("silu_exponential_row0", "silu_exponential", inter_ty,
                      i, "silu_exponential.row0", rows=logical_rows),
        emit_row_view("silu_denominator_row0", "silu_denominator", inter_ty,
                      i, "silu_denominator.row0", rows=logical_rows),
        emit_row_view("silu_gate_row0", "silu_gate", inter_ty, i,
                      "silu_gate.row0", rows=logical_rows),
        emit_row_view("up_row0", "up", inter_ty, i, "up.row0",
                      rows=logical_rows),
        emit_row_view("swiglu_row0", "swiglu", inter_ty, i, "swiglu.row0",
                      rows=logical_rows),
        emit_row_view("down_row0", "down", hidden_ty, h, "down.row0",
                      rows=logical_rows),
    ]
    if p.qk_head_norm:
        head_views = [
            (
                f"    %query_heads_row0 = memref.reinterpret_cast %query "
                f"to offset: [0], sizes: [{logical_rows * p.query_heads}, {p.head_dim}], "
                f"strides: [{p.head_dim}, 1] "
                f'{{npu.decode_view = "query.heads.row0"}} :\n'
                f"        {query_ty} to memref<{logical_rows * p.query_heads}x{p.head_dim}xf16, "
                f"strided<[{p.head_dim}, 1]>>"
            ),
            (
                f"    %key_heads_row0 = memref.reinterpret_cast %key "
                f"to offset: [0], sizes: [{logical_rows * p.kv_heads}, {p.head_dim}], "
                f"strides: [{p.head_dim}, 1] "
                f'{{npu.decode_view = "key.heads.row0"}} :\n'
                f"        {key_ty} to memref<{logical_rows * p.kv_heads}x{p.head_dim}xf16, "
                f"strided<[{p.head_dim}, 1]>>"
            ),
        ]
        views[4:4] = head_views
    view_text = "\n".join(views)
    qk_norm = ""
    if p.qk_head_norm:
        head_matrix_q = (
            f"memref<{logical_rows * p.query_heads}x{p.head_dim}xf16, "
            f"strided<[{p.head_dim}, 1]>>"
        )
        head_matrix_k = (
            f"memref<{logical_rows * p.kv_heads}x{p.head_dim}xf16, "
            f"strided<[{p.head_dim}, 1]>>"
        )
        qk_norm = f'''
    "etrinpu.rms_norm"(%query_heads_row0, %q_norm_gamma,
        %sqrt_head_dim_scalar, %epsilon_scalar, %input_norm_squares,
        %input_norm_sum_squares, %input_norm_mean_square, %input_norm_rms,
        %input_norm_normalized, %query_heads_row0) {{
        epsilon = 1.000000e-05 : f32, hidden_size = {p.head_dim} : i64,
        npu.stage = "q_head_rms_norm"}} :
        ({head_matrix_q}, memref<{p.head_dim}xf16>, memref<1xf16>, memref<1xf16>,
         memref<{h}xf16>, memref<1xf16>, memref<1xf16>, memref<1xf16>,
         memref<{h}xf16>, {head_matrix_q}) -> ()
    "etrinpu.rms_norm"(%key_heads_row0, %k_norm_gamma,
        %sqrt_head_dim_scalar, %epsilon_scalar, %input_norm_squares,
        %input_norm_sum_squares, %input_norm_mean_square, %input_norm_rms,
        %input_norm_normalized, %key_heads_row0) {{
        epsilon = 1.000000e-05 : f32, hidden_size = {p.head_dim} : i64,
        npu.stage = "k_head_rms_norm"}} :
        ({head_matrix_k}, memref<{p.head_dim}xf16>, memref<1xf16>, memref<1xf16>,
         memref<{h}xf16>, memref<1xf16>, memref<1xf16>, memref<1xf16>,
         memref<{h}xf16>, {head_matrix_k}) -> ()
'''
    if p.norm_placement == "pre":
        input_norm = f'''
    "etrinpu.rms_norm"(%hidden_row0, %input_norm_gamma,
        %sqrt_hidden_size_scalar, %epsilon_scalar, %input_norm_squares,
        %input_norm_sum_squares, %input_norm_mean_square, %input_norm_rms,
        %input_norm_normalized, %input_norm_output_row0) {{
        epsilon = 1.000000e-05 : f32, hidden_size = {h} : i64}} :
        ({row_h}, memref<{h}xf16>, memref<1xf16>, memref<1xf16>,
         memref<{h}xf16>, memref<1xf16>, memref<1xf16>, memref<1xf16>,
         memref<{h}xf16>, {row_h}) -> ()
'''
        attention_input = "%input_norm_output_row0"
        attention_transition = f'''
    "etrinpu.elementwise"(%hidden_row0, %projected_attention_row0,
        %attention_residual_row0) {{kind = "add"}} :
        ({row_h}, {row_h}, {row_h}) -> ()
    "etrinpu.rms_norm"(%attention_residual_row0,
        %post_attention_norm_gamma, %sqrt_hidden_size_scalar, %epsilon_scalar,
        %post_norm_squares, %post_norm_sum_squares, %post_norm_mean_square,
        %post_norm_rms, %post_norm_normalized, %post_norm_output_row0) {{
        epsilon = 1.000000e-05 : f32, hidden_size = {h} : i64}} :
        ({row_h}, memref<{h}xf16>, memref<1xf16>, memref<1xf16>,
         memref<{h}xf16>, memref<1xf16>, memref<1xf16>, memref<1xf16>,
         memref<{h}xf16>, {row_h}) -> ()
'''
        ffn_input = "%post_norm_output_row0"
        ffn_transition = f'''
    "etrinpu.elementwise"(%attention_residual_row0, %down_row0,
        %hidden_row0) {{kind = "add"}} : ({row_h}, {row_h}, {row_h}) -> ()
'''
    elif p.norm_placement == "post":
        input_norm = ""
        attention_input = "%hidden_row0"
        attention_transition = f'''
    "etrinpu.rms_norm"(%projected_attention_row0, %input_norm_gamma,
        %sqrt_hidden_size_scalar, %epsilon_scalar, %input_norm_squares,
        %input_norm_sum_squares, %input_norm_mean_square, %input_norm_rms,
        %input_norm_normalized, %input_norm_output_row0) {{
        epsilon = 1.000000e-05 : f32, hidden_size = {h} : i64}} :
        ({row_h}, memref<{h}xf16>, memref<1xf16>, memref<1xf16>,
         memref<{h}xf16>, memref<1xf16>, memref<1xf16>, memref<1xf16>,
         memref<{h}xf16>, {row_h}) -> ()
    "etrinpu.elementwise"(%hidden_row0, %input_norm_output_row0,
        %attention_residual_row0) {{kind = "add"}} :
        ({row_h}, {row_h}, {row_h}) -> ()
'''
        ffn_input = "%attention_residual_row0"
        ffn_transition = f'''
    "etrinpu.rms_norm"(%down_row0, %post_attention_norm_gamma,
        %sqrt_hidden_size_scalar, %epsilon_scalar, %post_norm_squares,
        %post_norm_sum_squares, %post_norm_mean_square, %post_norm_rms,
        %post_norm_normalized, %post_norm_output_row0) {{
        epsilon = 1.000000e-05 : f32, hidden_size = {h} : i64}} :
        ({row_h}, memref<{h}xf16>, memref<1xf16>, memref<1xf16>,
         memref<{h}xf16>, memref<1xf16>, memref<1xf16>, memref<1xf16>,
         memref<{h}xf16>, {row_h}) -> ()
    "etrinpu.elementwise"(%attention_residual_row0, %post_norm_output_row0,
        %hidden_row0) {{kind = "add"}} : ({row_h}, {row_h}, {row_h}) -> ()
'''
    else:
        raise ValueError("norm_placement must be pre or post")
    return f'''  func.func private @decode_layer(
{signature}) -> {hidden_ty} attributes {{etrinpu.decode_layer_template, {contract}}} {{
    %zero = arith.constant 0.000000e+00 : f16
{view_text}
{input_norm.rstrip()}

    linalg.fill ins(%zero : f16) outs(%query_row0 : {row_q})
    linalg.matmul indexing_maps = [#lhs, #rhs_transposed, #result]
        ins({attention_input}, %q_weight : {row_h}, memref<{qh}x{h}xf16>)
        outs(%query_row0 : {row_q})
    linalg.fill ins(%zero : f16) outs(%key_row0 : {row_k})
    linalg.matmul indexing_maps = [#lhs, #rhs_transposed, #result]
        ins({attention_input}, %k_weight : {row_h}, memref<{k}x{h}xf16>)
        outs(%key_row0 : {row_k})
{qk_norm}

    // V uses activation scratch; the persistent cache is never filled or
    // projected wholesale.  Only the logical query rows are copied into the
    // fixed staging block at the end of the KV domain.
    linalg.fill ins(%zero : f16) outs(%v_projection_row0 : {row_k})
    linalg.matmul indexing_maps = [#lhs, #rhs_transposed, #result]
        ins({attention_input}, %v_weight : {row_h}, memref<{k}x{h}xf16>)
        outs(%v_projection_row0 : {row_k})
    memref.copy %v_projection_row0, %value_cache_staging : {row_k} to {row_type(k, slot * k, logical_rows)}

    "etrinpu.rope"(%query_row0, %rope_cosine_row0, %rope_sine_row0,
        %rope_temp0, %rope_temp1, %query_rope_row0) {{heads = {q} : i64,
        head_dim = {p.head_dim} : i64}} :
        ({row_q}, {row_hh}, {row_hh}, memref<{hh}xf16>, memref<{hh}xf16>,
         {row_q}) -> ()
    "etrinpu.rope"(%key_row0, %rope_cosine_row0, %rope_sine_row0,
        %rope_temp0, %rope_temp1, %key_cache_staging) {{heads = {p.kv_heads} : i64,
        head_dim = {p.head_dim} : i64}} :
        ({row_k}, {row_hh}, {row_hh}, memref<{hh}xf16>, memref<{hh}xf16>,
         {row_type(k, slot * k, logical_rows)}) -> ()

    // The four semantic attention ops form one fused target schedule. The
    // target batches QK by GQA KV-head group, while softmax/PV retain the
    // per-query-row meaning represented here and reuse the same arenas.
    "etrinpu.gqa_qk"(%query_rope_row0, %key_cache_prefix,
        %attention_scores_bucket) {{head_dim = {p.head_dim} : i64,
        kv_heads = {p.kv_heads} : i64, query_heads = {q} : i64,
        npu.workspace_reuse = "one_query_head"}} :
        ({row_q}, memref<{bucket}x{p.kv_heads}x{p.head_dim}xf16,
         strided<[{k}, {p.head_dim}, 1]>>,
         {attention_view_ty}) -> ()
    "etrinpu.scale_mask"(%attention_scores_bucket, %attention_scale,
        %attention_mask_bucket, %probabilities_bucket, %masked_scores_bucket)
        {{npu.workspace_reuse = "one_query_head"}} :
        ({attention_view_ty}, memref<1xf16>, {attention_view_ty},
         {attention_view_ty}, {attention_view_ty}) -> ()
    "etrinpu.softmax"(%masked_scores_bucket, %softmax_chunk_max,
        %softmax_global_max, %attention_scores_bucket, %probabilities_bucket,
        %softmax_chunk_sum, %softmax_global_sum, %probabilities_bucket)
        {{npu.workspace_reuse = "one_query_head"}} :
        ({attention_view_ty}, memref<1xf16>, memref<1xf16>,
         {attention_view_ty}, {attention_view_ty}, memref<1xf16>,
         memref<1xf16>, {attention_view_ty}) -> ()
    "etrinpu.gqa_pv"(%probabilities_bucket, %value_cache_prefix,
        %attention_output_row0) {{head_dim = {p.head_dim} : i64,
        kv_heads = {p.kv_heads} : i64, query_heads = {q} : i64,
        npu.workspace_reuse = "one_query_head"}} :
        ({attention_view_ty}, memref<{bucket}x{p.kv_heads}x{p.head_dim}xf16,
         strided<[{k}, {p.head_dim}, 1]>>,
         {row_q}) -> ()

    linalg.fill ins(%zero : f16) outs(%projected_attention_row0 : {row_h})
    linalg.matmul indexing_maps = [#lhs, #rhs_transposed, #result]
        ins(%attention_output_row0, %o_weight : {row_q}, memref<{h}x{qh}xf16>)
        outs(%projected_attention_row0 : {row_h})
{attention_transition.rstrip()}

    linalg.fill ins(%zero : f16) outs(%gate_row0 : {row_i})
    linalg.matmul indexing_maps = [#lhs, #rhs_transposed, #result]
        ins({ffn_input}, %gate_weight : {row_h}, memref<{i}x{h}xf16>)
        outs(%gate_row0 : {row_i})
    "etrinpu.silu"(%gate_row0, %silu_one_scalar, %silu_negative_row0,
        %silu_exponential_row0, %silu_denominator_row0, %silu_gate_row0) :
        ({row_i}, memref<1xf16>, {row_i}, {row_i}, {row_i}, {row_i}) -> ()
    linalg.fill ins(%zero : f16) outs(%up_row0 : {row_i})
    linalg.matmul indexing_maps = [#lhs, #rhs_transposed, #result]
        ins({ffn_input}, %up_weight : {row_h}, memref<{i}x{h}xf16>)
        outs(%up_row0 : {row_i})
    "etrinpu.elementwise"(%silu_gate_row0, %up_row0, %swiglu_row0)
        {{kind = "mul"}} : ({row_i}, {row_i}, {row_i}) -> ()
    linalg.fill ins(%zero : f16) outs(%down_row0 : {row_h})
    linalg.matmul indexing_maps = [#lhs, #rhs_transposed, #result]
        ins(%swiglu_row0, %down_weight : {row_i}, memref<{h}x{i}xf16>)
        outs(%down_row0 : {row_h})
{ffn_transition.rstrip()}
    return %hidden : {hidden_ty}
  }}
'''


def layer_region(layer: int, local: str) -> str:
    return f"layer_{layer:02d}_{local}"


def emit_entry(profile: Profile, bucket: int, logical_rows: int = 1,
               physical_rows: int = 64,
               layout_query_rows: int | None = None,
               layout_kv_length: int | None = None) -> str:
    p = profile
    items = abi(profile, bucket, logical_rows, physical_rows,
                layout_query_rows, layout_kv_length)
    s, h, k, i = p.sequence, p.hidden, p.kv_hidden, p.intermediate
    hidden_ty = f"memref<{s}x{h}xf16>"
    final_row_ty = row_type(h, (logical_rows - 1) * h)
    args = helper_args(p, bucket, logical_rows, physical_rows,
                       layout_query_rows, layout_kv_length)
    lines = [
        '  func.func @decode_full() attributes {etrinpu.entry_point = "decode"} {'
    ]
    for name, ty in items["scalar_constants"]:
        lines.append(alloc(name, ty, "scalar_constant"))
    for name, ty in items["scalar_scratch"]:
        lines.append(alloc(name, ty, "scalar_scratch"))
    lines.append(alloc("hidden_state", hidden_ty, "runtime_input"))
    for name, ty in items["runtime"]:
        lines.append(alloc(name, ty, "runtime_input"))
    for layer in range(p.layers):
        for local, ty in items["weights"]:
            lines.append(alloc(layer_region(layer, local), ty, "constant"))
    for layer in range(p.layers):
        for local, ty in items["caches"]:
            kind = "value" if local.startswith("value") else "key"
            extra = (
                'npu.capacity_axis = 0 : i64, '
                'npu.capacity_from_config = "max_context", '
                f'npu.kv_kind = "{kind}"'
            )
            lines.append(alloc(layer_region(layer, local), ty, "state", extra))
    lines.append(alloc("final_norm_gamma", f"memref<{h}xf16>", "constant"))
    lines.append(alloc("token_embedding_lm_head_weight",
                       f"memref<{p.vocab}x{h}xf16>", "constant"))
    for name, ty in items["activation_scratch"]:
        lines.append(alloc(name, ty, "scratch"))
    lines.append(alloc("logits", f"memref<1x{p.vocab}xf16>", "runtime_output"))
    lines.append("")

    prefix = [name for name, _ in items["scalar_constants"] + items["runtime"]]
    scratch = [name for name, _ in items["scalar_scratch"] + items["activation_scratch"]]
    helper_types = types(args)
    previous = "%hidden_state"
    contract = (
        f"npu.decode.logical_query_length = {logical_rows} : i64, "
        f"npu.decode.physical_query_rows = {physical_rows} : i64, "
        f"npu.decode.kv_bucket = {bucket} : i64, "
        f"npu.decode.staging_slot = {bucket - logical_rows} : i64"
    )
    for layer in range(p.layers):
        operands = [previous]
        operands += [f"%{name}" for name in prefix]
        operands += [f"%{layer_region(layer, name)}" for name, _ in items["weights"]]
        operands += [f"%{layer_region(layer, name)}" for name, _ in items["caches"]]
        operands += [f"%{name}" for name in scratch]
        result = f"%hidden_{layer + 1:02d}"
        lines.append(
            f'    {result} = "func.call"({", ".join(operands)}) '
            f'{{callee = @decode_layer, npu.layer_index = {layer} : i64, {contract}}} : '
            f'({helper_types}) -> {hidden_ty}'
        )
        previous = result

    lines += [
        "",
        emit_row_view("final_input_row0", previous[1:], hidden_ty, h,
                      "final.input.row0", logical_rows - 1),
        emit_row_view("final_output_row0", "hidden_state", hidden_ty, h,
                      "final.output.row0", logical_rows - 1),
        f'    "etrinpu.rms_norm"(%final_input_row0, %final_norm_gamma,',
        "        %sqrt_hidden_size_scalar, %epsilon_scalar, %input_norm_squares,",
        "        %input_norm_sum_squares, %input_norm_mean_square, %input_norm_rms,",
        "        %input_norm_normalized, %final_output_row0) {",
        f"        epsilon = 1.000000e-05 : f32, hidden_size = {h} : i64,",
        '        npu.stage = "final_norm"} :',
        f"        ({final_row_ty}, memref<{h}xf16>, memref<1xf16>, memref<1xf16>,",
        f"         memref<{h}xf16>, memref<1xf16>, memref<1xf16>, memref<1xf16>,",
        f"         memref<{h}xf16>, {final_row_ty}) -> ()",
        "    %zero = arith.constant 0.000000e+00 : f16",
        f"    linalg.fill ins(%zero : f16) outs(%logits : memref<1x{p.vocab}xf16>)",
        "    linalg.matmul indexing_maps = [#lhs, #rhs_transposed, #result]",
        "        ins(%final_output_row0, %token_embedding_lm_head_weight :",
        f"            {final_row_ty}, memref<{p.vocab}x{h}xf16>)",
        f"        outs(%logits : memref<1x{p.vocab}xf16>)",
        "    return",
        "  }",
    ]
    return "\n".join(lines) + "\n"


def generate(profile: str = "llama3_2_3b", bucket: int = 128,
             logical_rows: int = 1, physical_rows: int = 64,
             layout_query_rows: int | None = None,
             layout_kv_length: int | None = None) -> str:
    p = PROFILES[profile]
    if p.query_hidden != p.query_heads * p.head_dim:
        raise ValueError("query hidden size must equal query_heads * head_dim")
    if bucket <= 0 or bucket % 64:
        raise ValueError("KV bucket must be a positive multiple of 64")
    if bucket > p.max_context:
        raise ValueError(
            f"KV bucket {bucket} exceeds {profile} max context {p.max_context}"
        )
    if logical_rows <= 0 or logical_rows > physical_rows:
        raise ValueError("logical query rows must be in 1..physical rows")
    if physical_rows <= 0 or physical_rows % 64 or physical_rows > p.sequence:
        raise ValueError("physical query rows must be a sequence-bounded multiple of 64")
    if logical_rows > 1 and logical_rows != physical_rows:
        raise ValueError("multi-row fixed query currently requires full query tiles")
    fixed_layout = layout_query_rows is not None or layout_kv_length is not None
    if fixed_layout:
        if layout_query_rows is None or layout_kv_length is None:
            raise ValueError("both fixed layout extents must be provided")
        if (layout_query_rows % 64 or layout_query_rows < physical_rows or
                layout_query_rows > p.sequence or
                layout_kv_length < bucket or layout_kv_length > p.max_context):
            raise ValueError("fixed layout extents are incompatible with execution")
    if logical_rows == 1 and not fixed_layout and 64 * p.kv_hidden > p.query_heads * p.sequence * p.sequence:
        raise ValueError("attention_scores arena cannot hold the 64-row V scratch")
    if logical_rows == 1 and not fixed_layout and bucket > p.sequence * p.sequence:
        raise ValueError("runtime mask arena cannot hold the requested KV bucket")
    effective_layout_rows = (
        layout_query_rows if fixed_layout else
        (logical_rows if logical_rows > 1 else None)
    )
    effective_layout_kv = (
        layout_kv_length if fixed_layout else
        (bucket if logical_rows > 1 else None)
    )
    layout_attrs = ""
    if effective_layout_rows is not None:
        layout_attrs = (
            f"  etrinpu.decode.layout_query_rows = {effective_layout_rows} : i64,\n"
            f"  etrinpu.decode.layout_kv_length = {effective_layout_kv} : i64,\n"
        )
    program = "full_chunked_prefill" if logical_rows > 1 else (
        "full_decode_tiny" if profile == "tiny" else "full_decode"
    )
    header = f'''// Generated by generate_full_decode_mlir.py. Do not hand-edit.
// CPU supplies the embedded token in hidden_state row zero and clears rows 1..63.
// Runtime RoPE uses the logical position; host commits staged K/V after the program.

#lhs = affine_map<(m, n, k) -> (m, k)>
#rhs_transposed = affine_map<(m, n, k) -> (n, k)>
#result = affine_map<(m, n, k) -> (m, n)>

module attributes {{
  etrinpu.model = "{p.name}",
  etrinpu.program = "{program}",
  etrinpu.sequence_length = {p.sequence} : i64,
  etrinpu.layer_count = {p.layers} : i64,
  etrinpu.vocab_size = {p.vocab} : i64,
  etrinpu.norm_placement = "{p.norm_placement}",
  etrinpu.decode.logical_query_length = {logical_rows} : i64,
  etrinpu.decode.physical_query_rows = {physical_rows} : i64,
  etrinpu.decode.kv_bucket = {bucket} : i64,
  etrinpu.decode.staging_slot = {bucket - logical_rows} : i64,
  etrinpu.decode.runtime_rope_rows = {logical_rows} : i64,
  etrinpu.decode.runtime_mask_elements = {logical_rows * bucket} : i64,
{layout_attrs.rstrip()}
  etrinpu.cpu_preembedded_input
}} {{
'''
    return (header + emit_helper(
                p, bucket, logical_rows, physical_rows,
                layout_query_rows, layout_kv_length) +
            "\n" + emit_entry(
                p, bucket, logical_rows, physical_rows,
                layout_query_rows, layout_kv_length) + "}\n")


def validate(text: str, profile: str, bucket: int, logical_rows: int = 1,
             physical_rows: int = 64) -> None:
    p = PROFILES[profile]
    required = {
        'etrinpu.entry_point = "decode"': 1,
        f"etrinpu.decode.logical_query_length = {logical_rows} : i64": 1,
        f"etrinpu.decode.physical_query_rows = {physical_rows} : i64": 1,
        f"etrinpu.decode.kv_bucket = {bucket} : i64": 1,
        f"etrinpu.decode.staging_slot = {bucket - logical_rows} : i64": 1,
        "func.func private @decode_layer": 1,
        '"func.call"': p.layers,
        'npu.capacity_from_config = "max_context"': 2 * p.layers,
        'npu.decode_view = "key_cache.prefix"': 1,
        'npu.decode_view = "key_cache.staging"': 1,
        'npu.decode_view = "value_cache.staging"': 1,
        'npu.decode_view = "v_projection_scratch.physical"': 1,
        'npu.decode_view = "attention_scores.bucket"': 1,
        "memref.copy %v_projection_row0, %value_cache_staging": 1,
        f"memref.subview %hidden_state[{logical_rows - 1}, 0]": 1,
    }
    for marker, expected in required.items():
        actual = text.count(marker)
        if actual != expected:
            raise ValueError(
                f"fixture marker {marker!r}: expected {expected}, found {actual}"
            )
    if "outs(%value_cache" in text:
        raise ValueError("decode must not fill or project the persistent V cache")
    call_bucket = f"npu.decode.kv_bucket = {bucket} : i64"
    if text.count(call_bucket) != p.layers + 2:
        raise ValueError("decode bucket is not repeated on helper and every call")
    for layer in range(p.layers):
        if text.count(f"npu.layer_index = {layer} : i64") != 1:
            raise ValueError(f"missing unique layer {layer} decode call")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=sorted(PROFILES),
                        default="llama3_2_3b")
    parser.add_argument("--kv-bucket", type=int, default=128)
    parser.add_argument("--query-rows", type=int, default=1)
    parser.add_argument("--physical-query-rows", type=int, default=64)
    parser.add_argument("--layout-query-rows", type=int)
    parser.add_argument("--layout-kv-length", type=int)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--check", type=pathlib.Path)
    args = parser.parse_args()
    try:
        text = generate(args.profile, args.kv_bucket, args.query_rows,
                        args.physical_query_rows, args.layout_query_rows,
                        args.layout_kv_length)
        validate(text, args.profile, args.kv_bucket, args.query_rows,
                 args.physical_query_rows)
    except ValueError as error:
        print(f"generate_full_decode_mlir.py: {error}", file=sys.stderr)
        return 1
    if args.check:
        if args.check.read_text() != text:
            print(f"generated fixture differs from {args.check}", file=sys.stderr)
            return 1
        return 0
    if args.output:
        args.output.write_text(text)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
