#!/usr/bin/env python3
"""Generate/check a supported dense Llama decoder on PLENA Program v7.

The default smoke test executes every prompt token through layer 0. Use
``--num-layers`` for a prefix, ``--with-lm-head`` for final norm/tied logits,
or ``--full-model`` for every checkpoint layer and the LM head. Computation
uses IEEE FP16 weight/activation boundaries. Matrix instructions use square-SA
tiles with M<=32, N<=32, and K<=64. RMSNorm, RoPE, softmax, SiLU, and
attention are lowered to generic Matrix/Vector/Scalar primitive instructions.

Linear weights are prefetched as all K tiles for one N=32 output block. Two
512-KiB private-L1 regions alternate: the first block is blocking, later blocks use
asynchronous raw DMA and become visible at their first Matrix read.

Only selected embeddings, requested decoder layers, and optional final
norm/LM-head weights are placed in the generated LP6 byte image.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import functools
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from safetensors import safe_open
from transformers import AutoTokenizer

from program_v7 import SystemProgramBuilder
from bounded_vector_emitter import BoundedVectorEmitterMixin
from plena_isa_encoding import (
    OP_C_SET_V2,
    matrix_load, matrix_mma, matrix_writeout,
    OP_L2_DMA,
    OP_M_MMA,
    OP_M_WRITEOUT,
    OP_S_ALU_F32,
    OP_V_BINARY_F16,
    OP_V_QUANT_I8,
    OP_V_REDUCE,
    OP_V_SCALAR_F16,
    OP_V_UNARY_F16,
    load_u32,
    rform,
)


M_TILE = 32
N_TILE = 32
K_TILE = 64
VECTOR_ZERO_ADDRESS = 0
SRAM_WEIGHT_REGION_BASE = 2 * 1024 * 1024
SRAM_WEIGHT_BATCH_CAPACITY = 512 * 1024
SRAM_WEIGHT_BATCH_ADDRESSES = (
    SRAM_WEIGHT_REGION_BASE,
    SRAM_WEIGHT_REGION_BASE + SRAM_WEIGHT_BATCH_CAPACITY,
)
SRAM_NORM_WEIGHT_ADDRESS = SRAM_WEIGHT_REGION_BASE + 2 * SRAM_WEIGHT_BATCH_CAPACITY

# C_SET_V2 funct values.
SET_VECTOR_ELEMENTS = 3
SET_DMA_BYTES = 7
OP_V_MEMORY = 0x35
OP_M_LOAD = 0x37


def align(value: int, amount: int = 64) -> int:
    return (value + amount - 1) // amount * amount


def round_f16(value: torch.Tensor) -> torch.Tensor:
    return value.to(torch.float16)


def round_scalar_f16(value: float) -> float:
    try:
        return struct.unpack("<e", struct.pack("<e", value))[0]
    except OverflowError:
        return math.copysign(math.inf, value)


@functools.lru_cache(maxsize=1)
def _host_expf_symbol():
    """Resolve the host ``expf`` used by Rust's ``f32::exp`` on Linux.

    The emulator evaluates scalar Vector-Unit transcendental operations with
    Rust ``f32`` methods.  PyTorch's vectorized ``torch.exp`` is allowed to use
    a different approximation and can therefore differ by one FP32 ULP.  The
    golden model must use the same host scalar libm operation when it promises
    bit-exact comparison with the Rust executable.
    """

    libm_name = ctypes.util.find_library("m")
    if libm_name is None:
        raise RuntimeError(
            "cannot resolve host libm; a scalar expf implementation is required "
            "for the Rust-compatible SiLU reference"
        )
    libm = ctypes.CDLL(libm_name)
    try:
        expf = libm.expf
    except AttributeError as error:
        raise RuntimeError(
            f"host math library {libm_name!r} does not export expf"
        ) from error
    expf.argtypes = [ctypes.c_float]
    expf.restype = ctypes.c_float
    # Keep the CDLL alive for as long as the cached function pointer is used.
    return libm, expf


@functools.lru_cache(maxsize=1)
def _host_sqrtf_symbol():
    """Resolve the scalar ``sqrtf`` used by Rust's ``f32::sqrt``."""

    libm_name = ctypes.util.find_library("m")
    if libm_name is None:
        raise RuntimeError("cannot resolve host libm sqrtf")
    libm = ctypes.CDLL(libm_name)
    try:
        sqrtf = libm.sqrtf
    except AttributeError as error:
        raise RuntimeError(
            f"host math library {libm_name!r} does not export sqrtf"
        ) from error
    sqrtf.argtypes = [ctypes.c_float]
    sqrtf.restype = ctypes.c_float
    return libm, sqrtf


def _round_host_f32(value: float) -> float:
    """Round one Python scalar at an explicit Rust-f32 operation boundary."""

    return float(ctypes.c_float(value).value)


@dataclass(frozen=True)
class SramTensor:
    name: str
    address: int
    shape: tuple[int, ...]
    element_bytes: int = 2

    @property
    def elements(self) -> int:
        return math.prod(self.shape)

    @property
    def bytes(self) -> int:
        return self.elements * self.element_bytes


class SramAllocator:
    def __init__(self, start: int = 4096) -> None:
        # [0, 128) remains an unwritten FP16-zero vector used for exact copies.
        assert start >= K_TILE * 2
        self.cursor = start
        self.tensors: dict[str, SramTensor] = {}

    def allocate(
        self, name: str, shape: Iterable[int], *, element_bytes: int = 2
    ) -> SramTensor:
        shape_tuple = tuple(int(value) for value in shape)
        assert shape_tuple and all(value > 0 for value in shape_tuple)
        assert element_bytes in (1, 2, 4)
        self.cursor = align(self.cursor)
        tensor = SramTensor(name, self.cursor, shape_tuple, element_bytes)
        self.tensors[name] = tensor
        self.cursor += tensor.bytes
        return tensor


class Lp6Image:
    """Streaming LP6 image writer with aligned region metadata."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.file = path.open("wb", buffering=8 * 1024 * 1024)
        self.cursor = 0
        self.regions: dict[str, dict[str, object]] = {}

    def add(self, name: str, data: bytes, **metadata: object) -> int:
        return self.add_chunks(name, (data,), **metadata)

    def add_chunks(
        self, name: str, chunks: Iterable[bytes], **metadata: object
    ) -> int:
        assert name not in self.regions
        base = align(self.cursor)
        self.file.write(bytes(base - self.cursor))
        self.cursor = base
        for chunk in chunks:
            self.file.write(chunk)
            self.cursor += len(chunk)
        self.regions[name] = {
            "address": base,
            "bytes": self.cursor - base,
            **metadata,
        }
        return base

    @property
    def size(self) -> int:
        return self.cursor

    def close(self) -> None:
        self.file.close()


@dataclass(frozen=True)
class TiledWeight:
    name: str
    base: int
    input_features: int
    output_features: int
    k_tiles: int
    n_tiles: int

    @property
    def tile_bytes(self) -> int:
        return K_TILE * N_TILE * 2

    @property
    def n_block_bytes(self) -> int:
        """All K tiles needed for one N=32 output block."""

        return self.k_tiles * self.tile_bytes

    def tile_address(self, n_tile: int, k_tile: int) -> int:
        assert 0 <= n_tile < self.n_tiles and 0 <= k_tile < self.k_tiles
        return self.base + (n_tile * self.k_tiles + k_tile) * self.tile_bytes

    def n_block_address(self, n_tile: int) -> int:
        assert 0 <= n_tile < self.n_tiles
        return self.base + n_tile * self.n_block_bytes


def add_tiled_weight(lp6: Lp6Image, name: str, weight: torch.Tensor) -> TiledWeight:
    """Store PyTorch `[N,K]` FP16 weight as consecutive `[K=64,N=32]` tiles."""

    weight = round_f16(weight).contiguous()
    output_features, input_features = map(int, weight.shape)
    assert input_features % K_TILE == 0
    assert output_features % N_TILE == 0
    def tiles() -> Iterable[bytes]:
        for n_start in range(0, output_features, N_TILE):
            for k_start in range(0, input_features, K_TILE):
                tile = weight[
                    n_start : n_start + N_TILE,
                    k_start : k_start + K_TILE,
                ].T.contiguous()
                yield tile.numpy().tobytes()

    base = lp6.add_chunks(
        name,
        tiles(),
        storage_dtype="fp16",
        logical_shape=[output_features, input_features],
        tile_order="N-major_then_K-major",
        tile_shape=[K_TILE, N_TILE],
    )
    return TiledWeight(
        name=name,
        base=base,
        input_features=input_features,
        output_features=output_features,
        k_tiles=input_features // K_TILE,
        n_tiles=output_features // N_TILE,
    )


class ProgramEmitter(BoundedVectorEmitterMixin):
    """Build unified GDMA and CORE_BEGIN/END records with core-local ISA."""

    def __init__(self, output: Path, *, vector_register_bytes: int = 64) -> None:
        if vector_register_bytes < 4 or vector_register_bytes % 4:
            raise ValueError("invalid vector register byte capacity")
        self.vector_register_bytes = vector_register_bytes
        self.program = SystemProgramBuilder(output)
        self.word_count = 0
        self.config: dict[int, int] = {}
        self._matrix_accumulate = False
        self._matrix_shape = (0, 0)

    def emit(self, word: int | list[int]) -> None:
        self.program.append(word)

    def emit_many(self, words: Iterable[int]) -> None:
        for word in words:
            self.emit(word)

    def close(self) -> None:
        self.program.finish()
        self.word_count = len(self.program)

    def load(self, register: int, value: int) -> None:
        self.emit_many(load_u32(register, value))

    def set_config(self, funct: int, value: int) -> None:
        value &= 0xFFFF_FFFF
        if self.config.get(funct) == value:
            return
        self.load(15, value)
        self.emit(rform(OP_C_SET_V2, rd=15, funct=funct))
        self.config[funct] = value

    def load_raw(self, destination: int, source: int, size: int) -> None:
        assert size > 0
        self.program.gdma_load(
            lp6_address=source,
            l2_address=destination,
            bytes=size,
        )
        self.set_config(SET_DMA_BYTES, size)
        self.load(1, destination)
        self.load(2, source)
        self.emit(rform(OP_L2_DMA, rd=1, rs1=1, funct=0))

    def load_raw_async(self, destination: int, source: int, size: int) -> None:
        """Explicitly stage LP6->L2, then issue L2->L1 asynchronously."""

        assert size > 0
        self.program.gdma_load(
            lp6_address=source,
            l2_address=destination,
            bytes=size,
            overlap_with_current_core_block=True,
        )
        self.set_config(SET_DMA_BYTES, size)
        self.load(1, destination)
        self.load(2, source)
        self.emit(rform(OP_L2_DMA, rd=1, rs1=1, funct=2))

    def set_vector_elements(self, elements: int) -> None:
        assert elements > 0
        self.set_config(SET_VECTOR_ELEMENTS, elements)


    def scalar_f32(
        self, destination: int, lhs: int, funct: int, rhs: int = 0
    ) -> None:
        self.load(3, destination)
        self.load(4, lhs)
        self.emit(rform(OP_V_MEMORY, rd=0, rs1=4, funct=9))
        if funct <= 5:
            self.load(5, rhs)
            self.emit(rform(OP_V_MEMORY, rd=1, rs1=5, funct=9))
        self.emit(
            rform(
                OP_S_ALU_F32,
                rd=2,
                rs1=0,
                rs2=1 if funct <= 5 else 0,
                funct=funct,
            )
        )
        self.emit(rform(OP_V_MEMORY, rd=3, rs1=2, funct=10))

    def local_copy_strided(
        self,
        destination: int,
        source: int,
        row_bytes: int,
        rows: int,
        source_stride: int,
        destination_stride: int,
    ) -> None:
        self.set_config(SET_DMA_BYTES, row_bytes)
        self.set_config(0, rows)
        self.set_config(1, source_stride)
        self.set_config(2, destination_stride)
        self.load(3, destination)
        self.load(4, source)
        self.emit(rform(OP_L2_DMA, rd=3, rs1=4, funct=9))

    def vector_add(self, destination: int, lhs: int, rhs: int, elements: int) -> None:
        self.vector_binary(destination, lhs, rhs, elements, 1)


    def attention(
        self,
        destination: int,
        query: int,
        key: int,
        value: int,
        rows: int,
        q_heads: int,
        kv_heads: int,
        head_dim: int,
        scratch: dict[str, SramTensor],
        score_scale: int,
    ) -> None:
        assert rows <= N_TILE and head_dim <= K_TILE and q_heads % kv_heads == 0
        q_width = q_heads * head_dim
        kv_width = kv_heads * head_dim
        group = q_heads // kv_heads
        for query_row in range(rows):
            keys = query_row + 1
            for query_head in range(q_heads):
                kv_head = query_head // group
                for dim in range(head_dim):
                    self.local_copy_strided(
                        scratch["attention_k_t"].address + dim * keys * 2,
                        key + (kv_head * head_dim + dim) * 2,
                        2,
                        keys,
                        kv_width * 2,
                        2,
                    )
                self.local_copy_strided(
                    scratch["attention_v"].address,
                    value + kv_head * head_dim * 2,
                    head_dim * 2,
                    keys,
                    kv_width * 2,
                    head_dim * 2,
                )
                query_address = query + (
                    query_row * q_width + query_head * head_dim
                ) * 2
                self.matrix_compute(
                    scratch["attention_k_t"].address,
                    query_address,
                    1,
                    keys,
                    head_dim,
                    head_dim,
                )
                self.matrix_write_f16(scratch["attention_score"].address, keys)
                self.softmax(
                    scratch["attention_probability"].address,
                    scratch["attention_score"].address,
                    keys,
                    scratch["primitive_temp_a"].address,
                    scratch["primitive_temp_b"].address,
                    scratch["scalar_max"].address,
                    scratch["scalar_sum"].address,
                    scratch["scalar_inverse"].address,
                    score_scale,
                )
                for dim_start in range(0, head_dim, N_TILE):
                    n = min(N_TILE, head_dim - dim_start)
                    self.matrix_compute(
                        scratch["attention_v"].address + dim_start * 2,
                        scratch["attention_probability"].address,
                        1,
                        n,
                        keys,
                        keys,
                        weight_row_stride=head_dim,
                    )
                    self.matrix_write_f16(
                        destination
                        + (
                            query_row * q_width
                            + query_head * head_dim
                            + dim_start
                        )
                        * 2,
                        head_dim,
                    )

    def matrix_compute(
        self,
        weight: int,
        activation: int,
        m: int,
        n: int,
        k: int,
        activation_row_stride: int,
        weight_row_stride: int | None = None,
    ) -> None:
        assert 0 < m <= M_TILE and 0 < n <= N_TILE and k > 0
        assert activation_row_stride >= k
        weight_row_stride = n if weight_row_stride is None else weight_row_stride
        assert weight_row_stride >= n
        self.load(7, weight)
        self.load(8, activation)
        if self._matrix_accumulate:
            assert self._matrix_shape == (m, n), "live Matrix accumulator shape changed"
        self.emit(matrix_load(7, k, n, weight_row_stride * 2, funct=3))
        self.emit(matrix_load(8, m, k, activation_row_stride * 2, funct=7))
        self.emit(matrix_mma(m, n, k, funct=3, accumulate=self._matrix_accumulate))
        self._matrix_accumulate = True
        self._matrix_shape = (m, n)

    def matrix_write_f16(self, destination: int, output_row_stride: int) -> None:
        self.load(9, destination)
        assert self._matrix_accumulate
        m, n = self._matrix_shape
        self.emit(matrix_writeout(9, m, n, output_row_stride * 2, funct=1))
        self._matrix_accumulate = False


def emit_linear(
    emitter: ProgramEmitter,
    source: SramTensor,
    destination: SramTensor,
    weight: TiledWeight,
) -> None:
    """Emit row-major FP16 linear with compact M/N/K Matrix Unit tiles."""

    rows, input_features = source.shape
    output_rows, output_features = destination.shape
    assert rows == output_rows
    assert input_features == weight.input_features
    assert output_features == weight.output_features
    assert input_features % K_TILE == 0 and output_features % N_TILE == 0

    assert weight.n_block_bytes <= SRAM_WEIGHT_BATCH_CAPACITY
    current_buffer = SRAM_WEIGHT_BATCH_ADDRESSES[0]
    emitter.load_raw(
        current_buffer,
        weight.n_block_address(0),
        weight.n_block_bytes,
    )
    for n_tile, n_start in enumerate(range(0, output_features, N_TILE)):
        next_buffer = SRAM_WEIGHT_BATCH_ADDRESSES[(n_tile + 1) % 2]
        for row_start in range(0, rows, M_TILE):
            m = min(M_TILE, rows - row_start)
            for k_tile, k_start in enumerate(range(0, input_features, K_TILE)):
                activation_address = (
                    source.address + (row_start * input_features + k_start) * 2
                )
                emitter.matrix_compute(
                    current_buffer + k_tile * weight.tile_bytes,
                    activation_address,
                    m,
                    N_TILE,
                    K_TILE,
                    input_features,
                )

            output_address = (
                destination.address + (row_start * output_features + n_start) * 2
            )
            emitter.matrix_write_f16(output_address, output_features)
        if n_tile + 1 < weight.n_tiles:
            # The central GDMA command and this completed core-program segment
            # share an input frontier, so next-buffer fill overlaps the current
            # block's Matrix execution without placing GDMA in the core ISA.
            emitter.load_raw_async(
                next_buffer,
                weight.n_block_address(n_tile + 1),
                weight.n_block_bytes,
            )
        current_buffer = next_buffer


def sequential_linear_f16(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Mirror K-ordered FP32 Matrix Unit accumulation and FP16 write-out."""

    x = round_f16(x).float()
    weight = round_f16(weight).float()
    rows, input_features = x.shape
    output_features, weight_input = weight.shape
    assert input_features == weight_input and input_features % K_TILE == 0
    accumulator = torch.zeros(rows, output_features, dtype=torch.float32)
    for k in range(input_features):
        accumulator.add_(x[:, k : k + 1] * weight[None, :, k])
    return round_f16(accumulator)


def hierarchical_tree_sum_f32(values: torch.Tensor, width: int = 32) -> torch.Tensor:
    """Mirror the VPU's per-vector and global pairwise reduction tree."""

    current = [value for value in values.float().reshape(-1)]
    if not current or width <= 1:
        raise ValueError("tree reduction requires values and width > 1")
    while len(current) > 1:
        partials: list[torch.Tensor] = []
        for begin in range(0, len(current), width):
            level = current[begin : begin + width]
            while len(level) > 1:
                level = [
                    level[index] + level[index + 1]
                    if index + 1 < len(level)
                    else level[index]
                    for index in range(0, len(level), 2)
                ]
            partials.append(level[0])
        current = partials
    return current[0]


def tree_rmsnorm_f16(
    x: torch.Tensor, weight: torch.Tensor, epsilon: float
) -> torch.Tensor:
    x32 = round_f16(x).float()
    weight32 = round_f16(weight).float()
    square_sum = torch.stack(
        [hierarchical_tree_sum_f32(row * row) for row in x32]
    )
    _, sqrtf = _host_sqrtf_symbol()
    inverse_hidden = _round_host_f32(1.0 / x32.shape[1])
    epsilon32 = _round_host_f32(epsilon)
    inverse_rms = torch.tensor(
        [
            _round_host_f32(
                1.0
                / float(
                    sqrtf(
                        ctypes.c_float(
                            _round_host_f32(
                                _round_host_f32(total.item() * inverse_hidden)
                                + epsilon32
                            )
                        )
                    )
                )
            )
            for total in square_sum
        ],
        dtype=torch.float32,
    )
    normalized = round_f16(x32 * inverse_rms[:, None]).float()
    return round_f16(normalized * weight32[None, :])


def vector_add_f16(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    return round_f16(round_f16(lhs).float() + round_f16(rhs).float())


def silu_mul_f16(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Mirror the six generic primitive instructions used for gated SiLU."""

    if gate.shape != up.shape:
        raise ValueError(
            f"SiLU gate/up shapes must match, got {tuple(gate.shape)} and "
            f"{tuple(up.shape)}"
        )
    gate_values = round_f16(gate).float().reshape(-1).tolist()
    up_values = round_f16(up).float().reshape(-1).tolist()
    _, expf = _host_expf_symbol()
    output: list[float] = []
    for gate_value, up_value in zip(gate_values, up_values, strict=True):
        gate_value = _round_host_f32(gate_value)
        up_value = _round_host_f32(up_value)
        temporary = round_scalar_f16(-gate_value)
        temporary = round_scalar_f16(float(expf(ctypes.c_float(temporary))))
        temporary = round_scalar_f16(temporary + 1.0)
        temporary = round_scalar_f16(_round_host_f32(1.0 / temporary))
        temporary = round_scalar_f16(gate_value * temporary)
        output.append(round_scalar_f16(temporary * up_value))
    output_tensor = torch.tensor(output, dtype=torch.float32).reshape(gate.shape)
    return round_f16(output_tensor)


def rope_frequencies(config: dict[str, object], head_dim: int) -> torch.Tensor:
    """Return Transformers-compatible plain or Llama-3 RoPE frequencies."""

    theta = float(config["rope_theta"])
    dimensions = torch.arange(0, head_dim, 2, dtype=torch.float32)
    frequencies = 1.0 / (
        torch.tensor(theta, dtype=torch.float32) ** (dimensions / head_dim)
    )
    scaling = config.get("rope_scaling") or {"rope_type": "none"}
    if not isinstance(scaling, dict):
        raise ValueError("rope_scaling must be an object or null")
    rope_type = scaling.get("rope_type", scaling.get("type", "none"))
    if rope_type in (None, "none", "plain"):
        return frequencies
    if rope_type != "llama3":
        raise ValueError(f"unsupported RoPE scaling type: {rope_type!r}")
    factor = float(scaling["factor"])
    low = float(scaling["low_freq_factor"])
    high = float(scaling["high_freq_factor"])
    original = float(scaling["original_max_position_embeddings"])
    if min(factor, low, high, original) <= 0 or high <= low:
        raise ValueError("invalid Llama-3 RoPE scaling parameters")
    wavelengths = (2.0 * math.pi) / frequencies
    low_wavelength = original / low
    high_wavelength = original / high
    scaled = torch.where(wavelengths > low_wavelength, frequencies / factor, frequencies)
    smooth = (original / wavelengths - low) / (high - low)
    blended = (1.0 - smooth) * scaled / factor + smooth * scaled
    medium = ~(wavelengths < high_wavelength) & ~(wavelengths > low_wavelength)
    return torch.where(medium, blended, scaled).to(torch.float32)


def rope_tables_f16(
    config: dict[str, object], head_dim: int, position_start: int, rows: int
) -> tuple[torch.Tensor, torch.Tensor]:
    frequencies = rope_frequencies(config, head_dim)
    positions = torch.arange(
        position_start, position_start + rows, dtype=torch.float32
    )
    angles = positions[:, None] * frequencies[None, :]
    return round_f16(torch.sin(angles)), round_f16(torch.cos(angles))


def rope_f16(
    value: torch.Tensor,
    heads: int,
    head_dim: int,
    sine_table: torch.Tensor,
    cosine_table: torch.Tensor,
) -> torch.Tensor:
    value32 = round_f16(value).float().reshape(value.shape[0], heads, head_dim)
    half = head_dim // 2
    assert head_dim % 2 == 0
    output = torch.empty_like(value32)
    for row in range(value32.shape[0]):
        for pair in range(half):
            cosine = cosine_table[row, pair].float()
            sine = sine_table[row, pair].float()
            first = value32[row, :, pair]
            second = value32[row, :, pair + half]
            first_cos = round_f16(first * cosine).float()
            second_sin = round_f16(second * sine).float()
            second_cos = round_f16(second * cosine).float()
            first_sin = round_f16(first * sine).float()
            output[row, :, pair] = round_f16(first_cos - second_sin).float()
            output[row, :, pair + half] = round_f16(second_cos + first_sin).float()
    return round_f16(output).reshape(value.shape[0], heads * head_dim)


def causal_gqa_attention_f16(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Mirror Matrix QK/PV and the six generic softmax primitives."""

    rows = query.shape[0]
    query32 = round_f16(query).float().reshape(rows, q_heads, head_dim)
    key32 = round_f16(key).float().reshape(rows, kv_heads, head_dim)
    value32 = round_f16(value).float().reshape(rows, kv_heads, head_dim)
    output = torch.zeros(rows, q_heads, head_dim, dtype=torch.float32)
    group_size = q_heads // kv_heads
    scale = torch.tensor(1.0 / math.sqrt(head_dim), dtype=torch.float32)

    for query_row in range(rows):
        for query_head in range(q_heads):
            kv_head = query_head // group_size
            scores: list[torch.Tensor] = []
            for key_row in range(query_row + 1):
                dot = torch.tensor(0.0, dtype=torch.float32)
                for column in range(head_dim):
                    dot = dot + (
                        query32[query_row, query_head, column]
                        * key32[key_row, kv_head, column]
                    )
                scores.append(round_f16(dot).float())
            scores = [round_f16(score * scale).float() for score in scores]
            maximum = torch.stack(scores).max()
            shifted = [round_f16(score - maximum).float() for score in scores]
            exponentials = [round_f16(torch.exp(score)).float() for score in shifted]
            denominator = hierarchical_tree_sum_f32(torch.stack(exponentials))
            reciprocal = torch.tensor(
                _round_host_f32(1.0 / _round_host_f32(denominator.item())),
                dtype=torch.float32,
            )
            probabilities = [
                round_f16(exponential * reciprocal).float()
                for exponential in exponentials
            ]
            for column in range(head_dim):
                accumulated = torch.tensor(0.0, dtype=torch.float32)
                for key_row, probability in enumerate(probabilities):
                    accumulated = accumulated + (
                        probability * value32[key_row, kv_head, column]
                    )
                output[query_row, query_head, column] = accumulated
    return round_f16(output).reshape(rows, q_heads * head_dim)


def build_reference_layer(
    hidden: torch.Tensor,
    tensors: dict[str, torch.Tensor],
    *,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    epsilon: float,
    rope_sine: torch.Tensor,
    rope_cosine: torch.Tensor,
) -> dict[str, torch.Tensor]:
    expected: dict[str, torch.Tensor] = {"input": round_f16(hidden)}
    expected["input_norm"] = tree_rmsnorm_f16(
        expected["input"], tensors["input_norm_weight"], epsilon
    )
    expected["q"] = sequential_linear_f16(expected["input_norm"], tensors["q_weight"])
    expected["k"] = sequential_linear_f16(expected["input_norm"], tensors["k_weight"])
    expected["v"] = sequential_linear_f16(expected["input_norm"], tensors["v_weight"])
    expected["q_rope"] = rope_f16(
        expected["q"], q_heads, head_dim, rope_sine, rope_cosine
    )
    expected["k_rope"] = rope_f16(
        expected["k"], kv_heads, head_dim, rope_sine, rope_cosine
    )
    expected["attention"] = causal_gqa_attention_f16(
        expected["q_rope"],
        expected["k_rope"],
        expected["v"],
        q_heads,
        kv_heads,
        head_dim,
    )
    expected["o_proj"] = sequential_linear_f16(
        expected["attention"], tensors["o_weight"]
    )
    expected["attention_residual"] = vector_add_f16(
        expected["input"], expected["o_proj"]
    )
    expected["post_norm"] = tree_rmsnorm_f16(
        expected["attention_residual"], tensors["post_norm_weight"], epsilon
    )
    expected["gate"] = sequential_linear_f16(expected["post_norm"], tensors["gate_weight"])
    expected["up"] = sequential_linear_f16(expected["post_norm"], tensors["up_weight"])
    expected["silu"] = silu_mul_f16(expected["gate"], expected["up"])
    expected["down"] = sequential_linear_f16(expected["silu"], tensors["down_weight"])
    expected["final"] = vector_add_f16(
        expected["attention_residual"], expected["down"]
    )
    return {name: value.cpu().contiguous() for name, value in expected.items()}


def load_layer_tensors(model: safe_open, layer: int) -> dict[str, torch.Tensor]:
    prefix = f"model.layers.{layer}"
    key_map = {
        "input_norm_weight": f"{prefix}.input_layernorm.weight",
        "q_weight": f"{prefix}.self_attn.q_proj.weight",
        "k_weight": f"{prefix}.self_attn.k_proj.weight",
        "v_weight": f"{prefix}.self_attn.v_proj.weight",
        "o_weight": f"{prefix}.self_attn.o_proj.weight",
        "post_norm_weight": f"{prefix}.post_attention_layernorm.weight",
        "gate_weight": f"{prefix}.mlp.gate_proj.weight",
        "up_weight": f"{prefix}.mlp.up_proj.weight",
        "down_weight": f"{prefix}.mlp.down_proj.weight",
    }
    return {
        local_name: round_f16(model.get_tensor(model_name).float()).contiguous()
        for local_name, model_name in key_map.items()
    }


def add_layer_to_lp6(
    lp6: Lp6Image, layer: int, tensors: dict[str, torch.Tensor]
) -> tuple[int, int, dict[str, TiledWeight]]:
    prefix = f"layer{layer}"
    input_norm = lp6.add(
        f"{prefix}.input_layernorm.weight",
        tensors["input_norm_weight"].numpy().tobytes(),
        storage_dtype="fp16",
        logical_shape=list(tensors["input_norm_weight"].shape),
    )
    tiled = {
        "q": add_tiled_weight(
            lp6, f"{prefix}.self_attn.q_proj.weight", tensors["q_weight"]
        ),
        "k": add_tiled_weight(
            lp6, f"{prefix}.self_attn.k_proj.weight", tensors["k_weight"]
        ),
        "v": add_tiled_weight(
            lp6, f"{prefix}.self_attn.v_proj.weight", tensors["v_weight"]
        ),
        "o": add_tiled_weight(
            lp6, f"{prefix}.self_attn.o_proj.weight", tensors["o_weight"]
        ),
    }
    post_norm = lp6.add(
        f"{prefix}.post_attention_layernorm.weight",
        tensors["post_norm_weight"].numpy().tobytes(),
        storage_dtype="fp16",
        logical_shape=list(tensors["post_norm_weight"].shape),
    )
    tiled.update(
        {
            "gate": add_tiled_weight(
                lp6, f"{prefix}.mlp.gate_proj.weight", tensors["gate_weight"]
            ),
            "up": add_tiled_weight(
                lp6, f"{prefix}.mlp.up_proj.weight", tensors["up_weight"]
            ),
            "down": add_tiled_weight(
                lp6, f"{prefix}.mlp.down_proj.weight", tensors["down_weight"]
            ),
        }
    )
    return input_norm, post_norm, tiled


def tiled_weight_metadata(weights: dict[str, TiledWeight]) -> dict[str, object]:
    return {
        name: {
            "lp6_address": value.base,
            "logical_shape": [value.output_features, value.input_features],
            "tile_shape": [K_TILE, N_TILE],
            "n_tiles": value.n_tiles,
            "k_tiles": value.k_tiles,
            "tile_bytes": value.tile_bytes,
            "n_block_bytes": value.n_block_bytes,
        }
        for name, value in weights.items()
    }


def emit_decoder_layer(
    emitter: ProgramEmitter,
    hidden_source: SramTensor,
    hidden_destination: SramTensor,
    work: dict[str, SramTensor],
    tiled_weights: dict[str, TiledWeight],
    input_norm_lp6: int,
    post_norm_lp6: int,
    rows: int,
    hidden: int,
    intermediate: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
) -> None:
    emitter.load_raw(SRAM_NORM_WEIGHT_ADDRESS, input_norm_lp6, hidden * 2)
    emitter.rmsnorm(
        work["input_norm"].address,
        hidden_source.address,
        SRAM_NORM_WEIGHT_ADDRESS,
        rows,
        hidden,
        work["primitive_temp_a"].address,
        work["scalar_sum"].address,
        work["scalar_inverse"].address,
        work["scalar_inverse_hidden"].address,
        work["scalar_epsilon"].address,
    )
    for projection in ("q", "k", "v"):
        emit_linear(
            emitter,
            work["input_norm"],
            work[projection],
            tiled_weights[projection],
        )

    emitter.rope(
        work["q_rope"].address,
        work["q"].address,
        rows,
        q_heads,
        head_dim,
        work["rope_sine"].address,
        work["rope_cosine"].address,
        work["primitive_temp_a"].address,
        work["primitive_temp_b"].address,
    )
    emitter.rope(
        work["k_rope"].address,
        work["k"].address,
        rows,
        kv_heads,
        head_dim,
        work["rope_sine"].address,
        work["rope_cosine"].address,
        work["primitive_temp_a"].address,
        work["primitive_temp_b"].address,
    )
    emitter.attention(
        work["attention"].address,
        work["q_rope"].address,
        work["k_rope"].address,
        work["v"].address,
        rows,
        q_heads,
        kv_heads,
        head_dim,
        work,
        work["scalar_score_scale"].address,
    )
    emit_linear(
        emitter, work["attention"], work["o_proj"], tiled_weights["o"]
    )
    emitter.vector_add(
        work["attention_residual"].address,
        hidden_source.address,
        work["o_proj"].address,
        rows * hidden,
    )

    emitter.load_raw(SRAM_NORM_WEIGHT_ADDRESS, post_norm_lp6, hidden * 2)
    emitter.rmsnorm(
        work["post_norm"].address,
        work["attention_residual"].address,
        SRAM_NORM_WEIGHT_ADDRESS,
        rows,
        hidden,
        work["primitive_temp_a"].address,
        work["scalar_sum"].address,
        work["scalar_inverse"].address,
        work["scalar_inverse_hidden"].address,
        work["scalar_epsilon"].address,
    )
    for projection in ("gate", "up"):
        emit_linear(
            emitter,
            work["post_norm"],
            work[projection],
            tiled_weights[projection],
        )
    emitter.silu_mul(
        work["silu"].address,
        work["gate"].address,
        work["up"].address,
        rows * intermediate,
        work["primitive_temp_a"].address,
        work["scalar_one"].address,
    )
    emit_linear(emitter, work["silu"], work["down"], tiled_weights["down"])
    emitter.vector_add(
        hidden_destination.address,
        work["attention_residual"].address,
        work["down"].address,
        rows * hidden,
    )


def generate(
    model_dir: Path,
    prompt: str,
    output: Path,
    position_start: int,
    num_layers: int,
    with_lm_head: bool,
    vector_register_bits: int = 512,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    hidden = int(config["hidden_size"])
    intermediate = int(config["intermediate_size"])
    total_layers = int(config["num_hidden_layers"])
    vocab_size = int(config["vocab_size"])
    q_heads = int(config["num_attention_heads"])
    kv_heads = int(config["num_key_value_heads"])
    head_dim = hidden // q_heads
    epsilon = float(config["rms_norm_eps"])
    theta = float(config["rope_theta"])
    assert hidden == q_heads * head_dim
    assert q_heads % kv_heads == 0
    assert head_dim <= K_TILE and head_dim % 2 == 0
    assert hidden % K_TILE == 0 and intermediate % K_TILE == 0
    assert 1 <= num_layers <= total_layers
    assert vocab_size % N_TILE == 0

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    token_ids = tokenizer(prompt, return_tensors="pt").input_ids[0]
    rows = int(token_ids.numel())
    assert 0 < rows <= N_TILE, (
        "the reference generator currently supports one prefill sequence tile; "
        "this is a generator limitation, not an NPU ISA limit"
    )

    allocator = SramAllocator()
    hidden_a = allocator.allocate("hidden_a", (rows, hidden))
    work = {
        "input_norm": allocator.allocate("input_norm", (rows, hidden)),
        "q": allocator.allocate("q", (rows, hidden)),
        "k": allocator.allocate("k", (rows, kv_heads * head_dim)),
        "v": allocator.allocate("v", (rows, kv_heads * head_dim)),
        "q_rope": allocator.allocate("q_rope", (rows, hidden)),
        "k_rope": allocator.allocate("k_rope", (rows, kv_heads * head_dim)),
        "attention": allocator.allocate("attention", (rows, hidden)),
        "o_proj": allocator.allocate("o_proj", (rows, hidden)),
        "attention_residual": allocator.allocate("attention_residual", (rows, hidden)),
        "post_norm": allocator.allocate("post_norm", (rows, hidden)),
        "gate": allocator.allocate("gate", (rows, intermediate)),
        "up": allocator.allocate("up", (rows, intermediate)),
        "silu": allocator.allocate("silu", (rows, intermediate)),
        "down": allocator.allocate("down", (rows, hidden)),
        "primitive_temp_a": allocator.allocate(
            "primitive_temp_a", (rows, max(hidden, intermediate))
        ),
        "primitive_temp_b": allocator.allocate(
            "primitive_temp_b", (rows, max(hidden, intermediate))
        ),
        "rope_sine": allocator.allocate("rope_sine", (rows, head_dim // 2)),
        "rope_cosine": allocator.allocate("rope_cosine", (rows, head_dim // 2)),
        "attention_k_t": allocator.allocate("attention_k_t", (head_dim, rows)),
        "attention_v": allocator.allocate("attention_v", (rows, head_dim)),
        "attention_score": allocator.allocate("attention_score", (rows,)),
        "attention_probability": allocator.allocate("attention_probability", (rows,)),
        "scalar_one": allocator.allocate("scalar_one", (1,), element_bytes=4),
        "scalar_epsilon": allocator.allocate("scalar_epsilon", (1,), element_bytes=4),
        "scalar_inverse_hidden": allocator.allocate(
            "scalar_inverse_hidden", (1,), element_bytes=4
        ),
        "scalar_score_scale": allocator.allocate(
            "scalar_score_scale", (1,), element_bytes=4
        ),
        "scalar_max": allocator.allocate("scalar_max", (1,), element_bytes=4),
        "scalar_sum": allocator.allocate("scalar_sum", (1,), element_bytes=4),
        "scalar_inverse": allocator.allocate("scalar_inverse", (1,), element_bytes=4),
    }
    checkpoints = [
        allocator.allocate(f"layer{layer}_checkpoint", (rows, hidden))
        for layer in range(num_layers)
    ]
    final_norm = (
        allocator.allocate("model_norm", (rows, hidden)) if with_lm_head else None
    )
    logits = (
        allocator.allocate("last_token_logits", (1, vocab_size))
        if with_lm_head
        else None
    )

    lp6 = Lp6Image(output / "lp6.bin")
    register_bits = vector_register_bits
    if register_bits < 32 or register_bits % 32:
        raise ValueError("vector-register-bits must be a positive multiple of 32")
    emitter = ProgramEmitter(output, vector_register_bytes=register_bits // 8)
    scalar_constants = {
        "scalar_one": 1.0,
        "scalar_epsilon": epsilon,
        "scalar_inverse_hidden": 1.0 / hidden,
        "scalar_score_scale": 1.0 / math.sqrt(head_dim),
    }
    for name, value in scalar_constants.items():
        address = lp6.add(
            f"constant.{name}",
            struct.pack("<f", value),
            storage_dtype="fp32",
            logical_shape=[1],
        )
        emitter.load_raw(work[name].address, address, 4)

    half = head_dim // 2
    rope_sine, rope_cosine = rope_tables_f16(
        config, head_dim, position_start, rows
    )
    sine_lp6 = lp6.add(
        "constant.rope_sine",
        rope_sine.contiguous().numpy().tobytes(),
        storage_dtype="fp16",
        logical_shape=[rows, half],
    )
    cosine_lp6 = lp6.add(
        "constant.rope_cosine",
        rope_cosine.contiguous().numpy().tobytes(),
        storage_dtype="fp16",
        logical_shape=[rows, half],
    )
    emitter.load_raw(work["rope_sine"].address, sine_lp6, work["rope_sine"].bytes)
    emitter.load_raw(work["rope_cosine"].address, cosine_lp6, work["rope_cosine"].bytes)
    expected: dict[str, torch.Tensor] = {}
    check_tensors: dict[str, dict[str, object]] = {}
    layer_metadata: list[dict[str, object]] = []

    with safe_open(model_dir / "model.safetensors", framework="pt", device="cpu") as model:
        embedding_table = model.get_tensor("model.embed_tokens.weight")
        embedding = round_f16(embedding_table[token_ids.long()].float()).contiguous()
        embedding_lp6 = lp6.add(
            "selected_embeddings",
            embedding.numpy().tobytes(),
            storage_dtype="fp16",
            logical_shape=[rows, hidden],
            token_ids=token_ids.tolist(),
        )
        emitter.load_raw(hidden_a.address, embedding_lp6, hidden_a.bytes)
        reference_hidden = embedding
        hidden_source = hidden_a

        for layer in range(num_layers):
            # The archived layer output is also the next layer's hidden input,
            # avoiding a second copy and preserving every boundary for checks.
            hidden_destination = checkpoints[layer]
            tensors = load_layer_tensors(model, layer)
            stages = build_reference_layer(
                reference_hidden,
                tensors,
                q_heads=q_heads,
                kv_heads=kv_heads,
                head_dim=head_dim,
                epsilon=epsilon,
                rope_sine=rope_sine,
                rope_cosine=rope_cosine,
            )
            input_norm_lp6, post_norm_lp6, tiled_weights = add_layer_to_lp6(
                lp6, layer, tensors
            )
            emit_decoder_layer(
                emitter,
                hidden_source,
                hidden_destination,
                work,
                tiled_weights,
                input_norm_lp6,
                post_norm_lp6,
                rows,
                hidden,
                intermediate,
                q_heads,
                kv_heads,
                head_dim,
            )

            checkpoint = hidden_destination
            final_key = f"layer{layer}.final"
            expected[final_key] = stages["final"]
            check_tensors[final_key] = {
                "address": checkpoint.address,
                "shape": list(checkpoint.shape),
                "storage_dtype": "fp16",
            }
            # Preserve the detailed one-layer smoke-test diagnostics without
            # growing full-model SRAM linearly for every intermediate.
            if num_layers == 1 or (layer == num_layers - 1 and not with_lm_head):
                if layer == 0:
                    expected["embedding"] = stages["input"]
                    check_tensors["embedding"] = {
                        "address": hidden_source.address,
                        "shape": list(hidden_source.shape),
                        "storage_dtype": "fp16",
                    }
                for stage_name, tensor in stages.items():
                    if stage_name in {"input", "final"}:
                        continue
                    key = f"layer{layer}.{stage_name}"
                    expected[key] = tensor
                    check_tensors[key] = {
                        "address": work[stage_name].address,
                        "shape": list(work[stage_name].shape),
                        "storage_dtype": "fp16",
                    }

            layer_metadata.append(
                {
                    "layer": layer,
                    "input_norm_lp6": input_norm_lp6,
                    "post_norm_lp6": post_norm_lp6,
                    "weights": tiled_weight_metadata(tiled_weights),
                    "checkpoint_sram": checkpoint.address,
                }
            )
            reference_hidden = stages["final"]
            hidden_source = hidden_destination
            del tensors, stages, tiled_weights

        lm_head_metadata: dict[str, object] | None = None
        if with_lm_head:
            assert final_norm is not None and logits is not None
            model_norm_weight = round_f16(model.get_tensor("model.norm.weight").float())
            model_norm_lp6 = lp6.add(
                "model.norm.weight",
                model_norm_weight.numpy().tobytes(),
                storage_dtype="fp16",
                logical_shape=[hidden],
            )
            emitter.load_raw(SRAM_NORM_WEIGHT_ADDRESS, model_norm_lp6, hidden * 2)
            emitter.rmsnorm(
                final_norm.address,
                hidden_source.address,
                SRAM_NORM_WEIGHT_ADDRESS,
                rows,
                hidden,
                work["primitive_temp_a"].address,
                work["scalar_sum"].address,
                work["scalar_inverse"].address,
                work["scalar_inverse_hidden"].address,
                work["scalar_epsilon"].address,
            )
            reference_norm = tree_rmsnorm_f16(
                reference_hidden, model_norm_weight, epsilon
            )
            expected["model.norm"] = reference_norm
            check_tensors["model.norm"] = {
                "address": final_norm.address,
                "shape": list(final_norm.shape),
                "storage_dtype": "fp16",
            }

            lm_weight = round_f16(embedding_table.float()).contiguous()
            lm_tiled = add_tiled_weight(lp6, "lm_head.tied_embedding", lm_weight)
            last_hidden = SramTensor(
                "last_hidden",
                final_norm.address + (rows - 1) * hidden * 2,
                (1, hidden),
            )
            emit_linear(emitter, last_hidden, logits, lm_tiled)
            reference_logits = sequential_linear_f16(
                reference_norm[-1:, :], lm_weight
            )
            expected["logits"] = reference_logits
            check_tensors["logits"] = {
                "address": logits.address,
                "shape": list(logits.shape),
                "storage_dtype": "fp16",
            }
            argmax_id = int(torch.argmax(reference_logits[0]).item())
            lm_head_metadata = {
                "weight": tiled_weight_metadata({"lm_head": lm_tiled})["lm_head"],
                "model_norm_lp6": model_norm_lp6,
                "last_prompt_row_only": True,
                "argmax_token_id": argmax_id,
                "argmax_token": tokenizer.convert_ids_to_tokens(argmax_id),
                "argmax_text": tokenizer.decode([argmax_id]),
            }
            del lm_weight, model_norm_weight, reference_logits

    lp6.close()
    emitter.close()
    torch.save(expected, output / "expected.pt")

    all_buffers = {
        "hidden_a": hidden_a,
        **work,
        **{checkpoint.name: checkpoint for checkpoint in checkpoints},
    }
    if final_norm is not None:
        all_buffers[final_norm.name] = final_norm
    if logits is not None:
        all_buffers[logits.name] = logits
    result_tensor = logits if logits is not None else hidden_source
    metadata = {
        "model": str(model_dir),
        "scope": (
            f"selected prompt embeddings plus decoder layers 0..{num_layers - 1}"
            + (" plus final norm and tied LM head" if with_lm_head else "")
        ),
        "prompt": prompt,
        "token_ids": token_ids.tolist(),
        "tokens": tokenizer.convert_ids_to_tokens(token_ids.tolist()),
        "position_start": position_start,
        "config": {
            "rows": rows,
            "hidden_size": hidden,
            "intermediate_size": intermediate,
            "q_heads": q_heads,
            "kv_heads": kv_heads,
            "head_dim": head_dim,
            "rms_epsilon": epsilon,
            "rope_theta": theta,
            "rope_scaling": config.get("rope_scaling"),
            "num_layers": num_layers,
            "model_total_layers": total_layers,
            "vocab_size": vocab_size,
        },
        "numeric_contract": {
            "weight_storage": "fp16",
            "activation_storage": "fp16",
            "matrix_accumulator": "fp32",
            "attention_score_storage": "fp16",
            "softmax_reduction_state": "fp32",
            "reduction_order": "32-element pairwise local tree plus hierarchical global tree",
            "scalar_transcendentals": "host scalar expf/sqrtf with explicit FP32 boundaries",
            "attention_probability_boundary": "fp16",
            "nonlinear_lowering": "generic Vector/Scalar primitive ISA only",
            "primitive_vector_boundary": "FP16 rounding after each Vector instruction",
            "matrix_tile_max": [M_TILE, N_TILE, K_TILE],
        },
        "weight_prefetch": {
            "batch_layout": "all K tiles for one N=32 output block",
            "sram_ping_pong_byte_addresses": list(SRAM_WEIGHT_BATCH_ADDRESSES),
            "bytes_per_buffer": SRAM_WEIGHT_BATCH_CAPACITY,
            "first_blocking_then_async": True,
            "readiness": "first overlapping Matrix weight read",
        },
        "lp6_image_bytes": lp6.size,
        "lp6_regions": lp6.regions,
        "layers": layer_metadata,
        "lm_head": lm_head_metadata,
        "core_instruction_words": emitter.word_count,
        "activation_sram_high_water": allocator.cursor,
        "activation_sram_buffers": {
            name: {
                "address": tensor.address,
                "shape": list(tensor.shape),
                "storage_dtype": {1: "bytes", 2: "fp16", 4: "fp32"}[
                    tensor.element_bytes
                ],
            }
            for name, tensor in all_buffers.items()
        },
        "check_tensors": check_tensors,
        "scratch": {
            "zero_vector_address": VECTOR_ZERO_ADDRESS,
            "matrix_weight_batch_addresses": list(SRAM_WEIGHT_BATCH_ADDRESSES),
            "norm_weight_address": SRAM_NORM_WEIGHT_ADDRESS,
        },
        "result_kind": "last_token_logits" if with_lm_head else "hidden_state",
        "result_expected_key": (
            "logits" if with_lm_head else f"layer{num_layers - 1}.final"
        ),
        "result_byte_address": result_tensor.address,
        "result_shape": list(result_tensor.shape),
    }
    assert allocator.cursor <= SRAM_WEIGHT_REGION_BASE, (
        f"activation allocation ends at {allocator.cursor:#x}, overlapping "
        f"the weight region at {SRAM_WEIGHT_REGION_BASE:#x}; compiler spill/tiling is required"
    )
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def read_sram_f16(dump: bytes, address: int, shape: list[int]) -> torch.Tensor:
    elements = math.prod(shape)
    end = address + elements * 2
    if end > len(dump):
        raise SystemExit(
            f"SRAM dump ends at {len(dump):#x}, tensor requires [{address:#x}, {end:#x})"
        )
    return torch.frombuffer(bytearray(dump[address:end]), dtype=torch.float16).reshape(shape)


def check(output: Path, atol: float) -> None:
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    expected: dict[str, torch.Tensor] = torch.load(
        output / "expected.pt", map_location="cpu", weights_only=True
    )
    dump = (output / "sram_dump.bin").read_bytes()
    stages: dict[str, dict[str, object]] = {}
    worst = 0.0
    actual_tensors: dict[str, torch.Tensor] = {}
    for name, tensor_metadata in metadata["check_tensors"].items():
        actual = read_sram_f16(
            dump, tensor_metadata["address"], tensor_metadata["shape"]
        )
        actual_tensors[name] = actual
        reference = expected[name]
        error = (actual.float() - reference.float()).abs()
        max_abs = float(error.max())
        rmse = float(torch.sqrt(torch.mean(error.square())))
        worst = max(worst, max_abs)
        stages[name] = {
            "max_abs_error": max_abs,
            "rmse": rmse,
            "exact_f16": bool(torch.equal(actual, reference)),
        }

    result_key = metadata["result_expected_key"]
    result: dict[str, object] = {
        "shape": metadata["result_shape"],
        "atol": atol,
        "worst_stage_max_abs_error": worst,
        "result_key": result_key,
        "result": stages[result_key],
        "stages": stages,
    }
    if metadata.get("lm_head") is not None:
        actual_argmax = int(torch.argmax(actual_tensors["logits"][0]).item())
        expected_argmax = int(metadata["lm_head"]["argmax_token_id"])
        result["argmax"] = {
            "actual_token_id": actual_argmax,
            "expected_token_id": expected_argmax,
            "match": actual_argmax == expected_argmax,
            "token": metadata["lm_head"]["argmax_token"],
            "text": metadata["lm_head"]["argmax_text"],
        }
    print(json.dumps(result, indent=2))
    (output / "check_result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    argmax_matches = result.get("argmax", {}).get("match", True)
    if not math.isfinite(worst) or worst > atol or not argmax_matches:
        raise SystemExit(
            f"decoder mismatch: worst stage max_abs_error={worst} exceeds atol={atol}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--position-start", type=int, default=0)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--with-lm-head", action="store_true")
    parser.add_argument(
        "--full-model",
        action="store_true",
        help="Generate all model layers, final norm, and tied LM head",
    )
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--atol", type=float, default=0.05)
    parser.add_argument("--vector-register-bits", type=int, default=512,
                        help="fixed register capacity used for explicit vector tiling")
    args = parser.parse_args()
    if args.check:
        check(args.output, args.atol)
    else:
        config = json.loads((args.model / "config.json").read_text(encoding="utf-8"))
        num_layers = int(config["num_hidden_layers"]) if args.full_model else args.num_layers
        generate(
            args.model,
            args.prompt,
            args.output,
            args.position_start,
            num_layers,
            args.with_lm_head or args.full_model,
            args.vector_register_bits,
        )


if __name__ == "__main__":
    main()
