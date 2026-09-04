"""Unified 32-bit command-ISA builder for PLENA v2 test programs.

The generated ``program.bin`` interleaves system-level GDMA records with
``CORE_BEGIN``/core-local words/``CORE_END`` records. Every physical word is
32-bit little-endian. JSON contains placement and debug symbols only; commands
and numeric completion-event dependencies are decoded from the binary.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Iterable


PROGRAM_MAGIC = int.from_bytes(b"PLN5", "little")
PROGRAM_VERSION = 5
PROGRAM_HEADER_WORDS = 5
OP_CORE_BEGIN = 0x26
OP_CORE_END = 0x27
OP_GDMA_LOAD = 0x2B
OP_GDMA_STORE = 0x2C
OP_PROGRAM_END = 0x2F
NONE_U32 = 0xFFFF_FFFF


def _u32(value: int, field: str) -> int:
    if not 0 <= value < 1 << 32:
        raise ValueError(f"{field}={value} does not fit 32 bits")
    return value


def _u64_words(value: int, field: str) -> tuple[int, int]:
    if not 0 <= value < 1 << 64:
        raise ValueError(f"{field}={value} does not fit 64 bits")
    return value & 0xFFFF_FFFF, value >> 32


def encode_unified_program(
    commands: list[dict[str, object]], core_isa: bytes
) -> tuple[bytes, list[dict[str, object]]]:
    """Encode system records and referenced core ranges into one 32-bit stream."""

    if len(core_isa) % 4:
        raise ValueError("temporary core ISA image contains a partial word")
    command_ids: dict[str, int] = {}
    symbols: list[dict[str, object]] = []
    for command_id, command in enumerate(commands):
        name = str(command["id"])
        if not name or name in command_ids:
            raise ValueError(f"duplicate or empty command name: {name!r}")
        command_ids[name] = command_id
        symbols.append({"id": command_id, "name": name})

    affinity_ids: dict[str, int] = {}
    alias_ids: dict[str, int] = {}

    def symbolic_id(value: object, table: dict[str, int]) -> int:
        if value is None:
            return NONE_U32
        name = str(value)
        return table.setdefault(name, len(table))

    output = bytearray(PROGRAM_HEADER_WORDS * 4)
    core_instruction_count = 0

    def append_words(values: Iterable[int]) -> None:
        words = [_u32(int(value), "program word") for value in values]
        if words:
            output.extend(struct.pack(f"<{len(words)}I", *words))

    for numeric_id, command in enumerate(commands):
        dependencies = [
            command_ids[str(name)] for name in command.get("dependencies", [])
        ]
        kind = str(command["kind"])
        if kind == "core_block":
            start = int(command["isa_start_word"])
            count = int(command["isa_word_count"])
            end = start + count
            if start < 0 or count < 0 or end * 4 > len(core_isa):
                raise ValueError(f"invalid core ISA range [{start}, {end})")
            target = command.get("target_logical_core")
            preferred = command.get("preferred_physical_core")
            allowed = [int(core) for core in command.get("allowed_physical_cores") or []]
            regions = list(command.get("l1_regions") or [])
            l1_low, l1_high = _u64_words(
                int(command.get("l1_bytes_required", 0)), "l1_bytes_required"
            )
            append_words(
                [
                    OP_CORE_BEGIN,
                    numeric_id,
                    NONE_U32 if target is None else int(target),
                    NONE_U32 if preferred is None else int(preferred),
                    symbolic_id(command.get("affinity_group"), affinity_ids),
                    l1_low,
                    l1_high,
                    count,
                    len(dependencies),
                    len(allowed),
                    len(regions),
                    *dependencies,
                    *allowed,
                ]
            )
            for region in regions:
                offset_low, offset_high = _u64_words(int(region["offset"]), "L1 offset")
                size_low, size_high = _u64_words(int(region["size"]), "L1 size")
                alignment_low, alignment_high = _u64_words(
                    int(region.get("alignment", 64)), "L1 alignment"
                )
                append_words(
                    [
                        offset_low,
                        offset_high,
                        size_low,
                        size_high,
                        alignment_low,
                        alignment_high,
                        symbolic_id(region.get("alias_group"), alias_ids),
                    ]
                )
            output.extend(core_isa[start * 4 : end * 4])
            append_words([OP_CORE_END, numeric_id])
            core_instruction_count += count
            continue

        if kind not in ("gdma_load", "gdma_store"):
            raise ValueError(f"unsupported unified command kind: {kind}")
        lp6_low, lp6_high = _u64_words(int(command["lp6_address"]), "LP6 address")
        l2_low, l2_high = _u64_words(int(command["l2_address"]), "L2 address")
        row_bytes = int(command["bytes"])
        rows = int(command.get("rows", 1))
        lp6_stride = int(command.get("lp6_stride", row_bytes))
        l2_stride = int(command.get("l2_stride", row_bytes))
        append_words(
            [
                OP_GDMA_LOAD if kind == "gdma_load" else OP_GDMA_STORE,
                numeric_id,
                len(dependencies),
                lp6_low,
                lp6_high,
                l2_low,
                l2_high,
                row_bytes,
                rows,
                lp6_stride,
                l2_stride,
                *dependencies,
            ]
        )

    append_words([OP_PROGRAM_END])
    total_words = len(output) // 4
    output[: PROGRAM_HEADER_WORDS * 4] = struct.pack(
        "<5I",
        PROGRAM_MAGIC,
        PROGRAM_VERSION,
        total_words,
        len(commands),
        core_instruction_count,
    )
    return bytes(output), symbols


def inspect_unified_program(program: bytes) -> dict[str, object]:
    """Decode structural records for tests and command-stream diagnostics."""

    if len(program) % 4:
        raise ValueError("unified program contains a partial word")
    words = list(struct.unpack(f"<{len(program) // 4}I", program))
    if len(words) < PROGRAM_HEADER_WORDS + 1:
        raise ValueError("unified program is shorter than its header")
    magic, version, total_words, command_count, core_count = words[:5]
    if (magic, version, total_words) != (
        PROGRAM_MAGIC,
        PROGRAM_VERSION,
        len(words),
    ):
        raise ValueError("invalid unified program header")
    cursor = PROGRAM_HEADER_WORDS
    records: list[dict[str, object]] = []
    observed_core = 0
    for _ in range(command_count):
        opcode = words[cursor]
        cursor += 1
        if opcode in (OP_GDMA_LOAD, OP_GDMA_STORE):
            command_id, dependency_count = words[cursor : cursor + 2]
            lp6_address = words[cursor + 2] | words[cursor + 3] << 32
            l2_address = words[cursor + 4] | words[cursor + 5] << 32
            row_bytes, rows, lp6_stride, l2_stride = words[cursor + 6 : cursor + 10]
            cursor += 10
            dependencies = words[cursor : cursor + dependency_count]
            cursor += dependency_count
            records.append(
                {
                    "id": command_id,
                    "kind": "gdma_load" if opcode == OP_GDMA_LOAD else "gdma_store",
                    "dependencies": dependencies,
                    "lp6_address": lp6_address,
                    "l2_address": l2_address,
                    "row_bytes": row_bytes,
                    "rows": rows,
                    "lp6_stride": lp6_stride,
                    "l2_stride": l2_stride,
                }
            )
            continue
        if opcode != OP_CORE_BEGIN:
            raise ValueError(f"invalid system opcode {opcode:#x}")
        (
            command_id,
            target,
            preferred,
            affinity,
            l1_low,
            l1_high,
            isa_count,
            dependency_count,
            allowed_count,
            region_count,
        ) = words[cursor : cursor + 10]
        cursor += 10
        dependencies = words[cursor : cursor + dependency_count]
        cursor += dependency_count
        allowed = words[cursor : cursor + allowed_count]
        cursor += allowed_count
        cursor += region_count * 7
        isa_start_word = cursor
        cursor += isa_count
        if words[cursor : cursor + 2] != [OP_CORE_END, command_id]:
            raise ValueError("CORE_BEGIN range is not terminated by matching CORE_END")
        cursor += 2
        observed_core += isa_count
        records.append(
            {
                "id": command_id,
                "kind": "core_block",
                "target_logical_core": None if target == NONE_U32 else target,
                "preferred_physical_core": None
                if preferred == NONE_U32
                else preferred,
                "affinity_id": None if affinity == NONE_U32 else affinity,
                "l1_bytes_required": l1_low | l1_high << 32,
                "dependencies": dependencies,
                "allowed_physical_cores": allowed,
                "isa_start_word": isa_start_word,
                "isa_word_count": isa_count,
            }
        )
    if words[cursor:] != [OP_PROGRAM_END] or observed_core != core_count:
        raise ValueError("invalid unified program terminator or core count")
    return {
        "program_word_count": len(words),
        "command_count": command_count,
        "core_instruction_count": core_count,
        "commands": records,
    }


class SystemProgramBuilder:
    def __init__(
        self,
        output: Path,
        *,
        logical_core: int = 0,
        l1_bytes_required: int = 4 * 1024 * 1024,
        l2_capacity_bytes: int = 8 * 1024 * 1024,
    ) -> None:
        self.output = output
        self.output.mkdir(parents=True, exist_ok=True)
        self.logical_core = logical_core
        self.l1_bytes_required = l1_bytes_required
        self.l2_capacity_bytes = l2_capacity_bytes
        self.commands: list[dict[str, object]] = []
        self._words: list[int] = []
        self._segment_dependencies: list[str] | None = None
        self._frontier: list[str] = []
        self._core_block_index = 0
        self._gdma_index = 0
        self._flushed_word_count = 0
        self._core_isa_name = ".core_isa.link.tmp"
        self._core_isa_file = (self.output / self._core_isa_name).open(
            "wb", buffering=8 * 1024 * 1024
        )
        self._finished = False
        self._l2_high_water = 0
        self._gp = [0] * 16
        self._control: dict[int, int] = {}
        self._replay_state = False

    def append(self, word: int) -> None:
        if not 0 <= word < 1 << 32:
            raise ValueError(f"core ISA word does not fit 32 bits: {word}")
        opcode = word & 0x3F
        if opcode in (0x2B, 0x3A):
            raise ValueError(
                "LP6 address/GDMA opcodes are system commands and cannot enter a core ISA stream"
            )
        self._ensure_segment()
        self._words.append(word)
        self._track_core_state(word)

    def extend(self, words: Iterable[int]) -> None:
        for word in words:
            self.append(word)

    def __iadd__(self, words: Iterable[int]) -> "SystemProgramBuilder":
        self.extend(words)
        return self

    def __len__(self) -> int:
        return self._flushed_word_count + len(self._words)

    def reserve_l2(self, address: int, size: int) -> None:
        if size <= 0 or address < 0 or address + size > self.l2_capacity_bytes:
            raise ValueError(
                f"invalid shared-L2 range [{address:#x}, {address + size:#x})"
            )
        self._l2_high_water = max(self._l2_high_water, address + size)

    def _ensure_segment(self) -> None:
        if self._segment_dependencies is not None:
            return
        self._segment_dependencies = list(self._frontier)
        if not self._replay_state:
            return
        saved_gp15 = self._gp[15]
        for register, value in enumerate(self._gp):
            self._words.extend(_load_u32_independent(register, value))
        for funct, value in sorted(self._control.items()):
            self._words.extend(_load_u32_independent(15, value))
            self._words.append(0x39 | 15 << 6 | funct << 22)
        self._words.extend(_load_u32_independent(15, saved_gp15))
        self._replay_state = False

    def _track_core_state(self, word: int) -> None:
        opcode = word & 0x3F
        rd = word >> 6 & 0xF
        rs1 = word >> 10 & 0xF
        if opcode == 0x22:
            immediate = word >> 14
            self._gp[rd] = (self._gp[rs1] + immediate) & 0xFFFF_FFFF
        elif opcode == 0x25:
            self._gp[rd] = ((word >> 10) & 0xF_FFFF) << 12
        elif opcode == 0x39:
            funct = word >> 22 & 0xF
            self._control[funct] = self._gp[rd]

    def gdma_load(
        self,
        *,
        lp6_address: int,
        l2_address: int,
        bytes: int,
        rows: int = 1,
        lp6_stride: int | None = None,
        l2_stride: int | None = None,
        overlap_with_current_core_block: bool = False,
    ) -> str:
        return self._gdma(
            kind="gdma_load",
            lp6_address=lp6_address,
            l2_address=l2_address,
            bytes=bytes,
            rows=rows,
            lp6_stride=lp6_stride,
            l2_stride=l2_stride,
            overlap_with_current_core_block=overlap_with_current_core_block,
        )

    def gdma_store(
        self,
        *,
        lp6_address: int,
        l2_address: int,
        bytes: int,
        rows: int = 1,
        lp6_stride: int | None = None,
        l2_stride: int | None = None,
        allow_following_core_block_overlap: bool = False,
    ) -> str:
        core_block = self._flush_core_block()
        dependencies = list(self._frontier)
        command = self._append_gdma(
            kind="gdma_store",
            dependencies=dependencies,
            lp6_address=lp6_address,
            l2_address=l2_address,
            bytes=bytes,
            rows=rows,
            lp6_stride=lp6_stride,
            l2_stride=l2_stride,
        )
        self._frontier = (
            [core_block]
            if allow_following_core_block_overlap and core_block
            else [command]
        )
        return command

    def _gdma(
        self,
        *,
        kind: str,
        lp6_address: int,
        l2_address: int,
        bytes: int,
        rows: int,
        lp6_stride: int | None,
        l2_stride: int | None,
        overlap_with_current_core_block: bool,
    ) -> str:
        if overlap_with_current_core_block and self._words:
            dependencies = list(self._segment_dependencies or self._frontier)
            core_block = self._flush_core_block(update_frontier=False)
            command = self._append_gdma(
                kind=kind,
                dependencies=dependencies,
                lp6_address=lp6_address,
                l2_address=l2_address,
                bytes=bytes,
                rows=rows,
                lp6_stride=lp6_stride,
                l2_stride=l2_stride,
            )
            self._frontier = [value for value in (core_block, command) if value]
            return command
        self._flush_core_block()
        command = self._append_gdma(
            kind=kind,
            dependencies=list(self._frontier),
            lp6_address=lp6_address,
            l2_address=l2_address,
            bytes=bytes,
            rows=rows,
            lp6_stride=lp6_stride,
            l2_stride=l2_stride,
        )
        self._frontier = [command]
        return command

    def _append_gdma(
        self,
        *,
        kind: str,
        dependencies: list[str],
        lp6_address: int,
        l2_address: int,
        bytes: int,
        rows: int,
        lp6_stride: int | None,
        l2_stride: int | None,
    ) -> str:
        if bytes <= 0 or rows <= 0:
            raise ValueError("GDMA bytes and rows must be positive")
        if lp6_address < 0 or l2_address < 0:
            raise ValueError("GDMA byte addresses must be non-negative")
        lp6_stride = bytes if lp6_stride is None else lp6_stride
        l2_stride = bytes if l2_stride is None else l2_stride
        if lp6_stride < bytes or l2_stride < bytes:
            raise ValueError("GDMA strides must cover one row")
        lp6_span = (rows - 1) * lp6_stride + bytes
        if lp6_address + lp6_span > 1 << 64:
            raise ValueError("GDMA LP6 byte range exceeds 64-bit address space")
        l2_span = (rows - 1) * l2_stride + bytes
        self.reserve_l2(l2_address, l2_span)
        command_id = f"gdma_{self._gdma_index:06d}"
        self._gdma_index += 1
        command: dict[str, object] = {
            "id": command_id,
            "kind": kind,
            "dependencies": dependencies,
            "lp6_address": lp6_address,
            "l2_address": l2_address,
            "bytes": bytes,
        }
        if rows != 1 or lp6_stride != bytes or l2_stride != bytes:
            command.update(
                rows=rows,
                lp6_stride=lp6_stride,
                l2_stride=l2_stride,
            )
        self.commands.append(command)
        return command_id

    def _flush_core_block(self, *, update_frontier: bool = True) -> str | None:
        if not self._words:
            return None
        command_id = f"core_block_{self._core_block_index:06d}"
        self._core_block_index += 1
        isa_start_word = self._flushed_word_count
        isa_word_count = len(self._words)
        self._core_isa_file.write(
            struct.pack(f"<{isa_word_count}I", *self._words)
        )
        self.commands.append(
            {
                "id": command_id,
                "kind": "core_block",
                "isa_start_word": isa_start_word,
                "isa_word_count": isa_word_count,
                "dependencies": list(self._segment_dependencies or []),
                "target_logical_core": self.logical_core,
                "l1_bytes_required": self.l1_bytes_required,
            }
        )
        self._flushed_word_count += len(self._words)
        self._words.clear()
        self._segment_dependencies = None
        self._replay_state = True
        if update_frontier:
            self._frontier = [command_id]
        return command_id

    def finish(self) -> Path:
        if self._finished:
            raise ValueError("System Program has already been finalized")
        self._flush_core_block()
        self._core_isa_file.flush()
        self._core_isa_file.close()
        self._finished = True
        if not self.commands:
            raise ValueError("System Program contains no commands")
        core_isa_path = self.output / self._core_isa_name
        unified_program, command_symbols = encode_unified_program(
            self.commands, core_isa_path.read_bytes()
        )
        program_name = "program.bin"
        (self.output / program_name).write_bytes(unified_program)
        core_isa_path.unlink()
        logical_cores = max(1, self.logical_core + 1)
        payload: dict[str, object] = {
            "schema": "plena.v2.unified_program.v5",
            "program": program_name,
            "program_word_count": len(unified_program) // 4,
            "command_count": len(self.commands),
            "core_instruction_count": self._flushed_word_count,
            "command_symbols": command_symbols,
            "scheduling_policy": "compiler_static",
            "logical_to_physical": list(range(logical_cores)),
            "l2_bytes_required": self._l2_high_water,
            "l2_regions": (
                [
                    {
                        "name": "compiler_workspace",
                        "offset": 0,
                        "size": self._l2_high_water,
                        "alignment": 64,
                    }
                ]
                if self._l2_high_water
                else []
            ),
        }
        target = self.output / "system.json"
        target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return target


def _load_u32_independent(register: int, value: int) -> list[int]:
    """Set a GP without relying on another GP's value."""

    value &= 0xFFFF_FFFF
    upper, lower = value >> 12, value & 0xFFF
    words = [0x25 | register << 6 | upper << 10]
    if lower:
        words.append(0x22 | register << 6 | register << 10 | lower << 14)
    return words
