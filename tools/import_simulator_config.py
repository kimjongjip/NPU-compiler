#!/usr/bin/env python3
"""Normalize compiler-relevant fields from PLENA simulator TOML to JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tomllib


def normalized_config(
    settings: dict[str, object], *, logical_cores: int | None, k_chunk: int
) -> dict[str, object]:
    transactional = settings["TRANSACTIONAL"]
    if not isinstance(transactional, dict):
        raise ValueError("TRANSACTIONAL table is missing")
    architecture = transactional["ARCHITECTURE"]
    system = transactional["SYSTEM"]
    memory = transactional["L2"]
    command = transactional["COMMAND_PROCESSOR"]
    if not all(isinstance(value, dict) for value in (architecture, system, memory, command)):
        raise ValueError("simulator architecture/system/L2/command tables are malformed")
    physical_cores = int(system["num_cores"])
    selected_cores = physical_cores if logical_cores is None else logical_cores
    if selected_cores <= 0 or selected_cores > physical_cores:
        raise ValueError(
            f"logical core count {selected_cores} is outside configured physical cores "
            f"1..{physical_cores}"
        )
    if k_chunk <= 0:
        raise ValueError("k_chunk must be positive")
    return {
        "schema_version": 1,
        "architecture": {
            "array_rows": int(architecture["array_rows"]),
            "array_columns": int(architecture["array_columns"]),
            "dataflow": str(architecture["dataflow"]),
            "physical_cores": physical_cores,
            "logical_cores": selected_cores,
            "logical_to_physical": list(range(selected_cores)),
        },
        "memory": {
            "l1_bytes_per_core": int(architecture["l1_sram_size_bytes"]),
            "l2_bytes": int(memory["size_bytes"]),
            "alignment": 64,
        },
        "tiling": {"k_chunk": k_chunk},
        "command_processor": {
            "completion_event_slots": int(command["completion_event_slots"])
        },
        "isa": {"version": 1 << 16, "word_bits": 32, "address_unit": "byte"},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("settings", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--vpu-only", action="store_true",
                        help="print the VPU settings as JSON for older model Python environments")
    parser.add_argument("--logical-cores", type=int)
    parser.add_argument("--k-chunk", type=int, default=64)
    args = parser.parse_args()
    with args.settings.open("rb") as stream:
        settings = tomllib.load(stream)
    if args.vpu_only:
        print(json.dumps(settings['TRANSACTIONAL']['VPU']))
        return
    if args.output is None:
        parser.error('--output is required unless --vpu-only is selected')
    config = normalized_config(
        settings, logical_cores=args.logical_cores, k_chunk=args.k_chunk
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8", newline="\n"
    )


if __name__ == "__main__":
    main()
