#!/usr/bin/env python3
"""Compile one FP16 matmul and execute it on the PLENA Rust simulator."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile


@dataclass(frozen=True)
class Problem:
    name: str
    m: int
    k: int
    n: int
    mlir: str


PROBLEMS = (
    Problem("aligned", 4, 64, 64, "fp16_matmul.mlir"),
    Problem("tail_k96", 5, 96, 37, "fp16_matmul_tail.mlir"),
    Problem("multi_mn_tile", 40, 33, 45, "fp16_matmul_multi_tile.mlir"),
)


def fp16_bytes(values: list[float]) -> bytes:
    return b"".join(struct.pack("<e", value) for value in values)


def decode_fp16(data: bytes) -> list[float]:
    return [value[0] for value in struct.iter_unpack("<e", data)]


def inputs(problem: Problem) -> tuple[list[float], list[float]]:
    activation = [
        float(((row * 3 + reduction) % 7) - 3) / 4
        for row in range(problem.m)
        for reduction in range(problem.k)
    ]
    weight = [
        float(((reduction * 5 + column * 2) % 9) - 4) / 8
        for reduction in range(problem.k)
        for column in range(problem.n)
    ]
    return activation, weight


def reference(
    problem: Problem, activation: list[float], weight: list[float]
) -> bytes:
    output: list[float] = []
    for row in range(problem.m):
        for column in range(problem.n):
            total = 0.0
            for reduction in range(problem.k):
                total += (
                    activation[row * problem.k + reduction]
                    * weight[reduction * problem.n + column]
                )
            output.append(total)
    return fp16_bytes(output)


def run(command: list[str], *, cwd: Path | None = None) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def check_invalid_input(
    *, work: Path, compiler: Path, compiler_root: Path
) -> dict[str, object]:
    case = work / "invalid_missing_zero_fill"
    case.mkdir()
    activation, weight = inputs(PROBLEMS[0])
    activation_path = case / "activation.bin"
    weight_path = case / "weight.bin"
    activation_path.write_bytes(fp16_bytes(activation))
    weight_path.write_bytes(fp16_bytes(weight))
    output = case / "bundle"
    result = subprocess.run(
        [
            str(compiler),
            str(compiler_root / "test" / "invalid_missing_zero_fill.mlir"),
            "--config",
            str(compiler_root / "configs" / "plena32_single_core.json"),
            "--activation-data",
            str(activation_path),
            "--weight-data",
            str(weight_path),
            "--output-dir",
            str(output),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode == 0 or output.exists():
        raise RuntimeError("invalid matmul unexpectedly compiled or published output")
    if "zero linalg.fill" not in result.stderr:
        raise RuntimeError(f"unexpected invalid-input diagnostic: {result.stderr}")
    return {"missing_zero_fill_rejected": True, "output_not_published": True}


def compile_and_run(
    *,
    work: Path,
    problem: Problem,
    cores: int,
    compiler: Path,
    optimizer: Path,
    simulator: Path,
    simulator_settings: Path,
    compiler_root: Path,
) -> dict[str, object]:
    case = work / f"{problem.name}_{cores}_core"
    case.mkdir()
    activation, weight = inputs(problem)
    activation_path = case / "activation.bin"
    weight_path = case / "weight.bin"
    activation_path.write_bytes(fp16_bytes(activation))
    weight_path.write_bytes(fp16_bytes(weight))
    bundle = case / "bundle"
    config = compiler_root / "configs" / (
        "plena32_single_core.json" if cores == 1 else "plena32_two_core.json"
    )
    run(
        [
            str(compiler),
            str(compiler_root / "examples" / problem.mlir),
            "--config",
            str(config),
            "--activation-data",
            str(activation_path),
            "--weight-data",
            str(weight_path),
            "--output-dir",
            str(bundle),
        ]
    )
    for artifact in (
        "planned.mlir",
        "tiled.mlir",
        "scheduled.mlir",
        "commands.mlir",
        "lowered.mlir",
    ):
        run([str(optimizer), str(bundle / artifact), "-o", os.devnull])

    settings_text = simulator_settings.read_text(encoding="utf-8")
    if settings_text.count("num_cores = 1") != 1:
        raise RuntimeError("simulator settings do not contain one default core count")
    case_settings = case / "plena_settings.toml"
    case_settings.write_text(
        settings_text.replace("num_cores = 1", f"num_cores = {cores}"),
        encoding="utf-8",
    )
    run(
        [
            str(simulator),
            "--system-program",
            "system.json",
            "--lp6-image",
            "lp6.bin",
            "--lp6-size",
            "64MiB",
            "--settings",
            str(case_settings),
            "--sram-timing-out",
            "timing.json",
            "--timeline-out",
            "timeline.json",
            "--log-level",
            "off",
        ],
        cwd=bundle,
    )

    manifest = json.loads((bundle / "model_manifest.json").read_text(encoding="utf-8"))
    output = manifest["output"]
    l2 = (bundle / "l2_sram_dump.bin").read_bytes()
    begin = int(output["byte_base"])
    size = int(output["size_bytes"])
    actual = l2[begin : begin + size]
    expected = reference(problem, activation, weight)
    if actual != expected:
        actual_values = decode_fp16(actual)
        expected_values = decode_fp16(expected)
        maximum = max(abs(lhs - rhs) for lhs, rhs in zip(actual_values, expected_values))
        raise RuntimeError(
            f"{problem.name} {cores}-core matmul mismatch: "
            f"max_abs_error={maximum}"
        )

    timing = json.loads((bundle / "timing.json").read_text(encoding="utf-8"))
    system = json.loads((bundle / "system.json").read_text(encoding="utf-8"))
    program = (bundle / "program.bin").read_bytes()
    header = struct.unpack("<5I", program[:20])
    if header[0] != int.from_bytes(b"PLN7", "little") or header[1] != 7:
        raise RuntimeError("compiler did not emit a Program v7 header")
    if header[2] * 4 != len(program):
        raise RuntimeError("Program v7 header length is inconsistent")
    if (
        int(system["program_word_count"]) != header[2]
        or int(system["command_count"]) != header[3]
        or int(system["core_instruction_count"]) != header[4]
    ):
        raise RuntimeError("Program v7 header and system sidecar disagree")
    if list(bundle.glob("*.mem")) or (bundle / "core_isa.bin").exists():
        raise RuntimeError("compiler emitted a removed per-core program artifact")
    if timing["program_abi"] != "unified_command_isa_v7":
        raise RuntimeError("simulator did not select the unified Program v7 ABI")
    tile_count = ((problem.m + 31) // 32) * ((problem.n + 31) // 32)
    if (
        timing["core_block_count"] != tile_count
        or timing["gdma_command_count"] != 2
    ):
        raise RuntimeError("incorrect N-axis core block or GDMA count")
    executions = [
        block
        for core_blocks in timing["core_block_executions"]
        for block in core_blocks
    ]
    physical_cores = sorted({int(block["physical_core_id"]) for block in executions})
    placement = [int(core) for core in system["logical_to_physical"]]
    used_logical = sorted({int(block["logical_core_id"]) for block in executions})
    expected_cores = sorted({placement[core] for core in used_logical})
    if physical_cores != expected_cores:
        raise RuntimeError(
            f"incorrect logical-to-physical placement: {physical_cores}"
        )
    for block in executions:
        logical = int(block["logical_core_id"])
        physical = int(block["physical_core_id"])
        if physical != placement[logical]:
            raise RuntimeError(
                f"block {block['block_id']} mapped logical {logical} to "
                f"physical {physical}, expected {placement[logical]}"
            )
    scheduled = (bundle / "scheduled.mlir").read_text(encoding="utf-8")
    if scheduled.count('"plena_sched.core_block"') != tile_count:
        raise RuntimeError("scheduled IR has the wrong output-tile count")
    if "plena_mem.binding" not in (bundle / "planned.mlir").read_text(
        encoding="utf-8"
    ):
        raise RuntimeError("planned IR has no explicit hierarchical memory bindings")
    commands = (bundle / "commands.mlir").read_text(encoding="utf-8")
    if commands.count('"plena_cmd.gdma_load"') != 2 or commands.count(
        '"plena_cmd.core_block"'
    ) != tile_count:
        raise RuntimeError("structured command IR does not match the binary records")
    return {
        "cores": cores,
        "problem": [problem.m, problem.k, problem.n],
        "program_words": header[2],
        "commands": header[3],
        "core_instruction_words": header[4],
        "total_cycles": timing["total_simulated_cycles"],
        "physical_cores_used": physical_cores,
        "logical_to_physical": placement,
        "output_bytes": size,
        "exact_fp16": True,
    }


def main() -> None:
    compiler_root = Path(__file__).resolve().parents[1]
    lp6_root = compiler_root.parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--compiler",
        type=Path,
        default=compiler_root / "build" / "bin" / "plena-compile",
    )
    parser.add_argument(
        "--optimizer",
        type=Path,
        default=compiler_root / "build" / "bin" / "plena-opt",
    )
    parser.add_argument(
        "--simulator",
        type=Path,
        default=lp6_root
        / "PLENA_Simulator"
        / "transactional_emulator"
        / "target"
        / "release"
        / "transactional_emulator",
    )
    parser.add_argument(
        "--simulator-settings",
        type=Path,
        default=lp6_root / "PLENA_Simulator" / "plena_settings.toml",
    )
    parser.add_argument("--work-dir", type=Path)
    args = parser.parse_args()
    for executable in (args.compiler, args.optimizer, args.simulator):
        if not executable.is_file():
            raise SystemExit(f"required executable is missing: {executable}")

    temporary = args.work_dir is None
    work = Path(tempfile.mkdtemp(prefix="plena-compiler-e2e.")) if temporary else args.work_dir
    if not temporary:
        if work.exists():
            raise SystemExit(f"work directory already exists: {work}")
        work.mkdir(parents=True)
    try:
        results = []
        for problem in PROBLEMS:
            for cores in (1, 2):
                results.append(
                    compile_and_run(
                        work=work,
                        problem=problem,
                        cores=cores,
                        compiler=args.compiler.resolve(),
                        optimizer=args.optimizer.resolve(),
                        simulator=args.simulator.resolve(),
                        simulator_settings=args.simulator_settings.resolve(),
                        compiler_root=compiler_root,
                    )
                )
        invalid = check_invalid_input(
            work=work,
            compiler=args.compiler.resolve(),
            compiler_root=compiler_root,
        )
        print(
            json.dumps(
                {
                    "schema": "plena.compiler.e2e.v1",
                    "cases": results,
                    "negative": invalid,
                },
                indent=2,
            )
        )
    finally:
        if temporary:
            shutil.rmtree(work)


if __name__ == "__main__":
    main()
