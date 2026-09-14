"""Hugging Face orchestration for the operation-driven MLIR pass pipeline."""

from __future__ import annotations

import gc
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

import numpy as np
import torch
from transformers import AutoTokenizer

from .graph import CompileError
from .huggingface import capture_bindings, read_outputs
from .lower import Hardware
from .pipeline import compile_linalg, write_json


def compile_model(args, frontend, run_simulator, source_root):
    if args.num_layers is not None or args.allow_partial_model:
        raise CompileError(
            "MLIR graph path always compiles the complete forward graph; partial-layer selection is unsupported"
        )
    if args.position_start != 0:
        raise CompileError(
            "nonzero position_start requires an explicit position_ids input; not yet wired in the HF driver"
        )
    if args.frontend_load_mode != "pretrained-cpu":
        raise CompileError(
            "MLIR model compilation requires --frontend-load-mode pretrained-cpu to bind actual checkpoint bytes"
        )
    destination = args.output_dir.resolve()
    if destination.exists():
        raise CompileError("output directory already exists: " + str(destination))
    model = args.hf_model.resolve()
    if not (model / "config.json").is_file():
        raise CompileError("not a local Hugging Face model directory")
    tokenizer = AutoTokenizer.from_pretrained(
        model, local_files_only=True, trust_remote_code=False
    )
    inputs = tokenizer(args.prompt, return_tensors="pt")
    ids = inputs.input_ids
    mask = inputs.get("attention_mask", torch.ones_like(ids))
    if ids.shape[1] <= 0:
        raise CompileError("empty prompt")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    started = time.monotonic()
    try:
        print(
            "[1/4] capture complete Hugging Face forward -> ATen -> Torch/Linalg MLIR",
            flush=True,
        )
        loaded = frontend.load_huggingface_model(
            model, sequence_length=ids.shape[1], load_mode="pretrained-cpu"
        )
        # The target's supported Matrix input contract is IEEE FP16. Capture
        # this exact model precision, rather than silently changing a BF16 IR.
        loaded.module.half().eval()
        loaded = replace(loaded, args=(ids, mask))
        capture = frontend.capture_loaded_model(loaded)
        front = staging / "frontend"
        frontend.emit_package(capture, front, output_types=("torch", "linalg"))
        bindings, static = capture_bindings(loaded, capture)
        expected = None
        if args.execute and not args.skip_hf_compare:
            with torch.no_grad():
                expected = loaded.module(*loaded.args).detach().cpu().numpy()
        print(
            "[2/4] legalize -> memory/liveness -> SA/VPU tiles -> core/event schedule",
            flush=True,
        )
        config_python = os.environ.get("PLENA_CONFIG_PYTHON", "python3")
        config_path = staging / "target_config.json"
        cmd = [
            config_python,
            str(source_root / "tools/import_simulator_config.py"),
            str(args.simulator_settings.resolve()),
            "-o",
            str(config_path),
            "--k-chunk",
            str(args.k_chunk),
        ]
        if args.logical_cores is not None:
            cmd.extend(["--logical-cores", str(args.logical_cores)])
        subprocess.run(cmd, check=True)
        cfg = json.loads(config_path.read_text())
        arch = cfg["architecture"]
        memory = cfg["memory"]
        vpu = json.loads(
            subprocess.check_output(
                [
                    config_python,
                    str(source_root / "tools/import_simulator_config.py"),
                    str(args.simulator_settings.resolve()),
                    "--vpu-only",
                ],
                text=True,
            )
        )
        hw = Hardware(
            rows=arch["array_rows"],
            columns=arch["array_columns"],
            k_chunk=args.k_chunk,
            cores=arch["logical_cores"],
            physical_cores=arch["physical_cores"],
            placement=tuple(arch["logical_to_physical"]),
            l1_bytes=memory["l1_bytes_per_core"],
            l2_bytes=memory["l2_bytes"],
            rf_bytes=int(vpu["vector_register_bits"]) // 8,
            registers=int(vpu["vector_registers"]),
            event_slots=cfg["command_processor"]["completion_event_slots"],
        )
        target = staging / "target"
        print(
            "[3/4] lower scheduled operations -> command MLIR -> native Program v7 encoder",
            flush=True,
        )
        report = compile_linalg(
            (front / "model.linalg.mlir").read_text(), bindings, target, hw, static
        )
        del bindings, capture, loaded
        gc.collect()
        execution = None
        if args.execute:
            print("[4/4] execute compiled Program v7 on Rust simulator", flush=True)
            execution = run_simulator(
                target,
                simulator=args.simulator.resolve(),
                settings=args.simulator_settings.resolve(),
                lp6_size=args.lp6_size,
            )
            actual = read_outputs(target, report)[0]
            np.save(target / "actual_logits.npy", actual)
            next_id = int(actual[0, -1].argmax())
            execution["next_token"] = {
                "id": next_id,
                "text": tokenizer.decode([next_id]),
            }
            if expected is not None:
                np.save(target / "expected_hf_logits.npy", expected)
                err = np.abs(actual.astype(np.float32) - expected.astype(np.float32))
                comparison = dict(
                    max_abs_error=float(err.max()),
                    mean_abs_error=float(err.mean()),
                    atol=args.graph_atol,
                    rtol=args.graph_rtol,
                    allclose=bool(
                        np.allclose(
                            actual, expected, atol=args.graph_atol, rtol=args.graph_rtol
                        )
                    ),
                    last_token_argmax_match=next_id == int(expected[0, -1].argmax()),
                    note="tolerance comparison; reductions are reassociated and division is RCP+MUL",
                )
                write_json(staging / "hf_comparison.json", comparison)
                if not comparison["allclose"]:
                    raise CompileError(
                        "compiled output failed Hugging Face tolerance comparison: "
                        + str(comparison)
                    )
                execution["huggingface_comparison"] = comparison
            write_json(staging / "execution.json", execution)
        else:
            print("[4/4] execution skipped", flush=True)
        write_json(
            staging / "compilation.json",
            dict(
                schema="plena.compiler.full_model.v2",
                model=str(model),
                full_model=True,
                prompt=args.prompt,
                prompt_token_ids=ids.tolist(),
                backend="mlir",
                operation_driven=True,
                model_specialized_backend=False,
                lowering=report,
                executed=bool(args.execute),
                wall_seconds=time.monotonic() - started,
                specialization="fixed prompt IDs/mask and model buffers; runtime parameters/activations",
                precision="captured model converted to FP16; explicit graph FP32 operations preserved",
            ),
        )
        os.replace(staging, destination)
        return destination
    except Exception:
        # Preserve failed compilation diagnostics, without publishing success.
        write_json(
            staging / "FAILED.json", dict(status="failed", destination=str(destination))
        )
        print("failed compilation artifacts:", staging, flush=True)
        raise
