#!/usr/bin/env python3
"""Compile a complete local Hugging Face forward through MLIR to Program v7.

The default backend consumes official Linalg IR operation-by-operation.
The historical model-specialized generator requires --backend reference and
is never selected automatically when graph legalization fails.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Sequence


SOURCE_ROOT = Path(__file__).resolve().parents[2]
PROJECT_TMP = SOURCE_ROOT.parent / "tmp"
PROJECT_TMP.mkdir(parents=True, exist_ok=True)
os.environ["TMPDIR"] = str(PROJECT_TMP)
tempfile.tempdir = str(PROJECT_TMP)
FRONTEND_ROOT = SOURCE_ROOT / "tools" / "frontend"
FULL_MODEL_ROOT = SOURCE_ROOT / "tools" / "full_model"
for source_directory in (FRONTEND_ROOT, FULL_MODEL_ROOT):
    sys.path.insert(0, str(source_directory))
sys.path.insert(0, str(SOURCE_ROOT / "tools"))

import plena_full_model_backend as backend  # noqa: E402
import plena_semantic_bridge as bridge  # noqa: E402
import plena_torch_frontend as frontend  # noqa: E402


class ModelCompileError(RuntimeError):
    pass


def _vpu_settings(path: Path) -> dict[str, Any]:
    # torch-MLIR may use Python 3.10 while the ordinary config tools use 3.11+.
    # Parse real TOML with the stdlib, never a partial regex/config approximation.
    try:
        import tomllib
    except ModuleNotFoundError:
        interpreter = os.environ.get('PLENA_CONFIG_PYTHON', 'python3')
        result = subprocess.run(
            [interpreter, str(SOURCE_ROOT / 'tools/import_simulator_config.py'),
             str(path), '--vpu-only'], text=True, capture_output=True,
        )
        if result.returncode:
            raise ModelCompileError(
                'TOML parsing requires Python 3.11+; set PLENA_CONFIG_PYTHON.\n'
                + result.stderr[-2000:]
            )
        return json.loads(result.stdout)
    with path.open('rb') as stream:
        return tomllib.load(stream)['TRANSACTIONAL']['VPU']


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: Path, root: Path, *, hash_contents: bool = True) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path.relative_to(root)),
        "bytes": path.stat().st_size,
    }
    if hash_contents:
        result["sha256"] = _sha256(path)
    return result


def _token_count(model: Path, prompt: str) -> tuple[int, list[int]]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model, local_files_only=True, trust_remote_code=False
    )
    token_ids = tokenizer(prompt, return_tensors="pt").input_ids[0].tolist()
    if not 0 < len(token_ids) <= backend.M_TILE:
        raise ModelCompileError(
            f"this initial full-model backend accepts 1..{backend.M_TILE} prefill "
            f"tokens, but the prompt produces {len(token_ids)}"
        )
    return len(token_ids), [int(value) for value in token_ids]


def _copy_certificate(certificate: bridge.SemanticCertificate, output: Path) -> None:
    _write_json(output / "semantic_certificate.json", certificate.document)
    _write_json(output / "model_spec.json", certificate.model_spec.to_dict())
    _write_json(output / "capability_report.json", certificate.capability_report.to_dict())


def _run_simulator(
    target: Path,
    *,
    simulator: Path,
    settings: Path,
    lp6_size: str,
) -> dict[str, Any]:
    if not simulator.is_file():
        raise ModelCompileError(f"Rust simulator executable does not exist: {simulator}")
    if not settings.is_file():
        raise ModelCompileError(f"simulator settings do not exist: {settings}")
    command = [
        str(simulator),
        "--system-program",
        "system.json",
        "--lp6-image",
        "lp6.bin",
        "--lp6-size",
        lp6_size,
        "--settings",
        str(settings),
        "--sram-timing-mode",
        "scheduled",
        "--sram-timing-out",
        "timing.json",
        "--log-level",
        "off",
    ]
    started = time.monotonic()
    with (target / "simulator.stdout.txt").open("w", encoding="utf-8") as stdout, (
        target / "simulator.stderr.txt"
    ).open("w", encoding="utf-8") as stderr:
        result = subprocess.run(command, cwd=target, stdout=stdout, stderr=stderr, check=False)
    elapsed = time.monotonic() - started
    if result.returncode:
        tail = (target / "simulator.stderr.txt").read_text(
            encoding="utf-8", errors="replace"
        )[-4000:]
        raise ModelCompileError(
            f"Rust simulator failed with exit code {result.returncode}:\n{tail}"
        )
    timing = json.loads((target / "timing.json").read_text(encoding="utf-8"))
    return {
        "command": command,
        "wall_seconds": elapsed,
        "total_simulated_cycles": timing.get("total_simulated_cycles"),
        "timing_schema": timing.get("schema"),
    }


def _compare_huggingface(model_dir: Path, prompt: str, target: Path) -> dict[str, Any]:
    """Compare simulator logits with ordinary Hugging Face FP16 eager output."""

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    transformers.utils.logging.disable_progress_bar()
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, local_files_only=True, trust_remote_code=False
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.float16,
        attn_implementation="eager",
    )
    model.eval()
    with torch.no_grad():
        hf_logits = model(**inputs, use_cache=False).logits[0, -1].float()
    metadata = json.loads((target / "metadata.json").read_text(encoding="utf-8"))
    dump = (target / "sram_dump.bin").read_bytes()
    actual_logits = backend.read_sram_f16(
        dump,
        int(metadata["result_byte_address"]),
        list(metadata["result_shape"]),
    )[0].float()
    error = (hf_logits - actual_logits).abs()
    hf_id = int(hf_logits.argmax())
    plena_id = int(actual_logits.argmax())

    def token(value: int) -> dict[str, Any]:
        return {
            "id": value,
            "token": tokenizer.convert_ids_to_tokens(value),
            "text": tokenizer.decode([value]),
        }

    def top_five(values: Any) -> list[dict[str, Any]]:
        return [
            token(int(index)) | {"logit": float(values[index])}
            for index in torch.topk(values, 5).indices
        ]

    report = {
        "prompt": prompt,
        "input_ids": inputs.input_ids[0].tolist(),
        "huggingface_execution": "AutoModelForCausalLM eager FP16",
        "plena_execution": "Rust simulator SRAM logits",
        "huggingface_argmax": token(hf_id),
        "plena_argmax": token(plena_id),
        "argmax_match": hf_id == plena_id,
        "max_abs_logit_error": float(error.max()),
        "mean_abs_logit_error": float(error.mean()),
        "rmse": float(torch.sqrt(torch.mean(error.square()))),
        "cosine_similarity": float(
            torch.nn.functional.cosine_similarity(hf_logits, actual_logits, dim=0)
        ),
        "huggingface_top5": top_five(hf_logits),
        "plena_top5": top_five(actual_logits),
        "note": (
            "Different FP16 rounding and reduction boundaries may change logits; "
            "simulator correctness is established separately against its exact numeric contract."
        ),
    }
    _write_json(target.parent / "hf_comparison.json", report)
    return report


def compile_model(args: argparse.Namespace) -> Path:
    if args.backend == "mlir":
        from graph_pipeline.model_driver import compile_model as compile_graph
        return compile_graph(args, frontend, _run_simulator, SOURCE_ROOT)
    model = args.hf_model.resolve()
    if not model.is_dir() or not (model / "config.json").is_file():
        raise ModelCompileError(f"not a local Hugging Face model directory: {model}")
    vpu_settings = _vpu_settings(args.simulator_settings)
    if not 7 <= int(vpu_settings.get('vector_registers', 16)) <= 16:
        raise ModelCompileError('dense decoder lowering requires 7..16 vector registers')
    register_bits = int(vpu_settings.get('vector_register_bits', 512))
    if register_bits < 32 or register_bits % 32:
        raise ModelCompileError('vector_register_bits must be a positive multiple of 32')
    destination = args.output_dir.resolve()
    if destination.exists():
        raise ModelCompileError(f"output directory already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    started = time.monotonic()
    try:
        rows, token_ids = _token_count(model, args.prompt)
        config = json.loads((model / "config.json").read_text(encoding="utf-8"))
        total_layers = int(config["num_hidden_layers"])
        num_layers = total_layers if args.num_layers is None else args.num_layers
        if not 1 <= num_layers <= total_layers:
            raise ModelCompileError(
                f"--num-layers must be in [1, {total_layers}], got {num_layers}"
            )
        if num_layers != total_layers and not args.allow_partial_model:
            raise ModelCompileError(
                "partial-model compilation requires --allow-partial-model; the default is the full decoder"
            )

        frontend_dir = staging / "frontend"
        print(f"[1/4] capture + torch-mlir import ({rows} prompt tokens)", flush=True)
        loaded = frontend.load_huggingface_model(
            model,
            batch_size=1,
            sequence_length=rows,
            load_mode=args.frontend_load_mode,
            seed=0,
        )
        capture = frontend.capture_loaded_model(
            loaded,
            strict=True,
            normalization_policy="plena-decoder-v1",
        )
        frontend_result = frontend.emit_package(
            capture,
            frontend_dir,
            output_types=("torch",),
            capture_only=False,
        )

        print("[2/4] validate captured decoder semantics", flush=True)
        certificate = bridge.certify_dense_decoder(
            frontend_dir,
            frontend_dir,
            sequence_length=rows,
            max_context=rows,
            decode_buckets=(),
        )
        _copy_certificate(certificate, staging)
        del loaded, capture
        gc.collect()

        target_dir = staging / "target"
        print(f"[3/4] emit PLENA Program v7 ({num_layers}/{total_layers} layers)", flush=True)
        backend.generate(
            model,
            args.prompt,
            target_dir,
            args.position_start,
            num_layers,
            num_layers == total_layers,
            vector_register_bits=register_bits,
        )

        execution: dict[str, Any] | None = None
        if args.execute:
            print("[4/4] execute Rust functional/timing simulator", flush=True)
            execution = _run_simulator(
                target_dir,
                simulator=args.simulator.resolve(),
                settings=args.simulator_settings.resolve(),
                lp6_size=args.lp6_size,
            )
            check_log = target_dir / "check.stdout.txt"
            with check_log.open("w", encoding="utf-8") as stream, redirect_stdout(stream):
                backend.check(target_dir, args.atol)
            execution["functional_check"] = json.loads(
                (target_dir / "check_result.json").read_text(encoding="utf-8")
            )
            if num_layers == total_layers and not args.skip_hf_compare:
                print("      compare next-token logits with Hugging Face eager", flush=True)
                execution["huggingface_comparison"] = _compare_huggingface(
                    model, args.prompt, target_dir
                )
            _write_json(staging / "execution.json", execution)
        else:
            print("[4/4] execution skipped", flush=True)

        source_files = sorted(FRONTEND_ROOT.glob("*.py"))
        compilation = {
            "schema": "plena.compiler.full_model.v1",
            "model": str(model),
            "prompt": args.prompt,
            "prompt_token_ids": token_ids,
            "prompt_tokens": rows,
            "layers": num_layers,
            "model_total_layers": total_layers,
            "full_model": num_layers == total_layers,
            "frontend": {
                "capture": "torch.export",
                "importer": "official torch-mlir FX importer",
                "package_kind": frontend_result.manifest["package_kind"],
                "source_ownership": "physically vendored regular files",
                "source_symlinks": [str(path) for path in source_files if path.is_symlink()],
            },
            "lowering": {
                "kind": "graph-certified dense-Llama reference lowering",
                "program_format": "PLENA unified Program v7",
                "instructions": "generic Matrix/Vector/Scalar plus DMA/control",
                "operation_driven_cpp_mlir_backend": False,
                "note": (
                    "The captured graph is fail-closed certified, then lowered by the "
                    "vendored model-specialized reference backend. General full-graph "
                    "MLIR pass lowering is not yet implemented."
                ),
            },
            "executed": bool(args.execute),
            "wall_seconds": time.monotonic() - started,
            "artifacts": {
                "torch_mlir": _artifact(frontend_dir / "model.torch.mlir", staging),
                "program": _artifact(target_dir / "program.bin", staging),
                # Hashing the multi-gigabyte LP6 image needlessly doubles compile I/O.
                "lp6_image": _artifact(target_dir / "lp6.bin", staging, hash_contents=False),
            },
        }
        _write_json(staging / "compilation.json", compilation)
        os.replace(staging, destination)
        return destination
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _default_simulator() -> Path:
    return SOURCE_ROOT.parent / "PLENA_Simulator" / "transactional_emulator" / "target" / "release" / "transactional_emulator"


def _default_settings() -> Path:
    return SOURCE_ROOT.parent / "PLENA_Simulator" / "plena_settings.toml"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compile a complete local Hugging Face forward through MLIR passes and optionally execute it"
    )
    parser.add_argument("--hf-model", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--position-start", type=int, default=0)
    parser.add_argument("--backend", choices=("mlir", "reference"), default="mlir",
                        help="operation-driven MLIR pipeline (default), or explicit old Llama reference generator")
    parser.add_argument("--logical-cores", type=int)
    parser.add_argument("--k-chunk", type=int, default=64)
    parser.add_argument("--graph-atol", type=float, default=0.002)
    parser.add_argument("--graph-rtol", type=float, default=0.02)
    parser.add_argument(
        "--frontend-load-mode",
        choices=("config-fake", "config-cpu", "pretrained-cpu"),
        default="pretrained-cpu",
    )
    parser.add_argument("--num-layers", type=int)
    parser.add_argument("--allow-partial-model", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--simulator", type=Path, default=_default_simulator())
    parser.add_argument("--simulator-settings", type=Path, default=_default_settings())
    parser.add_argument("--lp6-size", default="16GiB")
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument(
        "--skip-hf-compare",
        action="store_true",
        help="skip the extra Hugging Face eager comparison after full-model execution",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        result = compile_model(_parser().parse_args(argv))
    except Exception as error:
        print(f"plena-compile-model: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(f"bundle: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
