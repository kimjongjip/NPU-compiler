#!/usr/bin/env python3
"""Whole 2-layer HF Llama and non-model Linalg regressions, real Rust execution."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
PROJECT_TMP = ROOT.parent / "tmp"
PROJECT_TMP.mkdir(parents=True, exist_ok=True)
os.environ["TMPDIR"] = str(PROJECT_TMP)
tempfile.tempdir = str(PROJECT_TMP)

import numpy as np
import torch
from transformers import LlamaConfig, LlamaForCausalLM

sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "tools/frontend")]
import plena_torch_frontend as frontend
from graph_pipeline.pipeline import compile_linalg, write_json
from graph_pipeline.lower import Hardware
from graph_pipeline.huggingface import capture_bindings, read_outputs


def simulate(output, hw):
    sim = ROOT.parent / "PLENA_Simulator"
    settings = (sim / "plena_settings.toml").read_text()
    settings = settings.replace("num_cores = 1", f"num_cores = {hw.physical_cores}")
    (output / "settings.toml").write_text(settings)
    command = [
        str(sim / "transactional_emulator/target/release/transactional_emulator"),
        "--system-program",
        "system.json",
        "--lp6-image",
        "lp6.bin",
        "--lp6-size",
        "64MiB",
        "--settings",
        str(output / "settings.toml"),
        "--sram-timing-mode",
        "scheduled",
        "--sram-timing-out",
        "timing.json",
        "--log-level",
        "off",
    ]
    with (output / "simulator.log").open("w") as log:
        result = subprocess.run(command, cwd=output, stdout=log, stderr=log)
    if result.returncode:
        raise RuntimeError((output / "simulator.log").read_text()[-3000:])


def llama(work, cores):
    class WholeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder = (
                LlamaForCausalLM(
                    LlamaConfig(
                        vocab_size=96,
                        hidden_size=32,
                        intermediate_size=48,
                        num_hidden_layers=2,
                        num_attention_heads=2,
                        num_key_value_heads=1,
                        head_dim=16,
                        attn_implementation="eager",
                    )
                )
                .half()
                .eval()
            )

        def forward(self, ids, mask):
            return self.decoder(ids, attention_mask=mask, use_cache=False).logits

    torch.manual_seed(123)
    model = WholeModel().eval()
    args = (torch.tensor([[1, 2, 3]]), torch.ones((1, 3), dtype=torch.int64))
    loaded = frontend.LoadedModel(model, args, {}, dict(kind="test"), {}, {})
    capture = frontend.capture_loaded_model(loaded)
    front = work / "frontend"
    frontend.emit_package(capture, front, output_types=("torch", "linalg"))
    bindings, static = capture_bindings(loaded, capture)
    with torch.no_grad():
        expected = model(*args).numpy()
    hw = Hardware(
        cores=cores, physical_cores=cores, placement=tuple(reversed(range(cores)))
    )
    output = work / "target"
    report = compile_linalg(
        (front / "model.linalg.mlir").read_text(), bindings, output, hw, static
    )
    simulate(output, hw)
    actual = read_outputs(output, report)[0]
    np.save(output / "expected.npy", expected)
    np.save(output / "actual.npy", actual)
    error = np.abs(expected.astype(np.float32) - actual.astype(np.float32))
    metrics = dict(
        case="whole_hf_llama_2_layers",
        cores=cores,
        shape=list(actual.shape),
        max_abs_error=float(error.max()),
        mean_abs_error=float(error.mean()),
        argmax_match=bool(np.array_equal(actual.argmax(-1), expected.argmax(-1))),
        kernels=report["kernels"],
        tiles=report["tiles"],
        cycles=json.loads((output / "timing.json").read_text()).get(
            "total_simulated_cycles"
        ),
    )
    write_json(output / "comparison.json", metrics)
    np.testing.assert_allclose(actual, expected, atol=0.002, rtol=0.02)
    print(json.dumps(metrics), flush=True)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cores", type=int, default=1)
    parser.add_argument("--kernels-only", action="store_true")
    args = parser.parse_args()
    work = args.output or Path(tempfile.mkdtemp(prefix="plena-graph-e2e."))
    work.mkdir(exist_ok=True, parents=True)
    print("OUTPUT", work, flush=True)
    if not args.kernels_only:
        llama(work, args.cores)
    generic_cases(work)


def generic_cases(work):
    class LinearRelu(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(19, 37, bias=True)

        def forward(self, x):
            return torch.relu(self.linear(x)) + x[:, :1]

    class Spill(torch.nn.Module):
        def forward(self, x):
            a = x + x
            b = x * x
            c = a + b
            return -c

    class Reduce(torch.nn.Module):
        def forward(self, x):
            return x.square().mean(-1, keepdim=True).rsqrt() * x

    cases = [
        (
            "linear_relu",
            LinearRelu().half().eval(),
            torch.randn(5, 19).half(),
            Hardware(cores=2, physical_cores=2, placement=(1, 0)),
        ),
        (
            "spill",
            Spill().eval(),
            torch.arange(256).reshape(4, 64).half() / 128,
            Hardware(l2_bytes=1024, staging_bytes=128),
        ),
        ("reduce960", Reduce().eval(), torch.randn(2, 960).float(), Hardware()),
    ]
    for name, model, x, hw in cases:
        case = work / name
        case.mkdir()
        loaded = frontend.LoadedModel(model, (x,), {}, dict(kind="test"), {}, {})
        capture = frontend.capture_loaded_model(loaded)
        frontend.emit_package(capture, case / "frontend", output_types=("linalg",))
        bindings, static = capture_bindings(loaded, capture)
        with torch.no_grad():
            expected = model(x).numpy()
        source = (case / "frontend/model.linalg.mlir").read_text()
        report = compile_linalg(source, bindings, case / "target", hw, static)
        simulate(case / "target", hw)
        actual = read_outputs(case / "target", report)[0]
        np.testing.assert_allclose(actual, expected, atol=0.002, rtol=0.02)
        if name == "spill":
            assert report["spill_buffers"] > 0
            assert (
                '"plena_cmd.gdma_store"'
                in (case / "target/06-commands.mlir").read_text()
            )
        metrics = dict(
            case=name,
            max_abs_error=float(np.abs(actual.astype(float) - expected).max()),
            spill_buffers=report["spill_buffers"],
        )
        write_json(case / "comparison.json", metrics)
        print(json.dumps(metrics), flush=True)
        if name == "linear_relu":
            # Prove the imported MLIR, not the Python model, determines codegen.
            changed = source.replace("arith.addf", "arith.subf")
            assert changed != source
            modified = compile_linalg(changed, bindings, case / "modified", hw, static)
            simulate(case / "modified", hw)
            assert not np.allclose(read_outputs(case / "modified", modified)[0], actual)
            try:
                compile_linalg(
                    source.replace("arith.addf", "math.atan2"),
                    bindings,
                    case / "unsupported",
                    hw,
                    static,
                )
            except Exception as e:
                assert "unsupported" in str(e)
            else:
                raise AssertionError("unsupported operation silently compiled")


if __name__ == "__main__":
    main()
