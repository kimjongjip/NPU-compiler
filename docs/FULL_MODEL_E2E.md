# Full Hugging Face model path

## Current default versus historical results

**2026-09-14:** The default is now the operation-driven MLIR graph compiler,
documented in [GRAPH_COMPILER.md](GRAPH_COMPILER.md). It was validated on an
entire small 2-layer Hugging Face model and LM head, on 1/2 cores. The trained
Llama-3.2-1B checkpoint has not been validated with this new backend.

Everything below describes the **historical reference backend**, now selected
explicitly using `--backend reference`. Its old full-model accuracy and cycle
results do not establish correctness or performance of the new MLIR graph path.
New temporary artifacts are stored under `/home/jongjip/LP6/tmp`; old `/tmp`
paths below are historical records, not defaults for new runs.

## Historical reference-backend validation

Current Program v7 validation: 2026-09-10, actual Llama-3.2-1B layer 0,
six prompt tokens, all 16 FP16 checkpoints bit-exact with atol=0.
Continuous per-K FP32 reference accumulation matches the revised Matrix contract.
Total: 9,983,215 cycles; RF live-data peak: 448 B/core.
Artifacts: `/tmp/plena-generality-v7.ke5vTC/llama_layer`.
This layer check used the local model backend generator plus Rust execution;
it was not a new full-model torch.export/LM-head run.
The complete 16-layer model was **not rerun for v7**. Historical full-model
results below must not be reported as v7 timing. New generic ISA/queue details:
`PLENA_Simulator/transactional_emulator/docs/ISA_V7_GENERALITY.md`.

The model environment can use Python 3.10. TOML parsing then delegates to the
local config tool under `PLENA_CONFIG_PYTHON` (default `python3`, requires 3.11+).
Set that variable to a Python 3.11+ executable if the shell's python3 is also 3.10.
No additional parser is imported from the ETRI source tree, and RF limits are
validated before expensive model capture.

2026-09-10 update: Vector code generation now tiles to the simulator's finite
register capacity (default 16 registers × 512 bits per core). The full-model
driver reads the width from its simulator settings; its current dense decoder
assignment requires at least seven vector registers. `bounded_vector_emitter.py`
is physically vendored here, not linked/imported from the simulator project.
Older programs with oversized vector values must be regenerated. The full-model
cycle numbers below predate this change. A new full Llama layer (six prompt
tokens) was compiled and verified with zero FP16 error at every checkpoint;
the complete 16-layer model has not been rerun for this change.

Current compiler/simulator output uses Program v7 with inline Matrix shape and
retained Vector values. The 2026-09-04/08 results below are archived v5 results,
not instructions to run old binaries through the new decoder. Historical v6 validation
is recorded in `PLENA_Simulator/transactional_emulator/docs/ISA_V6_MIGRATION.md`.

The final v6 run completed all 16 layers and LM head with 18 bit-exact FP16
checkpoints and the same ` Paris` next token. It reports 199,181,725 cycles.
Its 46,588,072-byte program has 11,204,360 encoded core words, including Matrix
payloads. Results are retained at
`/home/jongjip/LP6/runs/llama32-1b-isa6-final-20260909`.

## What is implemented

`plena-compile-model` performs one end-to-end static prefill run:

```text
local Hugging Face checkpoint
  -> actual model.forward captured by torch.export
  -> official torch-mlir FX import (Torch dialect MLIR)
  -> graph/operator/parameter ABI/capability certificate
  -> checkpoint packing and generic PLENA ISA scheduling
  -> unified Program v7 + LP6 image
  -> Rust functional and resource-aware timing simulator
  -> SRAM logits versus target golden
  -> optional Hugging Face FP16 eager comparison
```

The frontend uses fake tensors only while capturing graph shape and state ABI.
The target generator reads every real weight used for execution from the local
`model.safetensors` checkpoint.

## Source ownership

The frontend implementation is stored as regular files under
`tools/frontend`; the full-model encoder is under `tools/full_model`.
Neither directory contains a symbolic link or imports Python source from the
ETRI repository. The CMake build and model wrapper still require compatible
external LLVM/MLIR and torch-mlir installations, just as they require PyTorch
and Transformers. Those are toolchain dependencies, not linked compiler
source.

The following check must print nothing:

```bash
find tools/frontend tools/full_model -type l -print
```

## Reproduced Llama-3.2-1B result

Command:

```bash
build/bin/plena-compile-model \
  --hf-model /home/jongjip/models/llama_3.2_1b_instruct \
  --prompt "The capital of France is" \
  --output-dir /tmp/plena-llama32-1b-full-v2 \
  --execute --atol 0
```

Result with the checked simulator configuration (32x32 FP16 SA, one core,
800 MHz):

| Item | Result |
|---|---:|
| Decoder layers | 16 / 16 |
| Captured graph nodes | 1,380 |
| `aten::linear` / `matmul` | 113 / 33 |
| Program commands | 31,648 |
| Core ISA words | 5,392,111 |
| `program.bin` | 23,339,076 bytes |
| `lp6.bin` | 2,471,654,400 bytes |
| Private-L1 activation high-water | 1,412,416 bytes |
| Simulated cycles | 199,366,514 |
| Target-contract worst FP16 error | 0.0 (bit-exact) |
| PLENA next token | ` Paris` (ID 12366) |
| Hugging Face FP16 next token | ` Paris` (ID 12366) |
| HF-vs-PLENA max absolute logit error | 0.017578125 |
| Logit cosine similarity | 0.9999957 |

The simulated cycle count is a simulator result, not measured silicon
performance. Host wall time was about 30 minutes, including tensor calculation,
per-clock DRAMSim3 processing and per-transaction SRAM/NoC/event scheduling.
That measurement alone does not identify the dominant component. Subsequent
host performance probes and equivalent optimizations are documented in
`PLENA_Simulator/transactional_emulator/docs/HOST_PERFORMANCE.md`.

The optimized simulator was rerun on 2026-09-08. It completed the same full
16-layer prefill plus LM head in 1631.333 s (27 min 11 s); compilation, execution
and validation together took 1816.662 s (30 min 17 s). All 18 FP16 checkpoint
tensors were bit-exact against the target golden. NPU cycles, next token and
HF logit comparison match the results above. The successful run had no disk
pause. This is approximately 9.4% less simulator wall time than the earlier
recorded 1800.372 s run, not a kernel-speedup extrapolation.

The complete new bundle is retained at
`/home/jongjip/LP6/runs/llama32-1b-optimized-20260908-retry`, including
`run_summary.json`, `execution.json`, `hf_comparison.json`, and target images,
timing and SRAM dumps. The retry launcher compiles first and preserves the
published inputs before executing Rust, and monitors free disk space.

## Important boundary

The complete imported graph is fail-closed certified, but the current
full-model backend is still a dense-Llama-specific Python reference lowering.
It emits only generic DMA, Matrix, Vector, Scalar, Reduction, and control ISA;
there are no fused `ATTENTION`, `RMSNORM`, or `SILU` instructions. However,
individual imported Torch operations are not yet lowered by the native C++
MLIR pass pipeline. `compilation.json` records this as
`operation_driven_cpp_mlir_backend: false`.

Current execution is static prefill for 1–32 prompt tokens and produces one
next-token logits vector. Stateful KV-cache decode, multiple generated tokens,
general model families, and native MLIR full-graph lowering remain future work.

The target packer implements both ordinary RoPE and the Llama-3 frequency
scaling contract. Its frequency tensor was checked bit-for-bit against the
Transformers `llama3` RoPE initializer before the full run above.
