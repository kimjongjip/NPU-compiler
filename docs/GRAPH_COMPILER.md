# Operation-driven graph compiler — 2026-09-14

**Migration handoff:** [RESUME_CPP_COMPILER.md](RESUME_CPP_COMPILER.md) records
the next native C++ implementation plan. The `NativeGraph*` draft files are
not built or tested. This document describes the still-active Python MLIR
baseline, not a completed C++ migration.

The default `build/bin/plena-compile-model` now compiles the actual imported
operation graph. It no longer validates a graph and then generates an unrelated
fixed Llama instruction sequence. `--backend reference` explicitly selects the
historical generator; there is **no automatic fallback**.

## End-to-end path

```text
local Hugging Face checkpoint, complete forward, static prompt
  -> torch.export: external parameters/buffers + ATen graph
  -> official torch-MLIR: Torch -> Linalg/Tensor/Arith/Math IR
  -> canonicalization and CSE
  -> operation legalization (registered plena_graph dialect)
  -> parameter layout packing and lifetime-aware memory planning
  -> SA/VPU tiles
  -> logical-core placement and completion-event dependencies
  -> plena_cmd: GDMA load/store + core instruction blocks
  -> native C++ plena-encode-program-v7 pass
  -> one program.bin + system.json + lp6.bin
  -> Rust simulator -> shared-L2 output tensors
```

Every graph-stage MLIR file is saved, verified by the native `plena-opt`, and
reparsed as the next pass's input. The imported graph's operands, shapes, indexing
maps, scalar regions and return values drive lowering. No decoder layer count,
model name or parameter-name pattern selects a canned computation.

## Implementation boundaries — important

The graph legalization, planning, tiling, scheduling and command lowering passes
are **Python transformations using MLIR IR bindings**. Their pass names below
identify Python pass classes; they are not all `plena-opt` command-line passes.
The graph dialect is registered using TableGen with native C++ structural
verification. Its experimental operations carry versioned descriptor attributes
and explicit buffer SSA operands. Detailed descriptor validation is also done
on Python reload. This is not yet a fully idiomatic C++ rewrite-pattern backend
with scalar expression regions in every graph operation.

The final command verifier/encoder is the existing **native C++ MLIR pass**.
Python extracts the encoded words; it does not use the old model generator to
create the final program. Python can be retained as an implementation language
or individual graph passes can later migrate to C++ without changing the ISA.

| Pass | Actual responsibility | Artifact |
|---|---|---|
| `canonicalize,cse` | Official MLIR cleanup | `01-canonical.linalg.mlir` |
| `plena-legalize-linalg` | Scalar-region legalization, views, constant folding, generic kernels | `02-legalized.mlir` |
| `plena-plan-graph-memory` | Weight layout, live ranges, L2 reuse, explicit LP6 spills | `03-memory.mlir` |
| `plena-tile-graph` | M/N SA tiles, temporal K, RF-bounded vectors and row reductions | `04-tiled.mlir` |
| `plena-schedule-graph` | Output-tile logical-core assignment and dependencies | `05-scheduled.mlir` |
| `plena-lower-graph-to-commands` | Concrete DMA, VPU/SA operands, registers and instructions | `06-commands.mlir` |
| `plena-encode-program-v7` | Native unified binary encoding | `07-isa.mlir`, `program.bin` |

## Supported baseline

- Complete static FP16 model forward, including every returned logits row.
- FP16 Matrix operands, FP32 accumulation and explicit FP16/FP32 writeout.
- `linalg.matmul`/`batch_matmul`; M/N tails and temporal K tails.
- General elementwise scalar regions used by add, multiply, square, casts,
  EXP, reciprocal-based division, sqrt/rsqrt and ReLU compare/select.
- Last-axis sum/max reductions, with RF-sized partial reductions and explicit
  FP32 scalar combination. An unused softmax argmax result is discarded; a
  live argmax result is not silently ignored.
- RMSNorm, softmax, SiLU and RoPE as imported primitive operations, not ISA macros.
- Static transpose, reshape, slices, concat and projected broadcast views.
- Compile-time position/mask/table expressions; discrete prompt IDs and masks
  are specialized. Runtime weights and activations are **not evaluated on CPU**
  to manufacture simulator outputs. Embeddings read selected weight entries.
- Parameter-only layout packing is a data-layout conversion, not inference.

## Memory and scheduling

Weights remain in LP6. A compiler-reserved L2 staging window per logical core
feeds private L1 through explicit GDMA then LDMA. Rectangular Matrix inputs use
2-D DMA descriptors where legal; a row is not automatically a separate GDMA.

Activation buffers use aligned shared-L2 allocations. A buffer's lifetime includes
all aliases and uses. Storage is reused only after the final consumer. If no L2
region fits, the buffer is allocated in LP6 and the compiler emits store/reload
commands. Returned outputs currently must fit shared L2; otherwise compilation
fails with an output-staging diagnostic.

Independent N output tiles are assigned round-robin to logical cores. M and K
are not split into cross-core partial sums. Tiles on one core are ordered;
downstream operations wait for all relevant producer cores. Physical placement
is explicit and separate from logical IDs. Vector and reduction kernels use
logical core 0 in this baseline.

VPU expressions keep temporary values in bounded vector registers **within one
tile**. Different tensor operations currently store/load through L2. There is
no graph fusion or automatic register spilling. SA operands for one spatial
tile's full K must fit L1 before K-chunk MMA issue; otherwise compilation fails.

## Numerical contract

The HF driver converts the loaded model to FP16 **before capturing it**; explicit
FP32 norm/softmax operations in that graph remain FP32. It does not quietly
reinterpret BF16 IR as FP16.

Reduction order is reassociated to RF-width partial trees plus sequential FP32
combination. Division uses RCP + MUL because the selected ISA has no vector DIV.
This is an inference-tolerance contract, **not a promise of bit-identical IEEE
evaluation or exact HF logits**. `--graph-atol` and `--graph-rtol` control the HF
comparison; defaults are 0.002 and 0.02. These are validation tolerances, not a
guaranteed error bound for arbitrary inputs. Exceptional NaN/Inf behavior is not
comprehensively validated; ReLU's unordered comparison is lowered explicitly.

## Run and test

```bash
# All new temporary files stay under the LP6 project.
export TMPDIR=/home/jongjip/LP6/tmp
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$TMPDIR"

PLENA_Compiler/build/bin/plena-compile-model \
  --hf-model /path/to/local/checkpoint \
  --prompt "The capital of France is" \
  --output-dir /home/jongjip/LP6/tmp/model-run --execute

# Use a Python containing the existing torch/torch-MLIR/transformers toolchain.
"$PLENA_TORCH_MLIR_PYTHON" PLENA_Compiler/test/run_graph_e2e.py --cores 2
"$PLENA_TORCH_MLIR_PYTHON" PLENA_Compiler/test/run_hf_graph_cli.py
```

The HF path binds actual checkpoint data (`pretrained-cpu`). It rejects partial
layer selection and nonzero `position_start` until that entry ABI is supported.
The public CLI test creates a small, randomly initialized on-disk HF checkpoint
and tokenizer. It tests the entire model; it does not test language quality.

Verified on 2026-09-14:

| Test | Result |
|---|---|
| Entire 2-layer HF Llama + final norm + LM head, 1 and 2 cores | Max absolute logits error 0.000244140625; argmax matched |
| 2-core logical-to-physical placement `[1,0]` | Same numerical result |
| Non-Llama Linear(19,37) + bias + ReLU + broadcast, five rows | Passed tolerance; covers M/N/K tails |
| L2 pressure requiring two spilled buffers | Explicit GDMA stores/reloads; exact test output |
| FP32 reduction over 960 elements | Passed numerical comparison |
| Change imported `arith.addf` to `arith.subf` | Simulator output changed accordingly |
| Unsupported imported operation | Compilation failed; no model-generator fallback |
| Public HF CLI, full 2-layer checkpoint, five prompt tokens | Passed HF comparison, argmax matched |
| Seven negative/unit cases | Existing-output preservation, dtype/operation rejection, finite events, hardware limits, runtime assertion and output-map rejection |

Final local artifacts (all inside this project):

- `../tmp/plena-graph-e2e.4url1hzq`: final 1-core graph tests; full-model
  simulated cycles 115,469.
- `../tmp/plena-graph-e2e.pi7m5wt4`: 2-core graph tests; full-model simulated
  cycles 112,620.
- `../tmp/plena-hf-graph-cli.woirss3v/bundle`: public CLI checkpoint capture,
  every intermediate MLIR stage, Program v7, Rust dumps and HF comparison.

These cycle counts are for the tiny regression model, not a trained 1B model
or hardware measurements. `test/run_all.sh` also passed the six existing
native matmul executions and four core-immediate tests.

These are small full-model tests. The previous trained Llama-3.2-1B checkpoint
was not present at the earlier path during this work, so **the new backend has
not been validated on that full checkpoint**. Historical large-model reference
results must not be attributed to the new graph compiler.

## Remaining limitations / next work

1. Large-model scalability: per-tile command/event counts and verbose MLIR can
   be large. The configured finite event scoreboard is enforced, not silently
   enlarged. Add command batching/event reuse before claiming arbitrary-size
   model support.
2. Move descriptor-heavy graph operations to richer typed attributes/regions;
   optionally port Python transformations to native C++ MLIR patterns.
3. Fusion, cross-operation RF reuse, asynchronous double-buffer scheduling and
   cost-based core placement are not implemented in the new graph path.
4. Persistent KV-cache decode, dynamic shapes/runtime gathers, general argmax,
   INT8/INT4 graph quantization, arbitrary reduction axes, and full-K L1 spill
   are not implemented. Unsupported cases fail closed.
5. Add independent instruction-level numerical differential tests and broader
   trained-model comparisons. Small HF tolerance tests alone are not proof of
   full compiler correctness.

This is an executable operation-driven **baseline compiler**, not a finished
optimizing compiler for arbitrary Hugging Face models.
