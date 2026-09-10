# PLENA MLIR Compiler

Standalone LLVM/MLIR compiler targeting the current PLENA NPU simulator.
The project reuses the proven multi-level organization of the ETRI compiler,
but its memory hierarchy, tiling, scheduling, command IR, and binary encoder
are native to the PLENA Program v7 ABI.

Two executable paths are available:

```text
static FP16 linalg.fill + linalg.matmul
  -> byte-addressed LP6/L2/L1 plan
  -> 32x32 output-stationary M/N tiles
  -> N-axis logical-core assignment
  -> GDMA + event-SSA core schedule
  -> structured command records
  -> unified 32-bit program.bin
  -> PLENA Rust simulator

local Hugging Face Llama checkpoint
  -> torch.export capture -> official torch-mlir Torch IR
  -> fail-closed graph/state/capability certificate
  -> dense-Llama reference lowering to generic PLENA ISA
  -> Program v7 + LP6 image -> PLENA Rust simulator
```

## Current support

- Exactly one static identity-layout `FP16` `linalg.matmul`.
- Canonical `[M,K] x [K,N] -> [M,N]` row-major layout.
- A preceding zero `linalg.fill`, preserving overwrite-style GEMM semantics.
- 32x32 spatial M/N tiling with legal M/N tails.
- Configurable temporal K chunking and FP32 accumulation across K chunks.
- Contiguous N-axis partitioning across logical cores; K is never split.
- Explicit LP6 to shared-L2 GDMA and per-core L2/L1 LDMA.
- Core-relative private-L1 byte offsets and shared-L2 byte offsets.
- Program v7 `CORE_BEGIN/CORE_END`, numeric dependencies, and configurable
  logical-to-physical placement.
- Compiler bundle execution on the Rust simulator with FP16 byte-exact tests.
- Bounded Vector ISA generation for the dense decoder: explicit tiles fit the
  configured RF width; RMSNorm/softmax reductions use a scalar pairwise tree.
  The current assignment requires at least seven vector registers (default 16).
- Physically vendored PyTorch/torch-mlir graph frontend; no ETRI source import
  or frontend symlink is used at runtime.
- Full supported Llama prefill through every decoder layer, final norm, and LM
  head, with Rust SRAM/logit checking and optional Hugging Face eager comparison.

The native C++ MLIR backend still accepts one matmul and requires its complete
activation, weight, and output to fit the 8 MiB shared L2. The full-model path
is intentionally identified as a graph-certified, model-specialized reference
lowering: it emits generic Matrix/Vector/Scalar ISA, but it is not yet a
general operation-by-operation Torch/Linalg-to-PLENA pass pipeline. This
boundary is recorded in every `compilation.json` rather than hidden.

## Build

The checked environment uses the LLVM/MLIR 24 build shared with the ETRI
torch-mlir toolchain:

```bash
cmake -S . -B build -G Ninja \
  -DMLIR_DIR=/home/jongjip/etri-mlir/third_party/torch-mlir/build-llvm/lib/cmake/mlir \
  -DLLVM_DIR=/home/jongjip/etri-mlir/third_party/torch-mlir/build-llvm/lib/cmake/llvm
cmake --build build --target plena-compile plena-opt -- -j4
```

`PLENA_MLIR_BUILD` can select another compatible LLVM/MLIR 24 build when using
`test/run_all.sh`.

## Compile and execute a Hugging Face model

The model directory must be local. The default compiles every layer, final
RMSNorm, and LM head, executes the Rust simulator, and compares the next-token
logits with Hugging Face FP16 eager execution:

```bash
build/bin/plena-compile-model \
  --hf-model /home/jongjip/models/llama_3.2_1b_instruct \
  --prompt "The capital of France is" \
  --output-dir /tmp/plena-llama32-1b \
  --execute
```

The wrapper uses `PLENA_TORCH_MLIR_PYTHON` when set. On this server it otherwise
selects the pinned torch-mlir Python under `/data2/jongjip/etri-mlir`. That is a
binary Python/toolchain dependency, not a source-code link. See
[full-model E2E](docs/FULL_MODEL_E2E.md) for ownership, artifacts, results, and
current restrictions.

## Test

```bash
./test/run_all.sh
```

The regression covers:

- aligned `4x64 x 64x64` GEMM;
- tail `5x96 x 96x37` GEMM;
- multi-M/N-tile `40x33 x 33x45` GEMM;
- K=96 lowered as 64+32 temporal accumulation;
- 1-core and 2-core N-axis placement;
- round-trip parsing of every emitted MLIR level;
- Rust simulator Program v7 decoding and exact FP16 output;
- rejection of a matmul without zero initialization.
- frontend source ownership/no-symlink checks and model-driver CLI smoke test.

Reference results in the checked configuration:

| Problem | Cores | NPU cycles | Exact |
|---|---:|---:|---:|
| `4x64 x 64x64` | 1 | 1,320 | yes |
| `4x64 x 64x64` | 2 | 860 | yes |
| `5x96 x 96x37` | 1 | 1,810 | yes |
| `5x96 x 96x37` | 2 | 1,326 | yes |
| `40x33 x 33x45` | 1 | 3,517 | yes |
| `40x33 x 33x45` | 2 | 2,248 | yes |

These are simulator results for regression, not measured hardware performance.
These are Program v7 results from 2026-09-10. The simulator uses atomic masked
L2 writes, bounded LDMA write pipelining and continuous FP32 Matrix accumulation.
Old v6 binaries must be regenerated. The native MLIR path still targets FP16
matmul; typed Vector and Q4 lowering helpers are local Python backend APIs,
not a claim that every new primitive has automatic graph legalization.

## Compile manually

Input payloads are little-endian row-major FP16 files.

```bash
build/bin/plena-compile examples/fp16_matmul.mlir \
  --config configs/plena32_single_core.json \
  --activation-data /path/to/activation.f16.bin \
  --weight-data /path/to/weight_kn.f16.bin \
  --output-dir /tmp/plena-matmul
```

Run the emitted bundle:

```bash
cd /tmp/plena-matmul
/home/jongjip/LP6/PLENA_Simulator/transactional_emulator/target/release/transactional_emulator \
  --system-program system.json \
  --lp6-image lp6.bin --lp6-size 64MiB \
  --settings /home/jongjip/LP6/PLENA_Simulator/plena_settings.toml \
  --sram-timing-out timing.json --timeline-out timeline.json
```

To derive compiler-visible hardware fields from the simulator configuration:

```bash
python3 tools/import_simulator_config.py \
  ../PLENA_Simulator/plena_settings.toml \
  --logical-cores 1 --k-chunk 64 \
  -o /tmp/plena-target.json
```

## Artifacts

| Artifact | Meaning |
|---|---|
| `planned.mlir` | LP6/L2/per-core-L1 byte bindings |
| `tiled.mlir` | 32x32 M/N tiles and logical-core ownership |
| `scheduled.mlir` | GDMA/core-block event SSA |
| `commands.mlir` | Structured GDMA and `CORE_BEGIN/END` records |
| `lowered.mlir` | Immutable `plena_isa.program` |
| `program.bin` | Simulator-executable unified Program v7 |
| `system.json` | Placement, L2 regions, command debug symbols |
| `model_manifest.json` | Tensor shapes, LP6/L2 byte locations, output ABI |
| `lp6.bin` | Activation and weight payload image |
| `compile_report.json` | Tiling, memory, command, and word counts |

Full-model bundles additionally contain `frontend/`, `semantic_certificate.json`,
`model_spec.json`, `capability_report.json`, `target/metadata.json`,
`target/expected.pt`, and—when executed—`target/timing.json`,
`target/sram_dump.bin`, `target/check_result.json`, `execution.json`, and
`hf_comparison.json`.

See [compiler architecture](docs/COMPILER_ARCHITECTURE.md) and
[implementation roadmap](docs/ROADMAP.md).
