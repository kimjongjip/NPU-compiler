# PLENA MLIR Compiler

Standalone LLVM/MLIR compiler targeting the current PLENA NPU simulator.
The project reuses the proven multi-level organization of the ETRI compiler,
but its memory hierarchy, tiling, scheduling, command IR, and binary encoder
are native to the PLENA Program v5 ABI.

The first executable vertical slice is complete:

```text
static FP16 linalg.fill + linalg.matmul
  -> byte-addressed LP6/L2/L1 plan
  -> 32x32 output-stationary M/N tiles
  -> N-axis logical-core assignment
  -> GDMA + event-SSA core schedule
  -> structured command records
  -> unified 32-bit program.bin
  -> PLENA Rust simulator
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
- Program v5 `CORE_BEGIN/CORE_END`, numeric dependencies, and configurable
  logical-to-physical placement.
- Compiler bundle execution on the Rust simulator with FP16 byte-exact tests.

The initial planner requires the complete activation, weight, and output to fit
the 8 MiB shared L2. Streaming L2 windows, compiler-generated double buffering,
generic VPU lowering, attention, and full Hugging Face models are subsequent
passes, not silently approximated features. The current output ABI leaves the
completed tensor in shared L2; an explicit final GDMA store/runtime handoff is
part of the full-model memory protocol.

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
- Rust simulator Program v5 decoding and exact FP16 output;
- rejection of a matmul without zero initialization.

Reference results in the checked configuration:

| Problem | Cores | NPU cycles | Exact |
|---|---:|---:|---:|
| `4x64 x 64x64` | 1 | 1,443 | yes |
| `4x64 x 64x64` | 2 | 923 | yes |
| `5x96 x 96x37` | 1 | 2,035 | yes |
| `5x96 x 96x37` | 2 | 1,464 | yes |
| `40x33 x 33x45` | 1 | 5,051 | yes |
| `40x33 x 33x45` | 2 | 3,149 | yes |

These are simulator results for regression, not measured hardware performance.

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
| `program.bin` | Simulator-executable unified Program v5 |
| `system.json` | Placement, L2 regions, command debug symbols |
| `model_manifest.json` | Tensor shapes, LP6/L2 byte locations, output ABI |
| `lp6.bin` | Activation and weight payload image |
| `compile_report.json` | Tiling, memory, command, and word counts |

See [compiler architecture](docs/COMPILER_ARCHITECTURE.md) and
[implementation roadmap](docs/ROADMAP.md).
