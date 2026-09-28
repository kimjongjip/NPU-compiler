# PLENA MLIR Compiler

**새 서버에서 이어서 작업할 때:** [C++ 전환 인수인계 문서](docs/RESUME_CPP_COMPILER.md)를
먼저 읽으세요. C++ 전체 그래프 전환은 **진행 중**이며, 추가된 `NativeGraph*` 3개 파일은
미컴파일·빌드 제외 초안입니다. 현재 기본 실행 경로는 아래의 Python MLIR baseline입니다.

Standalone LLVM/MLIR compiler targeting the current PLENA NPU simulator.
The project reuses the proven multi-level organization of the ETRI compiler,
but its memory hierarchy, tiling, scheduling, command IR, and binary encoder
are native to the PLENA Program v7 ABI.

The default full-model compiler now uses the imported MLIR operation graph.
See [the graph compiler](docs/GRAPH_COMPILER.md) for pass implementations,
validation and limitations. The small native matmul driver remains available.

Two entry paths are available:

```text
static FP16 linalg.fill + linalg.matmul
  -> byte-addressed LP6/L2/L1 plan
  -> 32x32 output-stationary M/N tiles
  -> N-axis logical-core assignment
  -> GDMA + event-SSA core schedule
  -> structured command records
  -> unified 32-bit program.bin
  -> PLENA Rust simulator

local Hugging Face checkpoint (default --backend mlir)
  -> torch.export capture -> official Torch -> Linalg MLIR
  -> graph legalization -> lifetime-aware LP6/L2/L1 planning
  -> SA/VPU tiling -> logical-core/event scheduling
  -> command MLIR -> native C++ Program v7 encoding
  -> Program v7 + LP6 image -> PLENA Rust simulator
```

## Native single-matmul support

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

The small `plena-compile` C++ driver retains its one-matmul/L2-fit restriction.
The default `plena-compile-model` instead uses operation-driven graph passes,
streams weights, reuses L2 by lifetime and spills intermediates when necessary.
Graph transformations use Python MLIR bindings and registered `plena_graph`
operations; final command encoding is a native C++ MLIR pass. Not all graph
passes are native C++ rewrite patterns. The old model-specialized backend is
available only with `--backend reference`; unsupported graphs never fall back.

New-path validation covers a complete **small, random 2-layer HF Llama model**
with LM head on 1/2 cores, generic non-Llama operations, and explicit L2 spills.
The old trained 1B checkpoint has **not** been rerun through the new graph path.
This is a functional baseline, not yet an arbitrary-model optimizing compiler.

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

The model directory must be local. The default compiles the entire forward,
including all layers and the LM head. `--execute` runs the simulator and checks
all returned logits against HF FP16 eager execution with explicit tolerances:

```bash
build/bin/plena-compile-model \
  --hf-model /home/jongjip/models/llama_3.2_1b_instruct \
  --prompt "The capital of France is" \
  --output-dir /home/jongjip/LP6/tmp/plena-llama32-1b \
  --execute
```

The wrapper uses `PLENA_TORCH_MLIR_PYTHON` when set. On this server it otherwise
selects the pinned torch-mlir Python under `/data2/jongjip/etri-mlir`. That is a
binary Python/toolchain dependency, not a source-code link. See
[graph compiler](docs/GRAPH_COMPILER.md) for current behavior. The older
[full-model E2E](docs/FULL_MODEL_E2E.md) records reference-backend history.
New temporary files use the project-local `../tmp` directory.

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
- Rust simulator ISA ver 1.0 decoding and exact FP16 output;
- rejection of a matmul without zero initialization.
- frontend source ownership/no-symlink checks and model-driver CLI smoke test.

Reference results in the checked configuration:

| Problem | Cores | NPU cycles | Exact |
|---|---:|---:|---:|
| `4x64 x 64x64` | 1 | 1,414 | yes |
| `4x64 x 64x64` | 2 | 953 | yes |
| `5x96 x 96x37` | 1 | 1,914 | yes |
| `5x96 x 96x37` | 2 | 1,429 | yes |
| `40x33 x 33x45` | 1 | 3,693 | yes |
| `40x33 x 33x45` | 2 | 2,363 | yes |

These are simulator results for regression, not measured hardware performance.
These are PLENA ISA ver 1.0 results from 2026-09-28: `M_MMA` is one word per
fixed 32x32x32 tile, so each K chunk emits `ceil(K/32)` MMAs after its W/A
loads. Against the same simulator at Program v7 the cycles were 1,414 / 954 /
1,912 / 1,428 / 3,569 / 2,300; K=33 pays one padded 32-wide slice per tile. The simulator uses atomic masked
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
