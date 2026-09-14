# 다른 서버에서 재개하기: C++ MLIR 컴파일러 전환 인수인계

작성일: 2026-09-14. **이 문서를 먼저 읽고 작업을 재개한다.**

## 1. 사용자가 원하는 것과 현재 상태

목표는 기능 검증용 Python 코드 생성기가 아니라, **메모리 플래닝·타일링·멀티코어
스케줄링 연구를 할 수 있는 C++ MLIR 컴파일러**다.

```text
Python의 책임
  HF 모델 로딩 / 실제 forward의 ATen 캡처 / 공식 torch-MLIR 호출
  Linalg IR, 입력·weight 데이터, 하드웨어 설정을 C++ 드라이버에 전달

C++의 책임 — 앞으로 구현할 최종 경로
  공식 canonicalize/CSE 등
  → Linalg legalization
  → 메모리 배치·검증
  → SA/VPU 타일링·검증
  → 코어·이벤트 스케줄링·검증
  → DMA/연산 명령 생성
  → Program v7 인코딩
```

**C++ 전환은 완료되지 않았다.** 현재 실행 가능한 기본 경로는 여전히 Python MLIR
패스 + C++ 최종 인코더다. C++ 초안은 빌드에 연결하지 않았고, 컴파일하지도 않았다.
디스크 부족으로 전환 작업이 중단되었으며, 이번 커밋은 이관용 보존 커밋이다.
공간이 다시 확보되더라도 이번 서버에서 개발을 계속하지 말고, 새 서버에서 재개한다.

| 구분 | 실제 상태 |
|---|---|
| 기존 단일 matmul C++ 패스 | 구현·회귀 검증됨 |
| 전체 그래프 Python MLIR 경로 | 작은 전체 HF 모델과 일반 연산에서 실행 검증됨 |
| Linalg 이후 전체 C++ 경로 | **초안 작성 중, 실행 불가** |
| C++ 메모리 플래너·타일러·스케줄러 | **전체 그래프용은 아직 미구현** |
| 모델별 Python reference generator | 명시적 `--backend reference` 선택용으로 남음 |
| 대형 학습 모델의 새 그래프 경로 | 이전 1B 체크포인트가 없어 검증하지 못함 |

이전 중요한 커밋:

- `792a462`: Python MLIR 전체 그래프 경로 추가.
- `4a4c3d5`: 그 상태의 파일 정리. C++ 전환 직전 기준점.
- 이 문서가 들어간 이관 커밋: 위 기준점 + **빌드 제외 C++ WIP 3개 파일** + 인수인계 문서.

## 2. 가져갈 저장소와 디렉터리 배치

```text
<작업공간>/                       예: 새 서버의 충분한 여유 공간이 있는 LP6 디렉터리
├── PLENA_Compiler/              github.com/kimjongjip/NPU-compiler, main
├── PLENA_Simulator/             github.com/buko9911/PLENA_Simulator
├── LP6-PIM-Simualator/          github.com/kimjongjip/lp6-pim-simulator
├── third_party/torch-mlir/      LLVM/MLIR 소스·빌드용, 별도 준비
├── toolchain/                  Python 환경, 별도 준비
├── models/                     체크포인트, 별도 준비
└── tmp/                        설치·검증·빌드 임시 파일
```

`LP6-PIM-Simualator` 철자는 현재 경로의 실제 철자다. 이름을 바꿀 경우
`LP6_DRAMSIM3_ROOT`를 명시한다. Compiler 테스트 중 일부는 simulator를 형제
디렉터리 `../PLENA_Simulator`에서 찾으므로 위 배치를 권장한다.

시뮬레이터 기준 커밋:
`9f4cdf7908d98e34fa5c33a7396d87d84e762e63`
(`feat/npu-simulator-v2` 브랜치에서 사용했던 버전).

LP6-PIM의 로컬 HEAD는 `ed18d9d4cfbd57bdbb60e94f7978f17306b673a0`이지만,
**그 저장소에는 별도 연구의 미커밋 변경들이 있다. 이번 compiler push에는 넣지 않았다.**
따라서 그 HEAD를 clone했다고 기존 로컬 DRAM backend와 완전히 같다고 단정하면 안 된다.
필요한 LP6 변경의 백업·커밋 여부를 별도로 확인하고, 새 서버에서 backend smoke test를 한다.
다른 프로젝트의 변경을 임의로 이 compiler 저장소에 섞어 올리지 않는다.

Git에 넣지 않은 것: LLVM 빌드, Python venv, 모델 weight, Rust/C++ 바이너리,
LP6 image, SRAM dump, 임시 테스트 결과, Black 설치 디렉터리.
이 파일들은 새 서버에서 재생성하거나 별도로 전송해야 한다.

## 3. 먼저 읽을 코드와 문서

| 목적 | 파일 |
|---|---|
| 현재 전체 실행 흐름 | `tools/graph_pipeline/pipeline.py` |
| 현재 Linalg 해석·constant/view 처리 | `tools/graph_pipeline/graph.py` |
| 현재 메모리·타일·이벤트·ISA 구현 | `tools/graph_pipeline/lower.py` |
| 실제 HF 입력·state binding | `tools/graph_pipeline/huggingface.py` |
| HF CLI orchestration | `tools/graph_pipeline/model_driver.py` |
| 현재 CLI/backend 선택 | `tools/plena-compile-model/plena_compile_model.py` |
| 기존 native PassManager 패턴 | `tools/plena-compile/plena-compile.cpp` |
| 기존 C++ 패스·명령 인코더 연결 | `lib/Transforms/MatmulPipeline.cpp` |
| ISA 인코딩·이벤트 ABI | `lib/Target/ProgramV7.cpp`, `include/PLENA/Target/ProgramV7.h` |
| 하드웨어 설정 import | `tools/import_simulator_config.py`, `configs/` |
| 현재 그래프 경로의 검증·제약 | `docs/GRAPH_COMPILER.md` |

참고했던 ETRI 트리는 기존 서버 `/data2/jongjip/etri-mlir`에 있다.
거기서는 Python semantic bridge 이후 메모리·스케줄·명령·인코딩을 C++ PassManager로
처리한다. 단, 최초 Torch IR부터 전부 일반적인 C++ 연산 lowering을 하는 것은 아니다.
이 프로젝트는 **ETRI의 패스 구성 방식은 참고하되 모델별 semantic template에 다시
의존하지 않고, 실제 Linalg 연산·SSA를 소비하는 C++ 경로**로 가야 한다.
새 서버에서 ETRI 소스 경로가 존재한다고 가정하지 않는다.

## 4. 중단 당시 저장된 C++ 초안

| 파일 | 들어 있는 내용 | 주의 |
|---|---|---|
| `include/PLENA/Transforms/NativeGraph.h` | 예정 pass factory와 image emission API 선언 | 구현과 등록 없음 |
| `lib/Transforms/NativeGraphInternal.h` | Tensor/Buffer/Expr/Kernel/Tile/Graph/Hardware 초안 | 안정화된 IR API 아님 |
| `lib/Transforms/NativeGraphLegalize.cpp` | Linalg/Tensor/Arith/Math를 읽는 C++ legalizer 초안 | 빌드·기능 검증 전 |

**저장되지 않은 것:** `NativeGraphIR.cpp`, native graph v2 TableGen op 변경,
메모리 플래너, 타일러, 스케줄러, native command lowering, image writer,
C++ driver, Passes.td 등록, CMake 연결. 디스크 오류로 해당 후속 패치는 적용되지 않았다.
파일이 있을 것이라고 가정하지 말고 `git ls-files`로 확인한다.

초안 검토 때 즉시 확인할 항목:

- LLVM/MLIR 24 API와 `getNumDpsInputs`, indexing map, region 접근 등의 실제 호환성.
- `llvm::APFloat f(float(v));`는 C++ most-vexing-parse 문제가 될 수 있다. 초기화 문법 확인.
- 초안의 `check()`는 `throw Error`를 쓴다. LLVM 빌드의 예외 비활성화 설정과 충돌할 수 있다.
  가능하면 `LogicalResult` / `FailureOr` / `llvm::Expected`로 바꾸고 `emitError` +
  `signalPassFailure`로 전파한다. 이를 해결하지 않고 CMake에 추가하지 않는다.
- compile-time 수치를 `double`로 보관한 초안은 일반적인 i64 모델이 아니다.
  실제 구현에서는 APInt/APFloat 또는 명시적 범위 제한을 검토한다.
- element count, byte size, offset×stride의 overflow 및 모든 view의 범위 검사.
- constant folding은 명시적 static 입력·buffer·상수에만 적용한다. weight와 activation의
  런타임 계산을 host에서 대신 수행해서 실행 결과를 만들면 안 된다.
- C++ 자료구조에 연산을 읽었다고 MLIR 패스가 완성된 것이 아니다. 패스 사이의 실제 IR,
  verifier, pass registration과 독립적인 실행 테스트가 필요하다.

## 5. 새 서버에서 환경 복구

### 버전 기록

관측한 환경은 아래와 같다. 다른 버전으로 자동 교체하지 말고 호환성을 확인한다.

| 구성 | 관측 버전 |
|---|---|
| C++ | C++17, CMake 프로젝트가 LLVM/MLIR **24.0.0** 요구 |
| LLVM 소스 | `6d1ace547c9f9ecf7df9ab87e2db74b5bd3f199d` |
| torch-mlir 소스 | `6d30f7d47a245425cb6aec7d943d94431a06ae04` |
| Python frontend | Python 3.10.18 |
| torch | `2.14.0.dev20260719+cpu` |
| torch-mlir wheel | `20260813.844` |
| transformers | `5.0.0rc1` |
| numpy / safetensors / tokenizers | `2.2.6` / `0.8.0` / `0.22.2` |
| TOML config helper | 별도 Python 3.11 이상, `tomllib` 사용 |
| simulator | Rust edition 2024를 지원하는 toolchain + C++ LP6 DRAMSim3 |

wheel URL·SHA256와 submodule pin은
[`handoff/toolchain-pins.env`](handoff/toolchain-pins.env)에 보존했다.
이 URL들의 새 서버 다운로드 가능 여부는 이관 작업에서 재검증하지 않았다.
오래된 nightly wheel이 사라졌다면 기존 wheel cache를 별도 전송하거나 동일 revision으로
toolchain을 재구축해야 한다. venv 디렉터리만 그대로 옮기면 shebang/절대경로가 깨질 수 있다.

### 경로 설정과 clone

아래는 새 서버에서 수행할 예시다. `NPU_WORKSPACE`는 실제 사용 가능한 경로로 바꾼다.
**모든 설치·빌드·cache·tmp는 이 작업공간 안에 둔다.**

```bash
export NPU_WORKSPACE=/path/to/LP6
mkdir -p "$NPU_WORKSPACE/tmp" "$NPU_WORKSPACE/toolchain" "$NPU_WORKSPACE/third_party"
export TMPDIR="$NPU_WORKSPACE/tmp"
export XDG_CACHE_HOME="$NPU_WORKSPACE/tmp/cache"
export PIP_CACHE_DIR="$NPU_WORKSPACE/tmp/pip-cache"
export CARGO_HOME="$NPU_WORKSPACE/toolchain/cargo"
export RUSTUP_HOME="$NPU_WORKSPACE/toolchain/rustup"
export PYTHONDONTWRITEBYTECODE=1
cd "$NPU_WORKSPACE"
git clone https://github.com/kimjongjip/NPU-compiler.git PLENA_Compiler
git clone --branch feat/npu-simulator-v2 https://github.com/buko9911/PLENA_Simulator.git PLENA_Simulator
git clone https://github.com/kimjongjip/lp6-pim-simulator.git LP6-PIM-Simualator
```

기존 baseline과 비교할 때는 simulator 커밋도 위 기록과 일치시키거나 차이를 기록한다.
LP6 작업트리의 미커밋 변경 문제는 앞 절의 주의를 따른다.

### Toolchain 준비 방향

```bash
source "$NPU_WORKSPACE/PLENA_Compiler/docs/handoff/toolchain-pins.env"
git clone https://github.com/llvm/torch-mlir.git "$NPU_WORKSPACE/third_party/torch-mlir"
git -C "$NPU_WORKSPACE/third_party/torch-mlir" checkout "$PLENA_TM_SOURCE_COMMIT"
git -C "$NPU_WORKSPACE/third_party/torch-mlir" submodule update --init --recursive

python3.10 -m venv "$NPU_WORKSPACE/toolchain/torch-mlir-python"
export PLENA_TORCH_MLIR_PYTHON="$NPU_WORKSPACE/toolchain/torch-mlir-python/bin/python"
```

pin 파일의 torch/torch-mlir/ml_dtypes wheel을 작업공간 내부로 내려받아 SHA256을
검증한 뒤 위 Python에 설치한다. numpy/transformers/tokenizers/safetensors도 pin에 맞춘다.
wheel은 Linux x86_64/Python 3.10용이므로 다른 아키텍처에서는 그대로 사용할 수 없다.

LLVM은 같은 torch-mlir checkout의 `externals/llvm-project/llvm`로 빌드한다.
기존 환경의 주요 CMake 옵션은 다음과 같다.

```bash
export PLENA_MLIR_BUILD="$NPU_WORKSPACE/third_party/torch-mlir/build-llvm"
cmake -S "$NPU_WORKSPACE/third_party/torch-mlir/externals/llvm-project/llvm" \
  -B "$PLENA_MLIR_BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_TARGETS_TO_BUILD=Native -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_ENABLE_RTTI=OFF -DBUILD_SHARED_LIBS=OFF \
  -DLLVM_INCLUDE_TESTS=OFF -DLLVM_BUILD_EXAMPLES=OFF \
  -DLLVM_ENABLE_ZLIB=OFF -DLLVM_ENABLE_ZSTD=OFF
```

필요한 MLIR libraries와 `mlir-tblgen`을 빌드해야 한다. 기존 toolchain bootstrap은
Arith/Func/Linalg/Tensor/Math/MemRef/ControlFlow/SCF/Shape/Bufferization/TOSA의
dialect·transform 라이브러리와 `MLIROptLib`, `MLIRTransforms`, `llvm-config`,
`mlir-tblgen`을 선택적으로 빌드했다. 전체 LLVM 빌드는 디스크를 많이 쓰므로 새 서버의
공간과 병렬도를 확인한다. 이 문서는 환경 기록이며 새 standalone bootstrap의 검증을
대신하지 않는다.

### Compiler와 simulator 빌드

```bash
export PLENA_CONFIG_PYTHON=/path/to/python3.11-or-newer
export LP6_DRAMSIM3_ROOT="$NPU_WORKSPACE/LP6-PIM-Simualator"

# LP6 README에 맞춰 CMake로 libdramsim3.so를 먼저 빌드한다.
cmake -S "$LP6_DRAMSIM3_ROOT" -B "$LP6_DRAMSIM3_ROOT/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$LP6_DRAMSIM3_ROOT/build" --target dramsim3 -- -j4

cargo build --release --manifest-path "$NPU_WORKSPACE/PLENA_Simulator/transactional_emulator/Cargo.toml"

cmake -S "$NPU_WORKSPACE/PLENA_Compiler" -B "$NPU_WORKSPACE/PLENA_Compiler/build" -G Ninja \
  -DMLIR_DIR="$PLENA_MLIR_BUILD/lib/cmake/mlir" \
  -DLLVM_DIR="$PLENA_MLIR_BUILD/lib/cmake/llvm"
cmake --build "$NPU_WORKSPACE/PLENA_Compiler/build" --target plena-compile plena-opt -- -j4
```

`LP6_DRAMSIM3_ROOT`는 `libdramsim3.so`, 필요한 C API와 config를 포함한 **backend root**다.
Rust linker에는 해당 library 경로의 rpath가 들어가므로 옛 서버 바이너리를 그대로 쓰지 말고
새 경로에서 다시 빌드한다. Compiler clone만으로 이 라이브러리가 제공되지는 않는다.

현재 wrapper에는 옛 서버 Python 경로 fallback이 남아 있으므로
`PLENA_TORCH_MLIR_PYTHON`을 반드시 설정한다. `test/run_all.sh`도 `PLENA_MLIR_BUILD`를
설정하지 않으면 옛 서버 LLVM 경로를 사용한다. 테스트 스크립트의 `python3`는 3.11 이상이
되어야 한다. 단순히 새 wheel만 설치하고 C++ MLIR을 다른 major로 쓰면 안 된다.

## 6. 전환 전에 재현할 검증 기준

```bash
cd "$NPU_WORKSPACE/PLENA_Compiler"
PLENA_TEST_GRAPH=1 bash test/run_all.sh
"$PLENA_TORCH_MLIR_PYTHON" test/run_graph_e2e.py --cores 1
```

- 기존 native matmul 6개: 1/2-core, M/N/K tail, FP16 byte-exact 유지.
- 전체 2-layer HF Llama + final norm + LM head: 1/2-core, argmax 일치,
  관측된 최대 logits 절대 오차 `0.000244140625`.
- Non-Llama linear+bias+ReLU+broadcast, L2 spill 2개, FP32 reduction 960.
- imported MLIR `ADD→SUB` 변경 시 실제 결과 변경.
- unsupported op, dtype mismatch, output overwrite, resource limits,
  runtime assertion과 output indexing 거부 테스트 7개.
- `run_hf_graph_cli.py`: 작은 **전체** HF 체크포인트와 tokenizer를 생성해 public CLI 실행.

작은 모델은 무작위 weight다. 언어 품질을 검증한 것이 아니고, 이전 학습된 1B 모델이
이 새 경로에서 검증되었다고 주장하면 안 된다. 수치 비교도 reduction 재결합·RCP+MUL에
대한 tolerance 비교이지 모든 입력에서 IEEE bit-exact임을 보장하는 것이 아니다.

## 7. 앞으로의 구현 순서와 완료 조건

### A. C++ pass 기반부터 연결

1. WIP를 먼저 검토하고 컴파일 오류·예외 정책을 해결한다.
2. `Passes.td`, pass factory, `Passes.cpp`, dependent dialect, CMake target을 연결한다.
3. Python은 입력 데이터와 **공식 Linalg IR**까지만 내보내게 한다.
4. C++ driver가 `PassManager`를 구성하고 각 단계 IR을 저장하게 한다.
5. full-graph C++ backend를 선택할 때 Python legalizer/planner/ISA emitter가 호출되지
   않는 테스트를 추가한다. 중간에 Python에 계획을 위탁하는 wrapper만 만들면 안 된다.

### B. IR과 official pass 사용 경계

- 가능한 동안 표준 Tensor/Linalg/Arith/Math/Func의 shape·SSA·indexing map을 유지한다.
- 공식 `canonicalize`, `CSE`, 필요한 shape/layout 정리 패스와 Linalg 인터페이스를 사용한다.
- 일반적인 Linalg fusion/tiling과 bufferization은 numeric/layout/메모리 계약을 확인한 뒤
  도입한다. **공식 pass를 무조건 나열하는 것 자체가 목표가 아니다.**
- 일반 bufferization은 tensor→buffer 변환이지 LP6/L2/L1의 물리 주소 배치 알고리즘이
  아니다. 후자는 타깃 전용 planner와 verifier가 필요하다.
- 현 Python graph IR의 JSON descriptor를 단순히 C++에서 다시 파싱하는 방식은 피한다.
  typed attributes, buffer SSA operands, 적절한 expression region·interfaces를 설계한다.
- 중단 직전에는 `native_buffer`, `native_kernel`, `native_tile`, `native_return`과
  typed attributes/SSA handle을 구상했지만 **정의는 저장되지 않았다**. 이 이름과
  C++ 내부 자료구조는 초안이지 확정 ABI가 아니다.

### C. 메모리 플래닝 — 최우선 연구 기반

- 논리 tensor/view와 실제 allocation을 분리하고 alias 수명까지 계산한다.
- L2 용량·정렬·범위·live-range overlap을 검증한다.
- weight는 LP6에 두고 tile만 staging. parameter layout packing은 명시적으로 기록한다.
- L2 부족 시 명시적인 spill/reload와 GDMA/LDMA 경로를 생성한다.
- shared L2와 core-local L1의 주소 공간을 구분한다. virtual/logical core placement도 기록한다.
- 선택한 배치의 이유, high-water, spill bytes, DMA bytes를 report로 내보낸다.
- 큰 tensor 전체에 proportional한 host 복사를 피하고 weight image도 streaming 방식으로 쓴다.
- 이후 bank mapping, stride, bandwidth를 고려하는 비용 모델로 확장한다.

### D. Tiling·스케줄링·instruction selection

- SA의 M/N 공간 타일과 temporal K chunk를 구분한다. M/N tail을 지원한다.
- Output Stationary이므로 K가 SA 한 변보다 길어도 된다. accumulator 유지·writeout 계약 준수.
- VPU는 physical lane 수와 architectural RF 용량을 구분한다. 모든 값이 RF에 들어맞는지
  확인하고 register allocation/spill 정책을 명시한다.
- 먼저 N/output 축 분배를 유지하고 split-K는 하지 않는다.
- Matrix/VPU/DMA의 데이터·자원 의존성, overwrite 위험을 검증한 뒤 overlap을 늘린다.
- 소규모 테스트마다 core block을 과도하게 만드는 현재 방식을 개선하고 command batching,
  finite event slot 재사용을 설계한다. scoreboard 용량을 몰래 키우지 않는다.
- 기존 Program v7 ABI는 가능한 유지한다. 이 전환 때문에 simulator ISA를 불필요하게 바꾸지 않는다.
- fused Attention/RMSNorm/SiLU ISA를 다시 만들지 않는다. 실제 primitive 연산으로 낮춘다.

### E. 검증 후 기본값 전환

1. 동일 Linalg IR과 입력으로 Python baseline / native C++ 결과를 비교한다.
2. C++ 각 pass를 개별 `plena-opt` pipeline으로 실행·재파싱·검증한다.
3. 기존 1/2-core·tail·spill·negative tests와 작은 전체 HF CLI를 모두 통과시킨다.
4. 충분한 공간에서 실제 학습된 모델의 **전체 레이어 + LM head**를 실행한다.
5. C++가 검증된 다음에만 기본 backend를 바꾼다. 이전 Python 경로는 필요하면 명시적
   테스트용으로 이동하고 production 경로에서 분리한다.
6. 큰 변경은 commit/push하고, 구현 완료와 계획 항목을 문서에서 구분한다.

그 후의 최적화: fusion, cross-op RF reuse, asynchronous double buffering,
cost-based multicore placement, persistent KV-cache decode, INT8/INT4 graph lowering.
이 항목들을 이번 C++ 전환이 이미 해결했다고 쓰면 안 된다.

## 8. 하드웨어 사양과 컴파일러 반영 수준

사양의 source of truth는 simulator의 `plena_settings.toml`이며, 아래는 위 기준 커밋의
스냅샷이다. 상용 NPU의 실측 수치가 아니라 이 연구 simulator의 configurable baseline이다.

| 항목 | 현재 기본값 | 현재 compiler가 사용하는 수준 |
|---|---|---|
| Core | 1개, configurable homogeneous cores | 개수·logical/physical placement |
| SA/core | 32×32, Output Stationary | M/N tile, tail 검사 |
| Clock | 800 MHz | 현 planner의 비용 선택에 사용하지 않음 |
| FP16 Matrix | FP16×FP16→FP32, PE당 1 MAC/cycle 가정 | dtype/accumulator 계약; timing은 simulator |
| INT8 Matrix | INT8×INT8→INT32, PE당 2 MAC/cycle·pipeline 2 가정 | 새 graph compiler의 자동 INT8 lowering은 미지원 |
| VPU/core | 16 physical lanes ×32 bits | lane 정보 기반 비용 모델은 아직 없음 |
| VPU issue width | FP16 32, FP32 16, INT8 64 elements/cycle 최대 가정 | RF 용량과 별개이며 현재 비용 최적화 미적용 |
| Vector RF/core | 16×512 bits = **1 KiB**, 2R/1W | register count/width로 capacity 검사·분할 |
| L1/core | 4 MiB, 32 banks, 32-bit word, bank당 1R/1W, latency 1 | capacity는 사용; bank-aware allocator 미구현 |
| Shared L2 | 8 MiB, 64 banks, 2 slices, 32-bit word, bank당 1R/1W, latency 4 | capacity·alignment·lifetime·spill |
| L1 ideal read/write | 각각 128 B/cycle, 800 MHz에서 102.4 GB/s | conflict-free 산술 상한; 실제 bandwidth 아님 |
| L2 ideal read/write | 각각 256 B/cycle, 800 MHz에서 204.8 GB/s | NoC/ports/layout 등 다른 bottleneck 고려 필요 |
| NoC | simple crossbar, 800 MHz, 32 B/cycle, latency 2, queue 16 | 비용 모델 미적용, simulator에서 timing |
| External DRAM | LPDDR6 16 GiB, 2 subchannel, 2 rank | image/주소/명시적 DMA; DRAMSim3가 timing |
| Command processor | 4 words/cycle fetch, issue 1, core FIFO 4, event slots 65536 | core/event ABI 및 finite event 용량 |
| ISA | Program v7, 물리 word 32 bits | Matrix/typed Vector는 multiword 명령 |
| K chunk | 기본 64 | **compiler 정책**; SA 물리 크기가 아님 |
| Compiler L2 staging | 기본 64 KiB/logical core | **compiler 예약 정책**; 별도 하드웨어 SRAM이 아님 |

새 C++ driver는 사용한 전체 hardware snapshot과 config hash를 bundle에 남기고,
각 필드가 legality/배치/cost 중 어디에 쓰이는지 문서화해야 한다.
현재 importer는 architecture·memory size·K chunk·event·ISA를 정규화하고 VPU는
별도 읽는다. bank/port/NoC 정보를 읽었다는 이유만으로 bandwidth-aware planner라고
주장하지 않는다.

## 9. 바로 다음 작업자에게 전달할 요청문

> docs/RESUME_CPP_COMPILER.md를 먼저 읽어라. 기존 Python MLIR 경로는 회귀 기준으로
> 보존하되, HF→ATen→공식 Linalg IR 이후는 실제 C++ MLIR PassManager 경로로 전환해라.
> WIP C++ 파일 3개는 미컴파일 초안이며 CMake에 연결되어 있지 않다. 공식 MLIR 패스와
> Linalg 인터페이스를 활용하고, 타깃 메모리 플래너·타일러·스케줄러·instruction selection을
> C++로 구현해라. 큰 모델을 한 레이어만 실행하고 전체 완료라고 보고하지 마라.
> 하드웨어 사양 및 실제 반영 범위를 기록하고, 먼저 기존 작은 전체 모델/negative tests를
> 재현해라. 모든 새 파일·toolchain·cache·임시 산출물은 이 프로젝트 작업공간 내부에 둬라.
> 다른 저장소의 미커밋 변경을 임의로 수정·업로드하지 마라. 큰 변경 후 commit/push해라.
