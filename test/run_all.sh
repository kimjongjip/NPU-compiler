#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

COMPILER_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
LP6_ROOT=$(cd -- "${COMPILER_ROOT}/.." && pwd)
MLIR_BUILD=${PLENA_MLIR_BUILD:-/home/jongjip/etri-mlir/third_party/torch-mlir/build-llvm}
SIMULATOR_ROOT=${PLENA_SIMULATOR_ROOT:-${LP6_ROOT}/PLENA_Simulator}
NORMALIZED_CONFIG=$(mktemp /tmp/plena-compiler-config.XXXXXX.json)
trap 'rm -f -- "${NORMALIZED_CONFIG}"' EXIT

cmake -S "${COMPILER_ROOT}" -B "${COMPILER_ROOT}/build" -G Ninja \
  -DMLIR_DIR="${MLIR_BUILD}/lib/cmake/mlir" \
  -DLLVM_DIR="${MLIR_BUILD}/lib/cmake/llvm"
cmake --build "${COMPILER_ROOT}/build" --target plena-compile plena-opt -- -j4

if [[ ! -x "${SIMULATOR_ROOT}/transactional_emulator/target/release/transactional_emulator" ]]; then
  cargo build --release --manifest-path \
    "${SIMULATOR_ROOT}/transactional_emulator/Cargo.toml"
fi

python3 "${COMPILER_ROOT}/tools/import_simulator_config.py" \
  "${SIMULATOR_ROOT}/plena_settings.toml" \
  --logical-cores 1 --k-chunk 64 -o "${NORMALIZED_CONFIG}"
cmp "${COMPILER_ROOT}/configs/plena32_single_core.json" \
  "${NORMALIZED_CONFIG}"
python3 "${COMPILER_ROOT}/test/run_matmul_e2e.py"

if find "${COMPILER_ROOT}/tools/frontend" "${COMPILER_ROOT}/tools/full_model" \
    -type l -print -quit | grep -q .; then
  echo "vendored model frontend/backend must not contain symbolic links" >&2
  exit 1
fi
if rg -n '/data2/jongjip/etri-mlir/compiler/(tools|examples)' \
    "${COMPILER_ROOT}/tools/frontend" "${COMPILER_ROOT}/tools/full_model"; then
  echo "vendored model sources must not import source from the ETRI tree" >&2
  exit 1
fi
"${COMPILER_ROOT}/build/bin/plena-compile-model" --help >/dev/null
