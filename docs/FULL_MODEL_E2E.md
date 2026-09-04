# Full Hugging Face model path

## What is implemented

`plena-compile-model` performs one end-to-end static prefill run:

```text
local Hugging Face checkpoint
  -> actual model.forward captured by torch.export
  -> official torch-mlir FX import (Torch dialect MLIR)
  -> graph/operator/parameter ABI/capability certificate
  -> checkpoint packing and generic PLENA ISA scheduling
  -> unified Program v5 + LP6 image
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
performance. Host wall time was about 30 minutes because the Rust functional
emulator computes the tensor values in addition to analytical/transactional
timing.

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
