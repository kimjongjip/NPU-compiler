# PLENA compiler roadmap

## 2026-09-14 graph-pipeline update

The default full-model path now consumes official Linalg IR operation-by-operation
through graph legalization, lifetime-aware memory planning, SA/VPU tiling,
multicore event scheduling, command lowering and native Program v7 encoding.
Weights stream from LP6; intermediate L2 overflow emits explicit spills/reloads.
See [GRAPH_COMPILER.md](GRAPH_COMPILER.md) for validated scope and remaining work.

Graph transformations are Python MLIR passes, not all native C++ patterns.
Outstanding priorities: larger-model command/event scalability, richer graph IR,
fusion/prefetch, output staging, broader numerical validation and stateful decode.
The old 1B reference-backend results are not validation of this new graph path.

The earlier milestones below are retained as historical context. Items about
connecting a full imported graph and basic L2 lifetime/spill planning have now
been implemented in the new baseline, but their optimization/native-C++ portions
remain open.

## Implemented baseline

- Standalone LLVM/MLIR 24 project and `plena-opt` driver.
- Memory, Tile, Schedule, Command, and ISA dialects.
- Static FP16 `linalg.matmul` verification.
- Byte-addressed LP6/shared-L2/private-L1 planning.
- 32x32 M/N tiling, M/N tails, temporal K chunk accumulation.
- Contiguous N-axis logical-core distribution without split-K.
- Explicit GDMA and per-core LDMA insertion.
- Event-SSA schedule lowered to numeric Program v7 completion events.
- Native C++ core ISA and unified Program v7 encoders.
- Simulator bundle writer and 1/2-core FP16-exact integration regression.
- Physically vendored torch.export/torch-mlir graph frontend and dense-decoder
  semantic certificate.
- Transitional full-Llama prefill reference lowering to generic PLENA ISA,
  including all decoder layers, final RMSNorm, and LM head.
- Full Llama-3.2-1B Rust simulator execution with exact target-contract logits
  and matching Hugging Face FP16 next token.

## Next milestones

### 1. Streaming shared-L2 planner

Replace the current whole-tensor L2 residency requirement with reusable L2
windows. Weights remain in LP6 and are brought in as contiguous output-channel
batches. Add lifetime and alias verification before enabling reuse, plus an
explicit result GDMA store or runtime-visible shared-L2 handoff contract.

### 2. Compiler-controlled double buffering

Add `off`, `l1`, and `l1-l2` scheduling modes. Allocate explicit Ping/Pong
ranges, emit tagged `L2_LOAD_*_ASYNC_EVENT` plus `C_WAIT_EVENT`, and overlap
the next LP6/L2 batch with the current core block when dependencies permit.

### 3. Native generic VPU lowering

Move the already executable reference lowering for semantic `silu`, `rms_norm`,
`softmax`, `rope`, and elementwise ops into native MLIR passes.
Lower them to generic Vector/Scalar/Reduction ISA while retaining intermediate
streams until an explicit store. Do not introduce model-specific fused ISA.

### 4. Attention and decode

Lower QK, scale/mask, row reduction, EXP, normalization, and PV as separate
scheduled stages. Support M=1 GEMV on the Matrix engine first, then compare a
dedicated GEMV mapping if simulator evidence justifies it.

### 5. Full model memory and multicore flow

Keep hidden/KV state in LP6 or shared L2 according to lifetime. Partition
projection output channels and attention head groups across logical cores.
Write disjoint shards into shared L2 and use multi-event dependencies before a
consumer requiring the complete tensor. Keep split-K disabled until a real
collective/reduction design exists.

### 6. Replace the transitional full-model lowering

The torch.export capture, package hashing, and dense-decoder graph certificate
are now vendored and operational. Replace the model-specialized Python
reference backend with actual Torch/Linalg-to-PLENA MLIR patterns so scheduling
and backend legality follow imported SSA rather than a fixed Llama stage
sequence.

### 7. Dtype expansion

After FP16 end-to-end is stable, preserve INT8 checkpoint payloads in LP6 and
add type legalization for W8A8 and INT8-to-FP16 dequant paths. INT4 remains a
later packed-storage extension.

## Acceptance target

Full static prefill is now executable. The next major endpoint is native-MLIR
prefill plus stateful decode:

```text
local Hugging Face checkpoint
  -> torch.export / torch-mlir
  -> PLENA semantic and memory schedule
  -> program.bin + system.json + lp6.bin
  -> Rust simulator
  -> exact/thresholded logits and generated token comparison
```
