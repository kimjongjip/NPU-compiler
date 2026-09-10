#include "PLENA/Target/CoreISAEmitter.h"

#include "llvm/ADT/Twine.h"

#include <algorithm>
#include <limits>

using namespace plena::target;

namespace {

constexpr uint32_t kOpAddImmediate = 0x22;
constexpr uint32_t kOpLoadUpper = 0x25;
constexpr uint32_t kOpWaitMatrix = 0x3f | (2u << 22);
constexpr uint32_t kOpSetControl = 0x39;
constexpr uint32_t kOpLocalDma = 0x38;
constexpr uint32_t kOpMatrixLoad = 0x37;
constexpr uint32_t kOpMatrixMma = 0x3b | (3u << 22);
constexpr uint32_t kOpMatrixWriteoutF16 = 0x3c | (1u << 22);

constexpr unsigned kActivationL1Register = 0;
constexpr unsigned kActivationL2Register = 1;
constexpr unsigned kWeightL1Register = 2;
constexpr unsigned kWeightL2Register = 3;
constexpr unsigned kOutputL1Register = 4;
constexpr unsigned kOutputL2Register = 5;

uint32_t encodeRegister(uint32_t opcode, unsigned rd = 0, unsigned rs1 = 0,
                        unsigned rs2 = 0, unsigned rs3 = 0,
                        unsigned function = 0) {
  return opcode | (rd << 6) | (rs1 << 10) | (rs2 << 14) | (rs3 << 18) |
         (function << 22);
}

llvm::Expected<uint32_t> narrow(uint64_t value, llvm::StringRef name) {
  if (value > UINT32_MAX)
    return llvm::createStringError(name + " does not fit the 32-bit core ISA");
  return static_cast<uint32_t>(value);
}

} // namespace

void CoreISAEmitter::loadU32(unsigned reg, uint32_t value) {
  const uint32_t upper = value >> 12;
  const uint32_t lower = value & 0xfffu;
  append(kOpLoadUpper | (reg << 6) | (upper << 10));
  if (lower != 0)
    append(kOpAddImmediate | (reg << 6) | (reg << 10) | (lower << 14));
}

void CoreISAEmitter::setControl(unsigned function, uint32_t value,
                                unsigned scratch) {
  loadU32(scratch, value);
  append(encodeRegister(kOpSetControl, scratch, 0, 0, 0, function));
}

void CoreISAEmitter::setDma2D(uint32_t rowBytes, uint32_t rows,
                              uint32_t sourceStride,
                              uint32_t destinationStride) {
  setControl(7, rowBytes);
  setControl(0, rows);
  setControl(1, sourceStride);
  setControl(2, destinationStride);
}

void CoreISAEmitter::emitL2Load(unsigned l1Register, unsigned l2Register,
                                uint32_t rowBytes, uint32_t rows,
                                uint32_t sourceStride,
                                uint32_t destinationStride) {
  setDma2D(rowBytes, rows, sourceStride, destinationStride);
  // The raw LDMA instruction consumes only C_SET_DMA_BYTES; it does not
  // multiply that value by C_SET_DMA_ROWS. Keep every multi-row descriptor
  // explicit even when both strides happen to equal row_bytes.
  const bool contiguous = rows == 1;
  append(encodeRegister(kOpLocalDma, l1Register, l2Register, 0, 0,
                        contiguous ? 0 : 4));
}

void CoreISAEmitter::emitL2Store(unsigned l1Register, unsigned l2Register,
                                 uint32_t rowBytes, uint32_t rows,
                                 uint32_t sourceStride,
                                 uint32_t destinationStride) {
  setDma2D(rowBytes, rows, sourceStride, destinationStride);
  const bool contiguous = rows == 1;
  append(encodeRegister(kOpLocalDma, l1Register, l2Register, 0, 0,
                        contiguous ? 1 : 5));
}

llvm::Expected<std::vector<uint32_t>>
CoreISAEmitter::emitMatmul(const CoreBlockPlan &plan) {
  words_.clear();
  const MatmulProblem &problem = plan.problem;
  const MatmulTile &tile = plan.tile;
  const MemoryLayout &memory = plan.memory;
  if (tile.m == 0 || tile.n == 0 || tile.totalK == 0 || tile.kChunk == 0)
    return llvm::createStringError("matmul tile dimensions must be non-zero");

  auto activationL1 = narrow(memory.l1Activation.base, "activation L1 address");
  auto weightL1 = narrow(memory.l1Weight.base, "weight L1 address");
  auto outputL1 = narrow(memory.l1Output.base, "output L1 address");
  if (!activationL1)
    return activationL1.takeError();
  if (!weightL1)
    return weightL1.takeError();
  if (!outputL1)
    return outputL1.takeError();

  for (uint32_t kOffset = 0; kOffset < tile.totalK;
       kOffset += tile.kChunk) {
    const uint32_t currentK =
        std::min(tile.kChunk, tile.totalK - kOffset);
    const uint64_t activationOffset =
        (uint64_t{tile.mOffset} * problem.k + kOffset) * 2;
    const uint64_t weightOffset =
        (uint64_t{kOffset} * problem.n + tile.nOffset) * 2;
    auto activationL2 = narrow(memory.l2Activation.base + activationOffset,
                               "activation L2 address");
    auto weightL2 = narrow(memory.l2Weight.base + weightOffset,
                           "weight L2 address");
    if (!activationL2)
      return activationL2.takeError();
    if (!weightL2)
      return weightL2.takeError();

    loadU32(kActivationL1Register, *activationL1);
    loadU32(kActivationL2Register, *activationL2);
    emitL2Load(kActivationL1Register, kActivationL2Register, currentK * 2,
               tile.m, problem.k * 2, currentK * 2);

    loadU32(kWeightL1Register, *weightL1);
    loadU32(kWeightL2Register, *weightL2);
    emitL2Load(kWeightL1Register, kWeightL2Register, tile.n * 2, currentK,
               problem.n * 2, tile.n * 2);

    loadU32(kWeightL1Register, *weightL1);
    append(encodeRegister(kOpMatrixLoad, 0, kWeightL1Register, 0, 0, 3));
    append(currentK);
    append(tile.n);
    append(tile.n * 2);
    loadU32(kActivationL1Register, *activationL1);
    append(
        encodeRegister(kOpMatrixLoad, 0, kActivationL1Register, 0, 0, 7));
    append(tile.m);
    append(currentK);
    append(currentK * 2);
    append(kOpMatrixMma | (kOffset != 0 ? 1u << 26 : 0));
    append(tile.m);
    append(tile.n);
    append(currentK);
  }

  loadU32(kOutputL1Register, *outputL1);
  append(kOpWaitMatrix);
  append(encodeRegister(kOpMatrixWriteoutF16, kOutputL1Register));
  append(tile.m);
  append(tile.n);
  append(tile.n * 2);

  const uint64_t outputOffset =
      (uint64_t{tile.mOffset} * problem.n + tile.nOffset) * 2;
  auto outputL2 =
      narrow(memory.l2Output.base + outputOffset, "output L2 address");
  if (!outputL2)
    return outputL2.takeError();
  loadU32(kOutputL2Register, *outputL2);
  emitL2Store(kOutputL1Register, kOutputL2Register, tile.n * 2, tile.m,
              tile.n * 2, problem.n * 2);
  return std::move(words_);
}
