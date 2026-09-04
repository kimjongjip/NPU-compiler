//===- CoreISAEmitter.h - PLENA v2 core ISA encoder ----------*- C++ -*-===//

#ifndef PLENA_TARGET_COREISAEMITTER_H
#define PLENA_TARGET_COREISAEMITTER_H

#include "PLENA/Target/Target.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <vector>

namespace plena::target {

struct CoreBlockPlan {
  MatmulProblem problem;
  MatmulTile tile;
  MemoryLayout memory;
};

class CoreISAEmitter {
public:
  llvm::Expected<std::vector<uint32_t>> emitMatmul(const CoreBlockPlan &plan);

private:
  void append(uint32_t word) { words_.push_back(word); }
  void loadU32(unsigned reg, uint32_t value);
  void setControl(unsigned function, uint32_t value, unsigned scratch = 6);
  void setDma2D(uint32_t rowBytes, uint32_t rows, uint32_t sourceStride,
                uint32_t destinationStride);
  void emitL2Load(unsigned l1Register, unsigned l2Register, uint32_t rowBytes,
                  uint32_t rows, uint32_t sourceStride,
                  uint32_t destinationStride);
  void emitL2Store(unsigned l1Register, unsigned l2Register,
                   uint32_t rowBytes, uint32_t rows,
                   uint32_t sourceStride, uint32_t destinationStride);

  std::vector<uint32_t> words_;
};

} // namespace plena::target

#endif // PLENA_TARGET_COREISAEMITTER_H
