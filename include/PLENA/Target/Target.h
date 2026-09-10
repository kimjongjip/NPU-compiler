//===- Target.h - PLENA target configuration and plans -------*- C++ -*-===//

#ifndef PLENA_TARGET_TARGET_H
#define PLENA_TARGET_TARGET_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"

#include <cstdint>
#include <string>
#include <vector>

namespace plena::target {

struct TargetConfig {
  uint32_t arrayRows = 32;
  uint32_t arrayColumns = 32;
  uint32_t physicalCores = 1;
  uint32_t logicalCores = 1;
  uint64_t l1BytesPerCore = 4ull * 1024 * 1024;
  uint64_t l2Bytes = 8ull * 1024 * 1024;
  uint32_t kChunk = 64;
  uint32_t alignment = 64;
  uint32_t completionEventSlots = 65536;
  uint32_t isaVersion = 7;
  std::vector<uint32_t> logicalToPhysical;

  llvm::Error validate() const;
  std::vector<uint32_t> physicalPlacement() const;
};

struct MatmulProblem {
  mlir::Value activation;
  mlir::Value weight;
  mlir::Value output;
  uint32_t m = 0;
  uint32_t k = 0;
  uint32_t n = 0;
};

struct ByteRegion {
  std::string name;
  std::string role;
  uint64_t base = 0;
  uint64_t bytes = 0;
  uint64_t alignment = 64;
};

struct MemoryLayout {
  ByteRegion lp6Activation;
  ByteRegion lp6Weight;
  uint64_t lp6ImageBytes = 0;

  ByteRegion l2Activation;
  ByteRegion l2Weight;
  ByteRegion l2Output;
  uint64_t l2BytesRequired = 0;

  ByteRegion l1Activation;
  ByteRegion l1Weight;
  ByteRegion l1Output;
  uint64_t l1BytesRequired = 0;
};

struct MatmulTile {
  uint32_t id = 0;
  uint32_t logicalCore = 0;
  uint32_t mOffset = 0;
  uint32_t nOffset = 0;
  uint32_t m = 0;
  uint32_t n = 0;
  uint32_t totalK = 0;
  uint32_t kChunk = 0;
};

llvm::Expected<TargetConfig> readTargetConfigFile(llvm::StringRef path);
void attachTargetConfig(mlir::ModuleOp module, const TargetConfig &config);
llvm::Expected<TargetConfig> readTargetConfig(mlir::ModuleOp module);

llvm::Expected<MatmulProblem> analyzeMatmul(mlir::ModuleOp module);
llvm::Expected<MemoryLayout> planMemory(const MatmulProblem &problem,
                                        const TargetConfig &config);
llvm::Expected<std::vector<MatmulTile>>
tileMatmul(const MatmulProblem &problem, const TargetConfig &config);

std::string buildModelManifestJSON(const MatmulProblem &problem,
                                   const TargetConfig &config,
                                   const MemoryLayout &layout);
std::string buildL2RegionsJSON(const MemoryLayout &layout);
std::string buildCompileReportJSON(const MatmulProblem &problem,
                                   const TargetConfig &config,
                                   const MemoryLayout &layout,
                                   llvm::ArrayRef<MatmulTile> tiles);

} // namespace plena::target

#endif // PLENA_TARGET_TARGET_H
