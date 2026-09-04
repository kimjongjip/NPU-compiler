//===- ProgramV5.h - Unified PLENA Program v5 encoder --------*- C++ -*-===//

#ifndef PLENA_TARGET_PROGRAMV5_H
#define PLENA_TARGET_PROGRAMV5_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <string>
#include <variant>
#include <vector>

namespace plena::target {

constexpr uint32_t kProgramV5Magic = 0x354e4c50u;
constexpr uint32_t kProgramV5Version = 5;
constexpr uint32_t kNoneU32 = 0xffff'ffffu;

struct GdmaLoadRecord {
  std::string name;
  uint32_t event = 0;
  std::vector<uint32_t> dependencies;
  uint64_t lp6Address = 0;
  uint64_t l2Address = 0;
  uint32_t rowBytes = 0;
  uint32_t rows = 1;
  uint32_t lp6Stride = 0;
  uint32_t l2Stride = 0;
};

struct L1RegionRecord {
  uint64_t offset = 0;
  uint64_t bytes = 0;
  uint64_t alignment = 64;
  uint32_t alias = kNoneU32;
};

struct CoreBlockRecord {
  std::string name;
  uint32_t event = 0;
  std::vector<uint32_t> dependencies;
  uint32_t logicalCore = 0;
  uint64_t l1BytesRequired = 0;
  std::vector<L1RegionRecord> l1Regions;
  std::vector<uint32_t> coreWords;
};

using ProgramRecord = std::variant<GdmaLoadRecord, CoreBlockRecord>;

struct ProgramV5Image {
  std::vector<uint32_t> words;
  uint32_t commandCount = 0;
  uint32_t coreInstructionCount = 0;
};

llvm::Expected<ProgramV5Image>
encodeProgramV5(llvm::ArrayRef<ProgramRecord> records,
                uint32_t completionEventSlots);

std::string buildSystemManifestJSON(
    const ProgramV5Image &image, llvm::ArrayRef<ProgramRecord> records,
    llvm::ArrayRef<uint32_t> logicalToPhysical, uint64_t l2BytesRequired,
    llvm::StringRef l2RegionsJSON);

llvm::Expected<std::string>
augmentCompileReportJSON(llvm::StringRef report,
                         const ProgramV5Image &image);

} // namespace plena::target

#endif // PLENA_TARGET_PROGRAMV5_H
