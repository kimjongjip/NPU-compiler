#include "PLENA/Target/ProgramV7.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <limits>

using namespace plena::target;

namespace {

constexpr uint32_t kHeaderWords = 5;
constexpr uint32_t kCoreBegin = 0x26;
constexpr uint32_t kCoreEnd = 0x27;
constexpr uint32_t kGdmaLoad = 0x2b;
constexpr uint32_t kProgramEnd = 0x2f;

void appendU64(std::vector<uint32_t> &words, uint64_t value) {
  words.push_back(static_cast<uint32_t>(value));
  words.push_back(static_cast<uint32_t>(value >> 32));
}

std::string jsonText(llvm::json::Value value) {
  std::string result;
  llvm::raw_string_ostream stream(result);
  stream << llvm::formatv("{0:2}", std::move(value));
  return result;
}

uint32_t eventOf(const ProgramRecord &record) {
  return std::visit([](const auto &value) { return value.event; }, record);
}

llvm::StringRef nameOf(const ProgramRecord &record) {
  return std::visit(
      [](const auto &value) -> llvm::StringRef { return value.name; }, record);
}

} // namespace

llvm::Expected<ProgramV7Image>
plena::target::encodeProgramV7(llvm::ArrayRef<ProgramRecord> records,
                               uint32_t completionEventSlots) {
  if (records.empty())
    return llvm::createStringError("unified program contains no commands");
  if (records.size() > UINT32_MAX)
    return llvm::createStringError("unified command count exceeds uint32");

  llvm::DenseSet<uint32_t> events;
  llvm::DenseSet<uint32_t> referencedEvents;
  for (const ProgramRecord &record : records) {
    const uint32_t event = eventOf(record);
    if (event >= completionEventSlots)
      return llvm::createStringError(
          "completion event exceeds configured scoreboard");
    if (!events.insert(event).second)
      return llvm::createStringError("duplicate completion event");
    std::visit(
        [&](const auto &value) {
          for (uint32_t dependency : value.dependencies)
            referencedEvents.insert(dependency);
        },
        record);
  }
  for (uint32_t dependency : referencedEvents)
    if (dependency >= completionEventSlots || !events.contains(dependency))
      return llvm::createStringError(
          "command references an unknown completion event");

  ProgramV7Image image;
  image.commandCount = static_cast<uint32_t>(records.size());
  image.words.resize(kHeaderWords, 0);
  uint64_t coreInstructions = 0;
  for (const ProgramRecord &record : records) {
    if (const auto *gdma = std::get_if<GdmaLoadRecord>(&record)) {
      if (gdma->rowBytes == 0 || gdma->rows == 0 ||
          gdma->lp6Stride < gdma->rowBytes ||
          gdma->l2Stride < gdma->rowBytes)
        return llvm::createStringError("invalid GDMA load dimensions");
      image.words.push_back(gdma->store ? 0x2c : kGdmaLoad);
      image.words.push_back(gdma->event);
      image.words.push_back(static_cast<uint32_t>(gdma->dependencies.size()));
      appendU64(image.words, gdma->lp6Address);
      appendU64(image.words, gdma->l2Address);
      image.words.push_back(gdma->rowBytes);
      image.words.push_back(gdma->rows);
      image.words.push_back(gdma->lp6Stride);
      image.words.push_back(gdma->l2Stride);
      image.words.insert(image.words.end(), gdma->dependencies.begin(),
                         gdma->dependencies.end());
      continue;
    }

    const auto &block = std::get<CoreBlockRecord>(record);
    if (block.coreWords.empty())
      return llvm::createStringError("CORE_BEGIN block contains no core ISA");
    if (block.dependencies.size() > UINT32_MAX ||
        block.l1Regions.size() > UINT32_MAX ||
        block.coreWords.size() > UINT32_MAX)
      return llvm::createStringError("CORE_BEGIN record field exceeds uint32");
    image.words.push_back(kCoreBegin);
    image.words.push_back(block.event);
    image.words.push_back(block.logicalCore);
    image.words.push_back(kNoneU32); // preferred physical core
    image.words.push_back(kNoneU32); // affinity ID
    appendU64(image.words, block.l1BytesRequired);
    image.words.push_back(static_cast<uint32_t>(block.coreWords.size()));
    image.words.push_back(static_cast<uint32_t>(block.dependencies.size()));
    image.words.push_back(0); // allowed physical-core count
    image.words.push_back(static_cast<uint32_t>(block.l1Regions.size()));
    image.words.insert(image.words.end(), block.dependencies.begin(),
                       block.dependencies.end());
    for (const L1RegionRecord &region : block.l1Regions) {
      appendU64(image.words, region.offset);
      appendU64(image.words, region.bytes);
      appendU64(image.words, region.alignment);
      image.words.push_back(region.alias);
    }
    image.words.insert(image.words.end(), block.coreWords.begin(),
                       block.coreWords.end());
    image.words.push_back(kCoreEnd);
    image.words.push_back(block.event);
    coreInstructions += block.coreWords.size();
    if (coreInstructions > UINT32_MAX)
      return llvm::createStringError("core instruction count exceeds uint32");
  }
  image.words.push_back(kProgramEnd);
  if (image.words.size() > UINT32_MAX)
    return llvm::createStringError("unified program word count exceeds uint32");
  image.coreInstructionCount = static_cast<uint32_t>(coreInstructions);
  image.words[0] = kProgramV7Magic;
  image.words[1] = kProgramV7Version;
  image.words[2] = static_cast<uint32_t>(image.words.size());
  image.words[3] = image.commandCount;
  image.words[4] = image.coreInstructionCount;
  return image;
}

std::string plena::target::buildSystemManifestJSON(
    const ProgramV7Image &image, llvm::ArrayRef<ProgramRecord> records,
    llvm::ArrayRef<uint32_t> logicalToPhysical, uint64_t l2BytesRequired,
    llvm::StringRef l2RegionsJSON) {
  llvm::json::Array symbols;
  for (const ProgramRecord &record : records)
    symbols.push_back(llvm::json::Object{
        {"id", eventOf(record)},
        {"name", nameOf(record)},
    });
  llvm::json::Array placement;
  for (uint32_t core : logicalToPhysical)
    placement.push_back(core);

  llvm::json::Array simulatorRegions;
  auto parsed = llvm::json::parse(l2RegionsJSON);
  if (parsed) {
    if (auto *regions = parsed->getAsArray()) {
      for (const llvm::json::Value &value : *regions) {
        const auto *region = value.getAsObject();
        if (!region)
          continue;
        simulatorRegions.push_back(llvm::json::Object{
            {"name", region->getString("name").value_or("unnamed")},
            {"offset", region->getInteger("byte_base").value_or(0)},
            {"size", region->getInteger("size_bytes").value_or(0)},
            {"alignment", region->getInteger("alignment").value_or(64)},
        });
      }
    }
  } else {
    llvm::consumeError(parsed.takeError());
  }

  llvm::json::Object root{
      {"schema", "plena.v2.unified_program.isa_v1.0"},
      {"program", "program.bin"},
      {"program_word_count", static_cast<int64_t>(image.words.size())},
      {"command_count", image.commandCount},
      {"core_instruction_count", image.coreInstructionCount},
      {"command_symbols", std::move(symbols)},
      {"scheduling_policy", "compiler_static"},
      {"logical_to_physical", std::move(placement)},
      {"l2_bytes_required", static_cast<int64_t>(l2BytesRequired)},
      {"l2_regions", std::move(simulatorRegions)},
  };
  return jsonText(std::move(root));
}

llvm::Expected<std::string> plena::target::augmentCompileReportJSON(
    llvm::StringRef report, const ProgramV7Image &image) {
  auto parsed = llvm::json::parse(report);
  if (!parsed)
    return parsed.takeError();
  auto *root = parsed->getAsObject();
  if (!root)
    return llvm::createStringError("compile report root is not an object");
  (*root)["program"] = llvm::json::Object{
      {"abi", "unified_command_isa_v1.0"},
      {"word_bits", 32},
      {"endianness", "little"},
      {"program_words", static_cast<int64_t>(image.words.size())},
      {"program_bytes", static_cast<int64_t>(image.words.size() * 4)},
      {"command_count", image.commandCount},
      {"core_instruction_words", image.coreInstructionCount},
  };
  return jsonText(std::move(*parsed));
}
