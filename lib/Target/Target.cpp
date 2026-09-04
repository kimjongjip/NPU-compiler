#include "PLENA/Target/Target.h"
#include "PLENA/Target/ProgramV5.h"

#include "llvm/ADT/Twine.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include <algorithm>
#include <limits>

using namespace mlir;
using namespace plena::target;

namespace {

llvm::Error readUnsigned(const llvm::json::Object &object,
                         llvm::StringRef name, uint32_t &result) {
  std::optional<int64_t> value = object.getInteger(name);
  if (!value)
    return llvm::createStringError("missing unsigned configuration field " +
                                   name);
  if (*value < 0 || static_cast<uint64_t>(*value) > UINT32_MAX)
    return llvm::createStringError("configuration field is not uint32: " +
                                   name);
  result = static_cast<uint32_t>(*value);
  return llvm::Error::success();
}

llvm::Error readUnsigned64(const llvm::json::Object &object,
                           llvm::StringRef name, uint64_t &result) {
  std::optional<int64_t> value = object.getInteger(name);
  if (!value)
    return llvm::createStringError("missing unsigned configuration field " +
                                   name);
  if (*value < 0)
    return llvm::createStringError("configuration field is negative: " +
                                   name);
  result = static_cast<uint64_t>(*value);
  return llvm::Error::success();
}

uint64_t alignUp(uint64_t value, uint64_t alignment) {
  return value + (alignment - value % alignment) % alignment;
}

llvm::Expected<uint64_t> checkedBytes(uint64_t lhs, uint64_t rhs,
                                      llvm::StringRef label) {
  if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs)
    return llvm::createStringError(label + " byte size overflows uint64");
  const uint64_t elements = lhs * rhs;
  if (elements > std::numeric_limits<uint64_t>::max() / 2)
    return llvm::createStringError(label + " FP16 byte size overflows uint64");
  return elements * 2;
}

llvm::Expected<uint32_t> dimensionToU32(int64_t value,
                                        llvm::StringRef label) {
  if (value <= 0 || static_cast<uint64_t>(value) > UINT32_MAX)
    return llvm::createStringError(label + " must be a positive uint32");
  return static_cast<uint32_t>(value);
}

bool isStaticIdentityF16(MemRefType type) {
  return type && type.hasStaticShape() && type.getElementType().isF16() &&
         type.getLayout().isIdentity();
}

std::string jsonText(llvm::json::Value value) {
  std::string result;
  llvm::raw_string_ostream stream(result);
  stream << llvm::formatv("{0:2}", std::move(value));
  return result;
}

llvm::json::Object regionJSON(const ByteRegion &region,
                              llvm::StringRef space) {
  return llvm::json::Object{
      {"name", region.name},
      {"role", region.role},
      {"space", space},
      {"byte_base", static_cast<int64_t>(region.base)},
      {"size_bytes", static_cast<int64_t>(region.bytes)},
      {"alignment", static_cast<int64_t>(region.alignment)},
  };
}

uint32_t coreForNTile(uint32_t nTile, uint32_t nTiles,
                      uint32_t logicalCores) {
  const uint32_t base = nTiles / logicalCores;
  const uint32_t remainder = nTiles % logicalCores;
  uint32_t begin = 0;
  for (uint32_t core = 0; core < logicalCores; ++core) {
    const uint32_t count = base + static_cast<uint32_t>(core < remainder);
    if (nTile >= begin && nTile < begin + count)
      return core;
    begin += count;
  }
  llvm_unreachable("N tile was not assigned to a logical core");
}

} // namespace

llvm::Error TargetConfig::validate() const {
  if (arrayRows != 32 || arrayColumns != 32)
    return llvm::createStringError(
        "the current PLENA functional target requires a 32x32 systolic array");
  if (physicalCores == 0 || logicalCores == 0 || logicalCores > physicalCores)
    return llvm::createStringError(
        "logical_cores must be in the configured physical-core range");
  if (!logicalToPhysical.empty()) {
    if (logicalToPhysical.size() != logicalCores)
      return llvm::createStringError(
          "logical_to_physical length must equal logical_cores");
    llvm::DenseSet<uint32_t> seenPhysical;
    for (uint32_t core : logicalToPhysical) {
      if (core >= physicalCores || !seenPhysical.insert(core).second)
        return llvm::createStringError(
            "logical_to_physical must be a one-to-one permutation");
    }
  }
  if (l1BytesPerCore == 0 || l1BytesPerCore > UINT32_MAX)
    return llvm::createStringError(
        "l1_bytes_per_core must fit the 32-bit core-local byte address space");
  if (l2Bytes == 0 || l2Bytes > UINT32_MAX)
    return llvm::createStringError(
        "l2_bytes must fit the 32-bit shared byte address space");
  if (kChunk == 0)
    return llvm::createStringError("k_chunk must be non-zero");
  if (alignment == 0 || !llvm::isPowerOf2_32(alignment))
    return llvm::createStringError("alignment must be a non-zero power of two");
  if (completionEventSlots < 3)
    return llvm::createStringError(
        "completion_event_slots must hold two GDMA events and one core block");
  if (isaVersion != kProgramV5Version)
    return llvm::createStringError("only PLENA unified Program version 5 is supported");
  return llvm::Error::success();
}

std::vector<uint32_t> TargetConfig::physicalPlacement() const {
  if (!logicalToPhysical.empty())
    return logicalToPhysical;
  std::vector<uint32_t> identity(logicalCores);
  for (uint32_t core = 0; core < logicalCores; ++core)
    identity[core] = core;
  return identity;
}

llvm::Expected<TargetConfig>
plena::target::readTargetConfigFile(llvm::StringRef path) {
  auto file = llvm::MemoryBuffer::getFile(path);
  if (!file)
    return llvm::createStringError(file.getError(),
                                   "cannot read target configuration " + path);
  auto parsed = llvm::json::parse((*file)->getBuffer());
  if (!parsed)
    return parsed.takeError();
  const llvm::json::Object *root = parsed->getAsObject();
  if (!root || root->getInteger("schema_version").value_or(0) != 1)
    return llvm::createStringError(
        "PLENA target configuration requires schema_version=1");
  const auto *architecture = root->getObject("architecture");
  const auto *memory = root->getObject("memory");
  const auto *tiling = root->getObject("tiling");
  const auto *command = root->getObject("command_processor");
  const auto *isa = root->getObject("isa");
  if (!architecture || !memory || !tiling || !command || !isa)
    return llvm::createStringError(
        "target configuration requires architecture, memory, tiling, "
        "command_processor, and isa objects");
  if (architecture->getString("dataflow").value_or("") !=
      "output_stationary")
    return llvm::createStringError(
        "architecture.dataflow must be output_stationary");
  if (isa->getInteger("word_bits").value_or(0) != 32 ||
      isa->getString("address_unit").value_or("") != "byte")
    return llvm::createStringError(
        "PLENA ISA requires 32-bit words and byte addresses");

  TargetConfig config;
  if (llvm::Error error =
          readUnsigned(*architecture, "array_rows", config.arrayRows))
    return std::move(error);
  if (llvm::Error error =
          readUnsigned(*architecture, "array_columns", config.arrayColumns))
    return std::move(error);
  if (llvm::Error error =
          readUnsigned(*architecture, "physical_cores", config.physicalCores))
    return std::move(error);
  if (llvm::Error error =
          readUnsigned(*architecture, "logical_cores", config.logicalCores))
    return std::move(error);
  if (const llvm::json::Array *placement =
          architecture->getArray("logical_to_physical")) {
    for (const llvm::json::Value &entry : *placement) {
      std::optional<int64_t> core = entry.getAsInteger();
      if (!core || *core < 0 || static_cast<uint64_t>(*core) > UINT32_MAX)
        return llvm::createStringError(
            "architecture.logical_to_physical must contain uint32 values");
      config.logicalToPhysical.push_back(static_cast<uint32_t>(*core));
    }
  } else if (architecture->get("logical_to_physical")) {
    return llvm::createStringError(
        "architecture.logical_to_physical must be an array");
  }
  if (llvm::Error error = readUnsigned64(*memory, "l1_bytes_per_core",
                                         config.l1BytesPerCore))
    return std::move(error);
  if (llvm::Error error = readUnsigned64(*memory, "l2_bytes", config.l2Bytes))
    return std::move(error);
  if (llvm::Error error = readUnsigned(*memory, "alignment", config.alignment))
    return std::move(error);
  if (llvm::Error error = readUnsigned(*tiling, "k_chunk", config.kChunk))
    return std::move(error);
  if (llvm::Error error = readUnsigned(*command, "completion_event_slots",
                                       config.completionEventSlots))
    return std::move(error);
  if (llvm::Error error = readUnsigned(*isa, "version", config.isaVersion))
    return std::move(error);
  if (llvm::Error error = config.validate())
    return std::move(error);
  return config;
}

void plena::target::attachTargetConfig(ModuleOp module,
                                       const TargetConfig &config) {
  Builder builder(module.getContext());
  auto set = [&](llvm::StringRef name, uint64_t value) {
    module->setAttr(name, builder.getI64IntegerAttr(value));
  };
  set("plena.target.array_rows", config.arrayRows);
  set("plena.target.array_columns", config.arrayColumns);
  set("plena.target.physical_cores", config.physicalCores);
  set("plena.target.logical_cores", config.logicalCores);
  set("plena.target.l1_bytes_per_core", config.l1BytesPerCore);
  set("plena.target.l2_bytes", config.l2Bytes);
  set("plena.target.k_chunk", config.kChunk);
  set("plena.target.alignment", config.alignment);
  set("plena.target.completion_event_slots", config.completionEventSlots);
  set("plena.target.isa_version", config.isaVersion);
  std::vector<int32_t> placement;
  for (uint32_t core : config.physicalPlacement())
    placement.push_back(static_cast<int32_t>(core));
  module->setAttr("plena.target.logical_to_physical",
                  builder.getDenseI32ArrayAttr(placement));
}

llvm::Expected<TargetConfig>
plena::target::readTargetConfig(ModuleOp module) {
  auto read = [&](llvm::StringRef name) -> llvm::Expected<uint64_t> {
    auto value = module->getAttrOfType<IntegerAttr>(name);
    if (!value || value.getValue().isNegative())
      return llvm::createStringError("module is missing nonnegative " + name);
    return value.getValue().getZExtValue();
  };
  TargetConfig config;
#define READ_FIELD(NAME, FIELD, TYPE)                                          \
  do {                                                                         \
    auto value = read(NAME);                                                   \
    if (!value)                                                                \
      return value.takeError();                                                \
    if (*value > std::numeric_limits<TYPE>::max())                             \
      return llvm::createStringError(NAME " exceeds its target field");       \
    config.FIELD = static_cast<TYPE>(*value);                                  \
  } while (false)
  READ_FIELD("plena.target.array_rows", arrayRows, uint32_t);
  READ_FIELD("plena.target.array_columns", arrayColumns, uint32_t);
  READ_FIELD("plena.target.physical_cores", physicalCores, uint32_t);
  READ_FIELD("plena.target.logical_cores", logicalCores, uint32_t);
  READ_FIELD("plena.target.l1_bytes_per_core", l1BytesPerCore, uint64_t);
  READ_FIELD("plena.target.l2_bytes", l2Bytes, uint64_t);
  READ_FIELD("plena.target.k_chunk", kChunk, uint32_t);
  READ_FIELD("plena.target.alignment", alignment, uint32_t);
  READ_FIELD("plena.target.completion_event_slots", completionEventSlots,
             uint32_t);
  READ_FIELD("plena.target.isa_version", isaVersion, uint32_t);
#undef READ_FIELD
  auto placement =
      module->getAttrOfType<DenseI32ArrayAttr>("plena.target.logical_to_physical");
  if (!placement)
    return llvm::createStringError(
        "module is missing plena.target.logical_to_physical");
  for (int32_t core : placement.asArrayRef()) {
    if (core < 0)
      return llvm::createStringError(
          "logical_to_physical contains a negative core ID");
    config.logicalToPhysical.push_back(static_cast<uint32_t>(core));
  }
  if (llvm::Error error = config.validate())
    return std::move(error);
  return config;
}

llvm::Expected<MatmulProblem>
plena::target::analyzeMatmul(ModuleOp module) {
  llvm::SmallVector<linalg::MatmulOp> matmuls;
  module.walk([&](linalg::MatmulOp op) { matmuls.push_back(op); });
  if (matmuls.size() != 1)
    return llvm::createStringError(
        "basic PLENA compiler requires exactly one linalg.matmul");
  linalg::MatmulOp matmul = matmuls.front();
  if (matmul.getInputs().size() != 2 || matmul.getOutputs().size() != 1)
    return matmul.emitError(
               "linalg.matmul requires two inputs and one output"),
           llvm::createStringError("invalid linalg.matmul operand count");

  auto activationType = dyn_cast<MemRefType>(matmul.getInputs()[0].getType());
  auto weightType = dyn_cast<MemRefType>(matmul.getInputs()[1].getType());
  auto outputType = dyn_cast<MemRefType>(matmul.getOutputs()[0].getType());
  if (!isStaticIdentityF16(activationType) ||
      !isStaticIdentityF16(weightType) || !isStaticIdentityF16(outputType) ||
      activationType.getRank() != 2 || weightType.getRank() != 2 ||
      outputType.getRank() != 2)
    return matmul.emitError(
               "PLENA matmul operands must be rank-2 static identity-layout FP16 memrefs"),
           llvm::createStringError("unsupported matmul type");

  auto m = dimensionToU32(activationType.getShape()[0], "M");
  auto k = dimensionToU32(activationType.getShape()[1], "K");
  auto weightK = dimensionToU32(weightType.getShape()[0], "weight K");
  auto n = dimensionToU32(weightType.getShape()[1], "N");
  auto outputM = dimensionToU32(outputType.getShape()[0], "output M");
  auto outputN = dimensionToU32(outputType.getShape()[1], "output N");
  if (!m)
    return m.takeError();
  if (!k)
    return k.takeError();
  if (!weightK)
    return weightK.takeError();
  if (!n)
    return n.takeError();
  if (!outputM)
    return outputM.takeError();
  if (!outputN)
    return outputN.takeError();
  if (*k != *weightK || *m != *outputM || *n != *outputN)
    return matmul.emitError(
               "PLENA expects [M,K] x [K,N] -> [M,N]"),
           llvm::createStringError("matmul shape mismatch");

  bool zeroInitialized = false;
  for (Operation &operation : *matmul->getBlock()) {
    if (&operation == matmul.getOperation())
      break;
    auto fill = dyn_cast<linalg::FillOp>(operation);
    if (!fill || fill.getOutputs().size() != 1 ||
        fill.getOutputs().front() != matmul.getOutputs().front())
      continue;
    auto constant = fill.getInputs().front().getDefiningOp<arith::ConstantOp>();
    auto value = constant ? dyn_cast<FloatAttr>(constant.getValue()) : FloatAttr{};
    if (value && value.getValue().isZero())
      zeroInitialized = true;
  }
  if (!zeroInitialized)
    return matmul.emitError(
               "linalg.matmul output must be initialized by a preceding zero linalg.fill"),
           llvm::createStringError("matmul output is not zero initialized");

  return MatmulProblem{matmul.getInputs()[0], matmul.getInputs()[1],
                       matmul.getOutputs()[0], *m, *k, *n};
}

llvm::Expected<MemoryLayout>
plena::target::planMemory(const MatmulProblem &problem,
                          const TargetConfig &config) {
  auto activationBytes = checkedBytes(problem.m, problem.k, "activation");
  auto weightBytes = checkedBytes(problem.k, problem.n, "weight");
  auto outputBytes = checkedBytes(problem.m, problem.n, "output");
  if (!activationBytes)
    return activationBytes.takeError();
  if (!weightBytes)
    return weightBytes.takeError();
  if (!outputBytes)
    return outputBytes.takeError();

  MemoryLayout layout;
  const uint64_t alignment = config.alignment;
  layout.lp6Activation = {"activation", "activation", 0, *activationBytes,
                          alignment};
  layout.lp6Weight = {"weight", "weight",
                      alignUp(layout.lp6Activation.base +
                                  layout.lp6Activation.bytes,
                              alignment),
                      *weightBytes, alignment};
  layout.lp6ImageBytes = layout.lp6Weight.base + layout.lp6Weight.bytes;

  layout.l2Activation = {"activation", "activation", 0, *activationBytes,
                         alignment};
  layout.l2Weight = {"weight", "weight",
                     alignUp(layout.l2Activation.base + layout.l2Activation.bytes,
                             alignment),
                     *weightBytes, alignment};
  layout.l2Output = {"output", "output",
                     alignUp(layout.l2Weight.base + layout.l2Weight.bytes,
                             alignment),
                     *outputBytes, alignment};
  layout.l2BytesRequired =
      alignUp(layout.l2Output.base + layout.l2Output.bytes, alignment);
  if (layout.l2BytesRequired > config.l2Bytes)
    return llvm::createStringError(
        "basic compiler requires the complete activation, weight, and output "
        "working set to fit shared L2");

  const uint64_t maxM = std::min(problem.m, config.arrayRows);
  const uint64_t maxN = std::min(problem.n, config.arrayColumns);
  const uint64_t maxK = std::min(problem.k, config.kChunk);
  auto l1ActivationBytes = checkedBytes(maxM, maxK, "L1 activation tile");
  auto l1WeightBytes = checkedBytes(maxK, maxN, "L1 weight tile");
  auto l1OutputBytes = checkedBytes(maxM, maxN, "L1 output tile");
  if (!l1ActivationBytes)
    return l1ActivationBytes.takeError();
  if (!l1WeightBytes)
    return l1WeightBytes.takeError();
  if (!l1OutputBytes)
    return l1OutputBytes.takeError();
  layout.l1Activation = {"activation_tile", "activation", 0,
                         *l1ActivationBytes, alignment};
  layout.l1Weight = {"weight_tile", "weight",
                     alignUp(layout.l1Activation.bytes, alignment),
                     *l1WeightBytes, alignment};
  layout.l1Output = {"output_tile", "output",
                     alignUp(layout.l1Weight.base + layout.l1Weight.bytes,
                             alignment),
                     *l1OutputBytes, alignment};
  layout.l1BytesRequired =
      alignUp(layout.l1Output.base + layout.l1Output.bytes, alignment);
  if (layout.l1BytesRequired > config.l1BytesPerCore)
    return llvm::createStringError(
        "one compact activation/weight/output tile does not fit private L1");
  return layout;
}

llvm::Expected<std::vector<MatmulTile>>
plena::target::tileMatmul(const MatmulProblem &problem,
                          const TargetConfig &config) {
  const uint32_t nTiles =
      (problem.n + config.arrayColumns - 1) / config.arrayColumns;
  const uint32_t mTiles =
      (problem.m + config.arrayRows - 1) / config.arrayRows;
  const uint64_t tileCount = uint64_t{nTiles} * mTiles;
  if (tileCount + 2 > config.completionEventSlots)
    return llvm::createStringError(
        "matmul command count exceeds the completion-event scoreboard");
  std::vector<MatmulTile> result;
  result.reserve(tileCount);
  uint32_t id = 0;
  for (uint32_t nTile = 0; nTile < nTiles; ++nTile) {
    const uint32_t nOffset = nTile * config.arrayColumns;
    const uint32_t n = std::min(config.arrayColumns, problem.n - nOffset);
    const uint32_t core =
        coreForNTile(nTile, nTiles, config.logicalCores);
    for (uint32_t mTile = 0; mTile < mTiles; ++mTile) {
      const uint32_t mOffset = mTile * config.arrayRows;
      const uint32_t m = std::min(config.arrayRows, problem.m - mOffset);
      result.push_back(MatmulTile{id++, core, mOffset, nOffset, m, n,
                                  problem.k,
                                  std::min(problem.k, config.kChunk)});
    }
  }
  return result;
}

std::string plena::target::buildL2RegionsJSON(const MemoryLayout &layout) {
  llvm::json::Array regions;
  for (const ByteRegion *region :
       {&layout.l2Activation, &layout.l2Weight, &layout.l2Output})
    regions.push_back(regionJSON(*region, "shared_l2"));
  return jsonText(std::move(regions));
}

std::string plena::target::buildModelManifestJSON(
    const MatmulProblem &problem, const TargetConfig &config,
    const MemoryLayout &layout) {
  llvm::json::Array placement;
  for (uint32_t core : config.physicalPlacement())
    placement.push_back(core);
  llvm::json::Array tensors;
  llvm::json::Object activation = regionJSON(layout.lp6Activation, "lp6");
  activation["shape"] = llvm::json::Array{problem.m, problem.k};
  activation["layout"] = "row_major_mk";
  activation["l2_byte_base"] = static_cast<int64_t>(layout.l2Activation.base);
  tensors.push_back(std::move(activation));
  llvm::json::Object weight = regionJSON(layout.lp6Weight, "lp6");
  weight["shape"] = llvm::json::Array{problem.k, problem.n};
  weight["layout"] = "row_major_kn";
  weight["l2_byte_base"] = static_cast<int64_t>(layout.l2Weight.base);
  tensors.push_back(std::move(weight));
  llvm::json::Object output = regionJSON(layout.l2Output, "shared_l2");
  output["shape"] = llvm::json::Array{problem.m, problem.n};
  output["layout"] = "row_major_mn";
  tensors.push_back(std::move(output));

  llvm::json::Object root{
      {"schema", "plena.compiler.model.v1"},
      {"program_kind", "fp16_matmul"},
      {"element_type", "fp16"},
      {"address_unit", "byte"},
      {"physical_cores", config.physicalCores},
      {"logical_cores", config.logicalCores},
      {"logical_to_physical", std::move(placement)},
      {"matrix_shape", llvm::json::Array{problem.m, problem.k, problem.n}},
      {"matrix_tile",
       llvm::json::Array{config.arrayRows, config.arrayColumns,
                         config.kChunk}},
      {"lp6_image", "lp6.bin"},
      {"lp6_image_bytes", static_cast<int64_t>(layout.lp6ImageBytes)},
      {"output",
       llvm::json::Object{{"tensor", "output"},
                          {"space", "shared_l2"},
                          {"byte_base",
                           static_cast<int64_t>(layout.l2Output.base)},
                          {"size_bytes",
                           static_cast<int64_t>(layout.l2Output.bytes)}}},
      {"tensors", std::move(tensors)},
  };
  return jsonText(std::move(root));
}

std::string plena::target::buildCompileReportJSON(
    const MatmulProblem &problem, const TargetConfig &config,
    const MemoryLayout &layout, llvm::ArrayRef<MatmulTile> tiles) {
  std::vector<uint32_t> perCore(config.logicalCores, 0);
  for (const MatmulTile &tile : tiles)
    ++perCore[tile.logicalCore];
  llvm::json::Array distribution;
  for (auto [core, count] : llvm::enumerate(perCore))
    distribution.push_back(llvm::json::Object{
        {"logical_core", static_cast<int64_t>(core)},
        {"output_tiles", static_cast<int64_t>(count)},
    });
  llvm::json::Array placement;
  for (uint32_t core : config.physicalPlacement())
    placement.push_back(core);
  llvm::json::Object root{
      {"schema", "plena.compiler.report.v1"},
      {"pipeline", "linalg_to_unified_program_v5"},
      {"problem", llvm::json::Object{{"m", problem.m},
                                     {"k", problem.k},
                                     {"n", problem.n}}},
      {"target", llvm::json::Object{{"array_rows", config.arrayRows},
                                    {"array_columns", config.arrayColumns},
                                    {"physical_cores", config.physicalCores},
                                    {"logical_cores", config.logicalCores},
                                    {"k_chunk", config.kChunk},
                                    {"logical_to_physical",
                                     std::move(placement)}}},
      {"memory", llvm::json::Object{
                     {"lp6_image_bytes",
                      static_cast<int64_t>(layout.lp6ImageBytes)},
                     {"l2_bytes_required",
                      static_cast<int64_t>(layout.l2BytesRequired)},
                     {"l1_bytes_required_per_block",
                      static_cast<int64_t>(layout.l1BytesRequired)}}},
      {"tiling", llvm::json::Object{
                     {"output_stationary", true},
                     {"partition_axis", "N"},
                     {"split_k", false},
                     {"double_buffering", "off"},
                     {"output_tile_count",
                      static_cast<int64_t>(tiles.size())},
                     {"per_core", std::move(distribution)}}},
  };
  return jsonText(std::move(root));
}
