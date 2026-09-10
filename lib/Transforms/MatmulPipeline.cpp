#include "PLENA/Transforms/Passes.h"

#include "PLENA/Dialect/Command/CommandDialect.h"
#include "PLENA/Dialect/Command/CommandOps.h"
#include "PLENA/Dialect/ISA/ISADialect.h"
#include "PLENA/Dialect/ISA/ISAOps.h"
#include "PLENA/Dialect/Memory/MemoryDialect.h"
#include "PLENA/Dialect/Memory/MemoryOps.h"
#include "PLENA/Dialect/Schedule/ScheduleDialect.h"
#include "PLENA/Dialect/Schedule/ScheduleOps.h"
#include "PLENA/Dialect/Tile/TileDialect.h"
#include "PLENA/Dialect/Tile/TileOps.h"
#include "PLENA/Target/CoreISAEmitter.h"
#include "PLENA/Target/ProgramV7.h"
#include "PLENA/Target/Target.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <variant>
#include <vector>

namespace mlir::plena {
#define GEN_PASS_DEF_VERIFYMATMUL
#define GEN_PASS_DEF_PLANMEMORY
#define GEN_PASS_DEF_TILEMATMUL
#define GEN_PASS_DEF_SCHEDULEMATMUL
#define GEN_PASS_DEF_LOWERTOCOMMANDS
#define GEN_PASS_DEF_ENCODEPROGRAMV7
#include "PLENA/Transforms/Passes.h.inc"
} // namespace mlir::plena

using namespace mlir;
using namespace ::plena::target;

namespace {

func::FuncOp entryFunction(ModuleOp module) {
  func::FuncOp entry = module.lookupSymbol<func::FuncOp>("main");
  return entry;
}

void addI64Attribute(OperationState &state, Builder &builder,
                     llvm::StringRef name, uint64_t value) {
  state.addAttribute(name, builder.getI64IntegerAttr(value));
}

void addSignedI64Attribute(OperationState &state, Builder &builder,
                           llvm::StringRef name, int64_t value) {
  state.addAttribute(name, builder.getI64IntegerAttr(value));
}

plena_mem::BindingOp createBinding(OpBuilder &builder, Location location,
                                   Value buffer, llvm::StringRef name,
                                   llvm::StringRef space,
                                   llvm::StringRef role, uint64_t base,
                                   uint64_t bytes, uint64_t alignment,
                                   int64_t logicalCore) {
  OperationState state(location, plena_mem::BindingOp::getOperationName());
  state.addOperands(buffer);
  state.addAttribute("name", builder.getStringAttr(name));
  state.addAttribute("space", builder.getStringAttr(space));
  state.addAttribute("role", builder.getStringAttr(role));
  addI64Attribute(state, builder, "byte_base", base);
  addI64Attribute(state, builder, "size_bytes", bytes);
  addI64Attribute(state, builder, "alignment", alignment);
  addSignedI64Attribute(state, builder, "logical_core", logicalCore);
  return cast<plena_mem::BindingOp>(builder.create(state));
}

plena_tile::MatmulOp createTile(OpBuilder &builder, Location location,
                                const MatmulProblem &problem,
                                const MatmulTile &tile) {
  OperationState state(location, plena_tile::MatmulOp::getOperationName());
  state.addOperands(
      {problem.activation, problem.weight, problem.output});
  addI64Attribute(state, builder, "tile_id", tile.id);
  addI64Attribute(state, builder, "logical_core", tile.logicalCore);
  addI64Attribute(state, builder, "m_offset", tile.mOffset);
  addI64Attribute(state, builder, "n_offset", tile.nOffset);
  addI64Attribute(state, builder, "tile_m", tile.m);
  addI64Attribute(state, builder, "tile_n", tile.n);
  addI64Attribute(state, builder, "total_k", tile.totalK);
  addI64Attribute(state, builder, "k_chunk", tile.kChunk);
  state.addAttribute("mode", builder.getStringAttr("fp16_fp16_fp32"));
  return cast<plena_tile::MatmulOp>(builder.create(state));
}

plena_sched::GdmaLoadOp createScheduledGdma(
    OpBuilder &builder, Location location, Value buffer, llvm::StringRef name,
    uint64_t lp6Address, uint64_t l2Address, uint64_t bytes) {
  OperationState state(location,
                       plena_sched::GdmaLoadOp::getOperationName());
  state.addOperands(buffer);
  state.addTypes(builder.getI32Type());
  state.addAttribute("name", builder.getStringAttr(name));
  addI64Attribute(state, builder, "lp6_address", lp6Address);
  addI64Attribute(state, builder, "l2_address", l2Address);
  addI64Attribute(state, builder, "row_bytes", bytes);
  addI64Attribute(state, builder, "rows", 1);
  addI64Attribute(state, builder, "lp6_stride", bytes);
  addI64Attribute(state, builder, "l2_stride", bytes);
  return cast<plena_sched::GdmaLoadOp>(builder.create(state));
}

plena_sched::CoreBlockOp createScheduledCoreBlock(
    OpBuilder &builder, Location location, const MatmulProblem &problem,
    const MemoryLayout &memory, plena_tile::MatmulOp tile,
    ValueRange waitEvents) {
  OperationState state(location,
                       plena_sched::CoreBlockOp::getOperationName());
  state.addOperands({problem.activation, problem.weight, problem.output});
  state.addOperands(waitEvents);
  state.addTypes(builder.getI32Type());
  const std::string name =
      llvm::formatv("matmul_m{0}_n{1}", tile.getMOffset(),
                    tile.getNOffset())
          .str();
  state.addAttribute("name", builder.getStringAttr(name));
  for (auto [key, value] : {
           std::pair<llvm::StringRef, uint64_t>{"logical_core",
                                                tile.getLogicalCore()},
           {"tile_id", tile.getTileId()},
           {"m_offset", tile.getMOffset()},
           {"n_offset", tile.getNOffset()},
           {"tile_m", tile.getTileM()},
           {"tile_n", tile.getTileN()},
           {"total_k", tile.getTotalK()},
           {"k_chunk", tile.getKChunk()},
           {"activation_l2", memory.l2Activation.base},
           {"weight_l2", memory.l2Weight.base},
           {"output_l2", memory.l2Output.base},
           {"activation_l1", memory.l1Activation.base},
           {"weight_l1", memory.l1Weight.base},
           {"output_l1", memory.l1Output.base},
           {"l1_bytes_required", memory.l1BytesRequired},
           {"activation_row_stride", uint64_t{problem.k} * 2},
           {"weight_row_stride", uint64_t{problem.n} * 2},
           {"output_row_stride", uint64_t{problem.n} * 2},
       })
    addI64Attribute(state, builder, key, value);
  state.addAttribute("mode", builder.getStringAttr("fp16_fp16_fp32"));
  return cast<plena_sched::CoreBlockOp>(builder.create(state));
}

std::vector<int32_t> signedWords(llvm::ArrayRef<uint32_t> words) {
  std::vector<int32_t> result;
  result.reserve(words.size());
  for (uint32_t word : words)
    result.push_back(static_cast<int32_t>(word));
  return result;
}

std::vector<uint32_t> unsignedWords(llvm::ArrayRef<int32_t> words) {
  std::vector<uint32_t> result;
  result.reserve(words.size());
  for (int32_t word : words)
    result.push_back(static_cast<uint32_t>(word));
  return result;
}

class VerifyMatmulPass
    : public mlir::plena::impl::VerifyMatmulBase<VerifyMatmulPass> {
public:
  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (!entryFunction(module)) {
      module.emitError("basic PLENA input requires func.func @main");
      signalPassFailure();
      return;
    }
    auto config = readTargetConfig(module);
    auto problem = analyzeMatmul(module);
    if (!config) {
      module.emitError(llvm::toString(config.takeError()));
      signalPassFailure();
      return;
    }
    if (!problem) {
      module.emitError(llvm::toString(problem.takeError()));
      signalPassFailure();
      return;
    }
    auto memory = planMemory(*problem, *config);
    if (!memory) {
      module.emitError(llvm::toString(memory.takeError()));
      signalPassFailure();
      return;
    }
    auto tiles = tileMatmul(*problem, *config);
    if (!tiles) {
      module.emitError(llvm::toString(tiles.takeError()));
      signalPassFailure();
      return;
    }
    module->setAttr("plena.verified", UnitAttr::get(module.getContext()));
  }
};

class PlanMemoryPass
    : public mlir::plena::impl::PlanMemoryBase<PlanMemoryPass> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<plena_mem::PLENAMemoryDialect>();
  }
  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (!module->hasAttr("plena.verified")) {
      module.emitError("memory planning requires verified input");
      signalPassFailure();
      return;
    }
    auto config = readTargetConfig(module);
    auto problem = analyzeMatmul(module);
    if (!config) {
      module.emitError(llvm::toString(config.takeError()));
      signalPassFailure();
      return;
    }
    if (!problem) {
      module.emitError(llvm::toString(problem.takeError()));
      signalPassFailure();
      return;
    }
    auto layout = planMemory(*problem, *config);
    if (!layout) {
      module.emitError(llvm::toString(layout.takeError()));
      signalPassFailure();
      return;
    }
    SmallVector<plena_mem::BindingOp> existing;
    module.walk([&](plena_mem::BindingOp op) { existing.push_back(op); });
    if (!existing.empty()) {
      module.emitError("memory planning cannot be applied twice");
      signalPassFailure();
      return;
    }
    func::FuncOp entry = entryFunction(module);
    OpBuilder builder(module.getContext());
    builder.setInsertionPointToStart(&entry.getBody().front());
    auto bind = [&](Value value, llvm::StringRef name, llvm::StringRef space,
                    llvm::StringRef role, const ByteRegion &region,
                    int64_t logicalCore) {
      createBinding(builder, module.getLoc(), value, name, space, role,
                    region.base, region.bytes, region.alignment, logicalCore);
    };
    bind(problem->activation, "activation", "lp6", "activation",
         layout->lp6Activation, -1);
    bind(problem->weight, "weight", "lp6", "weight", layout->lp6Weight,
         -1);
    bind(problem->activation, "activation", "shared_l2", "activation",
         layout->l2Activation, -1);
    bind(problem->weight, "weight", "shared_l2", "weight",
         layout->l2Weight, -1);
    bind(problem->output, "output", "shared_l2", "output",
         layout->l2Output, -1);
    for (uint32_t core = 0; core < config->logicalCores; ++core) {
      const std::string prefix = llvm::formatv("core_{0}.", core).str();
      bind(problem->activation, prefix + "activation_tile", "private_l1",
           "activation", layout->l1Activation, core);
      bind(problem->weight, prefix + "weight_tile", "private_l1", "weight",
           layout->l1Weight, core);
      bind(problem->output, prefix + "output_tile", "private_l1", "output",
           layout->l1Output, core);
    }
    module->setAttr("plena.memory_planned",
                    UnitAttr::get(module.getContext()));
  }
};

class TileMatmulPass
    : public mlir::plena::impl::TileMatmulBase<TileMatmulPass> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<plena_tile::PLENATileDialect>();
  }
  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (!module->hasAttr("plena.memory_planned")) {
      module.emitError("tiling requires planned memory");
      signalPassFailure();
      return;
    }
    auto config = readTargetConfig(module);
    auto problem = analyzeMatmul(module);
    if (!config) {
      module.emitError(llvm::toString(config.takeError()));
      signalPassFailure();
      return;
    }
    if (!problem) {
      module.emitError(llvm::toString(problem.takeError()));
      signalPassFailure();
      return;
    }
    auto tiles = tileMatmul(*problem, *config);
    if (!tiles) {
      module.emitError(llvm::toString(tiles.takeError()));
      signalPassFailure();
      return;
    }
    func::FuncOp entry = entryFunction(module);
    OpBuilder builder(module.getContext());
    builder.setInsertionPoint(entry.getBody().front().getTerminator());
    for (const MatmulTile &tile : *tiles)
      createTile(builder, module.getLoc(), *problem, tile);
    module->setAttr("plena.tiled", UnitAttr::get(module.getContext()));
  }
};

class ScheduleMatmulPass
    : public mlir::plena::impl::ScheduleMatmulBase<ScheduleMatmulPass> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<plena_sched::PLENAScheduleDialect>();
  }
  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (!module->hasAttr("plena.tiled")) {
      module.emitError("scheduling requires tiled input");
      signalPassFailure();
      return;
    }
    auto config = readTargetConfig(module);
    auto problem = analyzeMatmul(module);
    if (!config) {
      module.emitError(llvm::toString(config.takeError()));
      signalPassFailure();
      return;
    }
    if (!problem) {
      module.emitError(llvm::toString(problem.takeError()));
      signalPassFailure();
      return;
    }
    auto memory = planMemory(*problem, *config);
    if (!memory) {
      module.emitError(llvm::toString(memory.takeError()));
      signalPassFailure();
      return;
    }
    SmallVector<plena_tile::MatmulOp> tiles;
    module.walk([&](plena_tile::MatmulOp op) { tiles.push_back(op); });
    if (tiles.empty()) {
      module.emitError("tiled module contains no plena_tile.matmul");
      signalPassFailure();
      return;
    }
    func::FuncOp entry = entryFunction(module);
    OpBuilder builder(module.getContext());
    builder.setInsertionPoint(entry.getBody().front().getTerminator());
    auto activationReady = createScheduledGdma(
        builder, module.getLoc(), problem->activation, "load_activation",
        memory->lp6Activation.base, memory->l2Activation.base,
        memory->lp6Activation.bytes);
    auto weightReady = createScheduledGdma(
        builder, module.getLoc(), problem->weight, "load_weight",
        memory->lp6Weight.base, memory->l2Weight.base,
        memory->lp6Weight.bytes);
    for (plena_tile::MatmulOp tile : tiles)
      createScheduledCoreBlock(builder, module.getLoc(), *problem, *memory,
                               tile,
                               ValueRange{activationReady.getEvent(),
                                          weightReady.getEvent()});
    module->setAttr("plena.scheduled", UnitAttr::get(module.getContext()));
  }
};

class LowerToCommandsPass
    : public mlir::plena::impl::LowerToCommandsBase<LowerToCommandsPass> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<plena_cmd::PLENACommandDialect>();
  }
  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (!module->hasAttr("plena.scheduled")) {
      module.emitError("command lowering requires a scheduled module");
      signalPassFailure();
      return;
    }
    auto config = readTargetConfig(module);
    auto problem = analyzeMatmul(module);
    if (!config) {
      module.emitError(llvm::toString(config.takeError()));
      signalPassFailure();
      return;
    }
    if (!problem) {
      module.emitError(llvm::toString(problem.takeError()));
      signalPassFailure();
      return;
    }
    auto memory = planMemory(*problem, *config);
    auto tiles = tileMatmul(*problem, *config);
    if (!memory) {
      module.emitError(llvm::toString(memory.takeError()));
      signalPassFailure();
      return;
    }
    if (!tiles) {
      module.emitError(llvm::toString(tiles.takeError()));
      signalPassFailure();
      return;
    }

    llvm::DenseMap<Value, uint32_t> eventIDs;
    std::vector<ProgramRecord> records;
    uint32_t nextEvent = 0;
    func::FuncOp entry = entryFunction(module);
    for (Operation &operation : entry.getBody().front()) {
      if (auto gdma = dyn_cast<plena_sched::GdmaLoadOp>(operation)) {
        const uint32_t event = nextEvent++;
        eventIDs[gdma.getEvent()] = event;
        records.push_back(GdmaLoadRecord{
            gdma.getName().str(), event, {},
            static_cast<uint64_t>(gdma.getLp6Address()),
            static_cast<uint64_t>(gdma.getL2Address()),
            static_cast<uint32_t>(gdma.getRowBytes()),
            static_cast<uint32_t>(gdma.getRows()),
            static_cast<uint32_t>(gdma.getLp6Stride()),
            static_cast<uint32_t>(gdma.getL2Stride())});
        continue;
      }
      auto block = dyn_cast<plena_sched::CoreBlockOp>(operation);
      if (!block)
        continue;
      std::vector<uint32_t> dependencies;
      for (Value dependency : block.getWaitEvents()) {
        auto found = eventIDs.find(dependency);
        if (found == eventIDs.end()) {
          block.emitError("wait event does not dominate the core block");
          signalPassFailure();
          return;
        }
        dependencies.push_back(found->second);
      }
      const uint32_t event = nextEvent++;
      eventIDs[block.getEvent()] = event;
      MatmulTile tile{
          static_cast<uint32_t>(block.getTileId()),
          static_cast<uint32_t>(block.getLogicalCore()),
          static_cast<uint32_t>(block.getMOffset()),
          static_cast<uint32_t>(block.getNOffset()),
          static_cast<uint32_t>(block.getTileM()),
          static_cast<uint32_t>(block.getTileN()),
          static_cast<uint32_t>(block.getTotalK()),
          static_cast<uint32_t>(block.getKChunk()),
      };
      CoreISAEmitter emitter;
      auto words = emitter.emitMatmul(CoreBlockPlan{*problem, tile, *memory});
      if (!words) {
        block.emitError(llvm::toString(words.takeError()));
        signalPassFailure();
        return;
      }
      const uint64_t maxK = std::min<uint64_t>(tile.totalK, tile.kChunk);
      const uint64_t activationBytes = uint64_t{tile.m} * maxK * 2;
      const uint64_t weightBytes = maxK * tile.n * 2;
      const uint64_t outputBytes = uint64_t{tile.m} * tile.n * 2;
      records.push_back(CoreBlockRecord{
          block.getName().str(),
          event,
          std::move(dependencies),
          tile.logicalCore,
          static_cast<uint64_t>(block.getL1BytesRequired()),
          {{memory->l1Activation.base, activationBytes, config->alignment,
            kNoneU32},
           {memory->l1Weight.base, weightBytes, config->alignment, kNoneU32},
           {memory->l1Output.base, outputBytes, config->alignment, kNoneU32}},
          std::move(*words),
      });
    }
    if (records.size() != 2 + tiles->size() ||
        nextEvent > config->completionEventSlots) {
      module.emitError("scheduled command/event count is inconsistent");
      signalPassFailure();
      return;
    }

    const std::string modelManifest =
        buildModelManifestJSON(*problem, *config, *memory);
    const std::string l2Regions = buildL2RegionsJSON(*memory);
    const std::string compileReport =
        buildCompileReportJSON(*problem, *config, *memory, *tiles);

    module.getBody()->clear();
    OpBuilder builder(module.getContext());
    builder.setInsertionPointToStart(module.getBody());
    OperationState metadata(module.getLoc(),
                            plena_cmd::MetadataOp::getOperationName());
    metadata.addAttribute("model_manifest",
                          builder.getStringAttr(modelManifest));
    metadata.addAttribute("compile_report",
                          builder.getStringAttr(compileReport));
    metadata.addAttribute("l2_regions", builder.getStringAttr(l2Regions));
    addI64Attribute(metadata, builder, "logical_cores", config->logicalCores);
    addI64Attribute(metadata, builder, "l2_bytes_required",
                    memory->l2BytesRequired);
    builder.create(metadata);

    for (const ProgramRecord &record : records) {
      if (const auto *gdma = std::get_if<GdmaLoadRecord>(&record)) {
        OperationState state(module.getLoc(),
                             plena_cmd::GdmaLoadOp::getOperationName());
        state.addAttribute("name", builder.getStringAttr(gdma->name));
        addI64Attribute(state, builder, "event", gdma->event);
        state.addAttribute("dependencies",
                           builder.getDenseI32ArrayAttr(
                               signedWords(gdma->dependencies)));
        addI64Attribute(state, builder, "lp6_address", gdma->lp6Address);
        addI64Attribute(state, builder, "l2_address", gdma->l2Address);
        addI64Attribute(state, builder, "row_bytes", gdma->rowBytes);
        addI64Attribute(state, builder, "rows", gdma->rows);
        addI64Attribute(state, builder, "lp6_stride", gdma->lp6Stride);
        addI64Attribute(state, builder, "l2_stride", gdma->l2Stride);
        builder.create(state);
        continue;
      }
      const auto &blockRecord = std::get<CoreBlockRecord>(record);
      std::vector<int64_t> regions;
      for (const L1RegionRecord &region : blockRecord.l1Regions) {
        regions.push_back(static_cast<int64_t>(region.offset));
        regions.push_back(static_cast<int64_t>(region.bytes));
        regions.push_back(static_cast<int64_t>(region.alignment));
      }
      OperationState state(module.getLoc(),
                           plena_cmd::CoreBlockOp::getOperationName());
      state.addAttribute("name", builder.getStringAttr(blockRecord.name));
      addI64Attribute(state, builder, "event", blockRecord.event);
      state.addAttribute("dependencies",
                         builder.getDenseI32ArrayAttr(
                             signedWords(blockRecord.dependencies)));
      addI64Attribute(state, builder, "logical_core",
                      blockRecord.logicalCore);
      addI64Attribute(state, builder, "l1_bytes_required",
                      blockRecord.l1BytesRequired);
      state.addAttribute("l1_regions",
                         builder.getDenseI64ArrayAttr(regions));
      state.addAttribute("core_words",
                         builder.getDenseI32ArrayAttr(
                             signedWords(blockRecord.coreWords)));
      builder.create(state);
    }
  }
};

class EncodeProgramV7Pass
    : public mlir::plena::impl::EncodeProgramV7Base<EncodeProgramV7Pass> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<plena_isa::PLENAISADialect>();
  }
  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto metadata = dyn_cast_or_null<plena_cmd::MetadataOp>(
        module.getBody()->empty() ? nullptr
                                  : &module.getBody()->front());
    if (!metadata) {
      module.emitError("command module must begin with plena_cmd.metadata");
      signalPassFailure();
      return;
    }
    std::vector<ProgramRecord> records;
    for (Operation &operation : module.getBody()->getOperations()) {
      if (&operation == metadata.getOperation())
        continue;
      if (auto gdma = dyn_cast<plena_cmd::GdmaLoadOp>(operation)) {
        records.push_back(GdmaLoadRecord{
            gdma.getName().str(), static_cast<uint32_t>(gdma.getEvent()),
            unsignedWords(gdma.getDependencies()),
            static_cast<uint64_t>(gdma.getLp6Address()),
            static_cast<uint64_t>(gdma.getL2Address()),
            static_cast<uint32_t>(gdma.getRowBytes()),
            static_cast<uint32_t>(gdma.getRows()),
            static_cast<uint32_t>(gdma.getLp6Stride()),
            static_cast<uint32_t>(gdma.getL2Stride())});
        continue;
      }
      auto block = dyn_cast<plena_cmd::CoreBlockOp>(operation);
      if (!block) {
        operation.emitError("unexpected operation at command boundary");
        signalPassFailure();
        return;
      }
      llvm::ArrayRef<int64_t> flatRegions = block.getL1Regions();
      if (flatRegions.size() % 3 != 0) {
        block.emitError("flat L1 region array must contain triples");
        signalPassFailure();
        return;
      }
      std::vector<L1RegionRecord> regions;
      for (size_t index = 0; index < flatRegions.size(); index += 3)
        regions.push_back(
            {static_cast<uint64_t>(flatRegions[index]),
             static_cast<uint64_t>(flatRegions[index + 1]),
             static_cast<uint64_t>(flatRegions[index + 2]), kNoneU32});
      records.push_back(CoreBlockRecord{
          block.getName().str(), static_cast<uint32_t>(block.getEvent()),
          unsignedWords(block.getDependencies()),
          static_cast<uint32_t>(block.getLogicalCore()),
          static_cast<uint64_t>(block.getL1BytesRequired()),
          std::move(regions), unsignedWords(block.getCoreWords())});
    }
    auto config = readTargetConfig(module);
    if (!config) {
      module.emitError(llvm::toString(config.takeError()));
      signalPassFailure();
      return;
    }
    auto image = encodeProgramV7(records, config->completionEventSlots);
    if (!image) {
      module.emitError(llvm::toString(image.takeError()));
      signalPassFailure();
      return;
    }
    const std::string systemManifest = buildSystemManifestJSON(
        *image, records, config->physicalPlacement(),
        static_cast<uint64_t>(metadata.getL2BytesRequired()),
        metadata.getL2Regions());
    auto report =
        augmentCompileReportJSON(metadata.getCompileReport(), *image);
    if (!report) {
      module.emitError(llvm::toString(report.takeError()));
      signalPassFailure();
      return;
    }
    const std::string modelManifest = metadata.getModelManifest().str();

    module.getBody()->clear();
    OpBuilder builder(module.getContext());
    builder.setInsertionPointToStart(module.getBody());
    OperationState state(module.getLoc(),
                         plena_isa::ProgramOp::getOperationName());
    state.addAttribute("words",
                       builder.getDenseI32ArrayAttr(signedWords(image->words)));
    state.addAttribute("system_manifest",
                       builder.getStringAttr(systemManifest));
    state.addAttribute("model_manifest", builder.getStringAttr(modelManifest));
    state.addAttribute("compile_report", builder.getStringAttr(*report));
    builder.create(state);
  }
};

} // namespace

std::unique_ptr<Pass> mlir::plena::createVerifyMatmulPass() {
  return std::make_unique<VerifyMatmulPass>();
}

std::unique_ptr<Pass> mlir::plena::createPlanMemoryPass() {
  return std::make_unique<PlanMemoryPass>();
}

std::unique_ptr<Pass> mlir::plena::createTileMatmulPass() {
  return std::make_unique<TileMatmulPass>();
}

std::unique_ptr<Pass> mlir::plena::createScheduleMatmulPass() {
  return std::make_unique<ScheduleMatmulPass>();
}

std::unique_ptr<Pass> mlir::plena::createLowerToCommandsPass() {
  return std::make_unique<LowerToCommandsPass>();
}

std::unique_ptr<Pass> mlir::plena::createEncodeProgramV7Pass() {
  return std::make_unique<EncodeProgramV7Pass>();
}
