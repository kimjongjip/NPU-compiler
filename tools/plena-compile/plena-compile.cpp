#include "PLENA/Dialect/Command/CommandDialect.h"
#include "PLENA/Dialect/ISA/ISADialect.h"
#include "PLENA/Dialect/ISA/ISAOps.h"
#include "PLENA/Dialect/Memory/MemoryDialect.h"
#include "PLENA/Dialect/Schedule/ScheduleDialect.h"
#include "PLENA/Dialect/Tile/TileDialect.h"
#include "PLENA/Target/Target.h"
#include "PLENA/Transforms/Passes.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

using namespace mlir;

namespace {

llvm::cl::opt<std::string> inputFilename(llvm::cl::Positional,
                                         llvm::cl::Required,
                                         llvm::cl::desc("input MLIR file"));
llvm::cl::opt<std::string> configFilename(
    "config", llvm::cl::Required,
    llvm::cl::desc("PLENA target JSON configuration"));
llvm::cl::opt<std::string> activationFilename(
    "activation-data", llvm::cl::Required,
    llvm::cl::desc("row-major little-endian FP16 activation payload"));
llvm::cl::opt<std::string> weightFilename(
    "weight-data", llvm::cl::Required,
    llvm::cl::desc("row-major little-endian FP16 [K,N] weight payload"));
llvm::cl::opt<std::string> outputDirectory(
    "output-dir", llvm::cl::Required,
    llvm::cl::desc("new output bundle directory"));

int reportError(llvm::Error error) {
  llvm::errs() << "plena-compile: " << llvm::toString(std::move(error))
               << '\n';
  return 1;
}

llvm::Expected<std::vector<uint8_t>> readBinary(llvm::StringRef path,
                                                uint64_t expectedBytes,
                                                llvm::StringRef role) {
  auto file = llvm::MemoryBuffer::getFile(path, /*IsText=*/false,
                                          /*RequiresNullTerminator=*/false);
  if (!file)
    return llvm::createStringError(file.getError(), "cannot read " + role +
                                                        " payload " + path);
  llvm::StringRef bytes = (*file)->getBuffer();
  if (static_cast<uint64_t>(bytes.size()) != expectedBytes)
    return llvm::createStringError(
        role + " payload size mismatch: expected " +
        llvm::Twine(expectedBytes) + " bytes, got " +
        llvm::Twine(bytes.size()));
  return std::vector<uint8_t>(bytes.bytes_begin(), bytes.bytes_end());
}

llvm::Error writeText(llvm::StringRef path, llvm::StringRef contents) {
  std::error_code error;
  llvm::raw_fd_ostream output(path, error, llvm::sys::fs::OF_None);
  if (error)
    return llvm::createStringError(error, "cannot create " + path);
  output << contents;
  if (!contents.ends_with("\n"))
    output << '\n';
  return llvm::Error::success();
}

llvm::Error writeBytes(llvm::StringRef path, llvm::ArrayRef<uint8_t> bytes) {
  std::error_code error;
  llvm::raw_fd_ostream output(path, error, llvm::sys::fs::OF_None);
  if (error)
    return llvm::createStringError(error, "cannot create " + path);
  output.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
  return llvm::Error::success();
}

llvm::Error writeWords(llvm::StringRef path, llvm::ArrayRef<int32_t> words) {
  std::error_code error;
  llvm::raw_fd_ostream output(path, error, llvm::sys::fs::OF_None);
  if (error)
    return llvm::createStringError(error, "cannot create " + path);
  for (int32_t signedWord : words) {
    const uint32_t word = static_cast<uint32_t>(signedWord);
    const char bytes[4] = {
        static_cast<char>(word), static_cast<char>(word >> 8),
        static_cast<char>(word >> 16), static_cast<char>(word >> 24)};
    output.write(bytes, sizeof(bytes));
  }
  return llvm::Error::success();
}

std::string printModule(ModuleOp module) {
  std::string result;
  llvm::raw_string_ostream stream(result);
  module.print(stream);
  stream << '\n';
  return result;
}

std::string outputPath(llvm::StringRef leaf) {
  llvm::SmallString<256> path(outputDirectory.getValue());
  llvm::sys::path::append(path, leaf);
  return path.str().str();
}

} // namespace

int main(int argc, char **argv) {
  llvm::InitLLVM init(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "PLENA basic MLIR compiler\n");

  auto config = ::plena::target::readTargetConfigFile(configFilename);
  if (!config)
    return reportError(config.takeError());

  DialectRegistry registry;
  registry.insert<arith::ArithDialect, func::FuncDialect,
                  linalg::LinalgDialect, memref::MemRefDialect,
                  plena_cmd::PLENACommandDialect,
                  plena_isa::PLENAISADialect,
                  plena_mem::PLENAMemoryDialect,
                  plena_sched::PLENAScheduleDialect,
                  plena_tile::PLENATileDialect>();
  MLIRContext context(registry, MLIRContext::Threading::DISABLED);
  ParserConfig parserConfig(&context);
  OwningOpRef<ModuleOp> module =
      parseSourceFile<ModuleOp>(inputFilename, parserConfig);
  if (!module) {
    llvm::errs() << "plena-compile: cannot parse input MLIR\n";
    return 1;
  }
  ::plena::target::attachTargetConfig(*module, *config);

  auto problem = ::plena::target::analyzeMatmul(*module);
  if (!problem)
    return reportError(problem.takeError());
  auto memory = ::plena::target::planMemory(*problem, *config);
  if (!memory)
    return reportError(memory.takeError());
  auto activation = readBinary(activationFilename,
                               memory->lp6Activation.bytes, "activation");
  if (!activation)
    return reportError(activation.takeError());
  auto weight =
      readBinary(weightFilename, memory->lp6Weight.bytes, "weight");
  if (!weight)
    return reportError(weight.takeError());

  PassManager planning(&context);
  planning.addPass(::mlir::plena::createVerifyMatmulPass());
  planning.addPass(::mlir::plena::createPlanMemoryPass());
  if (failed(planning.run(*module))) {
    llvm::errs() << "plena-compile: verification/memory planning failed\n";
    return 1;
  }
  const std::string planned = printModule(*module);

  PassManager tiling(&context);
  tiling.addPass(::mlir::plena::createTileMatmulPass());
  if (failed(tiling.run(*module))) {
    llvm::errs() << "plena-compile: 32x32 tiling failed\n";
    return 1;
  }
  const std::string tiled = printModule(*module);

  PassManager scheduling(&context);
  scheduling.addPass(::mlir::plena::createScheduleMatmulPass());
  if (failed(scheduling.run(*module))) {
    llvm::errs() << "plena-compile: logical-core scheduling failed\n";
    return 1;
  }
  const std::string scheduled = printModule(*module);

  PassManager commandSelection(&context);
  commandSelection.addPass(::mlir::plena::createLowerToCommandsPass());
  if (failed(commandSelection.run(*module))) {
    llvm::errs() << "plena-compile: command selection failed\n";
    return 1;
  }
  const std::string commands = printModule(*module);

  PassManager encoding(&context);
  encoding.addPass(::mlir::plena::createEncodeProgramV5Pass());
  if (failed(encoding.run(*module))) {
    llvm::errs() << "plena-compile: Program v5 encoding failed\n";
    return 1;
  }
  const std::string lowered = printModule(*module);

  if (!llvm::hasSingleElement(module->getBody()->getOperations())) {
    llvm::errs() << "plena-compile: lowering did not produce one ISA program\n";
    return 1;
  }
  auto program = dyn_cast<plena_isa::ProgramOp>(module->getBody()->front());
  if (!program) {
    llvm::errs() << "plena-compile: lowering produced no plena_isa.program\n";
    return 1;
  }

  if (llvm::sys::fs::exists(outputDirectory.getValue()))
    return reportError(llvm::createStringError(
        "output directory already exists: " + outputDirectory.getValue()));
  if (std::error_code error =
          llvm::sys::fs::create_directories(outputDirectory.getValue()))
    return reportError(llvm::createStringError(
        error, "cannot create output directory " + outputDirectory.getValue()));

  if (memory->lp6ImageBytes > std::numeric_limits<size_t>::max())
    return reportError(
        llvm::createStringError("LP6 image is too large for the host"));
  std::vector<uint8_t> lp6(static_cast<size_t>(memory->lp6ImageBytes), 0);
  std::copy(activation->begin(), activation->end(),
            lp6.begin() + memory->lp6Activation.base);
  std::copy(weight->begin(), weight->end(),
            lp6.begin() + memory->lp6Weight.base);

  if (llvm::Error error = writeWords(outputPath("program.bin"),
                                     program.getWords()))
    return reportError(std::move(error));
  if (llvm::Error error = writeText(outputPath("system.json"),
                                    program.getSystemManifest()))
    return reportError(std::move(error));
  if (llvm::Error error = writeText(outputPath("model_manifest.json"),
                                    program.getModelManifest()))
    return reportError(std::move(error));
  if (llvm::Error error = writeText(outputPath("compile_report.json"),
                                    program.getCompileReport()))
    return reportError(std::move(error));
  if (llvm::Error error = writeBytes(outputPath("lp6.bin"), lp6))
    return reportError(std::move(error));
  for (auto [name, contents] : {
           std::pair<llvm::StringRef, llvm::StringRef>{"planned.mlir",
                                                       planned},
           {"tiled.mlir", tiled},
           {"scheduled.mlir", scheduled},
           {"commands.mlir", commands},
           {"lowered.mlir", lowered},
       })
    if (llvm::Error error = writeText(outputPath(name), contents))
      return reportError(std::move(error));

  llvm::outs() << "compiled FP16 matmul " << problem->m << 'x' << problem->k
               << " x " << problem->k << 'x' << problem->n << " to "
               << program.getWords().size()
               << " unified Program v5 words across " << config->logicalCores
               << " logical core(s)\n";
  return 0;
}
