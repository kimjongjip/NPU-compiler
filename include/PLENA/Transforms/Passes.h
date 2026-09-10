#ifndef PLENA_TRANSFORMS_PASSES_H
#define PLENA_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"

#include <memory>

namespace mlir::plena {

std::unique_ptr<Pass> createVerifyMatmulPass();
std::unique_ptr<Pass> createPlanMemoryPass();
std::unique_ptr<Pass> createTileMatmulPass();
std::unique_ptr<Pass> createScheduleMatmulPass();
std::unique_ptr<Pass> createLowerToCommandsPass();
std::unique_ptr<Pass> createEncodeProgramV7Pass();

#define GEN_PASS_DECL
#include "PLENA/Transforms/Passes.h.inc"

void registerPLENACompilerPasses();

} // namespace mlir::plena

#endif // PLENA_TRANSFORMS_PASSES_H
