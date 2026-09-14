// WIP handoff: declarations only; not wired into the build or pass registry.
// See docs/RESUME_CPP_COMPILER.md before enabling the native graph migration.
#ifndef PLENA_NATIVE_GRAPH_H
#define PLENA_NATIVE_GRAPH_H
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/Error.h"
#include <memory>
namespace mlir { class Pass; }
namespace mlir::plena {
std::unique_ptr<Pass> createLegalizeLinalgGraphPass();
std::unique_ptr<Pass> createPlanGraphMemoryPass();
std::unique_ptr<Pass> createTileGraphPass();
std::unique_ptr<Pass> createScheduleGraphPass();
std::unique_ptr<Pass> createLowerGraphCommandsPass();
std::unique_ptr<Pass> createVerifyNativeGraphPass();
// Only serialization/IO belongs in the driver. Planning decisions are passes.
llvm::Error emitNativeGraphImage(ModuleOp module, llvm::StringRef path);
}
#endif
