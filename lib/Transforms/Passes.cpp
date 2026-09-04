#include "PLENA/Transforms/Passes.h"

namespace mlir::plena {

#define GEN_PASS_REGISTRATION
#include "PLENA/Transforms/Passes.h.inc"

void registerPLENACompilerPasses() { registerPLENAPasses(); }

} // namespace mlir::plena
