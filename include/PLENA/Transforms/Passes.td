#ifndef PLENA_TRANSFORMS_PASSES_TD
#define PLENA_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def VerifyMatmul : Pass<"plena-verify-matmul", "ModuleOp"> {
  let summary = "Verify the basic static FP16 matmul input contract";
  let constructor = "mlir::plena::createVerifyMatmulPass()";
}

def PlanMemory : Pass<"plena-plan-memory", "ModuleOp"> {
  let summary = "Materialize byte-addressed LP6/shared-L2/private-L1 bindings";
  let constructor = "mlir::plena::createPlanMemoryPass()";
}

def TileMatmul : Pass<"plena-tile-matmul", "ModuleOp"> {
  let summary = "Tile output-stationary matmul for the 32x32 array and logical cores";
  let constructor = "mlir::plena::createTileMatmulPass()";
}

def ScheduleMatmul : Pass<"plena-schedule-matmul", "ModuleOp"> {
  let summary = "Create GDMA and logical-core event-SSA schedule operations";
  let constructor = "mlir::plena::createScheduleMatmulPass()";
}

def LowerToCommands : Pass<"plena-lower-to-commands", "ModuleOp"> {
  let summary = "Lower scheduled operations to structured PLENA command records";
  let constructor = "mlir::plena::createLowerToCommandsPass()";
}

def EncodeProgramV5 : Pass<"plena-encode-program-v5", "ModuleOp"> {
  let summary = "Encode command records into one unified PLENA Program v5";
  let constructor = "mlir::plena::createEncodeProgramV5Pass()";
}

#endif // PLENA_TRANSFORMS_PASSES_TD
