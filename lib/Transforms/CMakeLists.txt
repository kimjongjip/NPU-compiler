add_mlir_library(MLIRPLENATransforms
  MatmulPipeline.cpp
  Passes.cpp

  DEPENDS
  PLENAPassIncGen

  LINK_LIBS PUBLIC
  PLENATarget
  MLIRPLENACommand
  MLIRPLENAISA
  MLIRPLENAMemory
  MLIRPLENASchedule
  MLIRPLENATile
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRPass
  MLIRSupport
)
