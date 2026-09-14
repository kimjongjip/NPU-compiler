add_mlir_dialect_library(MLIRPLENAGraph
  GraphDialect.cpp GraphOps.cpp
  DEPENDS MLIRGraphOpsIncGen
  LINK_LIBS PUBLIC MLIRIR MLIRSupport
)
