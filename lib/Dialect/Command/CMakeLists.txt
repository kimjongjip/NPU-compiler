add_mlir_dialect_library(MLIRPLENACommand
  CommandDialect.cpp
  CommandOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/PLENA/Dialect/Command

  DEPENDS
  MLIRCommandOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
)
