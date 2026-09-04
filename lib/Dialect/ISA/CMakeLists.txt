add_mlir_dialect_library(MLIRPLENAISA
  ISADialect.cpp
  ISAOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/PLENA/Dialect/ISA

  DEPENDS
  MLIRISAOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
)
