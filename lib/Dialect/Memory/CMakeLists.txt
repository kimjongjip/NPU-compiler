add_mlir_dialect_library(MLIRPLENAMemory
  MemoryDialect.cpp
  MemoryOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/PLENA/Dialect/Memory

  DEPENDS
  MLIRMemoryOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
)
