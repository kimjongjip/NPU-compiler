add_mlir_library(PLENATarget
  Target.cpp
  CoreISAEmitter.cpp
  ProgramV7.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/PLENA/Target

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRMemRefDialect
  MLIRSupport
  LLVMSupport
)
