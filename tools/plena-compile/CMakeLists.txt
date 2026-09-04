add_llvm_executable(plena-compile plena-compile.cpp)

llvm_update_compile_flags(plena-compile)
target_link_libraries(plena-compile PRIVATE
  PLENATarget
  MLIRPLENACommand
  MLIRPLENAISA
  MLIRPLENAMemory
  MLIRPLENASchedule
  MLIRPLENATile
  MLIRPLENATransforms
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRMemRefDialect
  MLIRParser
  MLIRPass
  MLIRSupport
  MLIRTransforms
  LLVMSupport
)

mlir_check_all_link_libraries(plena-compile)
