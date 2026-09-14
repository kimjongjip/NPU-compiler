set(PLENA_OPT_LIBS
  MLIRArithDialect
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRMemRefDialect
  MLIRPLENACommand
  MLIRPLENAISA
  MLIRPLENAGraph
  MLIRPLENAMemory
  MLIRPLENASchedule
  MLIRPLENATile
  MLIRPLENATransforms
  MLIROptLib
  MLIRTransforms
)

add_llvm_executable(plena-opt plena-opt.cpp)
llvm_update_compile_flags(plena-opt)
target_link_libraries(plena-opt PRIVATE ${PLENA_OPT_LIBS})
mlir_check_all_link_libraries(plena-opt)
