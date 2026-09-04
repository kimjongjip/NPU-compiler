add_subdirectory(plena-opt)
add_subdirectory(plena-compile)

# Python model compiler.  The frontend/backend sources are owned by this
# repository; only the pinned torch/torch-mlir Python runtime is external.
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/plena-compile-model/plena-compile-model.in
  ${LLVM_RUNTIME_OUTPUT_INTDIR}/plena-compile-model
  @ONLY)
file(CHMOD ${LLVM_RUNTIME_OUTPUT_INTDIR}/plena-compile-model
  PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE
              WORLD_READ WORLD_EXECUTE)
