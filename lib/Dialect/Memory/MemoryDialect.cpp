#include "PLENA/Dialect/Memory/MemoryDialect.h"
#include "PLENA/Dialect/Memory/MemoryOps.h"

using namespace mlir;
using namespace mlir::plena_mem;

#include "PLENA/Dialect/Memory/MemoryOpsDialect.cpp.inc"

void PLENAMemoryDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "PLENA/Dialect/Memory/MemoryOps.cpp.inc"
      >();
}
