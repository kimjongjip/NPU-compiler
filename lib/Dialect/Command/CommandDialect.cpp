#include "PLENA/Dialect/Command/CommandDialect.h"
#include "PLENA/Dialect/Command/CommandOps.h"

using namespace mlir;
using namespace mlir::plena_cmd;

#include "PLENA/Dialect/Command/CommandOpsDialect.cpp.inc"

void PLENACommandDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "PLENA/Dialect/Command/CommandOps.cpp.inc"
      >();
}
