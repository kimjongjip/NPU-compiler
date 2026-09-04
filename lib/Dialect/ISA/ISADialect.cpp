#include "PLENA/Dialect/ISA/ISADialect.h"
#include "PLENA/Dialect/ISA/ISAOps.h"

using namespace mlir;
using namespace mlir::plena_isa;

#include "PLENA/Dialect/ISA/ISAOpsDialect.cpp.inc"

void PLENAISADialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "PLENA/Dialect/ISA/ISAOps.cpp.inc"
      >();
}
