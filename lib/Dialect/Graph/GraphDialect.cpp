#include "PLENA/Dialect/Graph/GraphDialect.h"
#include "PLENA/Dialect/Graph/GraphOps.h"
using namespace mlir;
using namespace mlir::plena_graph;
#include "PLENA/Dialect/Graph/GraphOpsDialect.cpp.inc"
void PLENAGraphDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "PLENA/Dialect/Graph/GraphOps.cpp.inc"
  >();
}

