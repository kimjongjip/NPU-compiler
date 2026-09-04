#include "PLENA/Dialect/Tile/TileDialect.h"
#include "PLENA/Dialect/Tile/TileOps.h"

using namespace mlir;
using namespace mlir::plena_tile;

#include "PLENA/Dialect/Tile/TileOpsDialect.cpp.inc"

void PLENATileDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "PLENA/Dialect/Tile/TileOps.cpp.inc"
      >();
}
