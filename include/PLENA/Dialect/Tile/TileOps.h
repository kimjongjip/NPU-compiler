//===- TileOps.h - PLENA tile operations -------------*- C++ -*-===//

#ifndef PLENA_DIALECT_TILE_TILEOPS_H
#define PLENA_DIALECT_TILE_TILEOPS_H

#include "PLENA/Dialect/Tile/TileDialect.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"

#define GET_OP_CLASSES
#include "PLENA/Dialect/Tile/TileOps.h.inc"

#endif // PLENA_DIALECT_TILE_TILEOPS_H
