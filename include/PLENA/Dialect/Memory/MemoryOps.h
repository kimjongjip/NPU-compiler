//===- MemoryOps.h - PLENA memory operations ----------*- C++ -*-===//

#ifndef PLENA_DIALECT_MEMORY_MEMORYOPS_H
#define PLENA_DIALECT_MEMORY_MEMORYOPS_H

#include "PLENA/Dialect/Memory/MemoryDialect.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"

#define GET_OP_CLASSES
#include "PLENA/Dialect/Memory/MemoryOps.h.inc"

#endif // PLENA_DIALECT_MEMORY_MEMORYOPS_H
