//===- ISAOps.h - PLENA ISA operations ---------------*- C++ -*-===//

#ifndef PLENA_DIALECT_ISA_ISAOPS_H
#define PLENA_DIALECT_ISA_ISAOPS_H

#include "PLENA/Dialect/ISA/ISADialect.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"

#define GET_OP_CLASSES
#include "PLENA/Dialect/ISA/ISAOps.h.inc"

#endif // PLENA_DIALECT_ISA_ISAOPS_H
