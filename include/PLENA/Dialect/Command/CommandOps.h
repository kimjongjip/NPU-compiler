//===- CommandOps.h - PLENA command operations -------*- C++ -*-===//

#ifndef PLENA_DIALECT_COMMAND_COMMANDOPS_H
#define PLENA_DIALECT_COMMAND_COMMANDOPS_H

#include "PLENA/Dialect/Command/CommandDialect.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"

#define GET_OP_CLASSES
#include "PLENA/Dialect/Command/CommandOps.h.inc"

#endif // PLENA_DIALECT_COMMAND_COMMANDOPS_H
