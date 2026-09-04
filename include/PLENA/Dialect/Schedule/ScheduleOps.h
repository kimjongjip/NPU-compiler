//===- ScheduleOps.h - PLENA schedule operations -----*- C++ -*-===//

#ifndef PLENA_DIALECT_SCHEDULE_SCHEDULEOPS_H
#define PLENA_DIALECT_SCHEDULE_SCHEDULEOPS_H

#include "PLENA/Dialect/Schedule/ScheduleDialect.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"

#define GET_OP_CLASSES
#include "PLENA/Dialect/Schedule/ScheduleOps.h.inc"

#endif // PLENA_DIALECT_SCHEDULE_SCHEDULEOPS_H
