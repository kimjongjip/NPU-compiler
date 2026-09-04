#include "PLENA/Dialect/Schedule/ScheduleDialect.h"
#include "PLENA/Dialect/Schedule/ScheduleOps.h"

using namespace mlir;
using namespace mlir::plena_sched;

#include "PLENA/Dialect/Schedule/ScheduleOpsDialect.cpp.inc"

void PLENAScheduleDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "PLENA/Dialect/Schedule/ScheduleOps.cpp.inc"
      >();
}
