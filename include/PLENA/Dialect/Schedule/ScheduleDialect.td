//===- ScheduleDialect.td - PLENA schedule dialect -----------*- tablegen -*-===//

#ifndef PLENA_DIALECT_SCHEDULE_SCHEDULEDIALECT_TD
#define PLENA_DIALECT_SCHEDULE_SCHEDULEDIALECT_TD

include "mlir/IR/OpBase.td"

def PLENASchedule_Dialect : Dialect {
  let name = "plena_sched";
  let cppNamespace = "::mlir::plena_sched";
  let summary = "Logical-core and completion-event PLENA schedules";
}

class PLENASchedule_Op<string mnemonic, list<Trait> traits = []>
    : Op<PLENASchedule_Dialect, mnemonic, traits>;

#endif // PLENA_DIALECT_SCHEDULE_SCHEDULEDIALECT_TD
