//===- ScheduleOps.td - PLENA schedule operations ------------*- tablegen -*-===//

#ifndef PLENA_DIALECT_SCHEDULE_SCHEDULEOPS_TD
#define PLENA_DIALECT_SCHEDULE_SCHEDULEOPS_TD

include "PLENA/Dialect/Schedule/ScheduleDialect.td"

def PLENASchedule_GdmaLoadOp : PLENASchedule_Op<"gdma_load"> {
  let summary = "One system-wide LP6 to shared-L2 transfer";
  let arguments = (ins AnyMemRef:$buffer,
                       StrAttr:$name,
                       I64Attr:$lp6_address,
                       I64Attr:$l2_address,
                       I64Attr:$row_bytes,
                       I64Attr:$rows,
                       I64Attr:$lp6_stride,
                       I64Attr:$l2_stride);
  let results = (outs I32:$event);
}

def PLENASchedule_CoreBlockOp : PLENASchedule_Op<"core_block"> {
  let summary = "One relocatable logical-core command block";
  let arguments = (ins AnyMemRef:$activation,
                       AnyMemRef:$weight,
                       AnyMemRef:$output,
                       Variadic<I32>:$wait_events,
                       StrAttr:$name,
                       I64Attr:$logical_core,
                       I64Attr:$tile_id,
                       I64Attr:$m_offset,
                       I64Attr:$n_offset,
                       I64Attr:$tile_m,
                       I64Attr:$tile_n,
                       I64Attr:$total_k,
                       I64Attr:$k_chunk,
                       I64Attr:$activation_l2,
                       I64Attr:$weight_l2,
                       I64Attr:$output_l2,
                       I64Attr:$activation_l1,
                       I64Attr:$weight_l1,
                       I64Attr:$output_l1,
                       I64Attr:$l1_bytes_required,
                       I64Attr:$activation_row_stride,
                       I64Attr:$weight_row_stride,
                       I64Attr:$output_row_stride,
                       StrAttr:$mode);
  let results = (outs I32:$event);
}

#endif // PLENA_DIALECT_SCHEDULE_SCHEDULEOPS_TD
