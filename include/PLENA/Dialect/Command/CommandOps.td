//===- CommandOps.td - PLENA command operations --------------*- tablegen -*-===//

#ifndef PLENA_DIALECT_COMMAND_COMMANDOPS_TD
#define PLENA_DIALECT_COMMAND_COMMANDOPS_TD

include "PLENA/Dialect/Command/CommandDialect.td"

def PLENACommand_MetadataOp : PLENACommand_Op<"metadata"> {
  let summary = "Bundle metadata carried to final Program v7 serialization";
  let arguments = (ins StrAttr:$model_manifest,
                       StrAttr:$compile_report,
                       StrAttr:$l2_regions,
                       I64Attr:$logical_cores,
                       I64Attr:$l2_bytes_required);
}

def PLENACommand_GdmaLoadOp : PLENACommand_Op<"gdma_load"> {
  let summary = "One encoded-shape system GDMA load command";
  let arguments = (ins StrAttr:$name,
                       I64Attr:$event,
                       DenseI32ArrayAttr:$dependencies,
                       I64Attr:$lp6_address,
                       I64Attr:$l2_address,
                       I64Attr:$row_bytes,
                       I64Attr:$rows,
                       I64Attr:$lp6_stride,
                       I64Attr:$l2_stride);
}

def PLENACommand_CoreBlockOp : PLENACommand_Op<"core_block"> {
  let summary = "One CORE_BEGIN/core-ISA/CORE_END command record";
  let arguments = (ins StrAttr:$name,
                       I64Attr:$event,
                       DenseI32ArrayAttr:$dependencies,
                       I64Attr:$logical_core,
                       I64Attr:$l1_bytes_required,
                       DenseI64ArrayAttr:$l1_regions,
                       DenseI32ArrayAttr:$core_words);
}

def PLENACommand_GdmaStoreOp : PLENACommand_Op<"gdma_store"> {
  let summary = "Explicit shared-L2 to LP6 spill/writeback";
  let arguments = (ins StrAttr:$name, I64Attr:$event,
                       DenseI32ArrayAttr:$dependencies,
                       I64Attr:$lp6_address, I64Attr:$l2_address,
                       I64Attr:$row_bytes, I64Attr:$rows,
                       I64Attr:$lp6_stride, I64Attr:$l2_stride);
}

#endif // PLENA_DIALECT_COMMAND_COMMANDOPS_TD
