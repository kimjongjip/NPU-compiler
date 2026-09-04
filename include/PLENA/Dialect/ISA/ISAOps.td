//===- ISAOps.td - PLENA ISA container operations ------------*- tablegen -*-===//

#ifndef PLENA_DIALECT_ISA_ISAOPS_TD
#define PLENA_DIALECT_ISA_ISAOPS_TD

include "PLENA/Dialect/ISA/ISADialect.td"

def PLENAISA_ProgramOp : PLENAISA_Op<"program"> {
  let summary = "Hold one complete little-endian unified Program v5 image";
  let arguments = (ins DenseI32ArrayAttr:$words,
                       StrAttr:$system_manifest,
                       StrAttr:$model_manifest,
                       StrAttr:$compile_report);
}

#endif // PLENA_DIALECT_ISA_ISAOPS_TD
