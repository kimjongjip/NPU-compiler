//===- ISADialect.td - PLENA ISA container dialect -----------*- tablegen -*-===//

#ifndef PLENA_DIALECT_ISA_ISADIALECT_TD
#define PLENA_DIALECT_ISA_ISADIALECT_TD

include "mlir/IR/OpBase.td"

def PLENAISA_Dialect : Dialect {
  let name = "plena_isa";
  let cppNamespace = "::mlir::plena_isa";
  let summary = "Immutable PLENA unified Program v5 images";
}

class PLENAISA_Op<string mnemonic, list<Trait> traits = []>
    : Op<PLENAISA_Dialect, mnemonic, traits>;

#endif // PLENA_DIALECT_ISA_ISADIALECT_TD
