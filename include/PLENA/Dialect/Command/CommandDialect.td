//===- CommandDialect.td - PLENA command dialect -------------*- tablegen -*-===//

#ifndef PLENA_DIALECT_COMMAND_COMMANDDIALECT_TD
#define PLENA_DIALECT_COMMAND_COMMANDDIALECT_TD

include "mlir/IR/OpBase.td"

def PLENACommand_Dialect : Dialect {
  let name = "plena_cmd";
  let cppNamespace = "::mlir::plena_cmd";
  let summary = "Structured PLENA Program v5 command records";
}

class PLENACommand_Op<string mnemonic, list<Trait> traits = []>
    : Op<PLENACommand_Dialect, mnemonic, traits>;

#endif // PLENA_DIALECT_COMMAND_COMMANDDIALECT_TD
