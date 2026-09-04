//===- MemoryDialect.td - PLENA hierarchical memory dialect --*- tablegen -*-===//

#ifndef PLENA_DIALECT_MEMORY_MEMORYDIALECT_TD
#define PLENA_DIALECT_MEMORY_MEMORYDIALECT_TD

include "mlir/IR/OpBase.td"

def PLENAMemory_Dialect : Dialect {
  let name = "plena_mem";
  let cppNamespace = "::mlir::plena_mem";
  let summary = "Byte-addressed PLENA LP6/L2/L1 memory plans";
}

class PLENAMemory_Op<string mnemonic, list<Trait> traits = []>
    : Op<PLENAMemory_Dialect, mnemonic, traits>;

#endif // PLENA_DIALECT_MEMORY_MEMORYDIALECT_TD
