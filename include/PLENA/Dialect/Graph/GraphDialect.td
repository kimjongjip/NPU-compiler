include "mlir/IR/OpBase.td"
def PLENAGraph_Dialect : Dialect {
  let name = "plena_graph";
  let cppNamespace = "::mlir::plena_graph";
  let summary = "Operation-driven static graph and tile planning IR";
}
class PLENAGraph_Op<string mnemonic> : Op<PLENAGraph_Dialect, mnemonic> {
  let hasVerifier = 1;
}

