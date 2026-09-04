//===- TileDialect.td - PLENA tile dialect -------------------*- tablegen -*-===//

#ifndef PLENA_DIALECT_TILE_TILEDIALECT_TD
#define PLENA_DIALECT_TILE_TILEDIALECT_TD

include "mlir/IR/OpBase.td"

def PLENATile_Dialect : Dialect {
  let name = "plena_tile";
  let cppNamespace = "::mlir::plena_tile";
  let summary = "Physical 32x32 PLENA matrix tiles";
}

class PLENATile_Op<string mnemonic, list<Trait> traits = []>
    : Op<PLENATile_Dialect, mnemonic, traits>;

#endif // PLENA_DIALECT_TILE_TILEDIALECT_TD
