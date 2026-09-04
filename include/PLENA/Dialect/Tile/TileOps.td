//===- TileOps.td - PLENA tile operations --------------------*- tablegen -*-===//

#ifndef PLENA_DIALECT_TILE_TILEOPS_TD
#define PLENA_DIALECT_TILE_TILEOPS_TD

include "PLENA/Dialect/Tile/TileDialect.td"

def PLENATile_MatmulOp : PLENATile_Op<"matmul"> {
  let summary = "One output-stationary M/N output tile with temporal K";
  let arguments = (ins AnyMemRef:$activation,
                       AnyMemRef:$weight,
                       AnyMemRef:$output,
                       I64Attr:$tile_id,
                       I64Attr:$logical_core,
                       I64Attr:$m_offset,
                       I64Attr:$n_offset,
                       I64Attr:$tile_m,
                       I64Attr:$tile_n,
                       I64Attr:$total_k,
                       I64Attr:$k_chunk,
                       StrAttr:$mode);
}

#endif // PLENA_DIALECT_TILE_TILEOPS_TD
