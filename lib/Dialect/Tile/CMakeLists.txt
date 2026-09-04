add_mlir_dialect_library(MLIRPLENATile
  TileDialect.cpp
  TileOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/PLENA/Dialect/Tile

  DEPENDS
  MLIRTileOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
)
