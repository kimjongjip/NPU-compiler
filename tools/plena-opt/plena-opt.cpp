#include "PLENA/Dialect/Command/CommandDialect.h"
#include "PLENA/Dialect/ISA/ISADialect.h"
#include "PLENA/Dialect/Memory/MemoryDialect.h"
#include "PLENA/Dialect/Schedule/ScheduleDialect.h"
#include "PLENA/Dialect/Tile/TileDialect.h"
#include "PLENA/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

int main(int argc, char **argv) {
  mlir::registerTransformsPasses();
  mlir::plena::registerPLENACompilerPasses();

  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::linalg::LinalgDialect, mlir::memref::MemRefDialect,
                  mlir::plena_cmd::PLENACommandDialect,
                  mlir::plena_isa::PLENAISADialect,
                  mlir::plena_mem::PLENAMemoryDialect,
                  mlir::plena_sched::PLENAScheduleDialect,
                  mlir::plena_tile::PLENATileDialect>();
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "PLENA optimizer driver\n", registry));
}
