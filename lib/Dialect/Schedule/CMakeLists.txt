add_mlir_dialect_library(MLIRPLENASchedule
  ScheduleDialect.cpp
  ScheduleOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/PLENA/Dialect/Schedule

  DEPENDS
  MLIRScheduleOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
)
