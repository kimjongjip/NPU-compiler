cmake_minimum_required(VERSION 3.20)

project(PLENACompiler LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)

find_package(MLIR 24.0.0 REQUIRED CONFIG)

if(NOT LLVM_VERSION_MAJOR EQUAL 24)
  message(FATAL_ERROR
    "PLENA Compiler requires the LLVM/MLIR 24 toolchain used by torch-mlir; "
    "got ${LLVM_PACKAGE_VERSION} from ${LLVM_DIR}")
endif()

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

# LLVM 24 headers instantiate a large number of generated casting helpers that
# trigger this GCC-only warning even though project code is warning-clean.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  add_compile_options(-Wno-unused-but-set-parameter)
endif()

set(PLENA_SOURCE_DIR ${PROJECT_SOURCE_DIR})
set(PLENA_BINARY_DIR ${PROJECT_BINARY_DIR})

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PLENA_SOURCE_DIR}/include)
include_directories(${PLENA_BINARY_DIR}/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)
