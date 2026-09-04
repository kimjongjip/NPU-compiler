# PLENA compiler architecture

## Separation from the simulator

The compiler owns graph legalization, tiling, logical-core placement, memory
layout, DMA insertion, dependencies, and binary generation. The simulator owns
functional execution and resource-aware timing. The compiler never embeds the
Rust emulator or predicts a fixed cycle count in ISA.

```text
PLENA_Compiler                         PLENA_Simulator
------------------------------         -----------------------------
linalg/semantic graph                  Program v5 parser
  -> tile and core partition             -> central command processor
  -> LP6/L2/L1 plan                       -> GDMA / NoC / LDMA
  -> command/event schedule               -> private L1 / Matrix / VPU
  -> program.bin + manifests              -> functional result + cycles
```

## IR boundaries

### Input MLIR

The initial frontend accepts one standard buffer-form operation:

```mlir
%zero = arith.constant 0.0 : f16
linalg.fill ins(%zero : f16) outs(%output : memref<MxNxf16>)
linalg.matmul ins(%a, %b : memref<MxKxf16>, memref<KxNxf16>)
              outs(%output : memref<MxNxf16>)
```

The verifier rejects dynamic shapes, non-FP16 types, nonidentity layouts,
shape mismatches, missing zero fill, and multiple matmuls.

### `plena_mem`

`plena_mem.binding` keeps address space and byte location explicit:

```text
space        = lp6 | shared_l2 | private_l1
byte_base    = byte address/offset
size_bytes   = physical byte span
logical_core = -1 for shared, otherwise logical core ID
```

The current layout is:

```text
LP6: activation | alignment | weight
L2 : activation | alignment | weight | alignment | output
L1 : compact activation tile | weight tile | output tile
```

Each logical core sees the same L1 offsets but owns physically separate SRAM.

### `plena_tile`

One `plena_tile.matmul` represents one disjoint M/N output tile. M and N are
bounded by the 32x32 array; `total_k` remains temporal and `k_chunk` controls
how much of K is staged into L1 at a time.

N tiles are assigned in contiguous balanced groups. For 30 N tiles on four
cores, ownership is 8/8/7/7. Every core receives the full K domain, so no
cross-core partial-sum reduction is required.

### `plena_sched`

Schedule operations express dependencies as SSA values before numeric event
allocation:

```mlir
%a_ready = "plena_sched.gdma_load"(...) : (...) -> i32
%w_ready = "plena_sched.gdma_load"(...) : (...) -> i32
%done = "plena_sched.core_block"(..., %a_ready, %w_ready)
    {logical_core = 0 : i64, ...} : (...) -> i32
```

This keeps graph dependencies independent of a fixed scoreboard number. The
command lowering pass assigns numeric IDs only after block formation.

### `plena_cmd`

The command boundary contains structured records rather than one opaque word
slice:

- `plena_cmd.gdma_load`
- `plena_cmd.core_block`
- `plena_cmd.metadata`

A core block stores its target logical core, dependencies, L1 region list, and
the selected core-local ISA words. System structural words are not mixed into
the core word slice.

### `plena_isa`

`plena_isa.program` owns the immutable Program v5 word stream and the three
runtime/reporting documents. The driver writes each uint32 word little-endian.

## Matmul lowering

For each output tile `(m0, n0, Mt, Nt)` and each temporal K chunk:

```text
L2_LOAD_STRIDED activation[m0:m0+Mt, k0:k0+Kt] -> compact L1
L2_LOAD_STRIDED weight[k0:k0+Kt, n0:n0+Nt]      -> compact L1
C_SET_TILE_M/N/K
C_SET_MATRIX_*_STRIDE
M_LOAD_WEIGHT_F16
M_LOAD_ACT_F16
M_MMA_F16F16F32
```

All K chunks accumulate into one FP32 Matrix accumulator. The final chunk is
followed by:

```text
C_WAIT_MATRIX
M_WRITEOUT_F16 -> compact L1 output
L2_STORE_STRIDED -> disjoint shared-L2 output slice
```

This is output-stationary. K=96 with `k_chunk=64` therefore issues two MMAs
with K=64 and K=32, not a cross-core reduction.

## Program v5 mapping

The command sequence is:

```text
GDMA_LOAD activation -> event 0
GDMA_LOAD weight     -> event 1
CORE_BEGIN core=N wait=[0,1] signal=2+
    core-local LDMA/Matrix words
CORE_END
PROGRAM_END
```

`system.json` supplies the configuration's logical-to-physical placement;
the simulator-derived default is identity, while the checked 2-core profile
uses `[1,0]` to test relocation. Changing placement does not rewrite core-local
addresses because private L1 uses core-relative byte offsets.

## Configuration ownership

The target JSON contains compiler-relevant legality fields only. The helper
`tools/import_simulator_config.py` derives array shape, core count, L1/L2 size,
and event slots from `plena_settings.toml`; K chunking remains an explicit
compiler policy. Timing-only FU and bank latency parameters remain in the
simulator.
