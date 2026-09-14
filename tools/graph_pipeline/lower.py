"""Memory planning, SIMD/SA tiling, event scheduling and command lowering.

The baseline streams parameters via LP6->L2->L1 and keeps live activations in
shared L2. L2 allocation is lifetime-aware; overflow buffers explicitly spill
to LP6. Only tile-sized operands/results occupy private L1. Independent output
tiles of a Matrix operation can execute on different logical cores.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
from .graph import CompileError, DTYPES, Tensor, Buffer, evaluate, affine, strides

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "full_model"))
from program_v7 import SystemProgramBuilder
from plena_isa_encoding import rform, load_u32, matrix_load, matrix_mma, matrix_writeout
from generic_vector_isa import typed_vector


def align(n, a=64):
    return (n + a - 1) // a * a


@dataclass
class Hardware:
    rows: int = 32
    columns: int = 32
    k_chunk: int = 64
    cores: int = 1
    physical_cores: int = 1
    placement: tuple = (0,)
    l1_bytes: int = 4 * 1024**2
    l2_bytes: int = 8 * 1024**2
    rf_bytes: int = 64
    registers: int = 16
    event_slots: int = 65536
    staging_bytes: int = 65536

    def validate(self):
        if (
            min(
                self.rows,
                self.columns,
                self.k_chunk,
                self.cores,
                self.l1_bytes,
                self.l2_bytes,
                self.rf_bytes,
                self.event_slots,
                self.staging_bytes,
            )
            <= 0
        ):
            raise CompileError("hardware sizes must be positive")
        if not 3 <= self.registers <= 16 or self.rf_bytes % 4:
            raise CompileError(
                "graph lowering requires 3..16 registers and RF width divisible by 32 bits"
            )
        if (
            len(self.placement) != self.cores
            or len(set(self.placement)) != self.cores
            or any(p < 0 or p >= self.physical_cores for p in self.placement)
        ):
            raise CompileError("invalid logical-to-physical placement")
        if self.staging_bytes * self.cores >= self.l2_bytes:
            raise CompileError("L2 must have space beyond per-core DMA staging")
        if self.staging_bytes % 64:
            raise CompileError("per-core L2 staging must be 64-byte aligned")
        if self.l1_bytes < align(self.rf_bytes) + 4:
            raise CompileError("L1 must fit one vector and reduction scalar state")


class Image:
    def __init__(self, path):
        self.stream = path.open("w+b")
        self.size = 0
        self.constants = {}

    def allocate(self, size, data=None):
        base = align(self.size)
        self.size = base + size
        self.stream.seek(base)
        if data is None:
            self.stream.truncate(self.size)
        else:
            self.stream.write(np.ascontiguousarray(data).tobytes())
        return base

    def constant(self, data):
        data = np.ascontiguousarray(data)
        key = (data.dtype.str, data.tobytes())
        if key not in self.constants:
            self.constants[key] = self.allocate(data.nbytes, data)
        return self.constants[key]


class PlanMemoryPass:
    name = "plena-plan-graph-memory"

    def __init__(self, graph, hardware, image):
        self.graph, self.hw, self.image = graph, hardware, image
        self.high_water = hardware.cores * hardware.staging_bytes
        self.spills = 0

    def run(self):
        g, h = self.graph, self.hw
        packed = {}
        for k in g.kernels:
            if k.kind != "matmul":
                continue
            for i, t in enumerate(k.inputs):
                b = t.buffer
                if b.data is None or t.steps[-1] == 1:
                    continue
                key = (b.id, t.shape, t.steps, t.offset)
                if key not in packed:
                    dtype = DTYPES[t.dtype]
                    original = np.ascontiguousarray(b.data)
                    view = np.ndarray(
                        t.shape,
                        dtype=dtype,
                        buffer=original,
                        offset=t.offset * dtype.itemsize,
                        strides=tuple(s * dtype.itemsize for s in t.steps),
                    )
                    # Compile-time parameter layout conversion, not matmul or
                    # activation evaluation. Source weights remain immutable.
                    nb = Buffer(
                        len(g.buffers), t.shape, t.dtype, np.ascontiguousarray(view)
                    )
                    g.buffers.append(nb)
                    packed[key] = Tensor(t.shape, t.dtype, nb, strides(t.shape))
                k.inputs[i] = packed[key]
        for i, k in enumerate(g.kernels):
            for t in k.sources() + [k.output]:
                if t.buffer is None:
                    continue
                b = t.buffer
                b.first = i if b.first < 0 else min(b.first, i)
                b.last = max(b.last, i)
        for t in g.outputs:
            t.buffer.last = len(g.kernels)
            if t.buffer.first < 0:
                t.buffer.first = 0
        active = []
        for b in sorted(g.buffers, key=lambda b: (b.first, b.id)):
            if b.last < 0:
                continue
            if b.dtype not in ("f16", "f32", "i8", "i32"):
                raise CompileError(
                    f"runtime buffer b{b.id} has unsupported dtype {b.dtype}"
                )
            if b.data is not None:
                b.space = "lp6"
                b.address = self.image.allocate(b.bytes, b.data)
                continue
            active = [a for a in active if a.last >= b.first]
            cursor = h.cores * h.staging_bytes
            for a in sorted(active, key=lambda x: x.address):
                if cursor + align(b.bytes) <= a.address:
                    break
                cursor = max(cursor, a.address + align(a.bytes))
            if cursor + align(b.bytes) <= h.l2_bytes:
                b.space = "l2"
                b.address = cursor
                active.append(b)
                self.high_water = max(self.high_water, cursor + align(b.bytes))
            else:
                b.space = "lp6"
                b.address = self.image.allocate(b.bytes)
                self.spills += 1
        # The bundle's observable ABI is a shared-L2 output, never an implicit
        # host-computed result. Do not publish an unobservable spilled output.
        if any(t.buffer.space != "l2" for t in g.outputs):
            raise CompileError(
                "returned tensors must fit shared L2; output staging is not yet supported"
            )
        return self


class TilePass:
    name = "plena-tile-graph"

    def __init__(self, graph, hardware):
        self.graph, self.hw = graph, hardware
        self.tiles = []

    def run(self):
        for i, k in enumerate(self.graph.kernels):
            if k.kind == "matmul":
                a, b = k.inputs
                m, kk, n = a.shape[-2], a.shape[-1], b.shape[-1]
                if b.shape[-2] != kk or a.shape[:-2] != b.shape[:-2]:
                    raise CompileError("incompatible batch matmul dimensions")
                for batch in np.ndindex(a.shape[:-2]):
                    for no in range(0, n, self.hw.columns):
                        core = (no // self.hw.columns) % self.hw.cores
                        for mo in range(0, m, self.hw.rows):
                            mm, nn = min(self.hw.rows, m - mo), min(
                                self.hw.columns, n - no
                            )
                            required = (
                                align(mm * kk * 2)
                                + align(kk * nn * 2)
                                + align(mm * nn * 4)
                            )
                            if required > self.hw.l1_bytes:
                                raise CompileError(
                                    "full-K Matrix tile exceeds L1: reduce spatial tile or add K spilling"
                                )
                            self.tiles.append(
                                dict(
                                    kernel=i,
                                    kind="matrix",
                                    core=core,
                                    batch=batch,
                                    m=mo,
                                    n=no,
                                    rows=mm,
                                    columns=nn,
                                    k=kk,
                                    l1_bytes=required,
                                )
                            )
            elif k.kind == "reduce":
                if k.output.dtype != "f32":
                    raise CompileError("reduction output must be FP32")
                for row in np.ndindex(k.shape[:-1]):
                    self.tiles.append(
                        dict(
                            kernel=i,
                            kind="reduce",
                            core=0,
                            row=row,
                            elements=k.shape[-1],
                        )
                    )
            else:
                # Bound both source and destination widths, including f16->f32.
                # f64 constants fold before code generation and don't consume RF.
                def element_width(e):
                    if e is None:
                        return 1
                    own = (
                        DTYPES[e.dtype].itemsize
                        if e.dtype in ("f16", "f32", "i8", "i32")
                        else 1
                    )
                    return max([own] + [element_width(a) for a in e.args])

                width = self.hw.rf_bytes // max(
                    DTYPES[k.output.dtype].itemsize,
                    element_width(k.expr),
                    *(DTYPES[t.dtype].itemsize for t in k.inputs),
                    1,
                )
                shape = k.shape
                for row in np.ndindex(shape[:-1]):
                    for start in range(0, shape[-1] if shape else 1, width):
                        self.tiles.append(
                            dict(
                                kernel=i,
                                kind="vector",
                                core=0,
                                row=row,
                                start=start,
                                elements=min(
                                    width, (shape[-1] if shape else 1) - start
                                ),
                            )
                        )
        return self


class SchedulePass:
    name = "plena-schedule-graph"

    def __init__(self, tiles):
        self.tiles = tiles

    def run(self):
        prior_kernel = []
        current = -1
        ends = {}
        for i, t in enumerate(self.tiles):
            if t["kernel"] != current:
                prior_kernel = list(ends.values()) or prior_kernel
                ends = {}
                current = t["kernel"]
            t["event"] = i
            t["wait_tiles"] = (
                [ends[t["core"]]] if t["core"] in ends else prior_kernel[:]
            )
            ends[t["core"]] = i
        return self


def expression_leaves(expr):
    # Constant subexpressions must fold before RF assignment (f64 epsilon etc.).
    if not list(expr.sources()) or expr.op in ("read", "extract"):
        yield expr
    else:
        for a in expr.args:
            yield from expression_leaves(a)


class LowerCommandsPass:
    name = "plena-lower-graph-to-commands"

    def __init__(self, graph, hardware, plan, tiles, output, image):
        self.g, self.h, self.plan, self.tiles, self.output, self.image = (
            graph,
            hardware,
            plan,
            tiles,
            output,
            image,
        )
        self.program = SystemProgramBuilder(
            output,
            l1_bytes_required=hardware.l1_bytes,
            l2_capacity_bytes=hardware.l2_bytes,
        )
        self.tile_events = {}
        self.stats = Counter()

    def emit(self, words):
        self.program.append(words)

    def gp(self, reg, value):
        if not 0 <= int(value) < 2**32:
            raise CompileError("core address/control value exceeds uint32")
        for w in load_u32(reg, int(value)):
            self.emit(w)

    def control(self, funct, value):
        self.gp(15, value)
        self.emit(rform(0x39, rd=15, funct=funct))

    def ldma(self, l1, l2, row_bytes, rows=1, stride=None, store=False):
        stride = row_bytes if stride is None else stride
        self.gp(0, l1)
        self.gp(1, l2)
        self.control(7, row_bytes)
        if rows > 1:
            self.control(0, rows)
            self.control(1, row_bytes if store else stride)
            self.control(2, stride if store else row_bytes)
        self.emit(
            rform(
                0x38, rd=0, rs1=1, funct=(5 if store else 4) if rows > 1 else int(store)
            )
        )
        self.stats["ldma_bytes"] += row_bytes * rows

    def transfer(self, space, address, l1, width, count, step, store=False):
        if step == 0 and count > 1:
            # Repeated source elements are explicitly copied, not free reads.
            for i in range(count):
                self.transfer(space, address, l1 + i * width, width, 1, width, store)
            return
        max_count = max(1, self.h.staging_bytes // width)
        for start in range(0, count, max_count):
            n = min(max_count, count - start)
            addr = address + start * step
            dst = l1 + start * width
            contiguous = step == width
            row_bytes = width * n if contiguous else width
            rows = 1 if contiguous else n
            stride = row_bytes if contiguous else step
            if space == "l2":
                self.ldma(dst, addr, row_bytes, rows, stride, store)
            else:
                stage = self.program.logical_core * self.h.staging_bytes
                if store:
                    self.ldma(dst, stage, width * n, store=True)
                    self.program.gdma_store(
                        lp6_address=addr,
                        l2_address=stage,
                        bytes=row_bytes,
                        rows=rows,
                        lp6_stride=stride,
                        l2_stride=row_bytes,
                    )
                else:
                    self.program.gdma_load(
                        lp6_address=addr,
                        l2_address=stage,
                        bytes=row_bytes,
                        rows=rows,
                        lp6_stride=stride,
                        l2_stride=row_bytes,
                    )
                    self.ldma(dst, stage, width * n)
                self.stats["gdma_bytes"] += width * n

    def access(self, tensor, indices, l1, store=False):
        width = DTYPES[tensor.dtype].itemsize
        array = np.asarray(indices, dtype=np.int64)
        idx = array.reshape(-1)
        if tensor.buffer is None:
            raise CompileError("attempt to access unallocated tensor")
        b = tensor.buffer
        if len(idx) == 0 or idx.min() < 0 or (idx.max() + 1) * width > b.bytes:
            raise CompileError(f"out-of-bounds access to buffer {b.id}")
        # Compact rectangular tiles use one 2-D descriptor, not a separate
        # GDMA for every K row. The physical byte stride remains explicit.
        if array.ndim == 2 and array.shape[1] * width <= self.h.staging_bytes:
            rows, columns = array.shape
            starts = array[:, 0]
            step = int(starts[1] - starts[0]) if rows > 1 else columns
            if (
                np.all(np.diff(array, axis=1) == 1)
                and step >= columns
                and (rows == 1 or np.all(np.diff(starts) == step))
            ):
                max_rows = max(1, self.h.staging_bytes // (columns * width))
                for start in range(0, rows, max_rows):
                    n = min(max_rows, rows - start)
                    size = columns * width
                    address = b.address + int(starts[start]) * width
                    local = l1 + start * size
                    if b.space == "l2":
                        self.ldma(local, address, size, n, step * width, store)
                    else:
                        stage = self.program.logical_core * self.h.staging_bytes
                        if store:
                            self.ldma(local, stage, size * n, store=True)
                            self.program.gdma_store(
                                lp6_address=address,
                                l2_address=stage,
                                bytes=size,
                                rows=n,
                                lp6_stride=step * width,
                                l2_stride=size,
                            )
                        else:
                            self.program.gdma_load(
                                lp6_address=address,
                                l2_address=stage,
                                bytes=size,
                                rows=n,
                                lp6_stride=step * width,
                                l2_stride=size,
                            )
                            self.ldma(local, stage, size * n)
                        self.stats["gdma_bytes"] += size * n
                return
        i = 0
        while i < len(idx):
            step = int(idx[i + 1] - idx[i]) if i + 1 < len(idx) else 1
            if step < 0:
                step = 1
                end = i + 1
            else:
                end = i + 1
                while end < len(idx) and int(idx[end] - idx[end - 1]) == step:
                    end += 1
            self.transfer(
                b.space,
                b.address + int(idx[i]) * width,
                l1 + i * width,
                width,
                end - i,
                step * width,
                store,
            )
            i = end

    def constant_to_l1(self, data, l1):
        data = np.ascontiguousarray(data)
        address = self.image.constant(data)
        self.transfer(
            "lp6", address, l1, data.dtype.itemsize, data.size, data.dtype.itemsize
        )

    def load_leaf(self, expr, coordinates, count, l1):
        if not list(expr.sources()):
            data = np.broadcast_to(evaluate(expr, coordinates), (count,))
            if expr.dtype not in ("f16", "f32", "i8", "i32"):
                raise CompileError(
                    f"constant {expr.dtype} is not legal at a runtime Vector boundary"
                )
            self.constant_to_l1(data, l1)
        elif expr.op == "read":
            coords = tuple(affine(x, coordinates) for x in expr.mapping)
            indices = np.broadcast_to(expr.tensor.indices(coords), (count,))
            self.access(expr.tensor, indices, l1)
        elif expr.op == "extract":
            if any(list(a.sources()) for a in expr.args):
                raise CompileError(
                    "runtime data-dependent tensor.extract requires a gather lowering"
                )
            coords = tuple(
                np.broadcast_to(evaluate(a, coordinates), (count,)).astype(np.int64)
                for a in expr.args
            )
            if any(
                np.any(c < 0) or np.any(c >= d)
                for c, d in zip(coords, expr.tensor.shape)
            ):
                raise CompileError("tensor.extract index outside dimension")
            self.access(expr.tensor, expr.tensor.indices(coords), l1)
        else:
            raise CompileError(f"cannot prepare expression {expr.op}")

    def vload(self, reg, l1, dtype, count):
        self.control(3, count)
        self.gp(3, l1)
        self.emit(
            rform(
                0x35,
                rd=reg,
                rs1=3,
                funct={"f16": 1, "f32": 3, "i8": 5, "i32": 7}[dtype],
            )
        )

    def vstore(self, reg, l1, dtype, count):
        self.control(3, count)
        self.gp(3, l1)
        self.emit(
            rform(
                0x35,
                rd=3,
                rs1=reg,
                funct={"f16": 2, "f32": 4, "i8": 6, "i32": 8}[dtype],
            )
        )

    def vector(self, k, tile):
        count = tile["elements"]
        coordinates = tuple(np.full(count, i, dtype=np.int64) for i in tile["row"])
        coordinates += (
            (np.arange(tile["start"], tile["start"] + count),) if k.shape else ()
        )
        if k.kind == "copy":
            t = k.inputs[0]
            self.access(t, t.indices(coordinates), 0)
            self.access(k.output, k.output.indices(coordinates), 0, True)
            return
        expr = k.expr
        leaves = list({id(a): a for a in expression_leaves(expr)}.values())
        addresses = {}
        cursor = 0
        # No RF value survives a GDMA-induced core-block boundary. Stage every
        # leaf first, then issue an uninterrupted register-resident expression.
        for leaf in leaves:
            addresses[id(leaf)] = cursor
            self.load_leaf(leaf, coordinates, count, cursor)
            cursor += align(count * DTYPES[leaf.dtype].itemsize)
        output_address = cursor
        if cursor + align(count * DTYPES[k.output.dtype].itemsize) > self.h.l1_bytes:
            raise CompileError("VPU expression staging exceeds L1")
        free = list(range(self.h.registers))
        live = {}
        uses = Counter()

        def count_uses(e):
            if id(e) in addresses:
                return
            for a in e.args:
                uses[id(a)] += 1
                count_uses(a)

        count_uses(expr)

        def alloc():
            if not free:
                raise CompileError(
                    "expression exceeds configured vector register count; spilling required"
                )
            return free.pop(0)

        def compile_expr(e):
            if id(e) in live:
                return live[id(e)]
            if id(e) in addresses:
                reg = alloc()
                self.vload(reg, addresses[id(e)], e.dtype, count)
                live[id(e)] = reg
                return reg
            regs = [compile_expr(a) for a in e.args]
            dt = e.args[0].dtype
            rd = alloc()

            def op(name, dtype=dt, r1=None, r2=0, r3=0, dst=rd):
                self.emit(
                    typed_vector(
                        name,
                        dtype.upper(),
                        dst,
                        regs[0] if r1 is None else r1,
                        r2,
                        r3,
                        elements=count,
                    )
                )

            binary = {
                "add": "ADD",
                "sub": "SUB",
                "mul": "MUL",
                "max": "MAX",
                "min": "MIN",
            }
            unary = {
                "neg": "NEG",
                "abs": "ABS",
                "exp": "EXP",
                "sqrt": "SQRT",
                "rsqrt": "RSQRT",
            }
            if e.op in binary:
                op(binary[e.op], r2=regs[1])
            elif e.op in unary:
                op(unary[e.op])
            elif e.op in ("eq", "lt", "le", "gt", "ge", "ugt"):
                compare = {
                    "eq": "CMP_EQ",
                    "lt": "CMP_LT",
                    "le": "CMP_LE",
                    "gt": "CMP_LT",
                    "ge": "CMP_LE",
                    "ugt": "CMP_LE",
                }[e.op]
                swap = e.op in ("gt", "ge")
                op(
                    compare,
                    r1=regs[1] if swap else regs[0],
                    r2=regs[0] if swap else regs[1],
                )
                if e.op == "ugt":
                    # Unordered greater-than = !(ordered less-or-equal).
                    # Preserve ReLU's NaN behavior instead of replacing it by
                    # a maxnum instruction with different NaN semantics.
                    one = alloc()
                    self.gp(3, 1)
                    self.emit(typed_vector("SPLAT", "I8", one, 3, elements=count))
                    self.emit(typed_vector("XOR", "I8", rd, rd, one, elements=count))
                    free.append(one)
            elif e.op == "select":
                op("SELECT", e.dtype, r1=regs[1], r2=regs[2], r3=regs[0])
            elif e.op == "cast":
                op("CAST_" + e.dtype.upper())
            elif e.op == "div":
                op("RCP", r1=regs[1])
                op("MUL", r2=rd)
            elif e.op == "pow":
                if np.all(evaluate(e.args[1], coordinates) == 2):
                    op("MUL", r2=regs[0])
                else:
                    raise CompileError(
                        "only square is currently lowered from runtime math.fpowi"
                    )
            else:
                raise CompileError(f"unsupported runtime scalar expression {e.op}")
            for a in e.args:
                uses[id(a)] -= 1
                if uses[id(a)] == 0:
                    free.append(live.pop(id(a)))
            live[id(e)] = rd
            return rd

        reg = compile_expr(expr)
        self.vstore(reg, output_address, k.output.dtype, count)
        self.access(k.output, k.output.indices(coordinates), output_address, True)
        self.stats["vector_tiles"] += 1

    def reduce(self, k, tile):
        count = tile["elements"]
        width = self.h.rf_bytes // DTYPES[k.expr.dtype].itemsize
        scalar_address = align(self.h.rf_bytes)
        self.constant_to_l1(np.array([k.initial], dtype=np.float32), scalar_address)
        for start in range(0, count, width):
            n = min(width, count - start)
            coords = tuple(np.full(n, i, dtype=np.int64) for i in tile["row"]) + (
                np.arange(start, start + n),
            )
            self.load_leaf(k.expr, coords, n, 0)
            self.vload(0, 0, k.expr.dtype, n)
            funct = (
                (4 if k.reduction == "add" else 5)
                if k.expr.dtype == "f32"
                else (1 if k.reduction == "add" else 2)
            )
            self.emit(rform(0x33, rd=0, rs1=0, funct=funct))
            self.gp(3, scalar_address)
            self.emit(rform(0x35, rd=1, rs1=3, funct=9))
            self.emit(
                rform(0x34, rd=0, rs1=1, rs2=0, funct=1 if k.reduction == "add" else 4)
            )
            self.emit(rform(0x35, rd=3, rs1=0, funct=10))
        flat = np.ravel_multi_index(tile["row"], k.shape[:-1]) if tile["row"] else 0
        coords = np.unravel_index(flat, k.output.shape) if k.output.shape else ()
        self.access(k.output, [k.output.indices(coords)], scalar_address, True)
        self.stats["reduction_rows"] += 1

    def matrix(self, k, tile):
        a, b = k.inputs
        m, n, kk = tile["rows"], tile["columns"], tile["k"]
        batch = tuple(tile["batch"])
        mo, no = tile["m"], tile["n"]
        ac = np.indices((m, kk))
        bc = np.indices((kk, n))
        ai = a.indices(batch + (ac[0] + mo, ac[1]))
        bi = b.indices(batch + (bc[0], bc[1] + no))
        wa = align(m * kk * 2)
        out = wa + align(kk * n * 2)
        self.access(a, ai, 0)
        self.access(b, bi, wa)
        for ko in range(0, kk, self.h.k_chunk):
            kc = min(self.h.k_chunk, kk - ko)
            self.gp(0, ko * 2)
            self.gp(1, wa + ko * n * 2)
            self.emit(matrix_load(1, kc, n, n * 2, 3))
            self.emit(matrix_load(0, m, kc, kk * 2, 7))
            self.emit(matrix_mma(m, n, kc, 3, accumulate=ko > 0))
        self.gp(2, out)
        self.emit(
            matrix_writeout(
                2,
                m,
                n,
                n * DTYPES[k.output.dtype].itemsize,
                4 if k.output.dtype == "f32" else 1,
            )
        )
        oc = np.indices((m, n))
        self.access(
            k.output, k.output.indices(batch + (oc[0] + mo, oc[1] + no)), out, True
        )
        self.stats["matrix_tiles"] += 1

    def run(self):
        for t in self.tiles:
            self.program._flush_core_block()
            self.program.logical_core = t["core"]
            self.program._frontier = [self.tile_events[i] for i in t["wait_tiles"]]
            before = len(self.program.commands)
            k = self.g.kernels[t["kernel"]]
            try:
                if t["kind"] == "matrix":
                    self.matrix(k, t)
                elif t["kind"] == "reduce":
                    self.reduce(k, t)
                else:
                    self.vector(k, t)
            except CompileError as e:
                raise CompileError(
                    f"kernel {t['kernel']} ({k.origin}), tile {t['event']}: {e}"
                ) from e
            end = self.program._flush_core_block()
            if end is None:
                end = self.program._frontier[-1]
            self.tile_events[t["event"]] = end
            t["first_command"] = before
            t["commands"] = len(self.program.commands) - before
        self.program._flush_core_block()
        self.program._core_isa_file.flush()
        words = np.fromfile(self.output / self.program._core_isa_name, dtype="<u4")
        return self.program.commands, words
