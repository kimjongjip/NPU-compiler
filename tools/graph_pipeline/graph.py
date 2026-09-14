"""Static Linalg legalization over real MLIR operations and SSA values.

Only explicit specialization inputs/buffers are constant-folded. Parameters
remain external memory, including embedding tables. No model class, layer
count, parameter-name pattern, or reference-model output drives lowering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np
from torch_mlir import ir


class CompileError(RuntimeError):
    pass


DTYPES = {
    "f16": np.dtype("<f2"),
    "f32": np.dtype("<f4"),
    "f64": np.dtype("<f8"),
    "i1": np.dtype("bool"),
    "i8": np.dtype("i1"),
    "i32": np.dtype("<i4"),
    "i64": np.dtype("<i8"),
    "index": np.dtype("<i8"),
}


def tensor_type(value):
    ty = ir.RankedTensorType(value.type)
    shape = tuple(ty.shape)
    if any(d <= 0 for d in shape):
        raise CompileError(f"positive static tensor shape required: {ty}")
    dt = str(ty.element_type)
    if dt not in DTYPES:
        raise CompileError(f"unsupported tensor type: {ty}")
    return shape, dt


def strides(shape):
    return tuple(math.prod(shape[i + 1 :]) for i in range(len(shape)))


@dataclass(eq=False)
class Buffer:
    id: int
    shape: tuple[int, ...]
    dtype: str
    data: Any = None
    space: str = "unplanned"
    address: int = 0
    first: int = -1
    last: int = -1

    @property
    def bytes(self):
        return math.prod(self.shape) * DTYPES[self.dtype].itemsize


@dataclass
class Tensor:
    shape: tuple[int, ...]
    dtype: str
    buffer: Buffer | None = None
    steps: tuple[int, ...] = ()
    offset: int = 0
    constant: Any = None

    def indices(self, coordinates):
        return self.offset + sum(c * s for c, s in zip(coordinates, self.steps))


@dataclass
class Expr:
    op: str
    dtype: str
    args: tuple = ()
    value: Any = None
    tensor: Tensor | None = None
    mapping: tuple = ()

    def sources(self):
        if self.tensor is not None and self.tensor.buffer is not None:
            yield self.tensor
        for a in self.args:
            yield from a.sources()


def const(value, dtype):
    return Expr("constant", dtype, value=np.asarray(value, dtype=DTYPES[dtype]))


def affine(expr, coordinates):
    if isinstance(expr, ir.AffineDimExpr):
        return coordinates[expr.position]
    if isinstance(expr, ir.AffineConstantExpr):
        return expr.value
    raise CompileError(f"unsupported indexing expression: {expr}")


def evaluate(expr, coordinates):
    """Evaluate compile-time-only expressions; never execute runtime tensors."""
    if expr.op == "constant":
        return expr.value
    if expr.op == "index":
        return coordinates[expr.value]
    if expr.op == "read":
        if expr.tensor.constant is None:
            raise CompileError("runtime value is not a compile-time constant")
        idx = tuple(affine(x, coordinates) for x in expr.mapping)
        return expr.tensor.constant[idx]
    if expr.op == "extract":
        if expr.tensor.constant is None:
            raise CompileError("runtime tensor.extract is not a constant")
        idx = tuple(evaluate(x, coordinates).astype(np.int64) for x in expr.args)
        return expr.tensor.constant[idx]
    args = [evaluate(x, coordinates) for x in expr.args]
    with np.errstate(all="ignore"):
        functions = {
            "add": np.add,
            "sub": np.subtract,
            "mul": np.multiply,
            "div": np.divide,
            "neg": np.negative,
            "max": np.maximum,
            "min": np.minimum,
            "exp": np.exp,
            "sqrt": np.sqrt,
            "rsqrt": lambda x: 1 / np.sqrt(x),
            "sin": np.sin,
            "cos": np.cos,
            "pow": np.power,
            "and": np.bitwise_and,
            "or": np.bitwise_or,
            "eq": np.equal,
            "ne": np.not_equal,
            "lt": np.less,
            "le": np.less_equal,
            "gt": np.greater,
            "ge": np.greater_equal,
            "select": np.where,
            "ugt": lambda a, b: np.greater(a, b) | np.isnan(a) | np.isnan(b),
            "cast": lambda x: x,
            "abs": np.abs,
        }
        if expr.op not in functions:
            raise CompileError(f"unsupported constant expression {expr.op}")
        return np.asarray(functions[expr.op](*args), dtype=DTYPES[expr.dtype])


@dataclass
class Kernel:
    kind: str
    output: Tensor
    inputs: list[Tensor] = field(default_factory=list)
    expr: Expr | None = None
    shape: tuple[int, ...] = ()
    reduction: str | None = None
    initial: float = 0
    origin: str = ""
    op: Any = None

    def sources(self):
        return self.inputs + (list(self.expr.sources()) if self.expr else [])


class LegalizeLinalgPass:
    name = "plena-legalize-linalg"

    def __init__(self, module, bindings, static_arguments=()):
        self.module = module
        self.values = {}
        self.buffers = []
        self.kernels = []
        self.outputs = []
        self.bindings = bindings
        self.static_arguments = set(static_arguments)
        self.current = None

    def new_tensor(self, shape, dtype, data=None):
        b = Buffer(len(self.buffers), shape, dtype, data)
        self.buffers.append(b)
        return Tensor(shape, dtype, b, strides(shape))

    def add(self, kernel):
        kernel.origin = self.current.operation.name
        kernel.op = self.current
        self.kernels.append(kernel)

    def materialize(self, tensor):
        if tensor.constant is not None:
            return self.new_tensor(tensor.shape, tensor.dtype, tensor.constant)
        return tensor

    def reshape(self, t, shape, reassociation, expand):
        if t.constant is not None:
            return Tensor(shape, t.dtype, constant=t.constant.reshape(shape))
        groups = [list(map(int, group)) for group in reassociation]
        if expand:
            steps = [0] * len(shape)
            for i, group in enumerate(groups):
                for j, dim in enumerate(group):
                    steps[dim] = t.steps[i] * math.prod(
                        shape[d] for d in group[j + 1 :]
                    )
        else:
            for group in groups:
                active = [d for d in group if t.shape[d] != 1]
                if any(
                    t.steps[a] != t.steps[b] * t.shape[b]
                    for a, b in zip(active, active[1:])
                ):
                    copy = self.new_tensor(t.shape, t.dtype)
                    self.add(Kernel("copy", copy, [t], shape=t.shape))
                    t = copy
                    break
            steps = [t.steps[g[-1]] for g in groups]
        return Tensor(shape, t.dtype, t.buffer, tuple(steps), t.offset)

    def scalar(self, value, local):
        if value in local:
            return local[value]
        if value in self.values and isinstance(self.values[value], Expr):
            return self.values[value]
        op = value.owner
        if not hasattr(op, "name"):
            raise CompileError(f"unbound scalar value {value}")
        name = op.name
        dt = str(value.type)
        if name == "arith.constant":
            return const(op.attributes["value"].value, dt)
        if name == "linalg.index":
            return Expr("index", "index", value=int(op.attributes["dim"].value))
        if name == "tensor.extract":
            t = self.values[op.operands[0]]
            idx = tuple(self.scalar(a, local) for a in list(op.operands)[1:])
            return Expr("extract", dt, idx, tensor=t)
        args = tuple(self.scalar(a, local) for a in op.operands)
        names = {
            "arith.addf": "add",
            "arith.addi": "add",
            "arith.subf": "sub",
            "arith.mulf": "mul",
            "arith.muli": "mul",
            "arith.divf": "div",
            "arith.negf": "neg",
            "arith.maximumf": "max",
            "arith.maxnumf": "max",
            "arith.minimumf": "min",
            "math.exp": "exp",
            "math.sqrt": "sqrt",
            "math.rsqrt": "rsqrt",
            "math.cos": "cos",
            "math.sin": "sin",
            "math.fpowi": "pow",
            "arith.andi": "and",
            "arith.ori": "or",
            "arith.select": "select",
            "arith.extf": "cast",
            "arith.truncf": "cast",
            "arith.index_cast": "cast",
            "arith.sitofp": "cast",
            "math.absf": "abs",
        }
        if name == "arith.cmpi":
            pred = int(op.attributes["predicate"].value)
            kind = {0: "eq", 1: "ne", 2: "lt", 3: "le", 4: "gt", 5: "ge"}.get(pred)
        elif name == "arith.cmpf":
            pred = int(op.attributes["predicate"].value)
            kind = {1: "eq", 2: "gt", 3: "ge", 4: "lt", 5: "le", 6: "ne", 9: "ugt"}.get(
                pred
            )
        else:
            kind = names.get(name)
        if kind is None:
            raise CompileError(f"unsupported scalar operation {name}")
        if (
            kind == "pow"
            and not list(args[1].sources())
            and np.all(evaluate(args[1], ()) == 2)
        ):
            return Expr("mul", dt, (args[0], args[0]))
        return Expr(kind, dt, args)

    def generic(self, op):
        rank = len(op.attributes["iterator_types"])
        maps = [tuple(a.value.results) for a in op.attributes["indexing_maps"]]
        block = op.regions[0].blocks[0]
        count_in = len(op.operands) - len(op.results)
        shape = [0] * rank
        local = {}
        for i, (arg, operand, mapping) in enumerate(
            zip(block.arguments, op.operands, maps)
        ):
            t = self.values[operand]
            for dim, size in zip(mapping, t.shape):
                if isinstance(dim, ir.AffineDimExpr):
                    shape[dim.position] = max(shape[dim.position], size)
            if i < count_in:
                local[arg] = Expr("read", t.dtype, tensor=t, mapping=mapping)
            else:
                local[arg] = Expr("accumulator", t.dtype)
        if any(d <= 0 for d in shape):
            raise CompileError("cannot infer static linalg iteration bounds")
        reduction_dims = [
            i
            for i, x in enumerate(op.attributes["iterator_types"])
            if "reduction" in str(x)
        ]
        allowed_scalar_ops = {
            "arith.constant",
            "arith.addf",
            "arith.addi",
            "arith.subf",
            "arith.mulf",
            "arith.muli",
            "arith.divf",
            "arith.negf",
            "arith.maximumf",
            "arith.maxnumf",
            "arith.minimumf",
            "arith.andi",
            "arith.ori",
            "arith.select",
            "arith.extf",
            "arith.truncf",
            "arith.index_cast",
            "arith.sitofp",
            "arith.cmpi",
            "arith.cmpf",
            "math.exp",
            "math.sqrt",
            "math.rsqrt",
            "math.cos",
            "math.sin",
            "math.fpowi",
            "math.absf",
            "tensor.extract",
            "linalg.index",
            "linalg.yield",
            "cf.assert",
        }
        for nested in block.operations:
            if nested.operation.name not in allowed_scalar_ops:
                raise CompileError(
                    "unsupported scalar region operation " + nested.operation.name
                )
            if nested.operation.name == "cf.assert":
                condition = self.scalar(nested.operands[0], local)
                if list(condition.sources()):
                    raise CompileError(
                        "runtime cf.assert requires an explicit assertion lowering"
                    )
                if not np.all(
                    evaluate(condition, tuple(np.indices(shape, sparse=True)))
                ):
                    raise CompileError("constant cf.assert failed in linalg region")
        terminator = list(block.operations)[-1]
        for result_index, result in enumerate(op.results):
            if not list(result.uses):
                continue  # e.g. dead softmax argmax indices, not a runtime result
            outshape, dtype = tensor_type(result)
            output_map = maps[count_in + result_index]
            output_dims = []
            for expr, extent in zip(output_map, outshape):
                if isinstance(expr, ir.AffineDimExpr):
                    output_dims.append(expr.position)
                elif not (
                    reduction_dims
                    and isinstance(expr, ir.AffineConstantExpr)
                    and expr.value == 0
                    and extent == 1
                ):
                    raise CompileError("unsupported generic output indexing map")
            if output_dims != list(range(rank - bool(reduction_dims))):
                raise CompileError(
                    "non-identity generic output indexing is not supported"
                )
            expression = self.scalar(terminator.operands[result_index], local)
            if reduction_dims:
                if reduction_dims != [rank - 1] or count_in != 1:
                    raise CompileError(
                        "only single trailing sum/max reductions are supported"
                    )
                if expression.op not in ("add", "max") or not any(
                    a.op == "accumulator" for a in expression.args
                ):
                    raise CompileError("unsupported reduction recurrence")
                source = next(a for a in expression.args if a.op != "accumulator")
                if source.op != "read":
                    raise CompileError("reduction source must be an explicit tensor")
                init = self.values[op.operands[count_in + result_index]]
                if init.constant is None or not np.all(
                    init.constant == init.constant.flat[0]
                ):
                    raise CompileError(
                        "reduction requires a uniform constant initializer"
                    )
                out = self.new_tensor(outshape, dtype)
                self.add(
                    Kernel(
                        "reduce",
                        out,
                        expr=source,
                        shape=tuple(shape),
                        reduction=expression.op,
                        initial=float(init.constant.flat[0]),
                    )
                )
                self.values[result] = out
                continue
            # Compile-time indices, causal masks and RoPE tables. Runtime
            # parameters never take this path, even when their bytes are known.
            if not list(expression.sources()):
                grid = tuple(np.indices(shape, sparse=True))
                data = np.broadcast_to(evaluate(expression, grid), shape).reshape(
                    outshape
                )
                self.values[result] = Tensor(outshape, dtype, constant=data)
                continue
            if expression.op == "read" and len(outshape) == len(shape):
                # A projected/broadcast identity region is a layout view, not
                # a runtime copy of an entire parameter matrix.
                t = expression.tensor
                steps = [0] * len(shape)
                offset = t.offset
                for dim, step in zip(expression.mapping, t.steps):
                    if isinstance(dim, ir.AffineDimExpr):
                        steps[dim.position] += step
                    else:
                        offset += dim.value * step
                self.values[result] = Tensor(
                    outshape, dtype, t.buffer, tuple(steps), offset
                )
                continue
            out = self.new_tensor(outshape, dtype)
            self.values[result] = out
            self.add(Kernel("vector", out, expr=expression, shape=tuple(shape)))

    def run(self):
        funcs = [
            o for o in self.module.body.operations if o.operation.name == "func.func"
        ]
        if len(funcs) != 1 or len(funcs[0].regions[0].blocks) != 1:
            raise CompileError("one single-block entry function is required")
        block = funcs[0].regions[0].blocks[0]
        if len(block.arguments) != len(self.bindings):
            raise CompileError("MLIR argument / checkpoint binding count mismatch")
        for i, (arg, data) in enumerate(zip(block.arguments, self.bindings)):
            shape, dt = tensor_type(arg)
            data = np.asarray(data)
            if data.shape != shape or data.dtype != DTYPES[dt]:
                raise CompileError(
                    f"argument {i}: expected {shape}/{dt}, got {data.shape}/{data.dtype}"
                )
            self.values[arg] = (
                Tensor(shape, dt, constant=data)
                if i in self.static_arguments
                else self.new_tensor(shape, dt, data)
            )
        for op in block.operations:
            self.current = op
            name = op.operation.name
            try:
                if name == "func.return":
                    self.outputs = [
                        self.materialize(self.values[x]) for x in op.operands
                    ]
                elif name == "arith.constant":
                    value = op.attributes["value"]
                    if isinstance(op.results[0].type, ir.RankedTensorType):
                        shape, dt = tensor_type(op.results[0])
                        data = np.array(value).astype(DTYPES[dt]).reshape(shape)
                        self.values[op.results[0]] = Tensor(shape, dt, constant=data)
                    else:
                        self.values[op.results[0]] = const(
                            value.value, str(op.results[0].type)
                        )
                elif name == "tensor.empty":
                    shape, dt = tensor_type(op.results[0])
                    self.values[op.results[0]] = Tensor(shape, dt)
                elif name == "linalg.fill":
                    shape, dt = tensor_type(op.results[0])
                    scalar = evaluate(self.values[op.operands[0]], ())
                    self.values[op.results[0]] = Tensor(
                        shape,
                        dt,
                        constant=np.broadcast_to(
                            np.asarray(scalar, dtype=DTYPES[dt]), shape
                        ),
                    )
                elif name == "linalg.generic":
                    self.generic(op)
                elif name in ("tensor.expand_shape", "tensor.collapse_shape"):
                    shape, dt = tensor_type(op.results[0])
                    self.values[op.results[0]] = self.reshape(
                        self.values[op.operands[0]],
                        shape,
                        op.attributes["reassociation"],
                        name == "tensor.expand_shape",
                    )
                elif name == "linalg.transpose":
                    t = self.values[op.operands[0]]
                    perm = tuple(op.attributes["permutation"])
                    shape = tuple(t.shape[i] for i in perm)
                    self.values[op.results[0]] = (
                        Tensor(shape, t.dtype, constant=t.constant.transpose(perm))
                        if t.constant is not None
                        else Tensor(
                            shape,
                            t.dtype,
                            t.buffer,
                            tuple(t.steps[i] for i in perm),
                            t.offset,
                        )
                    )
                elif name == "tensor.extract_slice":
                    t = self.values[op.operands[0]]
                    offsets = tuple(op.attributes["static_offsets"])
                    sizes = tuple(op.attributes["static_sizes"])
                    steps = tuple(op.attributes["static_strides"])
                    shape, dt = tensor_type(op.results[0])
                    if (
                        len(shape) != len(t.shape)
                        or any(x < 0 for x in offsets)
                        or any(s <= 0 for s in steps)
                    ):
                        raise CompileError("rank-changing/dynamic slice is unsupported")
                    if any(
                        o + (n - 1) * s >= d
                        for o, n, s, d in zip(offsets, sizes, steps, t.shape)
                    ):
                        raise CompileError("slice exceeds tensor bounds")
                    self.values[op.results[0]] = (
                        Tensor(
                            shape,
                            dt,
                            constant=t.constant[
                                tuple(
                                    slice(o, o + n * s, s)
                                    for o, n, s in zip(offsets, sizes, steps)
                                )
                            ],
                        )
                        if t.constant is not None
                        else Tensor(
                            shape,
                            dt,
                            t.buffer,
                            tuple(a * b for a, b in zip(t.steps, steps)),
                            t.offset + sum(a * b for a, b in zip(t.steps, offsets)),
                        )
                    )
                elif name == "tensor.concat":
                    tensors = [self.values[x] for x in op.operands]
                    dim = int(op.attributes["dim"].value)
                    shape, dt = tensor_type(op.results[0])
                    if all(t.constant is not None for t in tensors):
                        out = Tensor(
                            shape,
                            dt,
                            constant=np.concatenate(
                                [t.constant for t in tensors], axis=dim
                            ),
                        )
                    else:
                        out = self.new_tensor(shape, dt)
                        offset = 0
                        for t in tensors:
                            part = Tensor(
                                t.shape,
                                dt,
                                out.buffer,
                                out.steps,
                                offset * out.steps[dim],
                            )
                            self.add(
                                Kernel(
                                    "copy", part, [self.materialize(t)], shape=t.shape
                                )
                            )
                            offset += t.shape[dim]
                    self.values[op.results[0]] = out
                elif name in ("linalg.matmul", "linalg.batch_matmul"):
                    a, b, initial = [self.values[x] for x in op.operands]
                    shape, dt = tensor_type(op.results[0])
                    if initial.constant is None or np.any(initial.constant != 0):
                        raise CompileError(
                            "matmul currently requires a zero initializer"
                        )
                    if a.constant is not None and b.constant is not None:
                        # e.g. RoPE frequency outer product, never neural weights.
                        if math.prod(shape) > 1048576:
                            raise CompileError(
                                "compile-time matmul exceeds constant-folding budget"
                            )
                        out = Tensor(
                            shape,
                            dt,
                            constant=(
                                a.constant.astype(DTYPES[dt])
                                @ b.constant.astype(DTYPES[dt])
                            ),
                        )
                    else:
                        if (
                            a.dtype != "f16"
                            or b.dtype != "f16"
                            or dt not in ("f16", "f32")
                        ):
                            raise CompileError(
                                "runtime SA matmul requires FP16 operands and FP16/FP32 output"
                            )
                        out = self.new_tensor(shape, dt)
                        self.add(
                            Kernel(
                                "matmul",
                                out,
                                [self.materialize(a), self.materialize(b)],
                                shape=shape,
                            )
                        )
                    self.values[op.results[0]] = out
                else:
                    raise CompileError(f"unsupported operation: {name}")
            except (ValueError, KeyError, IndexError, TypeError) as e:
                raise CompileError(f"{name}: {e}") from e
        if not self.outputs:
            raise CompileError("entry has no tensor outputs")
        return self
