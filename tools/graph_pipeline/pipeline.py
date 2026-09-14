"""One pass pipeline: official Linalg IR -> graph -> memory -> tiles -> commands.

Graph/memory/tile passes are implemented using MLIR's Python IR bindings;
the final command verifier/encoder is the existing native C++ MLIR pass.
Every inter-pass artifact is reparsed and is the input to the next pass.
The registered graph operations use a versioned descriptor attribute and
explicit buffer operands. Their transformations are not yet native C++
Linalg rewrite patterns. This distinction is recorded in the bundle report.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
from torch_mlir import ir
from torch_mlir.passmanager import PassManager

from .graph import (
    Buffer,
    Tensor,
    Expr,
    Kernel,
    CompileError,
    LegalizeLinalgPass,
)
from .lower import (
    Hardware,
    Image,
    PlanMemoryPass,
    TilePass,
    SchedulePass,
    LowerCommandsPass,
)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def array_encode(data):
    a = np.ascontiguousarray(data)
    return dict(shape=list(data.shape), dtype=a.dtype.str, bytes=a.tobytes().hex())


def array_decode(data):
    return np.frombuffer(bytes.fromhex(data["bytes"]), dtype=data["dtype"]).reshape(
        data["shape"]
    )


def tensor_encode(t):
    return dict(
        shape=list(t.shape),
        dtype=t.dtype,
        buffer=t.buffer.id if t.buffer else None,
        steps=list(t.steps),
        offset=t.offset,
        constant=array_encode(t.constant) if t.constant is not None else None,
    )


def tensor_decode(d, buffers):
    return Tensor(
        tuple(d["shape"]),
        d["dtype"],
        buffers[d["buffer"]] if d["buffer"] is not None else None,
        tuple(d["steps"]),
        d["offset"],
        array_decode(d["constant"]) if d["constant"] else None,
    )


def expr_encode(e):
    if e is None:
        return None
    mapping = []
    for x in e.mapping:
        if isinstance(x, ir.AffineDimExpr):
            mapping.append(["dim", x.position])
        elif isinstance(x, ir.AffineConstantExpr):
            mapping.append(["constant", x.value])
        else:
            raise CompileError("non-projected indexing map")
    return dict(
        op=e.op,
        dtype=e.dtype,
        args=[expr_encode(a) for a in e.args],
        value=array_encode(e.value) if e.op == "constant" else e.value,
        tensor=tensor_encode(e.tensor) if e.tensor else None,
        mapping=mapping,
    )


def expr_decode(d, buffers):
    if d is None:
        return None
    return Expr(
        d["op"],
        d["dtype"],
        tuple(expr_decode(a, buffers) for a in d["args"]),
        array_decode(d["value"]) if d["op"] == "constant" else d["value"],
        tensor_decode(d["tensor"], buffers) if d["tensor"] else None,
        tuple(
            ir.AffineDimExpr.get(v) if k == "dim" else ir.AffineConstantExpr.get(v)
            for k, v in d["mapping"]
        ),
    )


def descriptor(data):
    # Versioned lowering payload; no executable Python, eval, or model names.
    return ir.StringAttr.get(json.dumps(data, separators=(",", ":"), allow_nan=False))


def graph_ir(graph, stage, tiles=None):
    m = ir.Module.create()
    values = {}
    with ir.InsertionPoint(m.body):
        for b in graph.buffers:
            ty = ir.Type.parse(
                "memref<" + "x".join([*(str(x) for x in b.shape), b.dtype]) + ">"
            )
            attrs = dict(
                id=b.id, space=b.space, address=b.address, first=b.first, last=b.last
            )
            op = ir.Operation.create(
                "plena_graph.buffer",
                results=[ty],
                attributes={"descriptor": descriptor(attrs)},
            )
            values[b.id] = op.results[0]
        for i, k in enumerate(graph.kernels):
            payload = dict(
                id=i,
                kind=k.kind,
                output=tensor_encode(k.output),
                inputs=[tensor_encode(t) for t in k.inputs],
                expression=expr_encode(k.expr),
                shape=list(k.shape),
                reduction=k.reduction,
                initial=str(k.initial),
                origin=k.origin,
            )
            ids = sorted({t.buffer.id for t in k.sources() + [k.output] if t.buffer})
            ir.Operation.create(
                "plena_graph." + k.kind,
                operands=[values[n] for n in ids],
                attributes={"descriptor": descriptor(payload)},
            )
        if tiles is not None:
            for t in tiles:
                ir.Operation.create(
                    "plena_graph."
                    + ("scheduled_tile" if stage == "scheduled" else "tile"),
                    attributes={"descriptor": descriptor(t)},
                )
        ir.Operation.create(
            "plena_graph.return",
            operands=[values[t.buffer.id] for t in graph.outputs],
            attributes={
                "descriptor": descriptor([tensor_encode(t) for t in graph.outputs])
            },
        )
    m.operation.attributes["plena.graph.version"] = ir.IntegerAttr.get(
        ir.IntegerType.get_signless(64), 1
    )
    m.operation.attributes["plena.graph.stage"] = ir.StringAttr.get(stage)
    return m


def read_graph(m, resources):
    buffers = []
    kernels = []
    outputs = []
    tiles = []
    ids = {}
    version = m.operation.attributes["plena.graph.version"].value
    if version != 1:
        raise CompileError("unsupported graph IR version")
    for op in m.body.operations:
        name = op.operation.name
        d = json.loads(op.attributes["descriptor"].value)
        if name == "plena_graph.buffer":
            if d["id"] != len(buffers):
                raise CompileError("noncanonical buffer ID")
            ty = ir.MemRefType(op.results[0].type)
            b = Buffer(
                d["id"],
                tuple(ty.shape),
                str(ty.element_type),
                resources.get(d["id"]),
                d["space"],
                d["address"],
                d["first"],
                d["last"],
            )
            ids[op.results[0]] = b.id
            buffers.append(b)
        elif name in ("plena_graph.tile", "plena_graph.scheduled_tile"):
            tiles.append(d)
        elif name == "plena_graph.return":
            outputs = [tensor_decode(t, buffers) for t in d]
            if [ids[x] for x in op.operands] != [t.buffer.id for t in outputs]:
                raise CompileError("return descriptor disagrees with SSA operands")
        elif name in (
            "plena_graph.vector",
            "plena_graph.copy",
            "plena_graph.reduce",
            "plena_graph.matmul",
        ):
            if d["id"] != len(kernels) or name != "plena_graph." + d["kind"]:
                raise CompileError("invalid kernel identity")
            k = Kernel(
                d["kind"],
                tensor_decode(d["output"], buffers),
                [tensor_decode(t, buffers) for t in d["inputs"]],
                expr_decode(d["expression"], buffers),
                tuple(d["shape"]),
                d["reduction"],
                float(d["initial"]),
                d["origin"],
            )
            expected = sorted(
                {t.buffer.id for t in k.sources() + [k.output] if t.buffer}
            )
            if [ids[x] for x in op.operands] != expected:
                raise CompileError("kernel descriptor disagrees with SSA operands")
            kernels.append(k)
        else:
            raise CompileError("unexpected graph IR operation " + name)
    if not outputs:
        raise CompileError("graph IR has no return")
    return SimpleNamespace(buffers=buffers, kernels=kernels, outputs=outputs), tiles


def checkpoint(module, path, optimizer=None):
    module.operation.verify()
    path.write_text(str(module) + "\n")
    reparsed = ir.Module.parse(path.read_text())
    reparsed.operation.verify()
    if optimizer is not None:
        result = subprocess.run(
            [
                str(optimizer),
                str(path),
                "--mlir-print-op-on-diagnostic=false",
                "-o",
                "/dev/null",
            ],
            text=True,
            capture_output=True,
        )
        if result.returncode:
            raise CompileError(
                "native graph IR verification failed:\n" + result.stderr[-3000:]
            )
    return reparsed


def command_ir(commands, words, hw, report, high_water):
    m = ir.Module.create()
    i64 = ir.IntegerType.get_signless(64)

    def integer(n):
        return ir.IntegerAttr.get(i64, int(n))

    attrs = {
        "array_rows": hw.rows,
        "array_columns": hw.columns,
        "physical_cores": hw.physical_cores,
        "logical_cores": hw.cores,
        "l1_bytes_per_core": hw.l1_bytes,
        "l2_bytes": hw.l2_bytes,
        "k_chunk": hw.k_chunk,
        "alignment": 64,
        "completion_event_slots": hw.event_slots,
        "isa_version": 7,
    }
    for k, v in attrs.items():
        m.operation.attributes["plena.target." + k] = integer(v)
    m.operation.attributes["plena.target.logical_to_physical"] = (
        ir.DenseI32ArrayAttr.get(hw.placement)
    )
    events = {str(c["id"]): i for i, c in enumerate(commands)}
    with ir.InsertionPoint(m.body):
        ir.Operation.create(
            "plena_cmd.metadata",
            attributes={
                "model_manifest": descriptor({"outputs": report["outputs"]}),
                "compile_report": descriptor(report),
                "l2_regions": descriptor(
                    [
                        dict(
                            name="graph_scratchpad",
                            byte_base=0,
                            size_bytes=high_water,
                            alignment=64,
                        )
                    ]
                ),
                "logical_cores": integer(hw.cores),
                "l2_bytes_required": integer(high_water),
            },
        )
        for i, c in enumerate(commands):
            attrs = {
                "name": ir.StringAttr.get(str(c["id"])),
                "event": integer(i),
                "dependencies": ir.DenseI32ArrayAttr.get(
                    [events[str(x)] for x in c.get("dependencies", [])]
                ),
            }
            if any(events[str(x)] >= i for x in c.get("dependencies", [])):
                raise CompileError("non-dominating command dependency")
            if c["kind"] == "core_block":
                start = int(c["isa_start_word"])
                count = int(c["isa_word_count"])
                attrs.update(
                    logical_core=integer(c["target_logical_core"]),
                    l1_bytes_required=integer(c["l1_bytes_required"]),
                    l1_regions=ir.DenseI64ArrayAttr.get([]),
                    core_words=ir.DenseI32ArrayAttr.get(
                        words[start : start + count].view(np.int32)
                    ),
                )
            else:
                for key in ("lp6_address", "l2_address"):
                    attrs[key] = integer(c[key])
                attrs.update(
                    row_bytes=integer(c["bytes"]),
                    rows=integer(c.get("rows", 1)),
                    lp6_stride=integer(c.get("lp6_stride", c["bytes"])),
                    l2_stride=integer(c.get("l2_stride", c["bytes"])),
                )
            ir.Operation.create("plena_cmd." + c["kind"], attributes=attrs)
    return m


def compile_linalg(
    text, bindings, output, hardware=None, static_arguments=(), optimizer=None
):
    """Compile the complete returned graph. Unsupported operations fail closed."""
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise CompileError("refusing to overwrite a nonempty compiler output directory")
    output.mkdir(parents=True, exist_ok=True)
    hw = hardware or Hardware()
    hw.validate()
    optimizer = Path(
        optimizer or Path(__file__).resolve().parents[2] / "build/bin/plena-opt"
    )
    passes = []
    image = Image(output / "lp6.bin")
    lower = None
    try:
        with ir.Context() as context, ir.Location.unknown():
            context.allow_unregistered_dialects = True
            m = ir.Module.parse(text)
            PassManager.parse("builtin.module(canonicalize,cse)").run(m.operation)
            m = checkpoint(m, output / "01-canonical.linalg.mlir")
            passes.append("canonicalize,cse")
            graph = LegalizeLinalgPass(m, bindings, static_arguments).run()
            resources = {b.id: b.data for b in graph.buffers if b.data is not None}
            m = checkpoint(
                graph_ir(graph, "legalized"), output / "02-legalized.mlir", optimizer
            )
            passes.append(LegalizeLinalgPass.name)
            graph, _ = read_graph(m, resources)
            plan = PlanMemoryPass(graph, hw, image).run()
            resources.update(
                {b.id: b.data for b in graph.buffers if b.data is not None}
            )
            m = checkpoint(
                graph_ir(graph, "planned"), output / "03-memory.mlir", optimizer
            )
            passes.append(PlanMemoryPass.name)
            graph, _ = read_graph(m, resources)
            tiles = TilePass(graph, hw).run().tiles
            m = checkpoint(
                graph_ir(graph, "tiled", tiles), output / "04-tiled.mlir", optimizer
            )
            passes.append(TilePass.name)
            graph, tiles = read_graph(m, resources)
            SchedulePass(tiles).run()
            m = checkpoint(
                graph_ir(graph, "scheduled", tiles),
                output / "05-scheduled.mlir",
                optimizer,
            )
            passes.append(SchedulePass.name)
            graph, tiles = read_graph(m, resources)
            lower = LowerCommandsPass(graph, hw, plan, tiles, output, image)
            commands, words = lower.run()
            passes.append(LowerCommandsPass.name)
            outputs = [
                dict(
                    shape=list(t.shape),
                    dtype=t.dtype,
                    space=t.buffer.space,
                    address=t.buffer.address,
                    offset_elements=t.offset,
                    strides=list(t.steps),
                )
                for t in graph.outputs
            ]
            report = dict(
                schema="plena.graph_compilation.v1",
                lowering="operation-driven Linalg MLIR passes",
                model_specialized_backend=False,
                passes=passes + ["plena-encode-program-v7"],
                pass_implementation="MLIR Python IR transformations + native C++ command encoder",
                graph_ir="experimental versioned descriptor operations with explicit buffer SSA operands",
                hardware=asdict(hw),
                kernels=len(graph.kernels),
                tiles=len(tiles),
                outputs=outputs,
                l2_high_water=plan.high_water,
                lp6_image_bytes=image.size,
                spill_buffers=plan.spills,
                stats=dict(lower.stats),
                numeric_contract={
                    "matrix": "FP16 multiply, FP32 accumulation, explicit output casts",
                    "vector": "MLIR scalar FP16/FP32 boundaries; division lowered to RCP + MUL",
                    "reduction": "RF-width partial tree, sequential FP32 combination; reassociated",
                },
            )
            cm = command_ir(commands, words, hw, report, plan.high_water)
            checkpoint(cm, output / "06-commands.mlir")
            image.stream.flush()
            result = subprocess.run(
                [
                    str(optimizer),
                    str(output / "06-commands.mlir"),
                    "--plena-encode-program-v7",
                    "--mlir-print-op-on-diagnostic=false",
                    "-o",
                    str(output / "07-isa.mlir"),
                ],
                text=True,
                capture_output=True,
            )
            if result.returncode:
                raise CompileError(
                    "native command encoding failed:\n" + result.stderr[-4000:]
                )
            encoded = ir.Module.parse((output / "07-isa.mlir").read_text())
            ops = list(encoded.body.operations)
            if len(ops) != 1 or ops[0].operation.name != "plena_isa.program":
                raise CompileError("native encoder did not produce one ISA program")
            program = ops[0]
            data = (
                np.asarray(list(program.attributes["words"]), dtype=np.int64)
                .astype("<u4")
                .tobytes()
            )
            (output / "program.bin").write_bytes(data)
            for attr, file in (
                ("system_manifest", "system.json"),
                ("model_manifest", "model.json"),
                ("compile_report", "compile_report.json"),
            ):
                (output / file).write_text(program.attributes[attr].value + "\n")
            write_json(
                output / "pass_pipeline.json",
                dict(
                    passes=report["passes"],
                    source_sha256=hashlib.sha256(text.encode()).hexdigest(),
                ),
            )
            return report
    finally:
        image.stream.close()
        if lower is not None:
            lower.program._core_isa_file.close()
            temp = output / lower.program._core_isa_name
            if temp.exists():
                temp.unlink()
