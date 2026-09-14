"""Bind actual captured arguments/checkpoint bytes, without model templates."""

from __future__ import annotations

import numpy as np

from .graph import CompileError, DTYPES


def capture_bindings(loaded, capture):
    import torch
    from torch.utils._pytree import tree_flatten

    mapping = capture.parameter_mapping
    if not mapping.get("parameters_externalized") and mapping.get("bindings"):
        raise CompileError(
            "graph pipeline requires the externalized-state frontend ABI"
        )
    bindings = []
    static = []
    # FX importer embeds lifted constant tensors as MLIR literals; only
    # externalized state and ordinary user tensors remain function arguments.
    entries = sorted(
        (b for b in mapping["bindings"] if b["kind"].startswith("external_")),
        key=lambda x: x["argument_index"],
    )
    for i, entry in enumerate(entries):
        if entry["argument_index"] != i:
            raise CompileError("non-contiguous external-state argument mapping")
        name = entry["module_name"]
        if entry["kind"] == "external_parameter":
            t = loaded.module.get_parameter(name)
        elif entry["kind"] == "external_buffer":
            t = loaded.module.get_buffer(name)
            static.append(i)
        else:
            raise CompileError("unsupported captured state binding " + entry["kind"])
        bindings.append(t.detach().cpu().numpy())
    user_values, _ = tree_flatten((loaded.args, loaded.kwargs))
    for t in user_values:
        if not isinstance(t, torch.Tensor):
            raise CompileError("only tensor entry arguments are supported")
        # Discrete input IDs/masks are compile-time specialization inputs.
        # Floating inputs remain runtime, even if their data is available.
        if not t.is_floating_point():
            static.append(len(bindings))
        bindings.append(t.detach().cpu().numpy())
    return bindings, static


def read_outputs(target, report):
    data = (target / "l2_sram_dump.bin").read_bytes()
    values = []
    for out in report["outputs"]:
        if out["space"] != "l2":
            raise CompileError("unsupported output memory space")
        dtype = DTYPES[out["dtype"]]
        values.append(
            np.ndarray(
                tuple(out["shape"]),
                dtype=dtype,
                buffer=data,
                offset=out["address"] + out["offset_elements"] * dtype.itemsize,
                strides=tuple(s * dtype.itemsize for s in out["strides"]),
            ).copy()
        )
    return values
