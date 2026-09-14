include "PLENA/Dialect/Graph/GraphDialect.td"
def PLENAGraph_BufferOp : PLENAGraph_Op<"buffer"> {
  let arguments = (ins StrAttr:$descriptor);
  let results = (outs AnyMemRef:$buffer);
}
class PLENAGraph_KernelOp<string mnemonic> : PLENAGraph_Op<mnemonic> {
  let arguments = (ins Variadic<AnyMemRef>:$buffers, StrAttr:$descriptor);
}
def PLENAGraph_VectorOp : PLENAGraph_KernelOp<"vector">;
def PLENAGraph_MatmulOp : PLENAGraph_KernelOp<"matmul">;
def PLENAGraph_CopyOp : PLENAGraph_KernelOp<"copy">;
def PLENAGraph_ReduceOp : PLENAGraph_KernelOp<"reduce">;
def PLENAGraph_ReturnOp : PLENAGraph_KernelOp<"return">;
def PLENAGraph_TileOp : PLENAGraph_Op<"tile"> {
  let arguments = (ins StrAttr:$descriptor);
}
def PLENAGraph_ScheduledTileOp : PLENAGraph_Op<"scheduled_tile"> {
  let arguments = (ins StrAttr:$descriptor);
}

