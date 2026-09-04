//===- MemoryOps.td - PLENA hierarchical memory operations ---*- tablegen -*-===//

#ifndef PLENA_DIALECT_MEMORY_MEMORYOPS_TD
#define PLENA_DIALECT_MEMORY_MEMORYOPS_TD

include "PLENA/Dialect/Memory/MemoryDialect.td"

def PLENAMemory_BindingOp : PLENAMemory_Op<"binding"> {
  let summary = "Bind one tensor to an LP6, shared-L2, or private-L1 byte range";
  let arguments = (ins AnyMemRef:$buffer,
                       StrAttr:$name,
                       StrAttr:$space,
                       StrAttr:$role,
                       I64Attr:$byte_base,
                       I64Attr:$size_bytes,
                       I64Attr:$alignment,
                       I64Attr:$logical_core);
}

#endif // PLENA_DIALECT_MEMORY_MEMORYOPS_TD
