#include "PLENA/Dialect/Graph/GraphOps.h"
#include "llvm/Support/JSON.h"
#include "llvm/ADT/DenseSet.h"
#include "mlir/IR/BuiltinOps.h"
#define GET_OP_CLASSES
#include "PLENA/Dialect/Graph/GraphOps.cpp.inc"

using namespace mlir;
using namespace mlir::plena_graph;

static LogicalResult verifyDescriptor(Operation *op) {
  auto attr = op->getAttrOfType<StringAttr>("descriptor");
  if (!attr)
    return op->emitOpError("requires a versioned descriptor");
  auto parsed = llvm::json::parse(attr.getValue());
  if (!parsed) {
    llvm::consumeError(parsed.takeError());
    return op->emitOpError("descriptor is not valid JSON");
  }
  auto module = op->getParentOfType<ModuleOp>();
  auto version = module->getAttrOfType<IntegerAttr>("plena.graph.version");
  if (!version || version.getInt() != 1)
    return op->emitOpError("requires plena.graph.version = 1");
  if (isa<ReturnOp>(op)) {
    auto *outputs = parsed->getAsArray();
    if (!outputs || outputs->size() != op->getNumOperands())
      return op->emitOpError("return descriptor/operand count mismatch");
    return success();
  }
  auto *object = parsed->getAsObject();
  if (!object)
    return op->emitOpError("descriptor must be an object");
  if (isa<BufferOp>(op)) {
    auto id = object->getInteger("id");
    auto address = object->getInteger("address");
    auto type = cast<MemRefType>(op->getResult(0).getType());
    if (!id || *id < 0 || !address || *address < 0 || !type.hasStaticShape())
      return op->emitOpError("invalid buffer ID/address/shape");
    for (int64_t dim : type.getShape())
      if (dim <= 0) return op->emitOpError("requires positive buffer dimensions");
    return success();
  }
  if (isa<TileOp, ScheduledTileOp>(op)) {
    auto kernel = object->getInteger("kernel");
    auto core = object->getInteger("core");
    if (!kernel || *kernel < 0 || !core || *core < 0)
      return op->emitOpError("invalid kernel/core ID");
    if (isa<ScheduledTileOp>(op)) {
      auto event = object->getInteger("event");
      auto *waits = object->getArray("wait_tiles");
      if (!event || *event < 0 || !waits)
        return op->emitOpError("scheduled tile lacks event dependencies");
      for (const auto &wait : *waits) {
        auto id = wait.getAsInteger();
        if (!id || *id < 0 || *id >= *event)
          return op->emitOpError("tile dependencies must dominate their consumer");
      }
    }
    return success();
  }
  auto kind = object->getString("kind");
  auto id = object->getInteger("id");
  if (!kind || op->getName().stripDialect() != *kind || !id || *id < 0 ||
      !object->getObject("output") || !object->getArray("shape"))
    return op->emitOpError("malformed kernel descriptor");
  for (Value operand : op->getOperands())
    if (!operand.getDefiningOp<BufferOp>())
      return op->emitOpError("kernel operands must be explicit graph buffers");
  return success();
}
LogicalResult BufferOp::verify() { return verifyDescriptor(*this); }
LogicalResult VectorOp::verify() { return verifyDescriptor(*this); }
LogicalResult MatmulOp::verify() { return verifyDescriptor(*this); }
LogicalResult CopyOp::verify() { return verifyDescriptor(*this); }
LogicalResult ReduceOp::verify() { return verifyDescriptor(*this); }
LogicalResult ReturnOp::verify() { return verifyDescriptor(*this); }
LogicalResult TileOp::verify() { return verifyDescriptor(*this); }
LogicalResult ScheduledTileOp::verify() { return verifyDescriptor(*this); }
