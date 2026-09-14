// WIP handoff: uncompiled internal design, not a stable compiler/IR API.
// See docs/RESUME_CPP_COMPILER.md (including the exception/error policy TODO).
#ifndef PLENA_NATIVE_GRAPH_INTERNAL_H
#define PLENA_NATIVE_GRAPH_INTERNAL_H
#include "PLENA/Transforms/NativeGraph.h"
#include "PLENA/Target/Target.h"
#include "PLENA/Target/ProgramV7.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/DenseMap.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
namespace plena::native {
using Shape = std::vector<int64_t>;
using namespace mlir;
struct Error : std::runtime_error { using std::runtime_error::runtime_error; };
inline void check(bool condition, const std::string &message) {
  if (!condition) throw Error(message);
}
inline int64_t count(const Shape &shape) {
  int64_t result=1;
  for (int64_t d:shape) {
    check(d>0 && result<=INT64_MAX/d, "invalid/overflowing static tensor shape");
    result*=d;
  }
  return result;
}
inline uint64_t align(uint64_t n) {
  check(n<=UINT64_MAX-63, "alignment overflow"); return (n+63)/64*64;
}
Shape strides(const Shape &shape);
Shape coords(int64_t flat, const Shape &shape);
unsigned bytes(Type type);
double roundTo(double value, Type type);
struct Tensor {
  Shape shape, step;
  Type type;
  int64_t buffer=-1, offset=0;
  std::shared_ptr<std::vector<double>> constant;
  int64_t index(const Shape &coordinates) const;
  double at(const Shape &coordinates) const;
};
struct Buffer {
  Shape shape;
  Type type;
  int64_t id=0, first=-1, last=-1;
  uint64_t address=0;
  std::string space="unplanned", file;
  // Explicit external-data or constant-view layout; no host inference.
  std::shared_ptr<Tensor> data;
  uint64_t size() const {return count(shape)*bytes(type);}
};
struct Expr {
  std::string op;
  Type type;
  double value=0;
  int64_t dimension=0;
  std::vector<std::shared_ptr<Expr>> args;
  std::shared_ptr<Tensor> tensor;
  // Nonnegative: loop dim. Negative: constant index -(value+1).
  Shape mapping;
};
using E = std::shared_ptr<Expr>;
struct Kernel {
  std::string kind, origin, reduction;
  Tensor output;
  std::vector<Tensor> inputs;
  E expression;
  Shape shape;
  double initial=0;
};
struct Tile {
  int64_t kernel=0, core=0, event=0;
  std::string kind;
  Shape row, batch, waits;
  int64_t start=0, elements=0, m=0,n=0,rows=0,columns=0,k=0;
  uint64_t l1Bytes=0;
};
struct Hardware {
  target::TargetConfig target;
  int64_t registerBits=512, registers=16, staging=65536;
  int64_t rfBytes() const {return registerBits/8;}
};
struct Graph {
  std::vector<Buffer> buffers;
  std::vector<Kernel> kernels;
  std::vector<Tensor> outputs;
  std::vector<Tile> tiles;
  uint64_t imageBytes=0,l2High=0;
  int64_t spills=0;
  Tensor allocate(Shape shape, Type type);
};
bool runtime(E expression);
double evaluate(E expression,const Shape &coordinates);
std::vector<Tensor> sources(const Kernel &kernel);
Graph legalize(ModuleOp module);
Hardware hardware(ModuleOp module);
void plan(Graph &g,const Hardware &hw);
void tile(Graph &g,const Hardware &hw);
void schedule(Graph &g);
Graph readGraph(ModuleOp module);
void writeGraph(ModuleOp module,const Graph &g,StringRef stage);
void verifyGraph(const Graph &g,const Hardware &hw,StringRef stage);
void lowerCommands(ModuleOp module,Graph &g,const Hardware &hw);
void writeImage(const Graph &g,StringRef path);
DictionaryAttr tensorAttr(Builder &b,const Tensor &t);
std::string reportJSON(const Graph &g,const Hardware &hw);
}
#endif
