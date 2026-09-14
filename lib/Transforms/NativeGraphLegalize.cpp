// WIP handoff: uncompiled legalization draft, deliberately excluded from CMake.
// This is NOT an executable native pass yet. See docs/RESUME_CPP_COMPILER.md.
#include "NativeGraphInternal.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Endian.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include <cstring>
#include <set>

using namespace plena::native;
using namespace mlir;

Shape plena::native::strides(const Shape &shape) {
  Shape result(shape.size()); int64_t step=1;
  for(int i=int(shape.size())-1;i>=0;--i){result[i]=step;step*=shape[i];}
  return result;
}
Shape plena::native::coords(int64_t flat,const Shape &shape) {
  Shape result(shape.size());
  for(int i=int(shape.size())-1;i>=0;--i){result[i]=flat%shape[i];flat/=shape[i];}
  check(flat==0,"coordinate outside shape");return result;
}
unsigned plena::native::bytes(Type t) {
  if(t.isIndex())return 8;
  if(auto f=dyn_cast<FloatType>(t))return f.getWidth()/8;
  if(auto i=dyn_cast<IntegerType>(t))return (i.getWidth()+7)/8;
  throw Error("unsupported element type");
}
double plena::native::roundTo(double v,Type t) {
  if(t.isF64())return v;
  if(t.isF32())return float(v);
  if(t.isF16()) {
    llvm::APFloat f(float(v)); bool loses=false;
    f.convert(llvm::APFloat::IEEEhalf(),llvm::APFloat::rmNearestTiesToEven,&loses);
    return f.convertToDouble();
  }
  check(std::isfinite(v) && std::abs(v)<=9007199254740991.,
        "compile-time integer exceeds exact supported range");
  if(t.isInteger(1))return v!=0;
  return std::trunc(v);
}
int64_t Tensor::index(const Shape &c) const {
  check(c.size()==shape.size() && step.size()==shape.size(),"tensor coordinate rank mismatch");
  int64_t result=offset;
  for(size_t i=0;i<c.size();++i){
    check(c[i]>=0 && c[i]<shape[i],"tensor index outside dimension");result+=c[i]*step[i];
  }
  return result;
}
double Tensor::at(const Shape &c) const {
  check(bool(constant),"runtime tensor cannot be evaluated on the host");
  int64_t i=index(c);check(i>=0 && size_t(i)<constant->size(),"constant view out of bounds");
  return (*constant)[i];
}
Tensor Graph::allocate(Shape shape,Type type) {
  count(shape);int64_t id=buffers.size();
  Buffer b;b.id=id;b.shape=shape;b.type=type;buffers.push_back(b);
  return Tensor{shape,strides(shape),type,id,0,nullptr};
}
bool plena::native::runtime(E e) {
  if(!e)return false;
  if(e->tensor && e->tensor->buffer>=0)return true;
  return std::any_of(e->args.begin(),e->args.end(),[](E a){return runtime(a);});
}
static Shape mapped(const Shape &mapping,const Shape &c) {
  Shape result;for(int64_t d:mapping){
    check(d<0 || size_t(d)<c.size(),"invalid indexing map dimension");
    result.push_back(d>=0?c[d]:-d-1);
  }return result;
}
double plena::native::evaluate(E e,const Shape &c) {
  check(bool(e),"missing scalar expression");
  if(e->op=="constant")return e->value;
  if(e->op=="index"){check(size_t(e->dimension)<c.size(),"invalid linalg.index");return c[e->dimension];}
  if(e->op=="read")return e->tensor->at(mapped(e->mapping,c));
  if(e->op=="extract"){
    Shape at;for(E a:e->args)at.push_back(int64_t(evaluate(a,c)));return e->tensor->at(at);
  }
  std::vector<double> a;for(E x:e->args)a.push_back(evaluate(x,c));
  check(!a.empty(),"unbound scalar accumulator");
  double v=0;auto op=e->op;
  if(op=="add")v=a[0]+a[1];else if(op=="sub")v=a[0]-a[1];
  else if(op=="mul")v=a[0]*a[1];else if(op=="div")v=a[0]/a[1];
  else if(op=="neg")v=-a[0];else if(op=="abs")v=std::abs(a[0]);
  else if(op=="max")v=std::isnan(a[0])?a[0]:std::max(a[0],a[1]);
  else if(op=="min")v=std::isnan(a[0])?a[0]:std::min(a[0],a[1]);
  else if(op=="exp")v=std::exp(a[0]);else if(op=="sqrt")v=std::sqrt(a[0]);
  else if(op=="rsqrt")v=1/std::sqrt(a[0]);else if(op=="sin")v=std::sin(a[0]);
  else if(op=="cos")v=std::cos(a[0]);else if(op=="cast")v=a[0];
  else if(op=="pow")v=std::pow(a[0],a[1]);
  else if(op=="and")v=int64_t(a[0])&int64_t(a[1]);else if(op=="or")v=int64_t(a[0])|int64_t(a[1]);
  else if(op=="eq")v=a[0]==a[1];else if(op=="ne")v=a[0]!=a[1];
  else if(op=="lt")v=a[0]<a[1];else if(op=="le")v=a[0]<=a[1];
  else if(op=="gt")v=a[0]>a[1];else if(op=="ge")v=a[0]>=a[1];
  else if(op=="ugt")v=!(a[0]<=a[1]);else if(op=="select")v=a[0]!=0?a[1]:a[2];
  else throw Error("unsupported constant expression "+op);
  return roundTo(v,e->type);
}
static void exprSources(E e,std::vector<Tensor>&out) {
  if(!e)return;if(e->tensor && e->tensor->buffer>=0)out.push_back(*e->tensor);
  for(E a:e->args)exprSources(a,out);
}
std::vector<Tensor> plena::native::sources(const Kernel &k) {
  auto out=k.inputs;exprSources(k.expression,out);return out;
}
static Shape shapeOf(Type type) {
  auto t=dyn_cast<ShapedType>(type);check(t && t.hasRank() && t.hasStaticShape(),"positive static shaped type required");
  Shape s(t.getShape().begin(),t.getShape().end());count(s);return s;
}
static Tensor emptyTensor(Type type) {
  auto s=shapeOf(type);return Tensor{s,strides(s),cast<ShapedType>(type).getElementType(),-1,0,nullptr};
}
static Tensor materializeConstant(Tensor t) {
  auto values=std::make_shared<std::vector<double>>();values->reserve(count(t.shape));
  for(int64_t i=0;i<count(t.shape);++i)values->push_back(t.at(coords(i,t.shape)));
  t.constant=values;t.offset=0;t.step=strides(t.shape);return t;
}
static double attrNumber(Attribute a) {
  if(auto f=dyn_cast<FloatAttr>(a))return f.getValueAsDouble();
  if(auto i=dyn_cast<IntegerAttr>(a)){
    if(i.getType().isInteger(1))return i.getInt()!=0;
    return roundTo(double(i.getInt()),i.getType());
  }throw Error("unsupported numeric constant attribute");
}
static Tensor denseConstant(Type type,DenseElementsAttr data) {
  Tensor t=emptyTensor(type);t.constant=std::make_shared<std::vector<double>>();
  for(Attribute a:data.getValues<Attribute>())t.constant->push_back(attrNumber(a));
  return t;
}
static Tensor fileConstant(Type type,const std::string &path) {
  Tensor t=emptyTensor(type);auto file=llvm::MemoryBuffer::getFile(path);
  check(bool(file),"cannot read input resource "+path);
  auto raw=(*file)->getBuffer();unsigned w=bytes(t.type);
  check(raw.size()==uint64_t(count(t.shape))*w,"input resource byte count mismatch: "+path);
  t.constant=std::make_shared<std::vector<double>>();
  for(int64_t i=0;i<count(t.shape);++i){
    const auto *p=reinterpret_cast<const uint8_t*>(raw.data()+i*w);double v=0;
    if(t.type.isF16())v=llvm::APFloat(llvm::APFloat::IEEEhalf(),llvm::APInt(16,llvm::support::endian::read16le(p))).convertToDouble();
    else if(t.type.isF32()){uint32_t bits=llvm::support::endian::read32le(p);float f;std::memcpy(&f,&bits,4);v=f;}
    else if(t.type.isF64()){uint64_t bits=llvm::support::endian::read64le(p);std::memcpy(&v,&bits,8);}
    else if(w==8)v=double(int64_t(llvm::support::endian::read64le(p)));
    else if(w==4)v=int32_t(llvm::support::endian::read32le(p));
    else if(w==1)v=t.type.isInteger(1)?bool(*p):int8_t(*p);
    else throw Error("unsupported input constant storage type");
    t.constant->push_back(roundTo(v,t.type));
  }return t;
}
namespace {
class Legalizer {
  ModuleOp module;
  Graph g;
  llvm::DenseMap<Value,Tensor> tensors;
  llvm::DenseMap<Value,E> scalars;
  Operation *current=nullptr;
  Tensor get(Value v){check(tensors.count(v),"unbound tensor SSA value");return tensors.lookup(v);}
  Tensor allocate(Type t){Tensor out=emptyTensor(t);return g.allocate(out.shape,out.type);}
  void add(Kernel k){k.origin=current->getName().getStringRef().str();g.kernels.push_back(std::move(k));}
  Tensor materialize(Tensor t) {
    if(!t.constant)return t;
    Tensor out=g.allocate(t.shape,t.type);g.buffers[out.buffer].data=std::make_shared<Tensor>(materializeConstant(t));return out;
  }
  E scalar(Value v,llvm::DenseMap<Value,E>&local) {
    if(local.count(v))return local.lookup(v);if(scalars.count(v))return scalars.lookup(v);
    Operation *op=v.getDefiningOp();check(op,"unbound scalar block argument");
    auto e=std::make_shared<Expr>();e->type=v.getType();std::string name=op->getName().getStringRef().str();
    if(name=="arith.constant"){e->op="constant";e->value=attrNumber(op->getAttr("value"));return e;}
    if(name=="linalg.index"){e->op="index";e->dimension=op->getAttrOfType<IntegerAttr>("dim").getInt();return e;}
    if(name=="tensor.extract"){
      e->op="extract";e->tensor=std::make_shared<Tensor>(get(op->getOperand(0)));
      for(Value a:op->getOperands().drop_front())e->args.push_back(scalar(a,local));return e;
    }
    for(Value a:op->getOperands())e->args.push_back(scalar(a,local));
    static const std::map<std::string,std::string> names={
      {"arith.addf","add"},{"arith.addi","add"},{"arith.subf","sub"},{"arith.mulf","mul"},
      {"arith.muli","mul"},{"arith.divf","div"},{"arith.negf","neg"},{"arith.maximumf","max"},
      {"arith.maxnumf","max"},{"arith.minimumf","min"},{"arith.andi","and"},{"arith.ori","or"},
      {"arith.select","select"},{"arith.extf","cast"},{"arith.truncf","cast"},{"arith.index_cast","cast"},
      {"arith.sitofp","cast"},{"math.exp","exp"},{"math.sqrt","sqrt"},{"math.rsqrt","rsqrt"},
      {"math.cos","cos"},{"math.sin","sin"},{"math.fpowi","pow"},{"math.absf","abs"}};
    if(name=="arith.cmpi" || name=="arith.cmpf"){
      int p=op->getAttrOfType<IntegerAttr>("predicate").getInt();
      std::map<int,std::string> n=name=="arith.cmpi"?std::map<int,std::string>{{0,"eq"},{1,"ne"},{2,"lt"},{3,"le"},{4,"gt"},{5,"ge"}}:
          std::map<int,std::string>{{1,"eq"},{2,"gt"},{3,"ge"},{4,"lt"},{5,"le"},{6,"ne"},{9,"ugt"}};
      check(n.count(p),"unsupported comparison predicate");e->op=n[p];
    }else{check(names.count(name),"unsupported scalar operation "+name);e->op=names.at(name);}
    if(e->op=="pow" && !runtime(e->args[1]) && evaluate(e->args[1],{})==2){e->op="mul";e->args[1]=e->args[0];}
    local[v]=e;return e;
  }
  void generic(linalg::GenericOp op) {
    // Official Linalg interfaces own indexing maps/iterator definitions.
    auto maps=op.getIndexingMapsArray();unsigned rank=op.getNumLoops();
    unsigned inputs=op.getNumDpsInputs();Shape shape(rank,0);std::vector<Shape> mapping;
    llvm::DenseMap<Value,E> local;Block &body=op.getRegion().front();
    for(auto [i,operand]:llvm::enumerate(op->getOperands())){
      Tensor t=get(operand);Shape map;
      for(auto [j,a]:llvm::enumerate(maps[i].getResults())){
        if(auto d=dyn_cast<AffineDimExpr>(a)){map.push_back(d.getPosition());shape[d.getPosition()]=std::max(shape[d.getPosition()],t.shape[j]);}
        else if(auto c=dyn_cast<AffineConstantExpr>(a)){check(c.getValue()>=0,"negative indexing constant");map.push_back(-c.getValue()-1);}
        else throw Error("unsupported non-projected affine indexing map");
      }
      mapping.push_back(map);auto e=std::make_shared<Expr>();e->type=t.type;
      if(i<inputs){e->op="read";e->tensor=std::make_shared<Tensor>(t);e->mapping=map;}else e->op="accumulator";
      local[body.getArgument(i)]=e;
    }
    count(shape);Shape reduction;
    for(auto [i,it]:llvm::enumerate(op.getIteratorTypesArray()))if(it==utils::IteratorType::reduction)reduction.push_back(i);
    for(Operation &nested:body){
      auto n=nested.getName().getStringRef();
      if(n=="cf.assert"){
        E condition=scalar(nested.getOperand(0),local);check(!runtime(condition),"runtime cf.assert requires explicit lowering");
        for(int64_t i=0;i<count(shape);++i)check(evaluate(condition,coords(i,shape))!=0,"constant cf.assert failed");
      }else if(n!="linalg.yield"){
        check(nested.getNumResults()==1 && nested.getNumRegions()==0,"unsupported scalar region operation "+n.str());
        (void)scalar(nested.getResult(0),local);
      }
    }
    Operation *yield=body.getTerminator();
    for(auto [i,result]:llvm::enumerate(op->getResults())){
      if(result.use_empty())continue;
      Tensor out=emptyTensor(result.getType());Shape outputDims;
      for(auto [j,d]:llvm::enumerate(mapping[inputs+i])){
        if(d>=0)outputDims.push_back(d);
        else check(!reduction.empty() && d==-1 && out.shape[j]==1,"unsupported generic output indexing map");
      }
      Shape expected(rank-(!reduction.empty()));std::iota(expected.begin(),expected.end(),0);
      check(outputDims==expected,"non-identity generic output indexing is unsupported");
      E e=scalar(yield->getOperand(i),local);
      if(!reduction.empty()){
        check(reduction==Shape{int64_t(rank)-1} && inputs==1,"only single trailing reductions supported");
        check((e->op=="add" || e->op=="max") && e->args.size()==2,"unsupported reduction recurrence");
        check(e->args[0]->op=="accumulator" || e->args[1]->op=="accumulator","reduction must use its initializer");
        E source=e->args[e->args[0]->op=="accumulator"?1:0];check(source->op=="read","reduction requires tensor input");
        Tensor initial=get(op->getOperand(inputs+i));check(bool(initial.constant),"nonconstant reduction initializer");
        double init=initial.at(Shape(initial.shape.size(),0));
        for(int64_t j=0;j<count(initial.shape);++j)check(initial.at(coords(j,initial.shape))==init,"nonuniform reduction initializer");
        out=g.allocate(out.shape,out.type);Kernel k;k.kind="reduce";k.output=out;k.expression=source;k.shape=shape;k.reduction=e->op;k.initial=init;add(k);
      }else if(!runtime(e)){
        check(count(shape)<=1048576,"constant folding element budget exceeded");
        out.constant=std::make_shared<std::vector<double>>();
        for(int64_t j=0;j<count(shape);++j)out.constant->push_back(evaluate(e,coords(j,shape)));
      }else if(e->op=="read" && out.shape.size()==shape.size()){
        Tensor t=*e->tensor;out.buffer=t.buffer;out.offset=t.offset;out.step=Shape(rank,0);
        for(auto [j,d]:llvm::enumerate(e->mapping)){if(d>=0)out.step[d]+=t.step[j];else out.offset+=(-d-1)*t.step[j];}
      }else{
        out=g.allocate(out.shape,out.type);Kernel k;k.kind="vector";k.output=out;k.expression=e;k.shape=shape;add(k);
      }tensors[result]=out;
    }
  }
  Tensor reshape(Operation *op,bool expand) {
    Tensor t=get(op->getOperand(0)),out=emptyTensor(op->getResult(0).getType());
    check(count(t.shape)==count(out.shape),"reshape element count mismatch");
    if(t.constant){t=materializeConstant(t);out.constant=t.constant;return out;}
    auto groups=op->getAttrOfType<ArrayAttr>("reassociation");check(bool(groups),"missing reshape reassociation");
    if(!expand){
      bool copy=false;
      for(Attribute a:groups){Shape active;for(Attribute d:cast<ArrayAttr>(a)){int64_t i=cast<IntegerAttr>(d).getInt();if(t.shape[i]!=1)active.push_back(i);}
        for(size_t j=1;j<active.size();++j)copy|=t.step[active[j-1]]!=t.step[active[j]]*t.shape[active[j]];
      }
      if(copy){Tensor c=g.allocate(t.shape,t.type);Kernel k;k.kind="copy";k.output=c;k.inputs={t};k.shape=t.shape;add(k);t=c;}
    }
    out.buffer=t.buffer;out.offset=t.offset;out.step=Shape(out.shape.size(),0);
    for(auto [i,a]:llvm::enumerate(groups)){
      auto group=cast<ArrayAttr>(a);
      if(expand){int64_t s=t.step[i];for(int j=group.size()-1;j>=0;--j){int64_t d=cast<IntegerAttr>(group[j]).getInt();out.step[d]=s;s*=out.shape[d];}}
      else out.step[i]=t.step[cast<IntegerAttr>(group[group.size()-1]).getInt()];
    }return out;
  }
public:
  explicit Legalizer(ModuleOp m):module(m){}
  Graph run(){
    auto funcs=module.getOps<func::FuncOp>();check(std::distance(funcs.begin(),funcs.end())==1,"one entry function required");
    func::FuncOp f=*funcs.begin();check(f.getBody().hasOneBlock(),"one entry block required");
    auto resources=module->getAttrOfType<ArrayAttr>("plena.input_resources");
    check(resources && resources.size()==f.getNumArguments(),"input resource binding count mismatch");
    for(auto [i,arg]:llvm::enumerate(f.getArguments())){
      auto r=cast<DictionaryAttr>(resources[i]);auto p=r.getAs<StringAttr>("file");check(bool(p),"input resource lacks file");
      bool stat=r.getAs<BoolAttr>("static") && r.getAs<BoolAttr>("static").getValue();
      if(stat)tensors[arg]=fileConstant(arg.getType(),p.str());
      else {
        Tensor t=allocate(arg.getType());auto file=llvm::MemoryBuffer::getFile(p.getValue());check(bool(file),"cannot read input resource "+p.str());
        check((*file)->getBufferSize()==g.buffers[t.buffer].size(),"input resource byte count mismatch");
        g.buffers[t.buffer].file=p.str();tensors[arg]=t;
      }
    }
    for(Operation &operation:f.front()){
      current=&operation;auto name=operation.getName().getStringRef();
      if(auto op=dyn_cast<func::ReturnOp>(operation)){
        for(Value v:op.getOperands())g.outputs.push_back(materialize(get(v)));continue;
      }
      check(operation.getNumResults()>0,"unsupported operation "+name.str());
      if(name=="arith.constant"){
        Attribute a=operation.getAttr("value");Value r=operation.getResult(0);
        if(auto dense=dyn_cast<DenseElementsAttr>(a))tensors[r]=denseConstant(r.getType(),dense);
        else{auto e=std::make_shared<Expr>();e->op="constant";e->type=r.getType();e->value=attrNumber(a);scalars[r]=e;}
      }else if(name=="tensor.empty")tensors[operation.getResult(0)]=emptyTensor(operation.getResult(0).getType());
      else if(name=="linalg.fill"){
        Tensor t=emptyTensor(operation.getResult(0).getType());llvm::DenseMap<Value,E> local;
        double v=roundTo(evaluate(scalar(operation.getOperand(0),local),{}),t.type);
        t.constant=std::make_shared<std::vector<double>>(1,v);t.step=Shape(t.shape.size(),0);tensors[operation.getResult(0)]=t;
      }else if(auto op=dyn_cast<linalg::GenericOp>(operation))generic(op);
      else if(name=="tensor.expand_shape" || name=="tensor.collapse_shape")tensors[operation.getResult(0)]=reshape(&operation,name=="tensor.expand_shape");
      else if(name=="linalg.transpose"){
        Tensor t=get(operation.getOperand(0)),out=emptyTensor(operation.getResult(0).getType());
        auto perm=operation.getAttrOfType<DenseI64ArrayAttr>("permutation");check(bool(perm),"missing transpose permutation");
        out.buffer=t.buffer;out.offset=t.offset;out.constant=t.constant;
        for(auto [i,d]:llvm::enumerate(perm.asArrayRef()))out.step[i]=t.step[d];tensors[operation.getResult(0)]=out;
      }else if(name=="tensor.extract_slice"){
        Tensor t=get(operation.getOperand(0)),out=emptyTensor(operation.getResult(0).getType());
        auto offsets=operation.getAttrOfType<DenseI64ArrayAttr>("static_offsets").asArrayRef();
        auto sizes=operation.getAttrOfType<DenseI64ArrayAttr>("static_sizes").asArrayRef();
        auto steps=operation.getAttrOfType<DenseI64ArrayAttr>("static_strides").asArrayRef();
        check(t.shape.size()==out.shape.size(),"rank-reducing slices not supported");
        out.buffer=t.buffer;out.offset=t.offset;out.constant=t.constant;
        for(size_t i=0;i<out.shape.size();++i){check(offsets[i]>=0 && steps[i]>0 && sizes[i]>0 && offsets[i]+(sizes[i]-1)*steps[i]<t.shape[i],"slice outside bounds");out.offset+=offsets[i]*t.step[i];out.step[i]=steps[i]*t.step[i];}
        tensors[operation.getResult(0)]=out;
      }else if(name=="tensor.concat"){
        std::vector<Tensor> ins;for(Value v:operation.getOperands())ins.push_back(get(v));
        auto dim=operation.getAttrOfType<IntegerAttr>("dim").getInt();Tensor out=emptyTensor(operation.getResult(0).getType());
        bool constant=llvm::all_of(ins,[](const Tensor&t){return bool(t.constant);});
        if(constant){out.constant=std::make_shared<std::vector<double>>(count(out.shape));int64_t off=0;
          for(const Tensor&t:ins){for(int64_t i=0;i<count(t.shape);++i){Shape c=coords(i,t.shape),dst=c;dst[dim]+=off;(*out.constant)[out.index(dst)]=t.at(c);}off+=t.shape[dim];}
        }else{out=g.allocate(out.shape,out.type);int64_t off=0;for(Tensor t:ins){Tensor part=out;part.shape=t.shape;part.offset=off*out.step[dim];Kernel k;k.kind="copy";k.output=part;k.inputs={materialize(t)};k.shape=t.shape;add(k);off+=t.shape[dim];}}
        tensors[operation.getResult(0)]=out;
      }else if(name=="linalg.matmul" || name=="linalg.batch_matmul"){
        Tensor a=get(operation.getOperand(0)),b=get(operation.getOperand(1)),init=get(operation.getOperand(2));
        check(bool(init.constant),"matmul requires zero constant initializer");
        for(int64_t i=0;i<count(init.shape);++i)check(init.at(coords(i,init.shape))==0,"matmul requires zero initializer");
        Tensor out=emptyTensor(operation.getResult(0).getType());
        if(a.constant && b.constant){
          check(count(out.shape)<=1048576,"constant matmul exceeds folding budget");out.constant=std::make_shared<std::vector<double>>();
          for(int64_t i=0;i<count(out.shape);++i){Shape c=coords(i,out.shape),ac=c,bc=c;double sum=0;
            for(int64_t k=0;k<a.shape.back();++k){ac.back()=k;bc[bc.size()-2]=k;sum=roundTo(sum+roundTo(a.at(ac)*b.at(bc),out.type),out.type);}out.constant->push_back(sum);}
        }else{check(a.type.isF16() && b.type.isF16() && (out.type.isF16()||out.type.isF32()),"runtime Matrix requires FP16 inputs and FP16/FP32 output");out=g.allocate(out.shape,out.type);Kernel k;k.kind="matmul";k.output=out;k.inputs={materialize(a),materialize(b)};k.shape=out.shape;add(k);}
        tensors[operation.getResult(0)]=out;
      }else throw Error("unsupported operation "+name.str());
    }check(!g.outputs.empty(),"entry has no returned tensor");return g;
  }
};
}
Graph plena::native::legalize(ModuleOp module){return Legalizer(module).run();}
