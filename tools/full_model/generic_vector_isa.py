"""Model-independent four-word Vector ISA and accumulator-load encoding."""
OPS = {"ADD":0,"SUB":1,"MUL":2,"FMA":3,"MAX":4,"MIN":5,"NEG":6,"ABS":7,"EXP":8,"RCP":9,"SQRT":10,"RSQRT":11,"ROUND_EVEN":12,"CMP_EQ":13,"CMP_LT":14,"CMP_LE":15,"SELECT":16,"MOV":17,"BCAST":18,"SLICE":19,"GATHER":20,"AND":21,"OR":22,"XOR":23,"SHL":24,"SHR_LOGICAL":25,"SHR":26,"CAST_F16":27,"CAST_F32":28,"CAST_I8":29,"CAST_I32":30,"SPLAT":31,"ADD_SCALAR":32,"SUB_SCALAR":33,"MUL_SCALAR":34,"MAX_SCALAR":35,"MIN_SCALAR":36,"QUANTIZE_I8":37,"DEQUANTIZE_F32":38,"DEQUANTIZE_F16":39,"BITCAST":40}
DTYPES = {"F16":0,"F32":1,"I8":2,"I32":3}

def valid_dtype(op,dtype):
    if op in (8,9,10,11,12,37): return dtype in (0,1)
    if op in (21,22,23,24,25,26,38,39): return dtype in (2,3)
    if op==40: return dtype in (1,3)
    return True

def vector_sources(op):
    if op==31: return 0
    if op in (3,16,37,38,39): return 3
    if op in (0,1,2,4,5,13,14,15,20,21,22,23,24,25,26): return 2
    return 1

def typed_vector(operation,dtype,rd,rs1=0,rs2=0,rs3=0,*,elements):
    op=OPS[operation.upper()] if isinstance(operation,str) else operation
    dt=DTYPES[dtype.upper()] if isinstance(dtype,str) else dtype
    if op not in OPS.values() or dt not in DTYPES.values() or not valid_dtype(op,dt):
        raise ValueError("unsupported typed Vector operation/dtype")
    if any(not 0<=r<16 for r in (rd,rs1,rs2,rs3)) or not 0<elements<1<<32:
        raise ValueError("invalid register index or element count")
    arity=vector_sources(op)
    if arity<3 and rs3: raise ValueError("reserved rs3")
    if arity<2 and op not in (19,32,33,34,35,36) and rs2: raise ValueError("reserved rs2")
    return [0x36 | rd<<6 | rs1<<10 | rs2<<14 | rs3<<18,op,dt,elements]

def matrix_load_acc(address,m,n,stride_bytes,*,integer=False):
    if not 0<=address<16 or any(not 0<x<1<<32 for x in (m,n,stride_bytes)):
        raise ValueError("invalid accumulator load")
    if stride_bytes<n*4 or stride_bytes%4: raise ValueError("invalid accumulator stride")
    return [0x3a | address<<10 | (2 if integer else 1)<<22,m,n,stride_bytes]

# Mnemonic -> (operation, dtype, operand namespaces before explicit length).
ALIASES={}
for dtype,dt in DTYPES.items():
    for name,op in OPS.items():
        if not valid_dtype(op,dt): continue
        if 27<=op<=30:
            mnemonic=f"V_CAST_{dtype}_{name[5:]}"
        elif op==37:
            mnemonic=f"V_QUANTIZE_{dtype}_I8"
        elif op in (38,39):
            mnemonic=f"V_DEQUANTIZE_{dtype}_{name[11:]}"
        elif op==40:
            mnemonic=f"V_BITCAST_{'I32' if dtype=='F32' else 'F32'}_{dtype}"
        else:
            mnemonic=f"V_{name}_{dtype}"
        arity=vector_sources(op)
        kinds=("v",)* (1+arity)
        if op==31: kinds=("v","s" if dt in (0,1) else "gp")
        if op in (32,33,34,35,36): kinds=("v","v","s" if dt in (0,1) else "gp")
        if op==19: kinds=("v","v","gp")
        ALIASES[mnemonic]=(op,dt,kinds)
