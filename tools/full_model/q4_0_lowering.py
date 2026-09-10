"""Q4_0 is a compiler lowering recipe, not a Matrix execution mode.

Raw block scale/bytes stay in L1. Bit operations, numeric casts and FP16 MUL
on the VPU produce ordinary FP16 weights for the unchanged Matrix datapath.
Clobbers v0..v5 and gp10..gp15; uses explicit vector lengths.
"""
from generic_vector_isa import typed_vector
from bounded_vector_emitter import BoundedProgram, rform


def emit_q4_0_to_fp16(words, source, destination, elements, register_bytes=64):
    if elements <= 0 or elements % 32:
        raise ValueError("Q4_0 requires complete 32-element blocks")
    if source < destination + elements*2 and destination < source + elements//32*18:
        raise ValueError("Q4_0 decode input/output must not overlap")
    out = BoundedProgram(words, register_bytes)
    chunk = min(16, register_bytes//2)
    if chunk < 1:
        raise ValueError("register too small")
    for reg, value in ((2,15),(3,4),(4,8)):
        out.load(13,value)
        words.append(typed_vector('SPLAT','I8',reg,13,elements=chunk))
    for block in range(elements//32):
        for start in range(0,16,chunk):
            n=min(chunk,16-start)
            out.set_vector_elements(n)
            out.load(10,source+block*18+2+start)
            words.append(rform(0x35,rd=0,rs1=10,funct=5))
            out.set_vector_elements(1)
            out.load(12,source+block*18)
            words.append(rform(0x35,rd=5,rs1=12,funct=1))
            words.append(typed_vector('BCAST','F16',5,5,elements=n))
            for high in (False,True):
                words.append(typed_vector('SHR_LOGICAL' if high else 'AND','I8',1,0,3 if high else 2,elements=n))
                words.append(typed_vector('SUB','I8',1,1,4,elements=n))
                words.append(typed_vector('CAST_F16','I8',1,1,elements=n))
                words.append(typed_vector('MUL','F16',1,1,5,elements=n))
                out.set_vector_elements(n)
                out.load(11,destination+(block*32+start+(16 if high else 0))*2)
                words.append(rform(0x35,rd=11,rs1=1,funct=2))
