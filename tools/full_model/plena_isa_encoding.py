"""Small native Python encoder for core-local PLENA 32-bit ISA words."""

OP_S_ADDI_INT = 0x22
OP_S_LUI_INT = 0x25
OP_V_BINARY_F16 = 0x30
OP_V_UNARY_F16 = 0x31
OP_V_SCALAR_F16 = 0x32
OP_V_REDUCE = 0x33
OP_S_ALU_F32 = 0x34
OP_V_MEMORY = 0x35
OP_M_LOAD = 0x37
OP_L2_DMA = 0x38
OP_C_SET_V2 = 0x39
OP_M_MMA = 0x3B
OP_M_WRITEOUT = 0x3C
OP_V_QUANT_I8 = 0x3D


def rform(opcode: int, rd: int = 0, rs1: int = 0, rs2: int = 0,
          rs3: int = 0, funct: int = 0) -> int:
    assert 0 <= opcode < 1 << 6
    assert 0 <= rd < 1 << 4
    assert 0 <= rs1 < 1 << 4
    assert 0 <= rs2 < 1 << 4
    assert 0 <= rs3 < 1 << 4
    assert 0 <= funct < 1 << 4
    return opcode | rd << 6 | rs1 << 10 | rs2 << 14 | rs3 << 18 | funct << 22


def _addi(rd: int, rs1: int, immediate: int) -> int:
    assert 0 <= immediate < 1 << 18
    return OP_S_ADDI_INT | rd << 6 | rs1 << 10 | immediate << 14


def load_u32(register: int, value: int) -> list[int]:
    """Materialize one uint32 in a GP register without host-width leakage."""

    value &= 0xFFFF_FFFF
    if value < 1 << 18:
        return [_addi(register, 0, value)]
    upper, lower = value >> 12, value & 0xFFF
    words = [OP_S_LUI_INT | register << 6 | upper << 10]
    if lower:
        words.append(_addi(register, register, lower))
    return words
