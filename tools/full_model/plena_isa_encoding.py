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
    """Set one uint32 independently of old GP contents, including writable GP0."""

    value &= 0xFFFF_FFFF
    upper, lower = value >> 12, value & 0xFFF
    words = [OP_S_LUI_INT | register << 6 | upper << 10]
    if lower:
        words.append(_addi(register, register, lower))
    return words

def matrix_load(address: int, rows: int, columns: int, stride_bytes: int,
                funct: int = 3) -> list[int]:
    """ISA ver 1.0: load W=[K,N] or A=[M,K]; K may exceed MMA_TILE_K."""
    if funct not in (1, 3, 5, 7):
        raise ValueError("invalid Matrix load mode")
    if not (0 < rows < 1 << 32 and 0 < columns < 1 << 32 and 0 <= stride_bytes < 1 << 32):
        raise ValueError("Matrix extents/stride must fit u32 and extents must be positive")
    width = 1 if funct in (1, 5) else 2
    if stride_bytes < columns * width or stride_bytes % width:
        raise ValueError("invalid Matrix byte stride")
    return [rform(0x37, rs1=address, funct=funct), rows, columns, stride_bytes]


# ISA ver 1.0: K elements consumed by one fixed-tile M_MMA (the simulator's
# [TRANSACTIONAL.MATRIX_MICROARCHITECTURE].mma_tile_k).
MMA_TILE_K = 32


def matrix_mma_tile(funct: int = 3, *, accumulate: bool = False) -> int:
    """ISA ver 1.0: one-word fixed-tile M_MMA (bit 26: 0 INIT, 1 ACC)."""
    if funct not in (1, 3):
        raise ValueError("invalid Matrix MMA mode")
    return rform(0x3b, funct=funct) | int(accumulate) << 26


def matrix_mma(m: int, n: int, k: int, funct: int = 3,
               *, accumulate: bool = False) -> list[int]:
    """ISA ver 1.0: the M_MMA sequence that covers one loaded W/A pair.

    Each one-word M_MMA consumes the next MMA_TILE_K elements of the loaded
    W [K,N] and A [M,K] rectangles; the hardware takes M/N/K from the loads.
    The first word is INIT unless ``accumulate``; the rest are ACC.
    """
    if funct not in (1, 3) or any(not 0 < x < 1 << 32 for x in (m, n, k)):
        raise ValueError("invalid Matrix MMA mode/extents")
    tiles = -(-k // MMA_TILE_K)
    return [matrix_mma_tile(funct, accumulate=accumulate)] + [
        matrix_mma_tile(funct, accumulate=True)
    ] * (tiles - 1)


def matrix_writeout(address: int, m: int, n: int, stride_bytes: int,
                    funct: int = 1) -> list[int]:
    width = 4 if funct in (2, 4) else 2
    if funct not in (1, 2, 4) or any(not 0 < x < 1 << 32 for x in (m, n, stride_bytes)):
        raise ValueError("invalid Matrix writeout mode/extents")
    if stride_bytes < n * width or stride_bytes % width:
        raise ValueError("invalid Matrix output byte stride")
    return [rform(0x3c, rd=address, funct=funct), m, n, stride_bytes]
