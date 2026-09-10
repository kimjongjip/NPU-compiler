"""Compile tensor operations into finite-register Vector/Scalar ISA.

These are code-generation helpers, not emulator-side implicit tiling or spills.
Each vector operand/result fits one register; intermediates stay in registers
within each tile. Global reductions use an explicit scalar pairwise tree.
"""
from __future__ import annotations


def rform(opcode, rd=0, rs1=0, rs2=0, funct=0):
    return opcode | rd << 6 | rs1 << 10 | rs2 << 14 | funct << 22


class BoundedVectorEmitterMixin:
    vector_register_bytes = 64

    def _vector_tiles(self, elements, element_bytes=2):
        width = self.vector_register_bytes // element_bytes
        if elements <= 0 or width < 1:
            raise ValueError("invalid vector size/register capacity")
        for start in range(0, elements, width):
            yield start, min(width, elements - start)

    def vector_binary(self, destination, lhs, rhs, elements, funct):
        for start, count in self._vector_tiles(elements):
            self.set_vector_elements(count)
            for gp, address in ((3, destination), (4, lhs), (5, rhs)):
                self.load(gp, address + start * 2)
            self.emit(rform(0x35, rd=0, rs1=4, funct=1))
            self.emit(rform(0x35, rd=1, rs1=5, funct=1))
            self.emit(rform(0x30, rd=2, rs1=0, rs2=1, funct=funct))
            self.emit(rform(0x35, rd=3, rs1=2, funct=2))

    def vector_unary(self, destination, source, elements, funct):
        for start, count in self._vector_tiles(elements):
            self.set_vector_elements(count)
            self.load(3, destination + start * 2)
            self.load(4, source + start * 2)
            self.emit(rform(0x35, rd=0, rs1=4, funct=1))
            self.emit(rform(0x31, rd=1, rs1=0, funct=funct))
            self.emit(rform(0x35, rd=3, rs1=1, funct=2))

    def vector_scalar(self, destination, source, scalar, elements, funct):
        self.load(5, scalar)
        self.emit(rform(0x35, rd=0, rs1=5, funct=9))
        for start, count in self._vector_tiles(elements):
            self.set_vector_elements(count)
            self.load(3, destination + start * 2)
            self.load(4, source + start * 2)
            self.emit(rform(0x35, rd=0, rs1=4, funct=1))
            self.emit(rform(0x32, rd=1, rs1=0, rs2=0, funct=funct))
            self.emit(rform(0x35, rd=3, rs1=1, funct=2))

    def vector_reduce(self, destination, source, elements, funct, rhs=0):
        if funct not in (1, 2, 3, 4, 5):
            raise ValueError("unsupported reduction")
        element_bytes = 2 if funct <= 3 else 4
        # Use a power-of-two tile to preserve the pairwise reduction tree.
        capacity = self.vector_register_bytes // element_bytes
        width = 1 << (capacity.bit_length() - 1)
        tiles = (elements + width - 1) // width
        if not 0 < tiles <= 2048:
            raise ValueError("reduction exceeds bounded scalar-tree capacity")
        occupied = set()
        combine = 4 if funct in (2, 5) else 1  # S_MAX or S_ADD
        for index, start in enumerate(range(0, elements, width)):
            level = 0
            while level in occupied:
                level += 1
            scalar = 4 + level
            self.set_vector_elements(min(width, elements - start))
            self.load(4, source + start * element_bytes)
            self.emit(rform(0x35, rd=0, rs1=4, funct=1 if funct <= 3 else 3))
            if funct == 3:
                self.load(5, rhs + start * element_bytes)
                self.emit(rform(0x35, rd=1, rs1=5, funct=1))
            self.emit(rform(0x33, rd=scalar, rs1=0, rs2=1 if funct == 3 else 0, funct=funct))
            for lower in range(level):
                self.emit(rform(0x34, rd=scalar, rs1=4 + lower, rs2=scalar, funct=combine))
                occupied.remove(lower)
            occupied.add(level)
        levels = sorted(occupied)
        scalar = 4 + levels[0]
        for level in levels[1:]:
            self.emit(rform(0x34, rd=scalar, rs1=4 + level, rs2=scalar, funct=combine))
        self.load(3, destination)
        self.emit(rform(0x35, rd=3, rs1=scalar, funct=10))

    def rmsnorm(self, destination, source, weight, rows, columns, temp,
                sum_scalar, inverse_scalar, inverse_columns, epsilon):
        self.load(11, inverse_columns)
        self.load(12, epsilon)
        self.emit(rform(0x35, rd=1, rs1=11, funct=9))
        self.emit(rform(0x35, rd=2, rs1=12, funct=9))
        for row in range(rows):
            src = source + row * columns * 2
            dst = destination + row * columns * 2
            self.vector_reduce(sum_scalar, src, columns, 3, rhs=src)
            self.load(3, sum_scalar)
            self.emit(rform(0x35, rd=0, rs1=3, funct=9))
            self.emit(rform(0x34, rd=0, rs1=0, rs2=1, funct=3))
            self.emit(rform(0x34, rd=0, rs1=0, rs2=2, funct=1))
            self.emit(rform(0x34, rd=0, rs1=0, funct=8))
            for start, count in self._vector_tiles(columns):
                self.set_vector_elements(count)
                for gp, base in ((3, src), (4, weight), (5, dst)):
                    self.load(gp, base + start * 2)
                self.emit(rform(0x35, rd=0, rs1=3, funct=1))
                self.emit(rform(0x35, rd=1, rs1=4, funct=1))
                self.emit(rform(0x32, rd=2, rs1=0, rs2=0, funct=3))
                self.emit(rform(0x30, rd=2, rs1=2, rs2=1, funct=3))
                self.emit(rform(0x35, rd=5, rs1=2, funct=2))

    def silu_mul(self, destination, gate, up, elements, temp, one):
        self.load(6, one)
        self.emit(rform(0x35, rd=0, rs1=6, funct=9))
        for start, count in self._vector_tiles(elements):
            self.set_vector_elements(count)
            for gp, base in ((3, destination), (4, gate), (5, up)):
                self.load(gp, base + start * 2)
            self.emit(rform(0x35, rd=0, rs1=4, funct=1))
            self.emit(rform(0x35, rd=1, rs1=5, funct=1))
            self.emit(rform(0x31, rd=2, rs1=0, funct=1))
            self.emit(rform(0x31, rd=2, rs1=2, funct=2))
            self.emit(rform(0x32, rd=2, rs1=2, rs2=0, funct=1))
            self.emit(rform(0x31, rd=2, rs1=2, funct=3))
            self.emit(rform(0x30, rd=2, rs1=0, rs2=2, funct=3))
            self.emit(rform(0x30, rd=2, rs1=2, rs2=1, funct=3))
            self.emit(rform(0x35, rd=3, rs1=2, funct=2))

    def rope(self, destination, source, rows, heads, head_dim, sine, cosine, temp_a, temp_b):
        assert head_dim % 2 == 0
        half = head_dim // 2
        for row in range(rows):
            for head in range(heads):
                base = source + (row * heads + head) * head_dim * 2
                dst = destination + (row * heads + head) * head_dim * 2
                for start, count in self._vector_tiles(half):
                    self.set_vector_elements(count)
                    addresses = [base, base + half * 2, cosine + row * half * 2,
                                 sine + row * half * 2, dst, dst + half * 2]
                    for gp, address in enumerate(addresses, start=3):
                        self.load(gp, address + start * 2)
                    for reg, gp in enumerate(range(3, 7)):
                        self.emit(rform(0x35, rd=reg, rs1=gp, funct=1))
                    self.emit(rform(0x30, rd=4, rs1=0, rs2=2, funct=3))
                    self.emit(rform(0x30, rd=5, rs1=1, rs2=3, funct=3))
                    self.emit(rform(0x30, rd=6, rs1=4, rs2=5, funct=2))
                    self.emit(rform(0x35, rd=7, rs1=6, funct=2))
                    self.emit(rform(0x30, rd=4, rs1=1, rs2=2, funct=3))
                    self.emit(rform(0x30, rd=5, rs1=0, rs2=3, funct=3))
                    self.emit(rform(0x30, rd=6, rs1=4, rs2=5, funct=1))
                    self.emit(rform(0x35, rd=8, rs1=6, funct=2))

    def softmax(self, destination, scores, elements, temp_a, temp_b,
                maximum, total, reciprocal, scale=None):
        if scale is not None:
            self.vector_scalar(temp_a, scores, scale, elements, 3)
            scores = temp_a
        self.vector_reduce(maximum, scores, elements, 2)
        self.load(6, maximum)
        self.emit(rform(0x35, rd=0, rs1=6, funct=9))
        for start, count in self._vector_tiles(elements):
            self.set_vector_elements(count)
            self.load(3, scores + start * 2)
            self.load(4, temp_b + start * 2)
            self.emit(rform(0x35, rd=0, rs1=3, funct=1))
            self.emit(rform(0x32, rd=1, rs1=0, rs2=0, funct=2))
            self.emit(rform(0x31, rd=1, rs1=1, funct=2))
            self.emit(rform(0x35, rd=4, rs1=1, funct=2))
        self.vector_reduce(total, temp_b, elements, 1)
        self.scalar_f32(reciprocal, total, 6)
        self.vector_scalar(destination, temp_b, reciprocal, elements, 3)

class BoundedProgram(BoundedVectorEmitterMixin):
    """Small ISA-generator adapter for tests and standalone tensor kernels."""
    def __init__(self, words, register_bytes=64):
        self.words = words
        self.vector_register_bytes = register_bytes

    def emit(self, word):
        self.words.append(word)

    def load(self, register, value):
        if not 0 <= value < 1 << 32:
            raise ValueError("GP value must fit u32")
        self.emit(0x25 | register << 6 | (value >> 12) << 10)
        if value & 4095:
            self.emit(0x22 | register << 6 | register << 10 | (value & 4095) << 14)

    def set_vector_elements(self, elements):
        self.load(15, elements)
        self.emit(rform(0x39, rd=15, funct=3))

    def scalar_f32(self, destination, lhs, funct, rhs=0):
        self.load(3, destination)
        self.load(4, lhs)
        self.emit(rform(0x35, rd=0, rs1=4, funct=9))
        if funct <= 5:
            self.load(5, rhs)
            self.emit(rform(0x35, rd=1, rs1=5, funct=9))
        self.emit(rform(0x34, rd=2, rs1=0, rs2=1 if funct <= 5 else 0, funct=funct))
        self.emit(rform(0x35, rd=3, rs1=2, funct=10))
