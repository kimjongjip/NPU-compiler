#!/usr/bin/env python3
"""Validate constant materialization against core scalar instruction semantics."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools/full_model'))
from plena_isa_encoding import load_u32, matrix_load, matrix_mma, matrix_writeout
from generic_vector_isa import matrix_load_acc, typed_vector


class ConstantMaterializationTests(unittest.TestCase):
    def test_every_destination_and_nonzero_gp0(self):
        constants = (0, 1, 6, 64, 0xfff, 0x1000, 0x3ffff, 0x40000,
                     0x80000000, 0xffffffff, -1, 0x100000005)
        for destination in range(16):
            for gp0 in (0, 64, 0xffffffff):
                for value in constants:
                    with self.subTest(destination=destination, gp0=gp0, value=value):
                        initial = [0x12345000 + index * 17 for index in range(16)]
                        initial[0] = gp0
                        actual = initial.copy()
                        for word in load_u32(destination, value):
                            opcode = word & 0x3f
                            rd = (word >> 6) & 0xf
                            if opcode == 0x25:
                                actual[rd] = ((word >> 10) & 0xfffff) << 12
                            elif opcode == 0x22:
                                rs1 = (word >> 10) & 0xf
                                actual[rd] = (actual[rs1] + (word >> 14)) & 0xffffffff
                            else:
                                self.fail(f'unexpected opcode {opcode:#x}')
                        expected = initial.copy()
                        expected[destination] = value & 0xffffffff
                        self.assertEqual(actual, expected)


class MatrixEncodingTests(unittest.TestCase):
    def test_common_fp16_and_accumulator_io(self):
        self.assertEqual(matrix_load(1, 64, 4, 8, funct=3),
                         [0x37 | 1 << 10 | 3 << 22, 64, 4, 8])
        self.assertEqual(matrix_load(2, 1, 64, 128, funct=7),
                         [0x37 | 2 << 10 | 7 << 22, 1, 64, 128])
        self.assertEqual(matrix_mma(1, 4, 64, funct=3, accumulate=True),
                         [0x3b | 3 << 22 | 1 << 26, 1, 4, 64])
        self.assertEqual(matrix_writeout(3, 1, 4, 8),
                         [0x3c | 3 << 6 | 1 << 22, 1, 4, 8])
        self.assertEqual(matrix_writeout(3, 1, 4, 16, funct=4),
                         [0x3c | 3 << 6 | 4 << 22, 1, 4, 16])
        self.assertEqual(matrix_load_acc(3, 1, 4, 16),
                         [0x3a | 3 << 10 | 1 << 22, 1, 4, 16])

    def test_generic_vector_encoding_and_retired_q4_matrix_mode(self):
        self.assertEqual(typed_vector('ADD', 'F32', 1, 2, 3, elements=16),
                         [0x36 | 1<<6 | 2<<10 | 3<<14, 0, 1, 16])
        with self.assertRaises(ValueError):
            matrix_load(1, 64, 4, 0, funct=2)
        with self.assertRaises(ValueError):
            matrix_mma(1, 4, 64, funct=4)

    def test_retired_bf16_modes_are_not_reinterpreted(self):
        with self.assertRaises(ValueError):
            matrix_load(1, 1, 64, 128, funct=6)
        with self.assertRaises(ValueError):
            matrix_mma(1, 4, 64, funct=2)
        with self.assertRaises(ValueError):
            matrix_writeout(1, 1, 4, 8, funct=3)


if __name__ == '__main__':
    unittest.main()
