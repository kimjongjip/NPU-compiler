"""Fail-closed and output-preservation tests for graph compilation."""

import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
PROJECT_TMP = ROOT.parent / "tmp"
PROJECT_TMP.mkdir(parents=True, exist_ok=True)
os.environ["TMPDIR"] = str(PROJECT_TMP)
tempfile.tempdir = str(PROJECT_TMP)
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np
from graph_pipeline.graph import CompileError
from graph_pipeline.lower import Hardware
from graph_pipeline.pipeline import compile_linalg

ADD = """
#id = affine_map<(d0) -> (d0)>
module {
  func.func @main(%a: tensor<4xf16>, %b: tensor<4xf16>) -> tensor<4xf16> {
    %empty = tensor.empty() : tensor<4xf16>
    %out = linalg.generic {indexing_maps = [#id, #id, #id], iterator_types = ["parallel"]}
      ins(%a, %b : tensor<4xf16>, tensor<4xf16>) outs(%empty : tensor<4xf16>) {
    ^bb0(%x: f16, %y: f16, %z: f16):
      %s = arith.addf %x, %y : f16
      linalg.yield %s : f16
    } -> tensor<4xf16>
    return %out : tensor<4xf16>
  }
}
"""


class Validation(unittest.TestCase):
    def setUp(self):
        self.work = Path(tempfile.mkdtemp(prefix="graph-validation.", dir=PROJECT_TMP))
        self.inputs = [np.ones(4, dtype=np.float16), np.ones(4, dtype=np.float16)]

    def test_preserve_existing_output(self):
        marker = self.work / "program.bin"
        marker.write_bytes(b"user-owned program")
        with self.assertRaisesRegex(CompileError, "nonempty"):
            compile_linalg(ADD, self.inputs, self.work)
        self.assertEqual(marker.read_bytes(), b"user-owned program")

    def test_unknown_operation_does_not_fallback(self):
        with self.assertRaisesRegex(CompileError, "unsupported"):
            compile_linalg(
                ADD.replace("arith.addf", "math.atan2"), self.inputs, self.work
            )
        self.assertFalse((self.work / "program.bin").exists())

    def test_wrong_input_dtype(self):
        inputs = [np.ones(4, dtype=np.float32), self.inputs[1]]
        with self.assertRaisesRegex(CompileError, "argument 0"):
            compile_linalg(ADD, inputs, self.work)

    def test_event_scoreboard_is_not_silently_expanded(self):
        with self.assertRaisesRegex(CompileError, "scoreboard"):
            compile_linalg(ADD, self.inputs, self.work, Hardware(event_slots=3))
        self.assertFalse((self.work / "program.bin").exists())

    def test_hardware_limits(self):
        for hw in (
            Hardware(registers=17),
            Hardware(placement=(1,)),
            Hardware(l1_bytes=32),
            Hardware(staging_bytes=65),
        ):
            with self.subTest(hardware=hw), self.assertRaises(CompileError):
                hw.validate()

    def test_runtime_assert_is_not_discarded(self):
        source = ADD.replace(
            "%s = arith.addf",
            "%check = arith.cmpf ogt, %x, %y : f16\n"
            '      cf.assert %check, "runtime contract"\n'
            "      %s = arith.addf",
        )
        with self.assertRaisesRegex(CompileError, "runtime cf.assert"):
            compile_linalg(source, self.inputs, self.work)

    def test_output_index_permutation_is_not_ignored(self):
        source = ADD.replace("tensor<4xf16>", "tensor<2x2xf16>")
        source = source.replace(
            "#id = affine_map<(d0) -> (d0)>",
            "#id = affine_map<(d0, d1) -> (d0, d1)>\n"
            "#swap = affine_map<(d0, d1) -> (d1, d0)>",
        ).replace("[#id, #id, #id]", "[#id, #id, #swap]")
        source = source.replace('["parallel"]', '["parallel", "parallel"]')
        inputs = [np.ones((2, 2), dtype=np.float16)] * 2
        with self.assertRaisesRegex(CompileError, "output indexing"):
            compile_linalg(source, inputs, self.work)


if __name__ == "__main__":
    unittest.main()
