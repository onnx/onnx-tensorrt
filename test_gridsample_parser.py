# SPDX-License-Identifier: Apache-2.0

"""Parser-only regression tests for GridSample padding attributes.

Run against the parser under test with:
    python -m unittest -v test_gridsample_parser
"""

import unittest

import onnx
from onnx import TensorProto, helper
import tensorrt as trt


class GridSamplePaddingTest(unittest.TestCase):
    def parse(self, opset, **attributes):
        node = helper.make_node(
            "GridSample", ["image", "grid"], ["output"], **attributes
        )
        graph = helper.make_graph(
            [node],
            "grid_sample_padding",
            [
                helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 1, 2, 2]),
                helper.make_tensor_value_info("grid", TensorProto.FLOAT, [1, 1, 1, 2]),
            ],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1, 1, 1])],
        )
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", opset)], ir_version=10
        )
        # ONNX's structural checker does not reject an unsupported padding string.
        onnx.checker.check_model(model)
        logger = trt.Logger(trt.Logger.ERROR)
        with trt.Builder(logger) as builder, builder.create_network(0) as network:
            with trt.OnnxParser(network, logger) as parser:
                accepted = parser.parse(model.SerializeToString())
                errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        return accepted, errors

    def test_omitted_padding_uses_default(self):
        for opset in (16, 20):
            with self.subTest(opset=opset):
                accepted, errors = self.parse(opset)
                self.assertTrue(accepted, errors)

    def test_supported_padding_modes(self):
        for opset in (16, 20):
            for padding_mode in ("zeros", "border", "reflection"):
                with self.subTest(opset=opset, padding_mode=padding_mode):
                    accepted, errors = self.parse(opset, padding_mode=padding_mode)
                    self.assertTrue(accepted, errors)

    def test_unsupported_padding_is_rejected(self):
        for opset in (16, 20):
            for padding_mode in ("unsupported", ""):
                with self.subTest(opset=opset, padding_mode=padding_mode):
                    accepted, errors = self.parse(opset, padding_mode=padding_mode)
                    self.assertFalse(
                        accepted, "Unsupported padding must not silently use zeros"
                    )
                    self.assertTrue(
                        any("Unsupported padding mode:" in error for error in errors),
                        errors,
                    )


if __name__ == "__main__":
    unittest.main()
