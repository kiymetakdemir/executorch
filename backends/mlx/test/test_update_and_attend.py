#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""AOT lowering tests for the off-graph kvcache::update_and_attend op on MLX."""

import unittest

# Registers torch.ops.kvcache.update_and_attend.
import executorch.extension.llm.cache.update_and_attend  # noqa: F401

import torch
from executorch.backends.mlx.builder.program_builder import MLXProgramBuilder
from executorch.backends.mlx.partitioner import MLXPartitioner
from executorch.exir import to_edge_transform_and_lower
from torch.export import export


class OneLayer(torch.nn.Module):
    def __init__(self, scale=0.125):
        super().__init__()
        self.scale = scale

    def forward(self, q, k, v, position):
        return torch.ops.kvcache.update_and_attend(
            q, k, v, position, 3, self.scale, torch.float32
        )


def _inputs(b=1, h=2, s=4, d=8):
    return (
        torch.randn(b, h, s, d),
        torch.randn(b, h, s, d),
        torch.randn(b, h, s, d),
        torch.arange(s, dtype=torch.long).unsqueeze(-1),
    )


class UpdateAndAttendLoweringTest(unittest.TestCase):
    def test_export_is_functional(self):
        ep = export(OneLayer(), _inputs(), strict=True)
        mutated = [
            s
            for s in ep.graph_signature.output_specs
            if s.kind.name == "BUFFER_MUTATION"
        ]
        self.assertEqual(mutated, [])
        calls = [
            n
            for n in ep.graph_module.graph.nodes
            if n.op == "call_function"
            and n.target is torch.ops.kvcache.update_and_attend.default
        ]
        self.assertEqual(len(calls), 1)

    def test_lowers_to_update_and_attend_node(self):
        ep = export(OneLayer(scale=0.25), _inputs(), strict=True)
        graph = MLXProgramBuilder(ep).build()
        nodes = [
            instr.op
            for chain in graph.instruction_chains
            for instr in chain.instructions
        ]
        ua = [n for n in nodes if type(n).__name__ == "UpdateAndAttendNode"]
        self.assertEqual(len(ua), 1)
        self.assertEqual(ua[0].layer_id, 3)
        self.assertAlmostEqual(ua[0].scale, 0.25, places=6)
        self.assertTrue(ua[0].causal)

    def test_partitioner_delegates_the_op(self):
        ep = export(OneLayer(), _inputs(), strict=True)
        lowered = to_edge_transform_and_lower(ep, partitioner=[MLXPartitioner()])
        targets = [
            str(n.target)
            for n in lowered.exported_program().graph_module.graph.nodes
            if n.op == "call_function"
        ]
        self.assertIn("executorch_call_delegate", targets)
        self.assertNotIn("kvcache.update_and_attend.default", targets)


if __name__ == "__main__":
    unittest.main()
