import tempfile
import unittest
from pathlib import Path

import torch

from CLIPs.openai_vit_L_14.conversion import make_conversion_plan
from CLIPs.openai_vit_L_14.model import VisionTransformer
from to_snn.ECMT import ECMTConversionParams, convert
from to_snn.prototype import SNNArtifact


class TestECMTConversion(unittest.TestCase):
    def setUp(self):
        self.model = VisionTransformer(8, 4, 8, 1, 2, 4).eval()
        self.images = torch.randn(2, 3, 8, 8)
        self.params = ECMTConversionParams(
            percent=0.9, linear_num=2, qkv_num=2, softmax_num=2,
            fused=False,
        )

    def test_plan_is_explicit_and_complete(self):
        plan = make_conversion_plan(self.model)
        kinds = {target.kind for target in plan.targets}
        self.assertTrue({"layernorm", "qkv", "q", "k", "v", "qk",
                         "softmax", "softmax_neuron", "av", "linear",
                         "quickgelu"} <= kinds)
        self.assertEqual(len(plan.targets), len({target.path for target in plan.targets}))

    def test_conversion_accepts_mapping_loader_and_resets(self):
        plan = make_conversion_plan(self.model)
        artifact, info = convert(self.model, plan, self.params,
                                  calibration_loader=[{"image": self.images}])
        self.assertIsInstance(artifact, SNNArtifact)
        self.assertIn("thresholds", info)
        artifact.reset_state()
        first = artifact(self.images)
        second = artifact(self.images)
        self.assertEqual(tuple(first.shape), (2, 4))
        self.assertTrue(torch.isfinite(second).all())
        artifact.reset_state()
        self.assertTrue(torch.equal(first, artifact(self.images)))

    def test_conversion_accepts_tuple_loader_and_checkpoint_roundtrip(self):
        plan = make_conversion_plan(self.model)
        artifact, _ = convert(self.model, plan, self.params,
                              calibration_loader=[(self.images, torch.zeros(2))])
        artifact.reset_state()
        expected = artifact(self.images)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ecmt.pt"
            artifact.save(path)
            loaded = SNNArtifact.load(path)
            loaded.reset_state()
            self.assertTrue(torch.equal(expected, loaded(self.images)))

    def test_one_shot_loader_is_rejected(self):
        plan = make_conversion_plan(self.model)
        loader = (batch for batch in [self.images])
        with self.assertRaises(ValueError):
            convert(self.model, plan, self.params, calibration_loader=loader)


if __name__ == "__main__":
    unittest.main()
