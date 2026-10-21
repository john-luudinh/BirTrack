"""
Unit Tests for RepBLGR and RepDWConv (Gate G1 in BiRTrack v0 plan)
"""
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import unittest
import torch
import torch.nn as nn
from lib.models.layers.rep_blgr import RepDWConv, RepBLGR
from lib.models.layers.fasa import FASA


class TestRepBLGR(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.B = 2
        self.C = 192
        self.H = 16
        self.W = 16
        self.x = torch.randn(self.B, self.C, self.H, self.W)

    def test_rep_dwconv_numerical_equivalence(self):
        """Test that fused DWConv produces identical outputs to parallel branches."""
        layer = RepDWConv(dim=self.C, deploy=False)
        layer.train()

        # Run several batches to populate running_mean and running_var with non-trivial values
        for _ in range(5):
            dummy = torch.randn(4, self.C, self.H, self.W)
            _ = layer(dummy)

        layer.eval()
        with torch.no_grad():
            out_train = layer(self.x)

        layer.switch_to_deploy()
        self.assertTrue(layer.deploy)

        with torch.no_grad():
            out_deploy = layer(self.x)

        max_diff = torch.max(torch.abs(out_train - out_deploy)).item()
        mean_diff = torch.mean(torch.abs(out_train - out_deploy)).item()
        print(f"\n[RepDWConv] Max abs difference: {max_diff:.8e}, Mean abs difference: {mean_diff:.8e}")
        self.assertLess(max_diff, 1e-5, f"Reparameterization error {max_diff} exceeds tolerance 1e-5")

    def test_rep_blgr_identity_init(self):
        """When gamma=0 (default initialization), output should exactly equal input."""
        model = RepBLGR(dim=self.C, num_heads=3, window_size=2, global_type="fasa", deploy=False)
        model.eval()
        with torch.no_grad():
            out = model(self.x)
        self.assertTrue(torch.equal(out, self.x), "Output with gamma=0 does not equal input tensor")

    def test_rep_blgr_numerical_equivalence(self):
        """Test overall RepBLGR reparameterization equivalence with non-zero gamma."""
        for global_type in ["fasa", "se"]:
            for bidirectional in [False, True]:
                with self.subTest(global_type=global_type, bidirectional=bidirectional):
                    model = RepBLGR(
                        dim=self.C,
                        num_heads=3,
                        window_size=2,
                        bidirectional=bidirectional,
                        global_type=global_type,
                        deploy=False
                    )
                    # Set gamma to non-zero to test residual path
                    with torch.no_grad():
                        model.gamma.fill_(1.0)

                    model.train()
                    for _ in range(5):
                        dummy = torch.randn(4, self.C, self.H, self.W)
                        _ = model(dummy)

                    model.eval()
                    with torch.no_grad():
                        out_train = model(self.x)

                    model.switch_to_deploy()
                    self.assertTrue(model.deploy)
                    self.assertTrue(model.local.deploy)

                    with torch.no_grad():
                        out_deploy = model(self.x)

                    max_diff = torch.max(torch.abs(out_train - out_deploy)).item()
                    print(f"[RepBLGR ({global_type}, bi={bidirectional})] Max diff: {max_diff:.8e}")
                    self.assertLess(max_diff, 1e-5, f"RepBLGR error {max_diff} exceeds tolerance 1e-5")

    def test_fasa_module_output_shape(self):
        """Test standalone FASA module."""
        fasa = FASA(dim=self.C, kernel_size=5, num_heads=3, window_size=2)
        fasa.eval()
        with torch.no_grad():
            out = fasa(self.x)
        self.assertEqual(out.shape, self.x.shape)


if __name__ == "__main__":
    unittest.main()
