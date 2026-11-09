import unittest
import torch
from torch import nn

from to_snn.ECMT.state import ExpectationCompensation, MatrixProductCompensation


class TestExpectationCompensation(unittest.TestCase):
    def test_temporal_recurrence_and_reset(self):
        fn = nn.Sigmoid()
        module = ExpectationCompensation(fn)
        x1 = torch.tensor([-1.0, 1.0])
        x2 = torch.tensor([1.0, 3.0])
        y1 = module(x1)
        self.assertTrue(torch.allclose(y1, torch.sigmoid(x1)))
        y2 = module(x2)
        expected = 2 * torch.sigmoid((x1 + x2) / 2) - torch.sigmoid(x1)
        self.assertTrue(torch.allclose(y2, expected))
        module.reset_state()
        self.assertTrue(torch.allclose(module(x1), y1))

    def test_input_shape_change_reinitializes(self):
        module = ExpectationCompensation(nn.Identity())
        module(torch.ones(2))
        output = module(torch.ones(3))
        self.assertEqual(tuple(output.shape), (3,))
        self.assertEqual(module.steps, 1)


class TestMatrixProductCompensation(unittest.TestCase):
    def test_two_input_recurrence(self):
        module = MatrixProductCompensation()
        a1 = torch.tensor([[1.0, 2.0]])
        b1 = torch.tensor([[3.0], [4.0]])
        a2 = torch.tensor([[2.0, -1.0]])
        b2 = torch.tensor([[1.0], [2.0]])
        self.assertTrue(torch.equal(module(a1, b1), a1 @ b1))
        out = module(a2, b2)
        expected = ((a1 + a2) @ (b1 + b2)) / 2 - (a1 @ b1)
        self.assertTrue(torch.allclose(out, expected))
        module.reset_state()
        self.assertTrue(torch.equal(module(a1, b1), a1 @ b1))

    def test_shape_change_reinitializes(self):
        module = MatrixProductCompensation()
        module(torch.ones(1, 2), torch.ones(2, 1))
        output = module(torch.ones(2, 2), torch.ones(2, 1))
        self.assertEqual(tuple(output.shape), (2, 1))
        self.assertEqual(module.steps, 1)


if __name__ == '__main__':
    unittest.main()
