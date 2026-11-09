import unittest

import torch

from to_snn.neurons import TwoSide, TwoSideNeuron


class TwoSideNeuronTests(unittest.TestCase):
    def test_positive_and_negative_trajectory(self):
        neuron = TwoSide(positive_threshold=1.0, negative_threshold=1.0,
                         num_thresholds=1, fused=False)
        self.assertTrue(torch.allclose(neuron(torch.tensor([0.4])), torch.tensor([0.0])))
        self.assertTrue(torch.allclose(neuron(torch.tensor([0.6])), torch.tensor([1.0])))
        # A negative excursion retracts the previously emitted positive output.
        self.assertTrue(torch.allclose(neuron(torch.tensor([-1.0])), torch.tensor([-1.0])))
        self.assertTrue(torch.allclose(neuron.cumulative_out, torch.tensor([0.0])))

    def test_geometric_levels_select_incremental_spikes(self):
        neuron = TwoSide(1.0, 1.0, num_thresholds=3, gap=2.0, fused=False)
        # Thresholds are 1, 1/2, 1/4 and the source boundary is 0.75 * threshold.
        out = neuron(torch.tensor([1.0]))
        self.assertTrue(torch.allclose(out, torch.tensor([1.0])))
        self.assertTrue(torch.allclose(neuron.mem, torch.tensor([0.0])))

    def test_geometric_levels_use_base_offset_boundaries(self):
        neuron = TwoSide(1.0, 1.0, num_thresholds=3, gap=2.0, fused=False)
        # Levels are 4, 2, 1; boundaries are 3.5, 1.5, 0.5.
        self.assertTrue(torch.allclose(neuron(torch.tensor([1.6])), torch.tensor([2.0])))

    def test_reset_restarts_trajectory_and_alias_is_exported(self):
        self.assertIs(TwoSideNeuron, TwoSide)
        neuron = TwoSide(1.0, 1.0, num_thresholds=1, fused=False)
        first = neuron(torch.tensor([1.0]))
        neuron.reset_state()
        second = neuron(torch.tensor([1.0]))
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(neuron.T, 1)

    def test_changed_shape_and_dtype_reinitializes_state(self):
        neuron = TwoSide(1.0, 1.0, num_thresholds=1, fused=False)
        neuron(torch.ones(2, dtype=torch.float32))
        neuron(torch.ones(3, dtype=torch.float64))
        self.assertEqual(tuple(neuron.mem.shape), (3,))
        self.assertEqual(neuron.mem.dtype, torch.float64)
        self.assertEqual(neuron.T, 1)

    def test_invalid_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            TwoSide(1.0, 1.0, num_thresholds=0)
        with self.assertRaises(ValueError):
            TwoSide(1.0, 1.0, gap=0)
        with self.assertRaises(ValueError):
            TwoSide(torch.ones(2), 1.0)


if __name__ == "__main__":
    unittest.main()
