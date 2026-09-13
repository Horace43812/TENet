import unittest
import torch
from model import TinyMNIST, TinyGraph


class TinyTests(unittest.TestCase):
    def test_budget_shape_and_gradients(self):
        torch.set_num_threads(1)
        for normalization in ("raw", "centered", "standardized"):
            model = TinyMNIST(normalization)
            self.assertEqual(sum(p.numel() for p in model.parameters()), 2295)
            logits = model(torch.randn(4, 1, 28, 28))
            self.assertEqual(logits.shape, (4, 10))
            torch.nn.functional.cross_entropy(logits, torch.tensor([1, 2, 3, 4])).backward()
            for parameter in model.parameters():
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_multichannel_windows_and_raw_formula(self):
        x = torch.arange(2 * 2 * 4 * 6, dtype=torch.float32).reshape(2, 2, 4, 6)
        layer = TinyGraph(2, 3)
        expected = []
        for sample in x:
            for row in range(0, 4, 2):
                for col in range(0, 6, 2):
                    v = sample[:, row:row + 2, col:col + 2].reshape(2, 4)
                    e = torch.stack([(v[:, a] - v[:, b]).abs() for a, b in layer.pairs], -1)
                    z = torch.cat((v, e), -1)
                    expected.append(torch.stack([torch.diag(channel) @ layer.L @ torch.diag(channel) for channel in z]))
        raw = layer.graph_matrix(x)
        torch.testing.assert_close(raw, torch.stack(expected))
        layer.normalization = "centered"
        torch.testing.assert_close(layer.graph_matrix(x), raw - raw.mean((-1, -2), keepdim=True))
        layer.normalization = "standardized"
        torch.testing.assert_close(layer.graph_matrix(x), (raw - raw.mean((-1, -2), keepdim=True)) / raw.std((-1, -2), keepdim=True).clamp_min(1e-6))


if __name__ == "__main__":
    unittest.main()
