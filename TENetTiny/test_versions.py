import importlib.util
from pathlib import Path
import unittest
import torch
from model import TENetTiny
from tiny_v1 import TinyMNIST


def historical(version):
    path = Path(__file__).resolve().parents[1] / 'Code/history' / f'model_arch_{version}_20260314.py'
    spec = importlib.util.spec_from_file_location('snapshot_' + version, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MNIST_TENet


class VersionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_historical_defaults_match_state_and_forward(self):
        for version, budget in [('current', 8555), ('legacy', 2360)]:
            with self.subTest(version=version):
                old = historical(version)().eval()
                new = TENetTiny(version).eval()
                self.assertEqual(sum(p.numel() for p in new.parameters()), budget)
                new.load_state_dict(old.state_dict(), strict=True)
                x = torch.randn(3, 1, 28, 28)
                torch.testing.assert_close(new(x), old(x), rtol=1e-6, atol=1e-6)

    def test_current_all_branches_match_snapshot(self):
        old = historical('current')(use_chebyshev=True, use_spatial_bridge=True).eval()
        new = TENetTiny('current', chebyshev=True, spatial=True).eval()
        new.load_state_dict(old.state_dict())
        x = torch.randn(3, 1, 28, 28)
        torch.testing.assert_close(new(x), old(x), rtol=1e-6, atol=1e-6)

    def test_2429_and_normalization_only_comparison(self):
        base = TENetTiny('current', head='linear', directed=False)
        self.assertEqual(sum(p.numel() for p in base.parameters()), 2429)
        for mode in ('raw', 'centered', 'standardized'):
            other = TENetTiny('current', head='linear', directed=False, normalization=mode)
            other.load_state_dict(base.state_dict(), strict=True)
            self.assertEqual(sum(p.numel() for p in other.parameters()), 2429)
            for i in (1, 2):
                layer = getattr(other, f'layer{i}')
                graph = torch.randn(4, i, 10, 10)
                expected = graph if mode == 'raw' else graph - graph.mean((2, 3), keepdim=True)
                if mode == 'standardized':
                    expected = expected / graph.std((2, 3), keepdim=True).clamp_min(1e-6)
                torch.testing.assert_close(layer.normalize_graph(graph), expected)

    def test_tiny_matches_original_with_mapped_weights(self):
        old, new = TinyMNIST().eval(), TENetTiny('tiny').eval()
        self.assertEqual(sum(p.numel() for p in new.parameters()), 2295)
        state = new.state_dict()
        for i in (1, 2):
            previous = getattr(old, f'layer{i}')
            prefix = f'layer{i}.'
            state[prefix + 'topology_extractor.weight'] = previous.projection.weight.reshape_as(state[prefix + 'topology_extractor.weight'])
            state[prefix + 'topology_extractor.bias'] = previous.projection.bias
            for name, value in previous.bn.state_dict().items():
                state[prefix + 'norm.' + name] = value
        state['classifier.1.weight'], state['classifier.1.bias'] = old.classifier.weight, old.classifier.bias
        new.load_state_dict(state)
        torch.testing.assert_close(new(torch.zeros(2, 1, 28, 28)), old(torch.zeros(2, 1, 28, 28)))
        x = torch.randn(2, 1, 28, 28)
        torch.testing.assert_close(new(x), old(x))

    def test_every_switch_for_every_version_has_finite_backward(self):
        for version in ('legacy', 'current', 'tiny'):
            for flag in ('learnable_phi', 'trace', 'edge', 'directed', 'cross', 'chebyshev', 'spatial', 'content', 'main_gate'):
                for enabled in (False, True):
                    options = {flag: enabled}
                    if flag == 'content' and not enabled:
                        options['cross'] = False
                    with self.subTest(version=version, flag=flag, enabled=enabled):
                        model = TENetTiny(version, in_channels=3, num_classes=7, image_size=8, **options)
                        output = model(torch.randn(2, 3, 8, 8))
                        torch.nn.functional.cross_entropy(output, torch.tensor([1, 2])).backward()
                        self.assertTrue(torch.isfinite(output).all())
                        grads = [p.grad for p in model.parameters() if p.grad is not None]
                        self.assertTrue(grads)
                        self.assertTrue(all(torch.isfinite(g).all() for g in grads))

    def test_corrected_windows_and_cross_validation(self):
        model = TENetTiny('current', window_layout='corrected')
        x = torch.arange(2 * 2 * 4 * 4.).reshape(2, 2, 4, 4)
        expected = torch.stack([sample[:, row:row+2, col:col+2].reshape(2, 4)
            for sample in x for row in (0, 2) for col in (0, 2)])
        torch.testing.assert_close(model.layer1.windows(x), expected)
        with self.assertRaisesRegex(ValueError, 'cross requires'):
            TENetTiny('current', content=False)


if __name__ == '__main__':
    unittest.main()
