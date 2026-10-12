import ast
from pathlib import Path
import unittest
import torch
from torch import nn
from to_snn.DCGS.threshold import _calculate_better, collect_thresholds
from to_snn.prototype import ConversionTarget

source = ast.parse(Path('ANN-to-SNN-DCGS/converter/threshold_getter.py').read_text())
selected = [n for n in source.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in ('calculate_better', 'ThreHook')]
ns = {'torch': torch, 'nn': nn}
exec(compile(ast.Module(body=selected, type_ignores=[]), '<dcgs-reference>', 'exec'), ns)

class VarReference(unittest.TestCase):
    def test_formula(self):
        mean = torch.tensor([-0.2, 0.5])
        var = torch.tensor([0.8, 1.2])
        scale = torch.tensor([2., 3.])
        torch.testing.assert_close(_calculate_better(mean, var, scale, 64), ns['calculate_better'](mean, var, scale, 64))

    def test_inplace_unequal_batches(self):
        torch.manual_seed(17)
        batches = [torch.randn(3, 2, 4, 4)-0.7, torch.randn(1, 2, 4, 4)+1.3]
        ref = ns['ThreHook'](mode='var', level='channel', out_layer=nn.ReLU(inplace=True))
        for batch in batches:
            ref(batch.clone())
        ref.get_scale_from_var(T=64)
        model = nn.Sequential(nn.ReLU(inplace=True))
        actual = collect_thresholds(model, [b.clone() for b in batches], [ConversionTarget('0')], mode='var', level='channel', time_steps=16, num_thresholds=2)
        torch.testing.assert_close(actual['0'], ref.scale.flatten())

if __name__ == '__main__':
    unittest.main()
