from collections import defaultdict
import torch
from torch import nn
from pathlib import Path
import json

def _calculate_better(mean, var, scale, n=64):
    # 确保输入张量的数据类型和设备一致
    device = mean.device
    dtype = mean.dtype
    
    # 定义基本参数
    sigma = torch.sqrt(var)  # 标准差
    theta = scale            # 缩放因子
    pi = torch.tensor(torch.pi, dtype=dtype, device=device)  # π 值
    
    # 生成 i 的序列 (1, 2, ..., n)
    i = torch.arange(1, n + 1, dtype=dtype, device=device).view(*([1] * mean.dim()), n)
    # 例如，如果 mean 的形状是 (batch_size, channels, height, width)，则 i 的形状将是 (1, 1, 1, 1, n)
    
    # 计算 coeff，确保维度匹配
    coeff = ((2 * i - 1) / (2 * n)) * theta.unsqueeze(-1)  # 在最后一个维度添加 n 的维度
    
    # 计算 (coeff - mean) / (sqrt(2) * sigma)
    sqrt2 = torch.sqrt(torch.tensor(2.0, dtype=dtype, device=device))
    erf_input = (coeff - mean.unsqueeze(-1)) / (sqrt2 * sigma.unsqueeze(-1))
    erf_term = torch.erf(erf_input)
    
    # 第一部分中的和
    term1_sum = torch.sum((1 / n) * erf_term, dim=-1)
    
    # 分母中的和
    term2_sum = torch.sum(((2 * i - 1) / n**2) * erf_term, dim=-1)
    
    # 计算 exp 项
    exp_input = -((coeff - mean.unsqueeze(-1))**2) / (2 * var.unsqueeze(-1))
    exp_term = torch.exp(exp_input)
    term3_sum = torch.sum((1 / n) * exp_term, dim=-1)
    
    # 分母
    denominator = 1 - term2_sum
    
    # 计算部分1和部分2
    part1 = (mean / theta) * (1 - term1_sum) / denominator
    sqrt_pi_over_2 = torch.sqrt(pi / 2)
    part2 = (sigma / (sqrt_pi_over_2 * theta)) * term3_sum / denominator
    
    # 合并得到 k1
    k1 = part1 + part2
    return k1



def collect_thresholds(model, loader, targets, mode='99.9', device=None,
                       level='layer', time_steps=1, num_thresholds=8):
    wanted=set(t.path for t in targets); vals=defaultdict(list); hooks=[]
    moments = {}
    def record_moments(name, x):
        dims = tuple(i for i in range(x.ndim) if level != 'channel' or i != 1)
        mean = x.mean(dim=dims).detach()
        var = x.var(dim=dims, unbiased=False).detach()
        count = x.shape[0]
        if name in moments:
            old_mean, old_var, seen = moments[name]
            weight = count / (seen + count)
            delta = mean - old_mean
            var = (1 - weight) * old_var + weight * (var + (1 - weight) * delta * delta)
            mean = (1 - weight) * old_mean + weight * mean
            count += seen
        moments[name] = (mean.clone(), var.clone(), count)
    for name in wanted:
        mod=model.get_submodule(name)
        if str(mode).lower() == 'var':
            hooks.append(mod.register_forward_pre_hook(
                lambda m,a,n=name: record_moments(n, a[0])))
        else:
            hooks.append(mod.register_forward_hook(lambda m,a,o,n=name: vals[n].append(o.detach().cpu())))
    model.eval(); device=device or next(model.parameters(), torch.tensor(0)).device
    with torch.no_grad():
        for batch in loader:
            model(batch.to(device) if torch.is_tensor(batch) else batch)
    for h in hooks: h.remove()
    out={}
    if str(mode).lower() == 'var':
        samples = int(time_steps) * (2 ** int(num_thresholds))
        for name, (mean, var, _) in moments.items():
            scale = torch.ones_like(mean)
            initial = scale.clone()
            for _ in range(100):
                scale = scale * _calculate_better(mean, var, scale, samples)
            mask = ~torch.isfinite(scale)
            scale[mask] = initial[mask]
            # Original converter zeros negative scales before constructing MTH.
            out[name] = (scale * (scale >= 0).float()).cpu()
        return out
    for name,v in vals.items():
        # Keep the original DCGS batch-wise calibration behavior.  Concatenating
        # all RN50 feature maps can exceed torch.quantile's tensor-size limit.
        if not v:
            x = torch.tensor(1.)
        else:
            x = v[0]
        # A channel threshold is one value per feature channel. Reduce batch
        # and spatial dimensions while retaining the channel dimension.
        if str(level).lower() == 'channel' and x.ndim >= 2:
            batches = [z.movedim(1, 0).reshape(z.shape[1], -1) for z in v]
        else:
            batches = [z.reshape(-1) for z in v]
        x = batches[0] if batches else torch.tensor(1.)
        mode_name = str(mode).lower()
        if mode_name == 'max':
            out[name] = x.max(dim=-1).values.clamp_min(1e-6) if x.ndim > 1 else x.max().clamp_min(1e-6)
        else:
            q = str(mode).strip().rstrip('%')
            q = float(q) / 100.0
            # Original DCGS computes a percentile per calibration batch and
            # updates the stored scale with a sample-weighted average.
            out[name] = torch.stack([torch.quantile(z, q, dim=-1) for z in batches]).mean(0).clamp_min(1e-6)
    return out

def save_statistics(statistics, directory, *, overwrite=False):
    directory=Path(directory)
    if directory.exists() and any(directory.iterdir()) and not overwrite:
        raise FileExistsError(f'statistics directory is not empty: {directory}')
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(statistics, directory/'thresholds.pt')
    (directory/'metadata.json').write_text(json.dumps({'format':'dcgs-thresholds-v1'}, indent=2))

def load_statistics(directory):
    return torch.load(Path(directory)/'thresholds.pt', map_location='cpu', weights_only=True)
