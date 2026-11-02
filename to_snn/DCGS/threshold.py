from dataclasses import dataclass
import math
import torch
from torch import nn
from pathlib import Path
import json
from to_snn.validation import reject_unsupported_modules

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



@dataclass(frozen=True)
class CalibrationStatistics:
    """ANN statistics independent of the eventual neuron, T and K.

    Each layer contains mean/var/count for var mode, or scale/count for
    percentile/max mode. Tensors retain singleton dimensions for broadcasting.
    """
    mode: str
    level: str
    layers: dict


def normalize_mode(mode):
    name = str(mode).strip().lower()
    if name in ('var', 'max'):
        return name
    percentile = float(name.rstrip('%'))
    if not math.isfinite(percentile) or not 0 < percentile < 100:
        raise ValueError('percentile must be strictly between 0 and 100')
    return f'{percentile:g}%'


def _reduction_dims(x, level):
    if not isinstance(x, torch.Tensor) or not x.ndim or not x.numel():
        raise ValueError('calibration activations must be nonempty batched tensors')
    if level == 'neuron':
        return (0,)
    if level == 'channel':
        if x.ndim < 2:
            raise ValueError('channel calibration requires a channel axis')
        return tuple(i for i in range(x.ndim) if i != 1)
    return tuple(range(x.ndim))


@torch.no_grad()
def collect_statistics(model, loader, targets, mode='99.9', device=None, level='layer'):
    """Collect ANN statistics in eval mode, cleaning hooks on every exit."""
    reject_unsupported_modules(model, 'DCGS')
    mode = normalize_mode(mode)
    level = str(level).lower()
    if level not in ('layer', 'channel', 'neuron'):
        raise ValueError('threshold_level must be layer, channel or neuron')
    targets = tuple(targets)
    wanted = [target.path for target in targets]
    if not wanted or len(wanted) != len(set(wanted)):
        raise ValueError('calibration requires nonempty, unique targets')
    if loader is None:
        raise ValueError('calibration loader is required')
    for name in wanted:
        if not isinstance(model.get_submodule(name), nn.ReLU):
            raise ValueError(f'DCGS target must be a ReLU: {name!r}')
    state = next(model.parameters(), None)
    if state is None:
        state = next(model.buffers(), torch.empty(0))
    device = device or state.device
    layers, hooks = {}, []
    training = [(module, module.training) for module in model.modules()]

    def record(name, activation):
        x = activation.detach()
        dims = _reduction_dims(x, level)
        count = x.shape[0]
        if mode == 'var':
            current = {'mean': x.mean(dim=dims, keepdim=True).cpu(),
                       'var': x.var(dim=dims, keepdim=True, unbiased=False).cpu()}
        elif mode == 'max':
            current = {'scale': x.amax(dim=dims, keepdim=True).cpu()}
        else:
            if x.dtype in (torch.float16, torch.bfloat16):
                x = x.float()
            remaining = tuple(i for i in range(x.ndim) if i not in dims)
            size = math.prod(x.shape[i] for i in dims)
            merged = x.permute(dims + remaining).reshape(size, -1)
            value = torch.quantile(merged, float(mode[:-1]) / 100, dim=0)
            shape = tuple(1 if i in dims else x.shape[i] for i in range(x.ndim))
            current = {'scale': value.reshape(shape).cpu()}
        if name in layers:
            old = layers[name]
            if any(old[key].shape != value.shape for key, value in current.items()):
                raise ValueError(f'calibration statistic shape changed at {name!r}')
            weight = count / (old['count'] + count)
            if mode == 'var':
                delta = current['mean'] - old['mean']
                current['var'] = ((1 - weight) * old['var']
                                  + weight * (current['var'] + (1 - weight) * delta.square()))
                current['mean'] = (1 - weight) * old['mean'] + weight * current['mean']
            else:
                current['scale'] = (1 - weight) * old['scale'] + weight * current['scale']
            count += old['count']
        layers[name] = {**current, 'count': count}

    print(f'[DCGS] calibrating {len(wanted)} layers (mode={mode}, level={level})', flush=True)
    try:
        model.eval()
        for name in wanted:
            module = model.get_submodule(name)
            if mode == 'var':
                hooks.append(module.register_forward_pre_hook(
                    lambda m, args, name=name: record(name, args[0])))
            else:
                hooks.append(module.register_forward_hook(
                    lambda m, args, output, name=name: record(name, output)))
        for index, batch in enumerate(loader, 1):
            images = batch[0] if isinstance(batch, (tuple, list)) and batch else batch
            if not isinstance(images, torch.Tensor):
                raise TypeError('calibration batches must be tensors or (images, labels)')
            model(images.to(device))
            if index == 1 or index % 10 == 0:
                print(f'[DCGS] calibration batch {index}', flush=True)
    finally:
        for hook in hooks:
            hook.remove()
        for module, was_training in training:
            module.training = was_training
    missing = set(wanted) - layers.keys()
    if missing:
        raise ValueError(f'no calibration activations for: {", ".join(sorted(missing))}')
    return CalibrationStatistics(mode, level, layers)


def thresholds_from_statistics(statistics, *, time_steps, num_thresholds, neuron, device=None):
    """Resolve raw statistics for the current conversion parameters."""
    if not isinstance(statistics, CalibrationStatistics):
        raise TypeError('statistics must be CalibrationStatistics')
    if neuron not in ('IF', 'MTH'):
        raise ValueError('neuron must be IF or MTH')
    if type(time_steps) is not int or time_steps <= 0:
        raise ValueError('time_steps must be a positive integer')
    if type(num_thresholds) is not int or num_thresholds <= 0:
        raise ValueError('num_thresholds must be a positive integer')
    samples = time_steps * (2 ** num_thresholds) if neuron == 'MTH' else time_steps
    result = {}
    for name, values in statistics.layers.items():
        if statistics.mode == 'var':
            mean = values['mean'].to(device=device)
            var = values['var'].to(device=device)
            scale = torch.ones_like(mean)
            # Neuron-level statistics can contain a full feature map. Solve
            # independent elements in chunks instead of allocating map_size*n.
            chunk_size = max(1, 1_048_576 // samples)
            flat_mean, flat_var, flat_scale = mean.reshape(-1), var.reshape(-1), scale.reshape(-1)
            for start in range(0, mean.numel(), chunk_size):
                stop = start + chunk_size
                current = flat_scale[start:stop].clone()
                for _ in range(100):
                    current = current * _calculate_better(
                        flat_mean[start:stop], flat_var[start:stop], current, samples)
                flat_scale[start:stop] = current
            # Match upstream: only NaN falls back to the initial value 1.
            scale = torch.where(torch.isnan(scale), torch.ones_like(scale), scale)
        else:
            scale = values['scale'].clone()
        result[name] = (scale * (scale >= 0).float()).detach().cpu()
    return result


def collect_thresholds(model, loader, targets, mode='99.9', device=None,
                       level='layer', time_steps=1, num_thresholds=8, neuron='MTH'):
    """Compatibility API returning final thresholds rather than raw statistics."""
    statistics = collect_statistics(model, loader, targets, mode, device, level)
    state = next(model.parameters(), None)
    if state is None:
        state = next(model.buffers(), torch.empty(0))
    return thresholds_from_statistics(statistics, time_steps=time_steps,
                                      num_thresholds=num_thresholds, neuron=neuron,
                                      device=device or state.device)


def save_statistics(statistics, directory, *, overwrite=False):
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()) and not overwrite:
        raise FileExistsError(f'statistics directory is not empty: {directory}')
    directory.mkdir(parents=True, exist_ok=True)
    if isinstance(statistics, CalibrationStatistics):
        payload = {'mode': statistics.mode, 'level': statistics.level,
                   'layers': statistics.layers}
        torch.save(payload, directory / 'statistics.pt')
        metadata = {'format': 'dcgs-statistics-v2', 'mode': statistics.mode,
                    'level': statistics.level}
    else:
        # Preserve the legacy final-threshold mapping API.
        torch.save(statistics, directory / 'thresholds.pt')
        metadata = {'format': 'dcgs-thresholds-v1'}
    (directory / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')


def load_statistics(directory):
    directory = Path(directory)
    metadata_path = directory / 'metadata.json'
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    version = metadata.get('format', 'dcgs-thresholds-v1')
    if version == 'dcgs-statistics-v2':
        payload = torch.load(directory / 'statistics.pt', map_location='cpu', weights_only=True)
        return CalibrationStatistics(**payload)
    if version != 'dcgs-thresholds-v1':
        raise ValueError(f'unknown DCGS statistics format: {version}')
    return torch.load(directory / 'thresholds.pt', map_location='cpu', weights_only=True)
