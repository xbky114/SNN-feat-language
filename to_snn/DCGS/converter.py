import copy

import torch
from torch import nn

from .neurons.if_rate import IFRate
from .neurons.mth_rate import MTHRate
from .threshold import (CalibrationStatistics, collect_statistics,
                        normalize_mode, thresholds_from_statistics)
from to_snn.prototype import SNNArtifact
from to_snn.validation import reject_unsupported_modules


def convert(visual, plan, params, calibration_loader=None, thresholds=None, *, statistics=None):
    """Convert from calibration data, raw statistics, or fixed thresholds."""
    reject_unsupported_modules(visual, 'DCGS')
    if params.neuron not in ('IF', 'MTH'):
        raise ValueError('neuron must be IF or MTH')
    for name in ('time_steps', 'num_thresholds'):
        value = getattr(params, name)
        if type(value) is not int or value <= 0:
            raise ValueError(f'{name} must be a positive integer')
    mode = normalize_mode(params.threshold_mode)
    level = str(params.threshold_level).lower()
    if level not in ('layer', 'channel', 'neuron'):
        raise ValueError('threshold_level must be layer, channel or neuron')
    paths = [target.path for target in plan.targets]
    if not paths or len(paths) != len(set(paths)):
        raise ValueError('DCGS requires nonempty, unique ReLU targets')
    for target in plan.targets:
        if (not target.path or target.kind not in ('relu', 'activation')
                or not isinstance(visual.get_submodule(target.path), nn.ReLU)):
            raise ValueError(f'DCGS target must expose a ReLU: {target.path!r}')
    if isinstance(thresholds, CalibrationStatistics):
        if statistics is not None:
            raise ValueError('provide statistics only once')
        statistics, thresholds = thresholds, None
    if thresholds is not None and statistics is not None:
        raise ValueError('provide either thresholds or statistics, not both')

    print(f'[DCGS] conversion started (neuron={params.neuron}, targets={len(paths)})', flush=True)
    model = copy.deepcopy(visual).eval()
    state = next(model.parameters(), None)
    if state is None:
        state = next(model.buffers(), torch.empty(0))
    if thresholds is None:
        if statistics is None:
            if calibration_loader is None:
                raise ValueError('calibration_loader, statistics or thresholds required')
            statistics = collect_statistics(model, calibration_loader, plan.targets,
                                            mode=mode, level=level)
        if not isinstance(statistics, CalibrationStatistics):
            raise TypeError('statistics must be CalibrationStatistics; use thresholds for legacy mappings')
        if normalize_mode(statistics.mode) != mode or statistics.level != level:
            raise ValueError('cached statistics mode/level do not match conversion parameters')
        thresholds = thresholds_from_statistics(statistics, time_steps=params.time_steps,
                                                num_thresholds=params.num_thresholds,
                                                neuron=params.neuron, device=state.device)
    missing = set(paths) - thresholds.keys()
    if missing:
        raise ValueError(f'missing thresholds for: {", ".join(sorted(missing))}')
    for target in plan.targets:
        threshold = torch.as_tensor(thresholds[target.path]).detach().clone()
        # Same boundary policy as upstream replacement: negatives -> zero,
        # zero remains zero. Never invent a positive epsilon threshold.
        threshold = threshold * (threshold >= 0).float()
        neuron = (IFRate(threshold) if params.neuron == 'IF'
                  else MTHRate(threshold, params.num_thresholds))
        parent, _, name = target.path.rpartition('.')
        holder = model.get_submodule(parent) if parent else model
        setattr(holder, name, neuron.to(state.device).eval())
    print(f'[DCGS] replaced {len(paths)} activation layers; conversion finished', flush=True)
    return SNNArtifact(model, params, plan).eval()
