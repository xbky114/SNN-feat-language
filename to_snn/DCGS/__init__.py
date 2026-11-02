from .config import DCGSConversionParams, ConversionParams
from .converter import convert, SNNArtifact
from .threshold import (CalibrationStatistics, collect_statistics, collect_thresholds,
                        thresholds_from_statistics, save_statistics, load_statistics)

def collect_activation_statistics(visual, dataloader, targets, *, model_name=None,
                                  mode='99.9', level='layer', device=None,
                                  directory=None, overwrite=False):
    """Save raw calibration statistics, without binding them to T/K/neuron."""
    from pathlib import Path
    from datetime import datetime, timezone
    model_name = model_name or visual.__class__.__name__.lower()
    method = str(mode).lower() if str(mode).lower() in ('var', 'max') else f'percentile_{str(mode).replace(".", "p")}'
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    root = Path(directory) if directory is not None else Path(__file__).parent / f'{model_name}_{method}_{stamp}'
    stats = collect_statistics(visual, dataloader, targets, mode=mode, level=level, device=device)
    save_statistics(stats, root, overwrite=overwrite)
    return root

Convert_by_DCGS = convert
