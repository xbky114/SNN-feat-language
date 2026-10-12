from .config import DCGSConversionParams
from .converter import convert, VisualSNNArtifact
from .threshold import collect_thresholds, save_statistics, load_statistics

def collect_activation_statistics(visual, dataloader, targets, *, model_name=None, mode='99.9', overwrite=False):
    from pathlib import Path
    from datetime import datetime, timezone
    model_name = model_name or visual.__class__.__name__.lower()
    method = 'var' if str(mode).lower() == 'var' else f'percentile_{str(mode).replace(".", "p")}'
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    root=Path(__file__).parent / f'{model_name}_{method}_{stamp}'
    stats=collect_thresholds(visual, dataloader, targets, mode=mode)
    save_statistics(stats, root, overwrite=overwrite)
    return root

Convert_by_DCGS = convert
