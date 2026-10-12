from collections import defaultdict
import torch
from torch import nn
from pathlib import Path
import json

def collect_thresholds(model, loader, targets, mode='99.9', device=None, level='layer'):
    wanted=set(t.path for t in targets); vals=defaultdict(list); hooks=[]
    for name in wanted:
        mod=model.get_submodule(name)
        hooks.append(mod.register_forward_hook(lambda m,a,o,n=name: vals[n].append(o.detach().cpu())))
    model.eval(); device=device or next(model.parameters(), torch.tensor(0)).device
    with torch.no_grad():
        for batch in loader:
            model(batch.to(device) if torch.is_tensor(batch) else batch)
    for h in hooks: h.remove()
    out={}
    for n,v in vals.items():
        x=torch.cat(v) if v else torch.tensor(1.)
        # A channel threshold is one value per feature channel. Reduce batch
        # and spatial dimensions while retaining the channel dimension.
        if str(level).lower() == 'channel' and x.ndim >= 2:
            x = x.movedim(1, 0).reshape(x.shape[1], -1)
        else:
            x = x.reshape(-1)
        mode_name = str(mode).lower()
        if mode_name == 'max':
            out[n] = x.max(dim=-1).values.clamp_min(1e-6) if x.ndim > 1 else x.max().clamp_min(1e-6)
        elif mode_name == 'var':
            # Variance calibration uses a conservative mean-plus-three-sigma
            # scale; this remains stable for small calibration sets.
            out[n]=(x.mean(dim=-1) + 3*x.std(dim=-1, unbiased=False)).clamp_min(1e-6)
        else:
            out[n]=torch.quantile(x,float(mode)/100, dim=-1).clamp_min(1e-6) if x.ndim > 1 else torch.quantile(x,float(mode)/100).clamp_min(1e-6)
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
