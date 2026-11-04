"""Evaluate an SNN visual tower with fixed three-level semantic labels."""
from __future__ import annotations
import argparse, importlib, json
from pathlib import Path
import torch
from CLIPs import get_clip
from clip_with_snn_visual import CLIPWithSNNVisual
from dataloader import build_dataloader
from prompt import build_prompt
from to_snn.prototype import SNNArtifact
from to_snn.coding import CODING

def _coding(name):
    value=getattr(importlib.import_module('to_snn.coding'),name,None)
    if not isinstance(value,CODING): raise ValueError(f'unknown coding instance: {name!r}')
    return value

@torch.no_grad()
def evaluate(model, loader, device, timesteps, images_per_file, output_dir, print_interval, log=None, prompt_father=False):
    ds = loader.dataset
    levels = list(ds.class_names)
    names = ds.class_names
    idx={k:{n:i for i,n in enumerate(names[k])} for k in levels}
    prompts={'basic':[build_prompt(n) for n in names['basic']],
             'subordinate':[build_prompt(n,ds.parent['subordinate'][n]) if prompt_father else build_prompt(n) for n in names['subordinate']],
             'fine_grained':[build_prompt(n,ds.parent['fine_grained'][n]) if prompt_father else build_prompt(n) for n in names['fine_grained']]}
    logits_text = {k: prompts[k] for k in levels}
    total = 0
    depth = torch.zeros(timesteps)
    correct = {k: torch.zeros(timesteps) for k in levels}
    selected = {k: torch.zeros(timesteps) for k in levels}
    shard = []
    shard_no = 0
    for batch_no,batch in enumerate(loader,1):
        images = batch['image'].to(device)
        b = images.shape[0]
        features=model.encode_image(images,timesteps)
        logits={k:model.calc_logits_from_image_features(features,logits_text[k]) for k in levels}
        target={k:torch.tensor([idx[k][v] for v in batch[k]],device=device) for k in levels}
        path=torch.stack([logits[k].gather(2,target[k][None,:,None].expand(timesteps,-1,1)).squeeze(-1) for k in levels],2)
        depth += (path.softmax(2)*torch.tensor([0.,.5,1.],device=device)).sum((1,2)).cpu()
        for i,k in enumerate(levels):
            correct[k] += (logits[k].argmax(2)==target[k][None,:]).sum(1).cpu()
            selected[k] += (path.argmax(2)==i).sum(1).cpu()
        total += b
        shard.append({'path_logits':path.cpu(), **{f'logits_{k}':logits[k].cpu() for k in levels}})
        if sum(x['path_logits'].shape[1] for x in shard) >= images_per_file:
            torch.save({k: torch.cat([x[k] for x in shard], 1) for k in shard[0]}, output_dir / f'logits_{shard_no:05d}.pt')
            shard = []
            shard_no += 1
        if log and batch_no % print_interval == 0:
            print(f'batch={batch_no} samples={total}', file=log, flush=True)
    if shard: torch.save({k:torch.cat([x[k] for x in shard],1) for k in shard[0]}, output_dir/f'logits_{shard_no:05d}.pt')
    per=[{'timestep':i+1,'semantic_depth':depth[i].item()/total,'accuracy':{k:correct[k][i].item()/total for k in levels},'selectivity':{k:selected[k][i].item()/total for k in levels}} for i in range(timesteps)]
    return {'samples':total,'timesteps':timesteps,'final':per[-1],'per_timestep':per}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--clip', required=True)
    parser.add_argument('--snn', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--coding', required=True)
    parser.add_argument('--T', type=int, required=True)
    parser.add_argument('--images-per-file', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--print-interval', type=int, default=20)
    parser.add_argument('--prompt-father', action='store_true')
    args = parser.parse_args()
    if args.T < 1:
        parser.error('--T must be positive')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ann = get_clip(args.clip, device=args.device)
    model = CLIPWithSNNVisual(ann, SNNArtifact.load(args.snn, map_location=args.device), _coding(args.coding))
    loader = build_dataloader(args.data, batch_size=args.batch_size, image_size=model.input_resolution, num_workers=args.workers)
    with (output_dir / 'evaluation.log').open('w') as log:
        summary = evaluate(model, loader, args.device, args.T, args.images_per_file, output_dir, args.print_interval, log, args.prompt_father)
        print(json.dumps(summary['final'], sort_keys=True), file=log)
    (output_dir / 'metrics.json').write_text(json.dumps(summary, indent=2) + '\n')
if __name__=='__main__': main()
