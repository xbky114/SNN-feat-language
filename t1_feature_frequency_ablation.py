#!/usr/bin/env python3
import argparse, json, sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data',type=Path,default=Path('./data/auxiliary_2000')); p.add_argument('--output-dir',type=Path,default=Path('./results/verifications/t1_feature_frequency_ablation')); p.add_argument('--clip-dir',type=Path,default=Path(__file__).parent/'CLIP'); p.add_argument('--device',default='cuda:0'); p.add_argument('--batch-size',type=int,default=64); p.add_argument('--num-workers',type=int,default=4); p.add_argument('--max-images',type=int,default=None); a=p.parse_args()
    sys.path.insert(0,str(a.clip_dir)); import clip
    model, preprocess=clip.load('RN50',device=a.device,jit=False); model.eval(); v=model.visual
    files=sorted([x for x in a.data.rglob('*') if x.suffix.lower() in {'.jpg','.jpeg','.png','.bmp','.webp'}])
    if a.max_images: files=files[:a.max_images]
    if not files: raise ValueError(f'No images found under {a.data}')
    def get(i):
        with Image.open(files[i]) as im: return preprocess(im.convert('RGB'))
    loader=DataLoader(range(len(files)),batch_size=a.batch_size,num_workers=a.num_workers,shuffle=False,collate_fn=lambda z:torch.stack([get(i) for i in z]))
    def feat(x):
        x=x.type(v.conv1.weight.dtype)
        x=v.relu1(v.bn1(v.conv1(x))); x=v.relu2(v.bn2(v.conv2(x))); x=v.relu3(v.bn3(v.conv3(x))); x=v.avgpool(x); x=v.layer1(x); x=v.layer2(x); x=v.layer3(x); return v.layer4(x)
    # First pass: channel-wise 99th percentile of h.
    vals=[]
    with torch.inference_mode():
        for x in loader: vals.append(feat(x.to(a.device, non_blocking=True)).float().cpu())
    hs=torch.cat(vals); vth=torch.quantile(hs.permute(1,0,2,3).reshape(hs.shape[1],-1),0.99,dim=1)
    H,W=hs.shape[-2:]; yy,xx=np.meshgrid(np.fft.fftshift(np.fft.fftfreq(H)),np.fft.fftshift(np.fft.fftfreq(W)),indexing='ij'); D2=yy*yy+xx*xx; sigmas=[.10,.20,.30,.40,.50]; deltas={f'sigma_{s:.2f}':[] for s in sigmas}
    for h in hs:
        base=torch.relu(h); spike=(h>=vth[:,None,None]).float(); denom=spike.sum().item()
        if denom==0: continue
        # Center the spectrum and apply Gaussian frequency attenuation.
        arr=np.fft.fftshift(np.fft.fft2(h.numpy(),axes=(-2,-1)),axes=(-2,-1))
        for s in sigmas:
            f=arr*np.exp(-D2/(2*s*s)); hf=torch.from_numpy(np.fft.ifft2(np.fft.ifftshift(f,axes=(-2,-1)),axes=(-2,-1)).real).float(); sf=(hf>=vth[:,None,None]).float(); deltas[f'sigma_{s:.2f}'].append(float((spike-sf).abs().sum()/denom))
    out=a.output_dir; out.mkdir(parents=True,exist_ok=True); means={k:float(np.mean(x)) for k,x in deltas.items()}; json.dump({'mean_delta':means,'num_images':len(files),'threshold_percentile':99},open(out/'results.json','w'),indent=2)
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    means=[np.mean(x) for x in deltas.values()]; stds=[np.std(x) for x in deltas.values()]; fig,ax=plt.subplots(figsize=(7,5)); ax.bar(list(deltas),means,yerr=stds,capsize=4); ax.set_ylabel('delta (mean +/- std)'); ax.set_xlabel('Gaussian sigma');
    fig.tight_layout(); fig.savefig(out/'delta_histograms.png',dpi=180)
if __name__=='__main__': main()
