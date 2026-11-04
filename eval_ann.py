"""Evaluate ANN CLIP on the fixed three-level hierarchy."""
import argparse,json
from pathlib import Path
import torch
from dataloader import build_dataloader
from prompt import build_prompt
from CLIPs import get_clip

def evaluate(model, loader, device="cpu", prompt_father=False):
    ds=loader.dataset; names=ds.class_names; levels=list(names)
    idx={k:{n:i for i,n in enumerate(names[k])} for k in levels}
    prompts={"basic":[build_prompt(n) for n in names["basic"]],"subordinate":[build_prompt(n,ds.parent["subordinate"][n]) if prompt_father else build_prompt(n) for n in names["subordinate"]],"fine_grained":[build_prompt(n,ds.parent["fine_grained"][n]) if prompt_father else build_prompt(n) for n in names["fine_grained"]]}
    text={k:model.encode_text(model.tokenize(prompts[k]).to(device)) for k in levels}; text={k:v/v.norm(dim=-1,keepdim=True) for k,v in text.items()}
    correct={k:0 for k in levels}; selected={k:0 for k in levels}; depth=total=0; saved={k:[] for k in levels}; paths=[]
    for batch in loader:
        x=batch['image'].to(device); b=x.shape[0]; feat=model.encode_image(x); feat=feat/feat.norm(dim=-1,keepdim=True)
        logits={k:model.logit_scale.exp()*feat@text[k].T for k in levels}; target={k:torch.tensor([idx[k][v] for v in batch[k]],device=device) for k in levels}
        path=torch.stack([logits[k][torch.arange(b,device=device),target[k]] for k in levels],1); p=path.softmax(1)
        paths.append(path.cpu()); [saved[k].append(logits[k].cpu()) for k in levels]
        depth+=(p*torch.tensor([0.,.5,1.],device=device)).sum(1).sum().item()
        for i,k in enumerate(levels): correct[k]+=(logits[k].argmax(1)==target[k]).sum().item(); selected[k]+=(path.argmax(1)==i).sum().item()
        total+=b
    return {'samples':total,'semantic_depth':depth/total,'accuracy':{k:correct[k]/total for k in levels},'selectivity':{k:selected[k]/total for k in levels},'_logits':{k:torch.cat(saved[k]) for k in levels},'_path_logits':torch.cat(paths)}
def main():
 p=argparse.ArgumentParser(); p.add_argument('--data',required=True); p.add_argument('--clip',required=True); p.add_argument('--output-dir',required=True); p.add_argument('--batch-size',type=int,default=128); p.add_argument('--workers',type=int,default=4); p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu'); p.add_argument('--prompt-father',action='store_true'); a=p.parse_args(); m=get_clip(a.clip,device=a.device).eval(); l=build_dataloader(a.data,batch_size=a.batch_size,image_size=m.input_resolution,num_workers=a.workers); s=evaluate(m,l,a.device,a.prompt_father); Path(a.output_dir).mkdir(parents=True,exist_ok=True); torch.save({**{f'logits_{k}':s.pop('_logits')[k] for k in l.dataset.class_names},'path_logits':s.pop('_path_logits')},Path(a.output_dir)/'logits.pt'); (Path(a.output_dir)/'metrics.json').write_text(json.dumps(s,indent=2)+'\n'); print(json.dumps(s))
if __name__=='__main__': main()
