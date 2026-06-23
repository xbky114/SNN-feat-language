import json
from functools import partial
from pathlib import Path

from llava.eval.run_llava import eval_model
from llava.mm_utils import get_model_name_from_path

from precision_funcs import topk_channel_precision

model_path = "liuhaotian/llava-v1.5-7b"
prompt = "Describe one main object in the picture in detail. Do not guess."
image_file = "./results/image.png"

base_args = {
    "model_path": model_path,
    "model_base": None,
    "model_name": get_model_name_from_path(model_path),
    "query": prompt,
    "conv_mode": None,
    "image_file": image_file,
    "sep": ",",
    "temperature": 0,
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 512,
}

ks = [None, 896, 768, 512, 384, 256, 128]
results = []

for k in ks:
    if k is None:
        args = type("Args", (), {
            **base_args,
            "set_precision": None,
        })()
    else:
        args = type("Args", (), {
            **base_args,
            "set_precision": partial(topk_channel_precision, k=k),
        })()
    output = eval_model(args)
    print(f"k={k}: {output}")
    results.append({"k": k, "output": output})

Path("results").mkdir(exist_ok=True)
with open("results/topk_channels.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
