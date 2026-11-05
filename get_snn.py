"""Convert a CLIP visual tower to an SNN artifact."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from dataloader import DATASETS, build_dataloader
from to_snn import convert_clip, load_conversion_params
from to_snn.prototype import SNNArtifact


def save_conversion_result(result, output):
    """Save a single artifact or an artifact accompanied by optional info."""
    info = None
    if isinstance(result, (tuple, list)):
        if len(result) not in (1, 2):
            raise ValueError("conversion must return artifact or (artifact, info)")
        artifact = result[0]
        if len(result) == 2:
            info = result[1]
    else:
        artifact = result
    if not isinstance(artifact, SNNArtifact):
        raise TypeError("the first conversion output must be an SNNArtifact")
    # Validate info before writing either output; reject NaN/Inf JSON values.
    encoded = json.dumps(info, indent=2, allow_nan=False) if info is not None else None
    output = Path(output)
    info_path = output.with_suffix(".json")
    if encoded is not None and info_path == output:
        raise ValueError("artifact output must not use the .json suffix")
    output.parent.mkdir(parents=True, exist_ok=True)
    artifact.cpu().eval().save(output)
    print(f"saved {output.resolve()}")
    if encoded is not None:
        info_path.write_text(encoded + "\n", encoding="utf-8")
        print(f"saved {info_path.resolve()}")

def main():
    p = argparse.ArgumentParser(description="Convert an OpenAI CLIP visual tower to SNN")
    p.add_argument("--dataset", required=True, choices=DATASETS)
    p.add_argument("--data", required=True, help="Dataset root directory used for calibration")
    p.add_argument("--clip", required=True, help="CLIP backend name, e.g. openai_rn50")
    p.add_argument("--meth", required=True, help="Conversion method, e.g. DCGS")
    p.add_argument("--params", required=True, help="YAML file containing conversion parameters")
    p.add_argument("--output", default="snn_mth_var_channel.pt")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="cpu", help="Device used to load and convert the ANN")
    a = p.parse_args()
    params = load_conversion_params(a.params, a.meth)
    # The bundled RN50 preprocessing uses 224px inputs.  convert_clip loads
    # the requested adapter and dispatches the requested conversion method.
    calibration = build_dataloader(a.dataset, a.data, batch_size=a.batch_size,
                                   image_size=224, num_workers=a.workers)
    # Reuse the public factory API; it loads the named adapter and dispatches
    # the requested conversion method.  The already-created calibration loader
    # is image-only by construction.
    result = convert_clip(a.clip, a.meth, params, dataloader=calibration,
                          device=a.device)
    save_conversion_result(result, a.output)

if __name__ == "__main__":
    main()
