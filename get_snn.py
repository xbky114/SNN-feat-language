"""Convert a CLIP visual tower to an SNN artifact."""
from __future__ import annotations
import argparse
from pathlib import Path
from dataloader import calibration_dataloader
from to_snn import convert_clip, load_conversion_params

def main():
    p = argparse.ArgumentParser(description="Convert an OpenAI CLIP visual tower to SNN")
    p.add_argument("--data", required=True, help="ImageFolder directory used for calibration")
    p.add_argument("--clip", required=True, help="CLIP backend name, e.g. openai_rn50")
    p.add_argument("--meth", required=True, help="Conversion method, e.g. DCGS")
    p.add_argument("--params", required=True, help="YAML file containing conversion parameters")
    p.add_argument("--output", default="snn_mth_var_channel.pt")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--calibration-size", type=int, default=None)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="cpu", help="Device used to load and convert the ANN")
    a = p.parse_args()
    # The bundled RN50 preprocessing uses 224px inputs.  convert_clip loads
    # the requested adapter and dispatches the requested conversion method.
    calibration = calibration_dataloader(a.data, batch_size=a.batch_size,
                                         calibration_size=a.calibration_size,
                                         image_size=224,
                                         num_workers=a.workers)
    params = load_conversion_params(a.params, a.meth)
    # Reuse the public factory API; it loads the named adapter and dispatches
    # the requested conversion method.  The already-created calibration loader
    # is image-only by construction.
    artifact = convert_clip(a.clip, a.meth, params, dataloader=calibration,
                            device=a.device)
    artifact.cpu().eval().save(a.output)
    print(f"saved {Path(a.output).resolve()}")

if __name__ == "__main__":
    main()
