#!/usr/bin/env python3
"""Measure CLIP RN50 image-feature invariance and Jacobian responses.

Here h is the input of the final ReLU in the ResNet image tower (the ReLU
immediately before AttentionPool2d).  The original feature F is obtained by a
normal image-tower forward pass.  S_1 measures an all-ones shift.  R compares
projected attention-pool Jacobian products for h > 0 and h > Vth, where Vth is
the 99th percentile of the dataset-wide pre-activation distribution.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class ImageFiles(Dataset):
    """Recursively enumerate images, without requiring ImageNet labels."""

    def __init__(self, root: Path, preprocess):
        self.paths = sorted(
            path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.paths:
            raise ValueError(f"No supported image files found under {root}")
        self.preprocess = preprocess

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            return self.preprocess(image.convert("RGB"))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("./data/auxiliary_2000"))
    parser.add_argument("--output-dir", type=Path, default=Path("./verifications/invariance"))
    parser.add_argument("--epsilon", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model", default="RN50", help="CLIP model name or local .pt checkpoint path")
    parser.add_argument("--download-root", type=Path, default=None, help="Optional directory containing cached CLIP weights")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.epsilon == 0:
        raise ValueError("--epsilon must be non-zero")
    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        raise RuntimeError(f"CUDA is unavailable, but --device={args.device} was requested")
    if str(args.device).startswith("cuda"):
        torch.cuda.set_device(torch.device(args.device))
        # torch.autograd.functional.jvp uses a reverse-over-reverse method.
        # The fused flash/memory-efficient SDP backward has no second
        # derivative in this PyTorch version, whereas the math backend does.
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    repo_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(repo_root / "CLIP"))
    import clip  # Imported after prioritizing the repository's CLIP package.

    model, preprocess = clip.load(
        args.model,
        device=args.device,
        jit=False,
        download_root=str(args.download_root) if args.download_root else None,
    )
    model.eval()
    visual = model.visual
    if not hasattr(visual, "layer4") or not hasattr(visual, "attnpool"):
        raise ValueError("This experiment requires a ResNet CLIP image tower such as RN50")

    # The final ReLU is in the last Bottleneck of layer4. clone() is essential:
    # the ReLU is inplace and would otherwise overwrite its pre-activation.
    final_relu = visual.layer4[-1].relu3
    captured_h = []

    def capture_h(_module, inputs):
        captured_h.append(inputs[0].detach().clone())

    handle = final_relu.register_forward_pre_hook(capture_h)
    dataset = ImageFiles(args.data, preprocess)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=str(args.device).startswith("cuda"),
        shuffle=False,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    h_values_path = args.output_dir / "h_values.npy"
    uniform_errors = []
    r_values = []
    dz_b_norms = []
    dz_a_norms = []
    projected_jg_b_norms = []
    projected_jg_a_norms = []
    num_h_neurons = None
    h_values = None
    h_offset = 0

    try:
        # Pass 1: save every flattened pre-activation h and compute the exact
        # dataset-wide 99th percentile. float16 preserves the model's h values
        # while keeping this 2000-image artifact to roughly 400 MB.
        with torch.inference_mode():
            for images in loader:
                images = images.to(args.device, non_blocking=True)
                captured_h.clear()
                model.encode_image(images)
                if len(captured_h) != 1:
                    raise RuntimeError(f"Expected one final-ReLU activation, captured {len(captured_h)}")
                h = captured_h[0]
                batch_num_h_neurons = h[0].numel()
                if num_h_neurons is None:
                    num_h_neurons = batch_num_h_neurons
                    h_values = np.lib.format.open_memmap(
                        h_values_path,
                        mode="w+",
                        dtype=np.float16,
                        shape=(len(dataset) * num_h_neurons,),
                    )
                elif num_h_neurons != batch_num_h_neurons:
                    raise RuntimeError("The number of h neurons unexpectedly changed between batches")
                flat_h = h.flatten().cpu().numpy()
                h_values[h_offset : h_offset + flat_h.size] = flat_h
                h_offset += flat_h.size

        if h_offset != h_values.size:
            raise RuntimeError(f"Saved {h_offset} h values, expected {h_values.size}")
        h_values.flush()
        vth = float(np.percentile(h_values, 99))

        # Pass 2: retain S_1 and differentiate the entire r -> G(r) -> z
        # chain. This gives dz = P_z J_G v / ||f|| directly.
        for images in loader:
            images = images.to(args.device, non_blocking=True)
            captured_h.clear()
            with torch.no_grad():
                features = model.encode_image(images).float()
                if len(captured_h) != 1:
                    raise RuntimeError(f"Expected one final-ReLU activation, captured {len(captured_h)}")
                h = captured_h[0]
                uniform_features = visual.attnpool(torch.relu(h + args.epsilon * torch.ones_like(h))).float()
                z = features / torch.linalg.vector_norm(features, dim=-1, keepdim=True)
                z_uniform = uniform_features / torch.linalg.vector_norm(uniform_features, dim=-1, keepdim=True)
                uniform_errors.append(torch.linalg.vector_norm(z_uniform - z, dim=-1).cpu())

            r = F.relu(h).detach().requires_grad_(True)
            b = (h > 0).to(dtype=h.dtype)
            a = (h >= vth).to(dtype=h.dtype)

            def z_of_r(r_value):
                f = visual.attnpool(r_value).float()
                return F.normalize(f, dim=-1)

            _, dz_b = torch.autograd.functional.jvp(
                z_of_r, (r,), (b,), create_graph=False, strict=True
            )
            _, dz_a = torch.autograd.functional.jvp(
                z_of_r, (r,), (a,), create_graph=False, strict=True
            )
            dz_b_norm = torch.linalg.vector_norm(dz_b.float(), dim=-1)
            dz_a_norm = torch.linalg.vector_norm(dz_a.float(), dim=-1)
            r_values.append((dz_b_norm / (dz_a_norm + 1e-12)).detach().cpu())
            dz_b_norms.append(dz_b_norm.detach().cpu())
            dz_a_norms.append(dz_a_norm.detach().cpu())

            # Since ||dz_v|| = ||P_z J_G v|| / ||f||, recover the requested
            # unnormalized projected-Jacobian norms for the ratio of means.
            f_norm = torch.linalg.vector_norm(features, dim=-1)
            projected_jg_b_norms.append((dz_b_norm * f_norm).detach().cpu())
            projected_jg_a_norms.append((dz_a_norm * f_norm).detach().cpu())
    finally:
        handle.remove()

    uniform_errors = torch.cat(uniform_errors).numpy()
    r_values = torch.cat(r_values).numpy()
    dz_b_norms = torch.cat(dz_b_norms).numpy()
    dz_a_norms = torch.cat(dz_a_norms).numpy()
    projected_jg_b_norms = torch.cat(projected_jg_b_norms).numpy()
    projected_jg_a_norms = torch.cat(projected_jg_a_norms).numpy()
    scale = np.sqrt(num_h_neurons) * args.epsilon
    s1 = float(uniform_errors.mean() / scale)
    r_mean = float(r_values.mean())
    r_p25, r_median, r_p75 = (float(value) for value in np.percentile(r_values, [25, 50, 75]))
    ratio_of_mean_projected_norms = float(projected_jg_b_norms.mean() / projected_jg_a_norms.mean())
    npz_path = args.output_dir / "errors.npz"
    json_path = args.output_dir / "results.json"
    np.savez_compressed(
        npz_path,
        errors=uniform_errors,
        errors_uniform=uniform_errors,
        R=r_values,
        dz_b_norm=dz_b_norms,
        dz_a_norm=dz_a_norms,
        projected_JG_b_norm=projected_jg_b_norms,
        projected_JG_a_norm=projected_jg_a_norms,
    )
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "model": args.model,
                "device": args.device,
                "dataset": str(args.data),
                "num_images": int(uniform_errors.size),
                "epsilon": args.epsilon,
                "num_h_neurons_m": num_h_neurons,
                "Vth_percentile": 99,
                "Vth": vth,
                "uniform_error_mean": float(uniform_errors.mean()),
                "S_1": s1,
                "R_mean": r_mean,
                "R_median": r_median,
                "R_percentile_25": r_p25,
                "R_percentile_75": r_p75,
                "mean_projected_JG_b_norm": float(projected_jg_b_norms.mean()),
                "mean_projected_JG_a_norm": float(projected_jg_a_norms.mean()),
                "ratio_of_mean_projected_JG_norms": ratio_of_mean_projected_norms,
                "h_values_npy": str(h_values_path),
                "errors_npz": str(npz_path),
            },
            file,
            indent=2,
        )
        file.write("\n")

    print(f"m = {num_h_neurons}")
    print(f"Vth (99th percentile) = {vth:.12g}")
    print(f"S_1 = {s1:.12g}")
    print(f"R.mean() = {r_mean:.12g}")
    print(f"R.median() = {r_median:.12g}")
    print(f"R 25th percentile = {r_p25:.12g}")
    print(f"R 75th percentile = {r_p75:.12g}")
    print(f"E[||P_z J_G b||] / E[||P_z J_G a||] = {ratio_of_mean_projected_norms:.12g}")
    print(f"Saved per-image errors to {npz_path}")
    print(f"Saved summary to {json_path}")


if __name__ == "__main__":
    main()
