"""Fixed three-level ImageNet loader used by evaluation and calibration."""
from __future__ import annotations
import json
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

_METADATA = Path(__file__).resolve().parent / "FreeGrainLearning" / "data"


def clip_preprocess(image_size=224):
    """The preprocessing shipped with OpenAI CLIP (kept local by design)."""
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size), transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                             (0.26862954, 0.26130211, 0.27577711)),
    ])


class IMNET_3L_Adapter(Dataset):
    """ImageNet-3L validation samples with fixed semantic string labels."""
    levels = ("basic", "subordinate", "fine_grained")

    def __init__(self, root, *, split="val", image_size=224, metadata_dir=_METADATA):
        self.root = Path(root).expanduser()
        metadata = Path(metadata_dir)
        self.transform = clip_preprocess(image_size)
        self.class_names = {}
        for level, suffix in (("basic", "basic"), ("subordinate", "subord"),
                              ("fine_grained", "finegrained")):
            values = json.loads((metadata / f"imagenet-3L-id_{suffix}_dic.json").read_text())
            self.class_names[level] = [values[str(i)] for i in range(len(values))]
        self.samples = []
        for line in (metadata / f"imagenet-F-{split}.txt").read_text().splitlines():
            if line.strip():
                path, basic, subordinate, fine = line.split()[:4]
                self.samples.append((path, self.class_names["basic"][int(basic)],
                                     self.class_names["subordinate"][int(subordinate)],
                                     self.class_names["fine_grained"][int(fine)]))
        self.parent = {"subordinate": {}, "fine_grained": {}}
        for _, basic, subordinate, fine in self.samples:
            self.parent["subordinate"].setdefault(subordinate, basic)
            self.parent["fine_grained"].setdefault(fine, subordinate)

    def __len__(self): return len(self.samples)

    def __getitem__(self, index):
        relative, basic, subordinate, fine = self.samples[index]
        with Image.open(self.root / relative) as image:
            image = image.convert("RGB")
        return {"image": self.transform(image), "basic": basic,
                "subordinate": subordinate, "fine_grained": fine}


def build_dataloader(root, *, batch_size=32, image_size=224, num_workers=4,
                     split="val", metadata_dir=_METADATA):
    dataset = IMNET_3L_Adapter(root, split=split, image_size=image_size,
                               metadata_dir=metadata_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      drop_last=False, num_workers=num_workers)


# Kept as a migration alias for callers that used the old name.
imagenet_dataloader = build_dataloader
