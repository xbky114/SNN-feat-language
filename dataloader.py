"""Three-level dataset adapters used by evaluation and calibration."""
from __future__ import annotations
import json
from io import BytesIO
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


class HierarchyDataLoader(Dataset):
    """Shared three-level labels, parent relationships and image preprocessing."""
    levels = ("basic", "subordinate", "fine_grained")

    def __init__(self, root, *, image_size, annotations, names, hierarchy=None):
        self.root = Path(root).expanduser()
        self.transform = clip_preprocess(image_size)
        self.class_names = {
            level: [mapping[i] for i in range(len(mapping))]
            for level, mapping in zip(self.levels, names)
        }
        self.parent = {"subordinate": {}, "fine_grained": {}}
        self.samples = [
            (path, *(self.class_names[level][i] for level, i in zip(self.levels, ids)))
            for path, ids in annotations
        ]
        if hierarchy is None:
            hierarchy = (ids for _, ids in annotations)
        for ids in hierarchy:
            basic, subordinate, fine = (
                self.class_names[level][i] for level, i in zip(self.levels, ids)
            )
            for level, child, parent in (("subordinate", subordinate, basic),
                                         ("fine_grained", fine, subordinate)):
                previous = self.parent[level].setdefault(child, parent)
                if previous != parent:
                    raise ValueError(f"inconsistent parent for {child}")

    def __len__(self):
        return len(self.samples)

    def _load_image(self, index):
        with Image.open(self.root / self.samples[index][0]) as image:
            return image.convert("RGB")

    def __getitem__(self, index):
        _, basic, subordinate, fine = self.samples[index]
        return {"image": self.transform(self._load_image(index)), "basic": basic,
                "subordinate": subordinate, "fine_grained": fine}


class IMNET_3L_Adapter(HierarchyDataLoader):
    """ImageNet-3L validation samples with fixed semantic string labels."""

    def __init__(self, root, *, split="val", image_size=224, metadata_dir=_METADATA):
        metadata = Path(metadata_dir)
        names = []
        for suffix in ("basic", "subord", "finegrained"):
            values = json.loads((metadata / f"imagenet-3L-id_{suffix}_dic.json").read_text())
            names.append({int(i): name for i, name in values.items()})
        tree = json.loads((metadata / "imagenet-3L-tree.json").read_text())
        hierarchy = [(int(row[2]), int(row[1]), fine_id)
                     for fine_id, row in enumerate(tree)]
        annotations = _read_annotations(metadata / f"imagenet-F-{split}.txt")
        super().__init__(root, image_size=image_size, annotations=annotations,
                         names=names, hierarchy=hierarchy)


def _read_annotations(path):
    return [(fields[0], tuple(map(int, fields[1:4])))
            for line in Path(path).read_text().splitlines()
            if (fields := line.split())]


class INAT21_Adapter(HierarchyDataLoader):
    """iNat21-F: order / family / species, using official annotation IDs."""

    def __init__(self, root, *, split="val", image_size=224, metadata_dir=_METADATA):
        if split not in ("val", "train"):
            raise ValueError("iNat21 supports split='val' or 'train'")
        annotations = _read_annotations(Path(metadata_dir) / f"inat21-F-{split}.txt")
        names = ({}, {}, {})
        for path, ids in annotations:
            # ID_kingdom_phylum_class_order_family_genus_species
            taxonomy = Path(path).parent.name.split("_")
            values = (taxonomy[4], taxonomy[5], " ".join(taxonomy[6:]))
            for mapping, i, name in zip(names, ids, values):
                previous = mapping.setdefault(i, name)
                if previous != name:
                    raise ValueError(f"inconsistent taxonomy for {path}")
        super().__init__(root, image_size=image_size, annotations=annotations, names=names)


# Names verified against Fine-Grained-or-Not/birds/Birds_name.xlsx at commit
# 8a188108bd1484d1bffe0a0a0c766c9f03ef9925, joined by species ID to
# birds_get_tree_target_2.py. Tuple offsets are the zero-based FGL label IDs.
# Source inconsistencies and display-name decisions: docs/CUB_LABEL_NAMES.md.
_CUB_ORDERS = (
    "Anseriformes", "Apodiformes", "Caprimulgiformes", "Charadriiformes",
    "Coraciiformes", "Cuculiformes", "Gaviiformes", "Passeriformes",
    "Pelecaniformes", "Piciformes", "Podicipediformes", "Procellariiformes",
    "Suliformes",
)
_CUB_FAMILIES = (
    "Anatidae", "Trochilidae", "Caprimulgidae", "Alcidae", "Laridae",
    "Stercorariidae", "Alcedinidae", "Laridae (Red legged Kittiwake)", "Cuculidae", "Gaviidae",
    "Alaudidae", "Bombycillidae", "Cardinalidae", "Certhiidae", "Corvidae",
    "Fringillidae", "Hirundinidae", "Icteridae", "Icteriidae", "Laniidae",
    "Mimidae", "Motacillidae", "Parulidae", "Passerellidae / Emberizidae", "Passeridae",
    "Ptilonorhynchidae", "Sittidae", "Sturnidae", "Troglodytidae",
    "Tyrannidae", "Vireonidae", "Pelecanidae", "Picidae", "Podicipedidae",
    "Diomedeidae", "Procellariidae", "Phalacrocoracidae", "Fregatidae",
)


class CUB_Adapter(HierarchyDataLoader):
    """CUB-F evaluation: split folders, original CUB images, or HF Parquet."""

    def __init__(self, root, *, split="val", image_size=224, metadata_dir=_METADATA):
        if split not in ("val", "test"):
            raise ValueError("CUB-F currently supports evaluation only (val/test)")
        annotations = _read_annotations(Path(metadata_dir) / "cub-Real-val.txt")
        fine_names = {ids[2]: Path(path).parent.name.split(".", 1)[1].replace("_", " ")
                      for path, ids in annotations}
        super().__init__(root, image_size=image_size, annotations=annotations,
                         names=(dict(enumerate(_CUB_ORDERS)),
                                dict(enumerate(_CUB_FAMILIES)), fine_names))
        self._parquet_locations = None
        self._cached_group = None
        if (self.root / "test").is_dir():
            return
        for image_root in (self.root / "images", self.root / "CUB_200_2011" / "images"):
            if image_root.is_dir():
                self.samples = [(str(image_root.relative_to(self.root) / Path(path).relative_to("test")), *labels)
                                for path, *labels in self.samples]
                return
        shards = sorted((self.root / "data").glob("test-*.parquet"))
        if not shards:
            raise FileNotFoundError(f"No CUB test images or data/test-*.parquet found in {self.root}")
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ImportError("Reading CUB Parquet requires pyarrow: pip install pyarrow") from exc
        locations = {}
        for shard in shards:
            parquet = pq.ParquetFile(shard)
            for group in range(parquet.num_row_groups):
                rows = parquet.read_row_group(group, columns=["image.path"]).to_pylist()
                for row, value in enumerate(rows):
                    filename = Path(value["image"]["path"]).name
                    if filename in locations:
                        raise ValueError(f"duplicate CUB image: {filename}")
                    locations[filename] = (shard, group, row)
        self._parquet_locations = [locations[Path(path).name] for path, *_ in self.samples]

    def _load_image(self, index):
        if self._parquet_locations is None:
            return super()._load_image(index)
        import pyarrow.parquet as pq
        shard, group, row = self._parquet_locations[index]
        key = (shard, group)
        if self._cached_group is None or self._cached_group[0] != key:
            table = pq.ParquetFile(shard).read_row_group(group, columns=["image"])
            self._cached_group = (key, table)
        record = self._cached_group[1]["image"][row].as_py()
        with Image.open(BytesIO(record["bytes"])) as image:
            return image.convert("RGB")


DATASETS = {
    "imnet": IMNET_3L_Adapter,
    "inat": INAT21_Adapter,
    "cub": CUB_Adapter,
}


def build_dataloader(dataset, root, *, batch_size=32, image_size=224,
                     num_workers=4, split="val", metadata_dir=_METADATA):
    """Select DATASETS[dataset], then construct its loader with root and options.

    Example: build_dataloader("inat", "./data/inat21").
    """
    try:
        adapter = DATASETS[dataset]
    except KeyError as exc:
        raise ValueError(f"Unknown dataset {dataset!r}; choose from {', '.join(DATASETS)}") from exc
    instance = adapter(root, split=split, image_size=image_size,
                       metadata_dir=metadata_dir)
    return DataLoader(instance, batch_size=batch_size, shuffle=False,
                      drop_last=False, num_workers=num_workers)
