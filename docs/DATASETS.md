# 数据集与三层标签

评测使用 FreeGrainLearning 提供的固定三层树元数据：

```text
basic → subordinate → fine_grained
```

通过 `DATASETS` 注册表选择数据集：`imnet`（ImageNet-3L）、`inat`（iNat21-F）、`cub`（CUB-F）。

## Dataloader 接口

```python
from dataloader import build_dataloader

loader = build_dataloader(
    dataset="inat",
    root="./data/inat21",
    batch_size=32,
    split="val",
)
```

`dataset` 和 `root` 均为必填参数。loader 固定使用 CLIP 标准图像预处理。

`eval_ann.py`、`eval_snn.py` 和 `get_snn.py` 均须传入 `--dataset {imnet,inat,cub}` 与 `--data 数据集根目录`。

## Batch 协议

每个 batch 是字典：

```python
{
    "image": Tensor,          # [B, 3, H, W]
    "basic": list[str],
    "subordinate": list[str],
    "fine_grained": list[str],
}
```

标签值是 FreeGrainLearning 元数据中的语义名称。
