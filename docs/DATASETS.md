# 数据集与三层标签

评测使用 FreeGrainLearning 提供的固定三层树元数据：

```text
basic → subordinate → fine_grained
```

当前实现支持 ImageNet-3L（`IMNET_3L_Adapter`），并预留扩展到 CUB-F、iNat-F 等具有相同三层标注协议的数据集。

## Dataloader 接口

```python
from dataloader import build_dataloader

loader = build_dataloader(
    root="/path/to/images",
    batch_size=32,
    split="val",
)
```

loader 固定使用 CLIP 标准图像预处理

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
