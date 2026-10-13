# WNTree JSON protocol

`wordnet_tree.py` writes a reusable ImageNet WordNet hierarchy.

## JSON fields

- `num_classes`: number of ImageNet leaf classes.
- `num_bins`: number of non-root semantic levels.
- `classes[wnid]`: leaf-class text, full path text, and representative WNID at each level.
- `levels[str(j)]`: ordered classes for level `j`.
  - `num_classes`: number of classes at this level.
  - `wnids[i]`: WNID represented by class ID `i`.
  - `texts[i]`: text of class-ID-`i`.
  - `class_to_idx`: WNID-to-class-ID mapping.

Level `num_bins` contains the detailed ImageNet classes and follows
`ImageFolder`'s sorted WNID order.

## Classification at level `j`


```python
level = tree["levels"][str(j)]
texts = level["texts"]
text_features = encode_clip_text(texts)
logits = ......
pred_id = logits.argmax(dim=1)
pred_wnid = [level["wnids"][i] for i in pred_id.tolist()]
```

The detailed WNID of an image is obtained from its `ImageFolder` label:

```python
dataset = dataloader.dataset

for images, labels in dataloader:
    true_wnids = [dataset.classes[i] for i in labels.tolist()]
```

`ImageFolder` assigns labels by sorting directory names. This is the same order
used by level `num_bins`, the final and most detailed level.

At level `j`, convert each detailed WNID to its representative target:

```python
target_wnids = [
    tree["classes"][wnid]["representatives"][j]
    for wnid in true_wnids
]
target_ids = torch.tensor(
    [level["class_to_idx"][wnid] for wnid in target_wnids],
    device=pred_id.device,
)
correct += (pred_id == target_ids).sum().item()
```
