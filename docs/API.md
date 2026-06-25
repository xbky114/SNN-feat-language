# CLIPWithSNNVisual API

`CLIPWithSNNVisual` 将 DCGS 转换后的 RN50 **SNN visual encoder** 与 OpenAI CLIP 的 **text tower** 组合，提供与 CLIP 类似的零样本图文匹配接口，并额外暴露 SNN 各时间步的输出。

定义位置：`DCGS/models/clip_wrapper.py`

## 构造

```python
from CLIP_with_SNN_visual import load_converted_snn, wrap_snn_as_clip
from utils import reset

device = "cuda:0"
visual = load_converted_snn(convert_attn=False, thre_path="...", T=256, device=device)
model = wrap_snn_as_clip(visual, device=device)
```

`wrap_snn_as_clip(snn_visual, clip_checkpoint=None, device=None) -> CLIPWithSNNVisual`

| 参数 | 说明 |
|------|------|
| `snn_visual` | `load_converted_snn` 返回的 SNN visual encoder |
| `clip_checkpoint` | CLIP RN50 权重路径，默认 `~/.cache/clip/RN50.pt` |
| `device` | 默认取 `snn_visual` 所在设备 |


## 推理前重置

每次对新 batch 做 SNN 推理前，需重置 visual 神经元状态：

```python
reset(model.visual)  # 或 reset(snn_visual)
```

`test_probs_at_all_t` 内部会调用 `encode_image`，调用方在多次独立推理之间自行 `reset`。

---

## `encode_image(image) -> Tensor`

编码图像，返回 **所有时间步** 的 visual 输出。


**输出**

| 形状 | 说明 |
|------|------|
| `(T, B, 1024)` | 第 `t` 步的 image feature（未 L2 归一化） |

其中 `T` 为加载 SNN 时设定的仿真步数（`load_converted_snn` 的 `T` 参数）。

### visual forward 语义

`output[t]` 的含义取决于 `convert_attn`。这个参数决定了，是只将RN50的卷积部分转换为SNN，还是把RN50最后的 Attention Pool模块也转换成SNN。推荐`False`, 转换质量更好。

| `convert_attn` | 计算逻辑 |
|----------------|----------|
| `False` | backbone 为 SNN，attention pool 为 ANN。`output[t] = ANN_attnpool( mean(backbone_spikes[:t+1]) )` |
| `True` | backbone 与 attention pool 均为 SNN。`output[t] = mean(SNN_attnpool(backbone_spikes[:t+1])` |


---

## `encode_text(text) -> Tensor`

编码文本，与原版 CLIP 相同。

---

## `forward(img_feat, text_feat) -> (logits_per_image, logits_per_text)`

在 **已编码特征** 上计算 CLIP 相似度 logits。与原版 CLIP 不同，此处不接受原始 `image, text`。

**输入**

| 参数 | 形状 | 说明 |
|------|------|------|
| `img_feat` | `(B, 1024)` | 单个时间步的 image feature |
| `text_feat` | `(N, 1024)` | 所有候选 label 的 text feature |

内部会对两者做 L2 归一化，再乘以 `exp(logit_scale)` 计算点积。

**输出**

| 返回值 | 形状 | 说明 |
|--------|------|------|
| `logits_per_image` | `(B, N)` | 每张图对各 label 的 logit |
| `logits_per_text` | `(N, B)` | 各 label 对每张图的 logit（转置） |

---

## `test_probs_at_all_t(image, text) -> ndarray`

对每个时间步分别计算 softmax 概率，用于观察 SNN 随时间的收敛过程。

**输入**

| 参数 | 形状 | 说明 |
|------|------|------|
| `image` | `(B, 3, 224, 224)` | 图像 batch |
| `text` | `(N, 77)` | tokenized 文本 |

**输出**

| 形状 | 说明 |
|------|------|
| `(B, T, N)` | `probs[b, t, n]` = batch 中第 `b` 张图，在第 `t` 步，对第 `n` 个 label 的概率 |

等价于对每个 `t` 执行 `forward(img_features[t], text_features)` 后取 softmax。


---

## 完整流程示例

见test_sample.py