# eval_snn 性能诊断（2026-11-02）

## 结论

优先共享跨层级的视觉特征，然后把 rate coding 改为流式前缀均值；进一步优化 MTH 神经元的 GPU kernel。当前模型已经使用 FP16。仅调整 DataLoader 或增加 batch size 不是首要方向。

本次只做诊断，未修改评估代码或正在运行的任务。

## 当前任务与测量边界

- eval_snn.sh：2000 张图、batch=256、T=256、cuda:1。
- WNTree 有 0–4 共 5 层，类别数依次为 1、12、49、256、1000。
- 实际 checkpoint：DCGS MTH，4 个阈值，channel 阈值，99.9 percentile；主干和 attention pool 均为 FP16。
- 现有输出 shard 的修改时间间隔约 15–16 分钟，与用户报告的约 2 小时一致。
- cuda:1 为 A100 80GB，采样时利用率 100%，进程占约 78GB 显存。
- 隔离测量使用空闲 cuda:2（同型号 A100 80GB）、PyTorch 2.1.2+cu121、真实 checkpoint、随机图像和随机主干特征；torch.no_grad，计时边界同步 CUDA。
- 这是组件测量，不是完整 2000 张 ImageNet 的优化前后对比。随机输入、GPU 时钟、缓存和 hooks 都可能影响耗时；不能把组件提速直接当作端到端提速。

## 测量结果

| 测量项 | 条件 | 秒 |
|---|---|---:|
| encode_image | B=256，T=16 | 11.3393 |
| SNN 主干，包含模块计时 hooks | B=256，8 步 | 5.6493 |
| 其中 MTHRate（CUDA events） | 同上 | 5.2355 |
| 其中 Conv2d（CUDA events） | 同上 | 0.1900 |
| 其中 BatchNorm2d（CUDA events） | 同上 | 0.1211 |
| 其中 AvgPool2d（CUDA events） | 同上 | 0.0552 |
| 原有全部前缀 rate_coding | B=256，T=256，FP16 [2048,7,7] 特征 | 43.2043 |
| 流式 rate coding，FP32 累加后转 FP16 | 同上 | 0.1146 |
| attention pool | B=256，256 次 | 0.6094 |
| 各层文本 tokenize+encode 合计 | 真实 prompts，包含首次调用开销 | 0.4025 |

小规模共享特征验证：真实模型，固定随机种子，B=2，T=4，5 组单条文本；原调用 0.8311 秒，共享图像特征后 0.1564 秒，所有 logits 最大绝对差为 0，严格相等断言通过。此处是一次小样本验证，首次文本调用等开销会影响倍率，不是完整数据集的 5.31 倍性能承诺。

## 1. 相同视觉计算重复 5 次：优先级最高

位置：eval_snn.py:85–87；clip_with_snn_visual.py:84–86。

每个层级都调用 calc_logits_through_time，它每次重置 SNN 并重新执行 encode_image。图像、权重、T 完全相同，文本不参与视觉计算。每 batch 主干执行 5×256=1280 步，实际上只需要 256 步。

建议：每 batch 计算一次 image_features[T,B,D]，所有层级共享；每个层级只做不同的文本相似度矩阵。所有 prompts 的文本特征也可以在 batch 循环外预计算并归一化。保留所有 timestep 和当前 shard 格式。

视觉部分计算减少 80%，理论接近 5 倍；若端到端主要由视觉部分支配，2 小时有望降至约 24 分钟，实际值需完整数据验证。现有小样本验证 logits 完全一致；完整验证还应覆盖真实图片和各层指标。

## 2. rate coding 为 O(T²)：明确且容易消除

位置：clip_with_snn_visual.py:65–75；to_snn/coding.py:48、64。

对每个 t 都重新 stack 前 t 个主干输出再求均值。T=256 时累计处理 32896 个 timestep 特征，而增量算法只处理 256 个。每步 FP16 特征占 49 MiB，历史列表占 12.25 GiB，最大 stack 又占 12.25 GiB，累计 stack 输出量约 1.54 TiB/视觉序列。

建议在 SNN 时间循环内使用 FP32 running_sum，rate_t=(running_sum/t).to(pool_dtype)，立即经过 pool，仅保留较小的池化后特征。不再保留全部 backbone_features。可显著降低峰值显存。

编码组件测量约提速 377 倍，但主干仍要执行，整体不可能提速 377 倍。按本次主干耗时外推，去掉该编码开销约能在共享特征后的基础上再获得约 1.2 倍收益；这是粗估，不是全量实测。

注意：只对已知的 rate_coding 使用此递推；通用自定义 coding 需保留原语义。FP32 累加可避免 FP16 长序列求和溢出，但与原 mean 的归约顺序不同，必须检查所有 timestep 特征、logits 和指标的误差，不能宣称逐 bit 相等。

## 3. MTH 神经元占主干约 93%：进一步加速的关键

位置：to_snn/DCGS/neurons/mth_rate.py:34–46。

当前将激活扩展到 4 个阈值维度，生成 positive、negative、clone、差分、乘法、归约等多个中间张量，反复读写大张量。主干的主要热点是这些神经元操作，卷积只占约 3.4%。此测量定位了算子热点；没有用硬件计数器直接证明带宽瓶颈。

下一步应先对纯张量状态更新尝试 torch.compile 或 fused Triton/CUDA kernel，将比较、阈值选择、膜电位和累计输出更新合并，减少阈值维度中间张量。不要先编译含 Python self.T 的整个有状态模型：应检查重编译和 graph break，优先提取纯张量更新函数。

需要回归测试负脉冲、阈值边界、channel 广播、reset、长时间序列状态，以及完整 logits。这里尚未实现融合或测得融合后的倍率，不能承诺具体提速。

## 次要优化与实验取舍

- 缓存文本特征、图像和文本归一化；批量计算时间维 logits；path.argmax(2) 只算一次；减少重复 .cpu() 同步。这些有收益，但不是分钟级重复主干的根因。
- shard 约 166 MiB，8 个 shard 总量约 1.3 GiB。未单独实测磁盘耗时，但组件计时已经解释了主要计算开销。增加 workers 不会消除这些开销。
- 当前已是 FP16，不应把 AMP 当作尚未启用的主要优化。显存占用接近上限时，也不应盲目增加 batch。
- T=256 降到 128/64 会减少计算，但改变时间轨迹和最终精度；应先读本次 per_timestep 指标判断收敛。如果实验要求全 256 步，优先采用前述保持 T 的优化。
- 多 GPU 按图片分片可以进一步并行，但应先去除重复计算；合并时指标按样本数加权，并避免 shard 文件名冲突。

测量脚本位于本次会话的 /tmp/profile_eval_snn.py 和 /tmp/check_snn_reuse.py。运行命令为 conda run -n llava python -c "exec(open('/tmp/profile_eval_snn.py').read())"，共享特征验证同理替换脚本名。它们是临时诊断脚本，不属于正式评估入口。

## MTH 融合实现后的实测

已实现 `to_snn/neurons/_mth_triton.py`，默认由 `MTH` 的 CUDA 推理调用。
以下结果均使用 A100 80GB PCIe、PyTorch 2.1.2+cu121、Triton 2.1.0、FP16、
4 个阈值，预热后计时，不含首次 JIT 编译。

### 单个 MTH

每组运行 3 轮，每轮 10 次，报告 GPU 时间的中位数。前后使用相同输入和阈值。
额外峰值显存指预热后已分配显存之上，测量期间新增的最大 allocated memory，
不是整个评估进程的显存。

| 输入形状 | PyTorch 毫秒/步 | 融合毫秒/步 | 加速 | 额外峰值显存 MiB，前→后 |
|---|---:|---:|---:|---:|
| [256,32,112,112] | 23.896 | 0.828 | 28.87× | 4312 → 588 |
| [256,256,56,56] | 47.724 | 1.628 | 29.32× | 8624 → 1176 |
| [256,2048,7,7] | 6.080 | 0.249 | 24.37× | 1078 → 147 |

复现：`conda run -n llava python tests/benchmark_mth.py --device cuda:2`。
可用 `MTH(..., fused=False)` 选择原 PyTorch 路径进行对照。

### 完整 RN50 视觉推理

使用预训练 RN50 权重，将 51 个激活转换为 MTH，channel 阈值统一设为 1；
这是控制变量性能实验，不使用旧 checkpoint 的校准阈值，也不代表 ImageNet 精度。
保持此前完成的视觉特征共享和流式 coding，唯一切换项是 MTH 的 `fused`。
测量 `CLIPWithSNNVisual.encode_image`，包含 SNN、coding 和 attention pool。
B=256、T=16，预热后运行三轮，包含状态重置和初始化，墙钟时间中位数：

- 原实现：11.3168 秒。
- 融合实现：1.2406 秒。
- 加速约 9.12×，耗时下降约 89.0%。

另用 B=2、T=32 比较完整视觉轨迹，所有池化特征零容差一致。
本次未测完整 2000 张图像的端到端耗时，不将上述倍率等同于全量评估提速。

### 正确性与适用范围

完整 GPU 测试集 45 项全部通过。融合专项覆盖 FP16、BF16、FP32，1–8 个阈值，
标量/channel/广播阈值，负脉冲、零阈值、小阈值、边界相邻浮点值、256 步状态、
重置及形状/dtype 变化。spike、mem、cumulative_out 均与原实现零容差比较。
保留独立新状态张量，避免修改旧输入或已返回的状态。

默认融合支持最多 4 维的 contiguous CUDA FP16/BF16/FP32 输入和 1–8 个阈值。
需要 autograd、CPU、非连续布局、FP64、更高维或更多阈值，以及缺少 Triton 时，
自动使用原实现。首次出现新的 kernel 配置需要编译，测量不包含此一次性开销。
