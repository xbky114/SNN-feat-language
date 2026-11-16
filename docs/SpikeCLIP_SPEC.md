## Problem Statement

用户希望将 SpikeCLIP finetune checkpoint 转换为本仓库可加载的 SNN visual artifact，搭配原版 OpenAI CLIP ViT-B/16 ANN text encoder，通过现有评测脚本完成评测。

实际训练版本的 visual encoder 是 Spikingformer，采用固定时间步的 rate coding。当前仓库评测则逐步推进 SNN，并累计编码。因此，需要适配权重协议和逐步推理接口，同时保持原训练时间步下的图文相似度近似一致，并允许继续扩展时间步。

修改应尽可能局部化，复用现有评测、编码和 artifact 接口。

## Solution

提供一次性离线转换工具，将 SpikeCLIP visual checkpoint 封装为本仓库支持的 artifact，并补充 ViT-B/16 ANN backend。

转换后的模型逐步运行 Spikingformer，保留神经元状态，由现有 rate coding 累计输出。用户在评测脚本中选择 ViT-B/16 backend 和转换后的权重即可使用，不需要填写神经元参数。

首轮使用已完成的 CUB finetune 权重验证，不使用仍在训练的 ImageNet 权重。

## User Stories

1. As an 评测用户, I want 将 finetune checkpoint 离线转换, so that 我能复用现有评测流程。
2. As an 评测用户, I want 使用训练得到的 Spikingformer visual, so that 评测对象保持为原 SNN。
3. As an 评测用户, I want 使用原版 ViT-B/16 ANN text encoder, so that 文本侧符合指定配置。
4. As an 评测用户, I want 仅更改模型 backend 和权重配置, so that 无需维护独立评测入口。
5. As an 评测用户, I want 自动恢复固定的神经元参数, so that 无需手工配置。
6. As an 评测用户, I want 原训练 T 下的图文相似度近似一致, so that 转换结果可信。
7. As an 评测用户, I want 每步产生累计编码结果, so that 能观察结果随时间变化。
8. As an 评测用户, I want 超过训练 T 后继续推理, so that 能探索更长时间步的表现。
9. As an 评测用户, I want 同一 batch 内持续保留神经元状态, so that 推理遵循连续时间演化。
10. As an 评测用户, I want 切换 batch 时重置状态, so that 不同图像之间不会相互影响。
11. As an 评测用户, I want 保留训练使用的图像缩放行为, so that 预处理差异不会掩盖转换误差。
12. As an 评测用户, I want 转换后的模型不依赖外部训练源码路径, so that 能在本仓库环境独立加载。
13. As an 评测用户, I want 不兼容权重明确报错, so that 不会误用部分加载的模型。
14. As an 评测用户, I want 首先验证 CUB 最佳权重, so that 验收不受持续训练影响。
15. As an 维护者, I want 保留现有 artifact 和 coding 契约, so that 修改范围可控。
16. As an 维护者, I want 原有模型评测行为保持兼容, so that 新增支持不会引入回归。
17. As an 维护者, I want 获得可复现的相似度误差报告, so that 能判断转换是否达标。

## Implementation Decisions

- 模型来源：以实际训练目录中的 rate-coding visual 实现为准，不采用当前仓库旧副本中的时间加权实现，也不引入已弃用的 SNN language encoder。
- 模型结构：保留训练的 tokenizer、Spikingformer blocks、空间聚合和线性投影；不将权重映射到 ANN ViT visual。
- 逐步接口：每次 visual 调用只推进一个时间步，返回该步的投影特征。返回值是由脉冲计算得到的连续特征，不要求为二值脉冲。
- 时间聚合：现有 rate coding 对逐步投影特征求平均。线性投影含 bias 仍与时间平均可交换，因此与原模型“先平均、后投影”数学等价，允许浮点误差。
- 状态生命周期：同一 batch 内持续推进状态；超过训练 T 后继续累计，不截断、不周期性重置。每个新 batch 开始前重置所有相关神经元状态。
- 原训练时间步：当前训练入口使用 T=8。转换信息记录该值，作为一致性验收位置；它不限制评测时间步。
- 神经元配置：保留训练定义，包括 τ=2、默认阈值 1、attention 特定阈值 0.5、硬重置到 0、detach_reset=True 和 decay_input=True。执行 backend 可适配环境，但不得有意改变神经元动力学。
- 预处理：保留 CLIP 预处理后缩放至 32×32 的行为，使用 bilinear、align_corners=False。缩放封装在 visual 适配层，避免改变其他模型的预处理。
- 文本侧：新增原版 OpenAI ViT-B/16 ANN backend，复用现有 CLIP 加载能力；不加载 SNN text 或 finetune text 权重。
- 权重转换：读取 checkpoint 的 visual state dictionary，严格检查必要参数和形状；不通过静默丢弃关键参数实现加载。
- artifact 协议：复用现有 SNNArtifact 保存、加载及 forward 契约，通过适配器连接训练模型的 reset 机制。
- 依赖范围：必要 visual 实现随本仓库提供，明确运行依赖，不依赖外部训练目录的绝对路径。
- 修改范围：集中在转换工具、visual 实现及适配器、ViT-B/16 backend 和模型选择配置，尽量不修改评测器、推理 wrapper 或通用 coding。
- 脚本使用：模型选择包括 ANN backend 与 SNN 权重两项。数据集和数据路径仍由现有评测参数控制，转换不自动切换数据集。

## Testing Decisions

以现有 artifact 加载 → CLIPWithSNNVisual 推理 → 图文相似度 为主要测试边界，验证外部行为，不绑定内部模块布局。

1. 原 T 一致性
   - 使用同一份 CUB 权重。
   - 参考侧运行实际训练版本的固定 T visual；转换侧加载保存后的 artifact，逐步运行至 T=8。
   - 两侧使用相同图像、文本、原版 ANN text encoder、预处理和精度配置，并处于 eval 模式。
   - 以归一化图文余弦相似度为主验收量，同时报告应用相同 logit scale 后的差异。
   - 报告最大绝对误差、平均绝对误差及测试样本范围；分类准确率或预测标签一致不能代替相似度验收。
   - 数值容差尚未约定：实现时需预先明确并记录阈值，不能只凭“看起来接近”判定通过。
2. 扩展时间步
   - 验证 T=1、T=8 和 T>8 能通过现有接口运行，输出形状正确且数值有限。
   - 超过 T=8 后继续推进状态并累计编码。
   - 不要求扩展时间步保持与固定八步结果一致，也不承诺准确率提升。
3. 状态隔离
   - 验证重复评测同一 batch 的结果可复现。
   - 验证先评测其他 batch 不影响重置后的目标 batch 结果。
4. 保存和加载
   - 在不引用外部训练源码目录的环境中加载 artifact 并完成推理。
   - 验证不兼容 checkpoint 明确失败。
5. 现有流程回归
   - 参考仓库已有共享 visual、推理 dtype 和 coding 测试方式。
   - 验证新增 backend 与适配器不破坏既有模型的加载和推理。

首轮验收包含真实 CUB 样本上的一致性检查和评测入口冒烟验证，不要求先跑完整数据集。

## Out of Scope

- 本轮实现代码、转换权重或启动评测。
- 重新训练、finetune 或改变神经元参数。
- SNN language encoder 和时间加权编码。
- 将 Spikingformer 转换为其他 visual 架构。
- 保证扩展时间步提高准确率。
- 完全复现原训练项目的提示词、分类策略和历史准确率。
- 使用尚未训练完成的 ImageNet checkpoint 作为首个验收对象。
- 大规模重构评测框架或新增独立评测系统。

## Further Notes

首个验收 checkpoint：

/data/home/bky/SpikeCLIP/output/cub/finetune/20261116-140105/model_best.pth.tar

已检查该文件：checkpoint 记录 epoch=267，包含 188 个 state entries，投影权重形状为 [512, 384]，包含投影 bias，没有 weights_tensor。

当前尚未提供项目 issue tracker 和 triage 配置，因此 spec 先完整列于此，未发布 issue。按已调用的 to-spec 技能 (/data/home/bky/.codex/plugins/cache/mattpocock/mattpocock-skills/1.3.2/skills/engineering/to-spec/SKILL.md) 要求：“If not, tell the user to run /setup-matt-pocock-skills.” 后续发布前需运行该命令完成配置。