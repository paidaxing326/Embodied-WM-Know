# GigaWorld-Policy 全景深度解读

> 论文：GigaWorld-Policy: An Efficient Action-Centered World–Action Model
> 团队：GigaAI
> 发表：arXiv 2603.17240，2026年3月
> 项目主页：https://gigaai-research.github.io/GigaWorld-Policy/
> 代码：https://github.com/open-gigaai/giga-world-policy

## 综合评价

| 维度 | 评分 | 评价 |
|------|------|------|
| 技术创新 | ⭐⭐⭐⭐⭐ | "训繁推简"是 2026 年世界模型领域最重要的架构创新。因果掩码精巧地解耦训练和推理，消融实验证明因果掩码反而提升了视频生成质量（PSNR 28.41 vs 27.87），说明设计本身是正确的。不是简单的工程 trick，而是对 WAM 范式的根本性重构。 |
| 工程化（商业） | ⭐⭐⭐⭐ | GigaAI 是商业公司，有产品化意识。CVPR 2026 Workshop 赛道是聪明的生态布局——通过竞赛吸引开发者，建立技术标准话语权。GigaTrain/GigaDataset 等工具链有 SDK 雏形。但目前模型只在 PiPER 6-DoF 机械臂上验证了 4 个任务，客户覆盖面窄。5B 参数模型的端侧部署是硬伤——客户的机器人上跑不动，必须依赖云端推理，这会增加延迟和成本。还没有看到明确的定价和商业模式。 |
| 落地价值（商业） | ⭐⭐⭐⭐ | "10% 数据达到 VLA 100% 效果"是最有说服力的商业故事——帮客户省 90% 数据采集成本，这个价值主张清晰。真机 SR 0.83 超过 π₀.₅ 和 Motus，性能有竞争力。但关键问题是：客户愿意为"预训练世界模型 backbone"付多少钱？机器人公司普遍倾向自研，不爱用三方服务。GigaAI 需要证明的不是技术好不好，而是客户离不开它——目前还没看到这种不可替代性。竞赛生态是好的开始，但从竞赛到付费产品还有很长的路。 |
> 论文：https://arxiv.org/pdf/2603.17240
> CVPR 2026 Workshop 赛道代码：https://github.com/open-gigaai/CVPR-2026-Workshop-WM-Track
> 赛道数据集：https://huggingface.co/datasets/open-gigaai/CVPR-2026-WorldModel-Track-Dataset

---

## 一、GigaAI 整体技术栈

GigaAI 围绕"世界模型"构建了一个完整的技术栈和生态：

```
GigaWorld-0-video (世界模型数据引擎，生成高保真视频)
    ↓
GigaWorld-0.5 (轻量级世界模型，多视角视频预测)
    ↓
GigaWorld-Policy (世界-动作模型，用世界模型做机器人策略) ← 本文核心
    ↓
GigaWorld-1 (登顶 WorldArena 榜单的完整世界模型)
    ↓
GigaBrain-0 (VLA 基础模型，用世界模型生成数据减少真机依赖)
    ↓
CVPR 2026 Workshop WM Track (世界模型竞赛平台)
```

---

## 二、GigaWorld-Policy 要解决什么问题

### 2.1 现有两条路线的痛点

**路线 A：VLA（Vision-Language-Action）**
- 代表：π₀、π₀.₅、OpenVLA
- 问题：**监督信号稀疏**。观测是高维的图像，但动作标签是低维稀疏的。模型容易走捷径，把很多不同情况映射到少数几个动作原型上

**路线 B：WAM（World-Action Model）**
- 代表：Motus、Cosmos-Policy、VideoVLA
- 思路：用视频生成模型预测未来画面，提供更密集的监督信号
- 问题：
  1. **推理太慢**——需要迭代采样生成未来视频，Motus 单步推理 3231ms
  2. **误差传播**——视频预测的像素级错误会传导到动作预测，长时间序列下误差累积

### 2.2 四种范式对比

论文用 Fig. 2 清晰地对比了四种范式：

| 范式 | 代表 | 推理时是否需要生成视频 | 问题 |
|------|------|----------------------|------|
| (a) VLA + 辅助未来监督 | Cen et al. | 不需要 | VLM 本质是判别式的，生成能力弱 |
| (b) 联合动作-视频预测（双向注意力） | Motus, VideoVLA | 需要 | 推理慢，双向注意力耦合 |
| (c) 两阶段：先生成视频再提取动作 | Mimic-video | 需要 | 继承视频预测误差，额外推理开销 |
| **(d) GigaWorld-Policy** | **本文** | **不需要（可选）** | **训练时用视频监督，推理时只出动作** |

---

## 三、核心创新："训繁推简"

这是论文最精彩的设计思想。

### 3.1 训练时（繁）

模型同时学两件事：
1. 根据当前观测预测未来动作序列
2. 根据预测的动作 + 当前观测生成未来视频

视频生成作为辅助监督，提供密集的像素级学习信号。视觉动态约束鼓励模型预测物理上合理的动作。

### 3.2 推理时（简）

- 完全丢弃视频生成分支
- 只保留动作预测，直接输出控制命令
- 单步推理 360ms（vs Motus 的 3231ms，快 9 倍）

### 3.3 因果掩码（Causal Mask）——让"训繁推简"成为可能

```
Token 序列: [观测 To | 状态 Ts | 动作 Ta | 未来视频 Tf]

注意力规则:
- To, Ts: 互相可见，但看不到 Ta 和 Tf
- Ta: 可以看到 To 和 Ts，但看不到 Tf  ← 关键！
- Tf: 可以看到 To, Ts, Ta（用动作条件生成未来视频）
```

因为动作 token 从来不依赖未来视频 token，所以推理时可以安全地去掉视频分支，不影响动作预测质量。这个设计非常精巧——训练时视频分支通过共享 Transformer 的梯度间接提升了动作预测的表征质量，但推理时不需要它。

---

## 四、模型架构详解

### 4.1 Backbone

- **Wan 2.2 5B**：大规模预训练视频生成扩散 Transformer
- 不是 VLM，而是视频生成模型——这是与 RDT2 等 VLA 方法的根本区别

### 4.2 多视角处理

将左、前、右三个相机视角拼接成一张图（Compose），不修改 backbone 结构：

```
o_t^comp = Compose(o_t^left, o_t^front, o_t^right)
```

好处：
- 不需要修改预训练模型的输入结构
- 在同一坐标系中保留各视角的空间结构
- 促进跨视角一致性

### 4.3 统一 Transformer

所有模态（视觉、状态、动作、语言）共享同一组 Q/K/V 投影矩阵：
- 不用 MoE，不用多分支——更简洁，更紧耦合
- 不同模态用不同位置编码（视觉用 2D 网格编码，动作/状态用 1D 时序编码）
- 语言指令通过 cross-attention 注入，不参与因果掩码序列

### 4.4 关键超参数

- 动作 chunk 长度：p = 48
- 未来帧采样间隔：Δ = 12
- 预测未来帧数：K = ⌊48/12⌋ = 4 帧
- 训练损失权重：λ_action = 5, λ_video = 1（强调动作预测为主）

---

## 五、三阶段训练 Pipeline

### 5.1 Stage 1：通用世界模型预训练

- **数据**：海量互联网视频
- **目标**：视频生成（仅 L_video）
- **作用**：学习通用物理规律和视觉动态
- **初始化**：从 Wan 2.2 5B 预训练权重开始

### 5.2 Stage 2：具身场景适配

- **数据**：~10,000 小时多源操作视频
- **目标**：视频生成（仅 L_video）
- **作用**：适配机器人视角和操作场景

数据来源非常丰富：

| 数据集 | 小时数 | 类型 |
|--------|--------|------|
| Ego4D | 3,500 | 第一人称人类视频 |
| Open X-Embodiment | 3,500 | 多机器人数据 |
| Agibot | 2,500 | 真机操作 |
| EgoDex | 800 | 第一人称灵巧操作 |
| DROID | 350 | 真机操作 |
| RoboMind | 300 | 真机操作 |
| Something-Something V2 | 200 | 人类物体交互 |
| RDT | 25 | 双臂操作 |
| ATARA | 10 | 真机操作 |

### 5.3 Stage 3：动作策略对齐（Post-Training）

- **数据**：少量真机动作标签数据
- **目标**：L_all = λ_video × L_video + λ_action × L_action
- **作用**：对齐"观测-动作-未来视觉"因果映射

### 5.4 训练目标：Flow Matching

对动作和视频两种模态都使用 Flow Matching：

```
x^(s) = (1-s)ε + sx        # 插值
ẋ^(s) = x - ε              # 目标速度

L_video = E[||g_Θ(z_f^(s), s | Ts, To, Ta, Tl) - ż_f^(s)||²]
L_action = E[||g_Θ(a^(s), s | Ts, To, Tl) - ȧ^(s)||²]
```

预训练阶段只优化 L_video，后训练阶段联合优化 L_all。

---

## 六、推理流程

推理时只保留动作解码路径：

1. 构建上下文：w_t = (T_l, T_s, T_o)
2. 初始化噪声：a^(0) ~ N(0, I)
3. 积分速度场从 s=0 到 s=1：da^(s)/ds = g_Θ(a^(s), s | w_t)
4. 得到 a^(1)，解码为连续动作 chunk â_{t:t+p-1}
5. 执行动作，观测新状态，重复

**不需要实例化任何未来视频 token**，避免了长序列视觉 token 推理的计算冗余。

如果需要未来预测（如调试或可视化），可以选择性地启用视频分支，复用动作解码时的 KV cache。

---

## 七、实验结果

### 7.1 推理速度对比（A100 GPU）

| 方法 | 类型 | 推理时间 | 仿真 SR | 真机 SR |
|------|------|---------|---------|---------|
| π₀.₅ | VLA | 225ms | 0.48 | 0.69 |
| GigaBrain-0 | VLA | 452ms | – | 0.68 |
| Cosmos-Policy | WAM | 1413ms | – | 0.58 |
| Motus | WAM | 3231ms | 0.88 | 0.76 |
| **GigaWorld-Policy** | **WAM** | **360ms** | **0.86** | **0.83** |

关键发现：
- 比 Motus 快 **9 倍**，真机成功率还高 **7%**
- 比 π₀.₅ 慢 135ms，但真机成功率高 **14%**
- 在 RoboTwin 2.0 仿真 50 个任务上，比 π₀.₅ 提升 **95%**（0.44 → 0.86）

### 7.2 RoboTwin 2.0 仿真结果（50 个任务）

| 方法 | Clean 平均 SR | Randomized 平均 SR |
|------|-------------|-------------------|
| π₀.₅ | 0.43 | 0.44 |
| X-VLA | 0.73 | 0.73 |
| Motus | 0.89 | 0.87 |
| **GigaWorld-Policy** | **0.87** | **0.85** |

GigaWorld-Policy 在快 9 倍的情况下，仿真性能与 Motus 基本持平。

### 7.3 真机实验（4 个任务）

| 任务 | GigaWorld-Policy | Motus | π₀.₅ | Cosmos-Policy |
|------|-----------------|-------|------|--------------|
| 清理桌面 | **0.90** | 0.80 | 0.75 | 0.65 |
| 扫描二维码 | **0.75** | 0.75 | 0.55 | 0.50 |
| 扫垃圾 | **0.75** | 0.70 | 0.65 | 0.45 |
| 叠碗 | **0.90** | 0.80 | 0.80 | 0.70 |
| **平均** | **0.83** | 0.76 | 0.69 | 0.58 |

### 7.4 数据效率（杀手级结果）

**GigaWorld-Policy 只用 10% 的训练数据就能达到 π₀.₅ 用 100% 数据的成功率。**

世界模型预训练带来的先验知识大幅降低了对标注数据的需求。这对实际部署意义重大——真机数据采集成本极高。

### 7.5 消融实验

#### 预训练的重要性

| 配置 | 真机 SR |
|------|---------|
| 从零训练 | 0.45 |
| 只有视频预训练 | 0.57 |
| 只有具身数据预训练 | 0.73 |
| **两者都有** | **0.83** |

结论：视频预训练和具身数据预训练提供互补的收益。

#### 未来帧预测数量的影响

| 采样间隔 Δ | 预测帧数 K | 真机 SR |
|-----------|-----------|---------|
| 不预测 | 0 | 0.60 |
| 4 | 12 | 0.76 |
| 8 | 6 | 0.78 |
| **12** | **4** | **0.83** |
| 24 | 2 | 0.80 |
| 48 | 1 | 0.76 |

结论：适量的未来建模就够了（4帧最优），过多反而有害。

#### 因果掩码 vs 全注意力

| 方法 | SR | PSNR | SSIM |
|------|-----|------|------|
| 全注意力 | 0.81 | 27.87 | 0.892 |
| **因果掩码** | **0.83** | **28.41** | **0.901** |

因果掩码不仅让推理时可以去掉视频分支，还因为防止了信息泄漏，视频生成质量反而更好。

#### 具身数据量的影响

增加具身预训练数据量，真机成功率从 57%（无具身预训练）稳步提升到 83%（全量数据），呈现清晰的正相关趋势。

---

## 八、CVPR 2026 Workshop 世界模型竞赛赛道

### 8.1 赛道定位

GigaBrain Challenge 2026 CVPR Workshop 的世界模型赛道，评估世界模型在机器人操作中的能力。

### 8.2 数据集结构

数据集包含多个操作任务（task1-task8），每个任务有三个 split：

| Split | GT 视频 | 轨迹数据 | 初始状态 | 用途 |
|-------|---------|---------|---------|------|
| Train | 有 | 有 | 有 | 模型训练 |
| Video Quality | 无 | 有 | 有 | 视频生成质量评测 |
| Evaluator | 无 | 无（仅初始） | 有（仅初始） | 世界模型+VLA 闭环交互评测 |

每个任务的数据包含：
- 多视角视频（cam_high, cam_left_wrist, cam_right_wrist）
- 轨迹数据（.pkl 格式的状态序列）
- 元数据（JSON 格式的任务指令）
- 额外提供深度图和仿真器渲染

### 8.3 两种评测模式

**Offline（离线）**：
- 世界模型直接消费轨迹数据生成未来视频帧
- 评测视频生成质量（PSNR、SSIM 等）
- 不涉及策略交互
- 命令：`python scripts/inference.py --mode offline`

**Online（在线）**：
- 世界模型与 VLA 策略闭环交互
- 策略输出动作 → 世界模型预测下一帧 → 策略根据预测帧继续输出动作
- 评测世界模型作为"环境模拟器"的能力
- 需要启动 RoboTwin 2.0 仿真器服务
- 命令：`python scripts/inference.py --mode online`

### 8.4 技术栈

赛道提供了完整的 baseline 代码和工具链：

- **GigaTrain**：训练框架（https://github.com/open-gigaai/giga-train）
- **GigaDataset**：数据加载框架（https://github.com/open-gigaai/giga-datasets）
- **RoboTwin 2.0**：仿真器（用于将 qpos 动作渲染为图像）
- **GigaBrain Policy**：用于在线评测的策略模型

### 8.5 环境搭建

```bash
conda create -n giga_torch python=3.11.10
conda activate giga_torch

# 安装训练框架
cd third_party/giga-train && pip3 install -e .

# 安装数据集框架
cd third_party/giga-datasets && pip3 install -e .

# 下载预训练模型
python scripts/download_pretrained_models.py
python scripts/download_gigabrain_policy.py
```

### 8.6 训练和推理流程

```bash
# 打包训练数据
python scripts/pack_training_data.py --task all

# 启动训练
python scripts/launch_train.py --config_path cvpr_2026_workshop_wm_track.configs.baseline_wm_task4.config

# 离线推理（视频质量评测）
python scripts/inference.py --transformer_model_path /path/to/model --mode offline --task task4

# 在线推理（闭环交互评测）
python simulator/script/run_simulator_server.py --host_port 9151  # 先启动仿真器
python scripts/inference.py --transformer_model_path /path/to/model --mode online --task task4
```

---

## 九、GigaWorld-Policy vs RDT2 对比

| 维度 | RDT2 | GigaWorld-Policy |
|------|------|-----------------|
| 技术路线 | VLA（纯动作预测） | WAM（世界模型+动作） |
| Backbone | Qwen2.5-VL 7B（VLM） | Wan 2.2 5B（视频生成模型） |
| 参数量 | 7B + 400M action expert | 5B |
| 训练数据 | 10,000h UMI 人类数据 | 10,000h 混合数据（人类+机器人） |
| 动作建模 | RVQ 离散化 + Flow Matching | Flow Matching |
| 视频生成 | 不涉及 | 训练时联合，推理时可选 |
| 跨 embodiment | 通过 UMI 实现 zero-shot | 未强调 |
| 推理速度 | 蒸馏后极快（RDT2-UltraFast） | 360ms（去掉视频分支） |
| Scaling Laws | 首次验证 | 未涉及 |
| 数据效率 | 未专门研究 | 10% 数据达到 VLA 100% 效果 |
| 核心创新 | 三阶段训练（离散→扩散→蒸馏） | 训繁推简（因果掩码解耦） |

两者代表了 2026 年具身智能的两条技术路线：
- **RDT2**：数据驱动 + VLM backbone + 跨 embodiment 泛化
- **GigaWorld-Policy**：世界模型驱动 + 视频生成 backbone + 密集监督

---

## 十、关键洞察与战略意义

### 10.1 世界模型用于策略学习的价值已被验证

GigaWorld-Policy 的数据效率结果（10% 数据 = VLA 100% 数据）是杀手级的。这意味着世界模型预训练可以大幅降低真机数据采集成本，这对商业化至关重要。

### 10.2 "训繁推简"是可落地的工程范式

不需要推理时生成视频，解决了世界模型最大的部署痛点（推理慢）。360ms 的推理延迟已经接近 VLA 的水平（π₀.₅ 为 225ms），同时性能大幅领先。

### 10.3 世界模型的竞争焦点已经转移

从"能不能生成漂亮的视频"转向"怎么用视频监督来提升动作质量"。这是一个更务实、更有商业价值的方向。

### 10.4 开源生态已经形成

CVPR 2026 Workshop 赛道提供了现成的评测基础设施——数据集、baseline 代码、评测平台都是开源的，可以直接用来做研究或产品验证。

### 10.5 预训练数据的分层策略值得借鉴

GigaWorld-Policy 的三阶段数据策略（互联网视频 → 具身操作视频 → 少量真机标注）是一个非常实用的范式，消融实验证明每一层都有不可替代的贡献。
