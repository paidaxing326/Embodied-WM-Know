# WMPO：基于世界模型的 VLA 策略优化深度解读

> **论文**: WMPO: World Model-based Policy Optimization for Vision-Language-Action Models
> **作者**: Fangqi Zhu¹², Zhengyang Yan¹, Zicong Hong¹, Quanxin Shou¹, Xiao Ma²*, Song Guo¹*
> **机构**: ¹香港科技大学 　²字节跳动 Seed
> **发表**: arXiv 2511.09515 (2025年11月)
> **论文链接**: arxiv.org/abs/2511.09515 ｜ **项目主页**: wm-po.github.io ｜ **代码**: github.com/WM-PO/WMPO
> **解读时间**: 2026-06-11 ｜ 解读基于论文全文（含附录）与官方源码逐模块交叉验证

---

## 目录

1. [一句话概览与核心贡献](#1-一句话概览与核心贡献)
2. [背景与动机：VLA 为什么需要 RL、又为什么做不了 RL](#2-背景与动机vla-为什么需要-rl又为什么做不了-rl)
3. [核心方法：WMPO 框架全解析](#3-核心方法wmpo-框架全解析)
4. [像素空间生成世界模型](#4-像素空间生成世界模型)
5. [奖励模型：VideoMAE 轨迹分类器](#5-奖励模型videomae-轨迹分类器)
6. [On-Policy GRPO 策略优化](#6-on-policy-grpo-策略优化)
7. [源码架构全景（逐模块验证）](#7-源码架构全景逐模块验证)
8. [实验设置与结果](#8-实验设置与结果)
9. [涌现行为分析：自纠正与高效执行](#9-涌现行为分析自纠正与高效执行)
10. [泛化能力与终身学习](#10-泛化能力与终身学习)
11. [真实机器人实验](#11-真实机器人实验)
12. [与相关工作的对比](#12-与相关工作的对比)
13. [关键创新与贡献总结](#13-关键创新与贡献总结)
14. [局限性、未来方向与反思](#14-局限性未来方向与反思)
15. [对"世界模型 + VLA RL"赛道的学习价值](#15-对世界模型--vla-rl赛道的学习价值)
16. [参数参考表](#16-参数参考表)

---

## 1. 一句话概览与核心贡献

> **WMPO 解决的核心问题**：VLA 模型依赖专家演示的模仿学习训练，无法从失败中学习、不能自我纠正。直接在真实机器人上做 RL 代价高昂且难以实现 on-policy。WMPO 的核心洞察是——**在一个高保真像素空间世界模型中"想象"完整轨迹，然后完全在想象空间中执行 on-policy GRPO**，无需真实环境交互。

三个关键贡献：

| 贡献 | 一句话本质 | 解决什么 |
|---|---|---|
| **① 像素空间世界模型 + 策略行为对齐** | 基于 OpenSora 的视频扩散模型，在 OXE 上预训练后用策略自身数据微调，生成完整的机器人交互视频 | 消除专家演示与策略行为之间的分布偏移；使世界模型忠实模拟失败场景 |
| **② 世界模型内的 On-Policy GRPO** | 在世界模型中从同一初始状态采样多条完整轨迹，用奖励模型判断成功/失败，执行 GRPO 策略更新 | 避免真实机器人交互的样本效率问题；on-policy 比 off-policy（DPO）更强 |
| **③ 涌现的自纠正行为** | 策略在世界模型中学到了从失败状态恢复的策略，这些行为从未出现在专家演示中 | 超越模仿学习的能力天花板，实现真正的"学习从失败中恢复" |

**核心成绩**：在 MimicGen 四个精细操控任务上，WMPO（P=1280）平均成功率 **57.6%**，比 Base Policy（33.6%）提升 **+24.0pp**，比最强 baseline DPO（42.4%）高出 **+15.2pp**。真实机器人上成功率 **70%**，比 Base Policy（53%）高 **+17pp**。

---

## 2. 背景与动机：VLA 为什么需要 RL、又为什么做不了 RL

### 2.1 模仿学习的天花板

当前 VLA 模型（RT-2、OpenVLA、π₀ 等）的主流训练范式是**模仿学习（Imitation Learning, IL）**：从大规模人类演示中学习视觉-语言到动作的映射。这套范式有效但存在根本缺陷：

- **只能学专家做过的**：从未见过失败状态，遇到 OOD 状态时因复合错误导致彻底失败
- **无法利用失败经验**：机器人执行失败的数据被浪费，策略无法从错误中改进
- **执行效率低**：纯 IL 策略常在次优状态"卡住"，反复尝试无效动作直到超时

### 2.2 RL 的自然解法与实际困境

强化学习天然适合解决上述问题——让策略通过与环境的交互、从成功和失败中学习。但将 RL 应用到 VLA 面临两大瓶颈：

```
                 真实机器人上的 RL 困境
                        │
        ┌───────────────┼───────────────┐
        ▼                               ▼
  ① 物理交互瓶颈                ② On-Policy 难以实现
  每次策略更新需要数百条          Off-policy（DPO）方法弱
  真实轨迹，耗时耗力且危险        On-policy（GRPO）需要从同一
  G=8 就需要 8×64=512 条         状态重复采样，真实世界做不到
```

**关键观察**：GRPO（Group Relative Policy Optimization）需要从**同一初始状态采样一组轨迹**，比较组内相对优劣来计算优势函数。这在真实世界中**物理上不可能**——你不能让机器人回到完全相同的初始状态做 8 次不同的尝试。而 off-policy 方法（如 DPO）虽然可用，但性能显著弱于 on-policy。

### 2.3 WMPO 的核心洞察

WMPO 的解决方案极其优雅：

> **如果有一个足够好的世界模型，能把"交互环境"替换成"视频生成环境"，那么：(1) 无需真实交互；(2) 从同一初始状态反复采样变得轻而易举；(3) 可以执行 on-policy GRPO。**

而且 WMPO 特别强调**像素空间**世界模型（而非隐空间），原因是：VLA 模型在网页规模图像上预训练，其视觉特征与真实图像对齐。如果世界模型在隐空间操作，生成的"想象数据"与 VLA 预训练知识之间会产生根本性不匹配。

---

## 3. 核心方法：WMPO 框架全解析

### 3.1 问题建模：MDP 形式化

WMPO 将 VLA 操控任务建模为 MDP $\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, R)$：

| 要素 | 定义 | 说明 |
|------|------|------|
| **状态空间** $\mathcal{S} = \mathcal{I} \times \mathcal{G}$ | 图像观测 $\mathcal{I}$（图像序列 $I_{0:K}$）× 语言指令 $\mathcal{G}$ | 假设机器人状态可由图像观测完全定义 |
| **动作空间** $\mathcal{A}$ | 动作块（action chunk）序列 $a_t \in \mathbb{R}^{K \times D}$ | $K$ 步 × $D$ 自由度；离散化为 256 bins |
| **转移函数** $P$ | 世界模型 $s_{t+1} \sim p_\phi(s_{t+1} \mid s_t, a_t)$ | 像素空间视频扩散生成 |
| **奖励函数** $R$ | 学习的奖励模型 $R_\psi(\tau) \in \{0, 1\}$ | 轨迹级二值奖励（成功/失败） |

**优化目标**：

$$\max_\theta \mathbb{E}_{\tau \sim \pi_\theta, p_\phi} [R_\psi(\tau)]$$

这揭示了一个通用范式：**VLA 的 RL 可以完全解耦于真实世界交互，前提是有足够好的世界模型**。

### 3.2 三阶段训练流水线

WMPO 的训练过程分为三个组件循环迭代：

```
┌─────────────────────────────────────────────────────────────────┐
│                      WMPO 训练循环                               │
│                                                                 │
│  ① Imagined Trajectory Generation                               │
│  ┌─────────────┐      ┌─────────────┐                           │
│  │ VLA Policy  │─────▶│ World Model │──┐                        │
│  │ π_{θ_old}   │◀─────│   p_φ       │  │ 交替交互               │
│  └─────────────┘      └─────────────┘  │ 直到最大长度 N          │
│        │                                │                        │
│        │ 生成完整想象轨迹 τ              │                        │
│        ▼                                ▼                        │
│  ② Trajectory Sampling                                          │
│  从同一初始状态采样 G 条轨迹 {τ₁,...,τ_G}                         │
│  用奖励模型 R_ψ 判断每条轨迹成功/失败                               │
│  动态采样：丢弃全成功或全失败的组                                    │
│        │                                                        │
│        ▼                                                        │
│  ③ Policy Update (GRPO)                                         │
│  计算优势 Â_i = (R_i - mean) / std                               │
│  用裁剪目标更新策略 θ                                             │
│  θ_old ← θ，回到 ①                                               │
└─────────────────────────────────────────────────────────────────┘
```

#### 想象轨迹生成（Imagined Trajectory Generation）

给定 $c$ 帧初始图像 $I_{0:c}$，策略和世界模型交替交互：

1. **策略预测**：$a_{i:i+K} \sim \pi_\theta(I_{i-m:i}, g)$ —— 策略取最近 $m$ 帧 + 语言指令，预测长度为 $K$ 的动作块
2. **世界模型生成**：$I_{i:i+K} \sim p_\phi(I_{i-c:i}, a_{i:i+K})$ —— 世界模型取最近 $c$ 帧条件 + 动作块，生成未来 $K$ 帧
3. 重复直到最大长度 $N$，得到完整轨迹 $\tau = \{I_{0:N}, a_{0:N}\}$

**关键参数**：$c = 4$（条件帧数），$K = 8$（动作块长度/预测帧数），$m$（策略输入帧数）。

#### 轨迹采样与奖励评估

从初始状态数据集 $\mathcal{D}$ 中采样初始帧 $I_{0:c}$，用当前策略 $\pi_{\theta_{\text{old}}}$ 在世界模型中采样 $G$ 条轨迹。奖励模型 $R_\psi$ 评估每条轨迹，输出二值标签 $R_\psi(\tau) \in \{0, 1\}$。

**动态采样策略**（来自 DAPO）：如果一组 $G$ 条轨迹全部成功或全部失败，则丢弃该组重新采样。这保证了优势计算有意义，避免梯度消失。

---

## 4. 像素空间生成世界模型

### 4.1 架构选择：为什么是 OpenSora + SDXL VAE

WMPO 的世界模型基于 **OpenSora**（开源视频生成框架），但做了关键修改：

| 组件 | 原始 OpenSora | WMPO 修改 | 原因 |
|------|--------------|----------|------|
| **VAE** | 3D VAE（时空联合压缩） | **SDXL 2D VAE** | 更好保留精细运动细节，避免时间维度过度压缩导致的时间扭曲 |
| **扩散空间** | VAE 隐空间 | VAE 隐空间 | 保持生成效率 |
| **输出** | 隐空间表示 | **解码回像素空间** | 让 VLA 利用预训练视觉知识 |

**核心设计哲学**：世界模型在 VAE 隐空间中做扩散生成（效率），但在喂给 VLA 之前**解码回像素空间**（兼容性）。这样 VLA 看到的是与其预训练数据一致的像素级图像，而非陌生的隐空间表示。

### 4.2 Noisy-Frame Conditioning：解决长程生成退化

世界模型以自回归方式生成长轨迹——先前生成的帧作为后续预测的条件。这会导致**误差累积**：早期的小误差在长程生成中不断放大，最终导致画面崩溃。

**解决方案：噪声帧条件化（Noisy-Frame Conditioning）**

> 在训练时，条件帧 $I_{i-m:i}$ 不保持干净，而是**注入扩散噪声（50/1000 步）**。

这个看似简单的设计使世界模型能够稳定生成**数百帧**的长轨迹而无明显质量下降。其原理是：通过在训练时让模型适应"不完美"的条件帧，推理时即使前面的生成有微小偏差，模型也能稳健地继续生成。

### 4.3 帧级动作控制：AdaLN 注入

为了实现精确的动作条件控制，WMPO 扩展了 **AdaLN**（Adaptive Layer Normalization）机制，在**帧级别**注入动作信号和扩散时间步：

$$\mathbf{x}^i = \mathbf{x}^i + (1 + \alpha_1^i) \cdot \text{Block}(\gamma_1^i \cdot \text{LayerNorm}(\mathbf{x}^i) + \beta_1^i)$$

其中 $\gamma_1^i$（缩放）、$\beta_1^i$（偏移）、$\alpha_1^i$（残差缩放）由一个 MLP 从每个动作 $a_i$ 生成。这个设计灵感来自 IRASim（同一作者此前的世界模型工作）。

**帧级 vs 视频级**：传统方法在整个视频级别注入一个全局条件。WMPO 的帧级设计确保**每一帧都能感知对应的精细动作**，实现更准确的动作-帧对齐。

### 4.4 策略行为对齐（Policy Behavior Alignment）

这是 WMPO 的一个核心创新点：

```
          OXE 数据集（数百万条专家轨迹）
                 │ 预训练
                 ▼
          世界模型获得广泛的物理动力学知识
          但：主要包含成功演示，缺乏失败场景
                 │
                 ▼ 策略行为对齐
          用策略自身收集的真实轨迹微调世界模型
          │
          ├── 适应策略的 (state, action) 分布
          ├── 学会忠实模拟失败场景
          └── 缩小专家演示与策略行为之间的分布偏移
```

**为什么这一步至关重要？**

如果世界模型只在专家演示上训练：
1. 它**从未见过策略产生的失败轨迹**，无法忠实模拟失败场景
2. 想象的轨迹将**偏向成功**，GRPO 无法获得足够的负样本对比
3. 策略无法学到"如何从失败中恢复"——因为世界模型想象不出失败

通过用策略自身的行为数据（包括成功和失败）微调，世界模型变得能忠实模拟策略的所有可能行为，包括那些导致失败的行为。

---

## 5. 奖励模型：VideoMAE 轨迹分类器

### 5.1 设计思路

WMPO 需要自动判断一条想象轨迹是"成功"还是"失败"，以提供 GRPO 所需的奖励信号。设计要求：
- **无需复杂奖励工程**：避免手工设计密集奖励
- **避免奖励作弊（Reward Hacking）**：短视的密集奖励容易被策略钻空子
- **可靠**：错误判断会直接误导策略优化

解决方案：一个**轻量级轨迹级二值分类器**，判断完整轨迹是否表示任务成功。

### 5.2 架构与训练

**模型**：VideoMAE（Video Masked AutoEncoder）+ 线性分类头

**训练数据构造**：

| 样本类型 | 定义 | 标签 |
|----------|------|------|
| **正样本** | 成功轨迹的**末端片段** $c_N = I_{N-L:N}$ | 1 |
| **负样本-类型A** | 成功轨迹的**非末端片段** $\{c_i : L \leq i \leq N-L\}$ | 0 |
| **负样本-类型B** | 失败轨迹的**任意片段** | 0 |

其中 $L = 8$（视频片段长度）。**类别平衡**：每个训练 batch 中正负样本数量相等。

**损失函数**：二元交叉熵（Binary Cross-Entropy）。

### 5.3 推理过程

推理时，奖励模型用**滑动窗口**（stride = 1）在轨迹 $\tau$ 上计算每个片段的成功概率：

```python
# 伪代码（来自源码 predict_success 方法）
for video in videos:
    finish_step = total_frames - 1
    complete = 0
    for end in range(total_frames, min_steps + window_size - 1, -stride):
        clip = video[end - window_size:end]
        prob = reward_model(clip)
        if prob >= threshold and end - 1 < finish_step:
            finish_step = end - 1
            complete = 1
            break  # 找到最早的完成帧即停止
```

**阈值选择**：$\tau_{\text{thr}}$ 通过验证实验选择（代码中默认搜索范围 0.3~1.0，步数 20）。

**关键设计**：轨迹被判定为成功，当且仅当**任意一个片段超过阈值**。这种"任一命中"的设计确保即使轨迹中只有部分帧展示成功迹象也能被正确识别。

### 5.4 性能

论文报告奖励模型在所有任务上达到 **F1 > 0.95**，能可靠区分成功与失败，有效缓解奖励作弊。

---

## 6. On-Policy GRPO 策略优化

### 6.1 为什么选 GRPO

GRPO（Group Relative Policy Optimization）是 DeepSeek-R1 证明其威力的策略优化算法。它的核心思想：

> 从同一初始状态采样一组轨迹 $G$ 条，**在组内比较相对优劣**来计算优势函数，无需训练价值函数（Critic）。

这天然适合 WMPO 的设定：
- **世界模型中可以轻松实现"从同一初始状态重复采样"**（这在真实世界做不到）
- **稀疏二值奖励**（只有最终成功/失败），GRPO 不需要密集奖励
- **无需 Critic 网络**，减少计算开销和训练复杂度

### 6.2 优势函数计算

对每组 $G$ 条轨迹，优势函数为：

$$\hat{A}_i = \frac{R_i - \text{mean}(\{R_i\}_{i=1}^G)}{\text{std}(\{R_i\}_{i=1}^G)}$$

其中 $R_i = R(\tau_i) \in \{0, 1\}$ 是奖励模型对第 $i$ 条轨迹的判断。

**源码实现**（`core_algos.py`）：

```python
def compute_grpo_outcome_advantage(token_level_rewards, eos_mask, index, epsilon=1e-6):
    scores = token_level_rewards.sum(dim=-1)
    id2score = defaultdict(list)
    for i in range(bsz):
        id2score[index[i]].append(scores[i])
    # 按 group 计算均值和标准差
    for idx in id2score:
        id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
        id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
    # 标准化
    for i in range(bsz):
        scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
    return scores, scores
```

### 6.3 策略更新目标

WMPO 遵循 DAPO 的设计，**移除了 KL 正则化**（不需要参考模型，节省内存），最终目标为：

$$\mathcal{J}(\theta) = \mathbb{E}_{s_0, \{\tau_i\}} \left[ \frac{1}{G} \sum_{i=1}^G \frac{1}{T} \sum_{t=0}^T \min\left(r_{i,t}(\theta)\hat{A}_i, \text{clip}(r_{i,t}(\theta), 1-\epsilon_{\text{low}}, 1+\epsilon_{\text{high}})\hat{A}_i \right) \right]$$

其中概率比：

$$r_{i,t}(\theta) = \frac{\pi_\theta(a_{i,t} \mid s_{i,t})}{\pi_{\theta_{\text{old}}}(a_{i,t} \mid s_{i,t})}$$

**非对称裁剪**：$\epsilon_{\text{low}} = 0.20$, $\epsilon_{\text{high}} = 0.28$。注意这里用的是**非对称裁剪**（DAPO 风格），而非标准 PPO 的对称 $\epsilon$。

### 6.4 动作的对数概率计算

每个动作块被离散化为 $K \times D$ 个 token（$K$ 步，每步 $D$ 个自由度，每维度 256 bins）：

$$\log \pi_{\theta_{\text{old}}}(a_t \mid s_t) = \sum_{i=1}^{K} \sum_{j=1}^{D} \log \pi_{\theta_{\text{old}}}(a_t^{i,j} \mid s_t)$$

### 6.5 动态采样

如果一组中所有轨迹都被判定为成功（全 1）或全部失败（全 0），则丢弃该组重新采样。这确保优势函数 $\hat{A}_i$ 有意义（组内有区分度），避免梯度消失。

---

## 7. 源码架构全景（逐模块验证）

### 7.1 代码库结构

```
WMPO/
├── verl/                          # 核心训练框架（基于 verl 改造）
│   ├── trainer/
│   │   ├── ppo/
│   │   │   ├── core_algos.py      # GRPO/DPO/RLOO 等算法核心实现
│   │   │   └── ray_trainer.py     # Ray 分布式训练主循环
│   │   └── config/
│   │       └── ppo_trainer.yaml   # 训练超参数配置
│   ├── workers/
│   │   ├── rollout/
│   │   │   ├── robwm_rollout.py   # ★ 世界模型 Rollout 核心（想象轨迹生成）
│   │   │   └── rob_rollout.py     # 真实环境 Rollout（用于 baseline）
│   │   ├── actor/
│   │   │   └── dp_rob.py          # VLA Actor（策略梯度更新）
│   │   └── fsdp_workers.py        # ★ FSDP Worker（世界模型构建、初始化）
│   └── utils/
│       ├── dataset/
│       │   ├── rl_dataset.py       # RL 训练数据集
│       │   └── rob_dataset.py      # 机器人状态数据集
│       └── vla_utils/             # OpenVLA/OpenVLA-OFT 模型工具
├── reward_model/
│   ├── videomae.py                # ★ VideoMAE 奖励模型训练代码
│   ├── inference_videomae.py      # 奖励模型推理
│   └── find_thre.py              # 阈值搜索
├── dependencies/
│   ├── opensora/                  # OpenSora 视频生成框架（世界模型骨干）
│   └── openvla-oft/              # OpenVLA-OFT（基础 VLA 策略）
├── examples/
│   ├── mimicgen/                  # 各任务 WMPO 训练脚本
│   └── opensora/                  # 世界模型训练脚本
└── download_hf.py                 # 下载预训练模型和数据
```

### 7.2 核心模块深度解析

#### `robwm_rollout.py`：世界模型 Rollout

这是 WMPO 最核心的模块，实现了"在世界模型中想象完整轨迹"的逻辑：

```python
class RobWMHFRollout(BaseRollout):
    def __init__(self, module, world_model_mapping, config):
        # 加载 VLA 策略、世界模型、VAE、调度器、奖励模型
        self.vae = world_model_mapping["vae"]              # SDXL 2D VAE
        self.world_model = world_model_mapping["model"]    # STDiT-v3 扩散模型
        self.scheduler = world_model_mapping["scheduler"]  # 扩散采样调度器
        self.rm_model = world_model_mapping["rm_model"]    # VideoMAE 奖励模型
        self.queue_len = 4   # 条件帧数 c = 4
```

**`run_wm_inference` 方法**：核心的想象轨迹生成循环

```python
def run_wm_inference(self, image_paths, max_steps, repeat=1):
    # 1. 加载初始帧，编码到 VAE 隐空间
    latents = self.vae.encode(init_frames_for_vae)
    image_history_tensor = latents.repeat(1, 1, queue_len, 1, 1)
    
    while frame_num <= max_steps:
        # 2. VLA 策略预测动作块
        vla_output = self._generate_one_step(vla_input)
        actions = vla_output["action"]
        
        # 3. 世界模型：条件帧 + 噪声 → 扩散去噪 → 生成未来帧
        z = torch.randn(...)  # 随机噪声
        z_combined = concat([image_history_tensor, z])  # 拼接条件帧和噪声
        masks = ...  # 标记哪些帧是待生成的
        samples = self.scheduler.sample(self.world_model, z=z_combined, y=actions, ...)
        
        # 4. 解码回像素空间
        pred_latents = samples[:, :, -chunk:]
        decoded_images = self.vae.decode(pred_latents)
        
        # 5. 更新历史帧（用于下一轮自回归）
        image_history_tensor = pred_latents[:, :, -queue_len:]
        current_frames_np = decoded_images 的最后一帧
```

**关键实现细节**：
- 世界模型在 VAE 隐空间生成（效率），但解码回像素空间后才喂给 VLA（兼容性）
- 每轮生成 $K = 8$ 帧，对应一个动作块
- 用最近 $c = 4$ 帧作为条件帧（`queue_len = 4`）
- 动作通过 AdaLN 以帧级粒度注入世界模型

**`predict_success` 方法**：奖励模型推理

```python
def predict_success(self, videos, batch_size=128):
    for video in videos:
        clips = []
        # 滑动窗口从后往前扫描
        for end in range(total_frames, min_steps + window_size - 1, -stride):
            clip = video[end - window_size:end]
            clips.append(clip)
        
        # 找到最早的成功帧
        for batch in clip_batches:
            inputs = self.rm_feature_extractor(clip_imgs)
            logits = self.rm_model(pixel_values=inputs).logits
            probs = torch.sigmoid(logits)
            preds = [1 if p[1] >= threshold else 0 for p in probs]
            if pred == 1:
                finish_step = end - 1
                complete = 1
                break
```

#### `fsdp_workers.py`：世界模型构建

`_build_world_model` 方法展示了世界模型的完整初始化流程：

```python
def _build_world_model(self):
    # 1. 加载配置
    cfg = read_config(self.config.wm.inference_config_path)
    
    # 2. 构建 SDXL 2D VAE
    vae = build_module(cfg.vae, MODELS).to(device, dtype).eval()
    
    # 3. 构建扩散模型（STDiT-v3）
    model = build_module(cfg.model, MODELS, input_size=latent_size, 
                         in_channels=vae.out_channels).to(device, dtype).eval()
    
    # 4. 构建调度器
    scheduler = build_module(cfg.scheduler, SCHEDULERS)
    
    # 5. 构建奖励模型（VideoMAE）
    rm_model = VideoMAEForVideoClassification.from_pretrained(
        "MCG-NJU/videomae-base", num_frames=8, num_labels=2)
    rm_model.load_state_dict(torch.load(reward_model_path))
    
    return {"model": model, "scheduler": scheduler, "vae": vae,
            "rm_model": rm_model, "rm_threshold": threshold}
```

当 `update_wm=True` 时（终身学习模式），还支持用策略新数据在线更新世界模型：

```python
# 终身学习模式：ColossalAI + Zero2 并行训练世界模型
booster = Booster(plugin=plugin)
model, optimizer, _, _, lr_scheduler = booster.boost(model=model, ...)
# EMA 模型
ema = deepcopy(model).cpu().to(torch.float32)
```

#### `core_algos.py`：GRPO 算法实现

```python
def compute_policy_loss(old_log_prob, log_prob, advantages, eos_mask, 
                        clip_ratio_high, clip_ratio_low):
    ratio = torch.exp(log_prob - old_log_prob)
    pg_losses = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)
    pg_loss = masked_mean(torch.max(pg_losses, pg_losses2), eos_mask)
    return pg_loss, pg_clipfrac, ppo_kl
```

注意**非对称裁剪**：`clip_ratio_low = 0.20`, `clip_ratio_high = 0.28`，这是 DAPO 的设计。

#### `ray_trainer.py`：训练主循环

```python
class RayTrainer:
    def _create_dataloader(self):
        # 加载初始状态数据集
        state_path = f'./verl/utils/dataset/{task_name}_d0_states.pkl'
        self.train_dataset = StateDataset(state_path)
        self.train_dataloader = BufferedDataLoader(...)
    
    def _validate(self, global_steps):
        # 验证：从初始状态生成轨迹 + 奖励模型评估
        for test_data in self.val_dataloader:
            test_output = self.actor_rollout_wg.generate_sequences(test_batch)
            verifier_score = self.val_reward_fn.verify(test_batch)
```

### 7.3 分布式训练架构

```
┌─────────────────────────────────────────────┐
│              Ray Cluster                      │
│                                               │
│  Head Node: RayTrainer (主控)                 │
│    ├── ActorRollout Worker (FSDP)             │
│    │     ├── VLA Policy (OpenVLA-OFT)         │
│    │     ├── World Model (STDiT-v3 + SDXL VAE)│
│    │     └── Reward Model (VideoMAE)          │
│    └── (可选) Ref Policy Worker               │
│                                               │
│  Worker Nodes: 并行 rollout 生成               │
└─────────────────────────────────────────────┘
```

- **32 × H100 GPU** 用于世界模型训练和策略优化
- **8 × H100 GPU** 用于 OpenVLA-OFT 的监督微调
- 支持**多节点 Ray 集群**训练（`launch_head.sh` / `launch_worker.sh`）

---

## 8. 实验设置与结果

### 8.1 仿真实验

**环境**：MimicGen 仿真基准

| 任务 | 描述 | 最大步数 | Base Policy 成功率 |
|------|------|---------|-------------------|
| **Coffee** | 制作咖啡 | 256 | 43.8% |
| **StackThree** | 三物体堆叠 | 320 | 46.9% |
| **ThreePieceAssembly** | 三件组装 | 384 | 19.5% |
| **Square** | 方块插入杆（5mm 间隙） | 184 | 24.2% |

**基础策略**：OpenVLA-OFT，每任务用 300 条专家轨迹微调，动作块长度 $K=8$。

**评估**：128 个不同初始状态，报告平均成功率。

### 8.2 主实验结果

| Rollout Budget $P$ | Methods | Coffee | StackThree | ThreePiece | Square | **Mean** |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| — | Base Policy | 43.8 | 46.9 | 19.5 | 24.2 | 33.6 |
| **P=128** | GRPO (online) | 38.3 | 52.3 | 17.2 | 25.0 | 33.2 |
| | DPO (offline) | 43.8 | 53.9 | 23.4 | 28.1 | 37.3 |
| | **WMPO (ours)** | **61.7** | **56.3** | **37.5** | **32.8** | **47.1** |
| **P=1280** | GRPO (online) | 47.7 | 54.7 | 20.3 | 25.8 | 37.1 |
| | DPO (offline) | 52.3 | 57.0 | 26.7 | 33.6 | 42.4 |
| | **WMPO (ours)** | **75.0** | **64.1** | **46.1** | **45.3** | **57.6** |

**核心发现**：

1. **数据效率**：WMPO 仅用 $P=128$ 条轨迹就超过 DPO（P=1280）的 37.3%，达到 47.1%
2. **可扩展性**：从 P=128 到 P=1280，WMPO 提升 **+10.5pp**（47.1→57.6），而 DPO 仅提升 +5.1pp
3. **GRPO 在真实环境表现差**：P=128 时甚至低于 Base Policy（33.2% vs 33.6%），因为 batch size 限制导致更新次数不足
4. **WMPO 在所有任务上一致领先**：没有某个任务特别弱的短板

### 8.3 Baseline 实现细节

**GRPO（online）**：直接在仿真环境中收集轨迹并更新。主要挑战是 batch size 对性能影响极大：batch=64 需要 64×8=512 条轨迹才能做一次更新，P=128 时不可行。论文报告了 batch=8 和 batch=64 的最佳结果。

**DPO（offline）**：用 Base Policy 收集的轨迹构建偏好对（成功 vs 失败），用标准 DPO 损失优化。优势是可以重复利用数据，但缺乏在线更新能力。

---

## 9. 涌现行为分析：自纠正与高效执行

### 9.1 自纠正行为（Self-Correction）

这是 WMPO 最引人注目的发现。在 Square 任务（将方块插入杆）中：

```
时间 →                                                         
                                                                
Base Policy:  接近 → 碰撞 → 持续推 → 持续推 → ... → 超时失败
                                      ↑ 卡住了，一直重复无效动作
WMPO:        接近 → 碰撞 → 退后 → 重新对准 → 插入 → 成功！
                         ↑ 自主学会了从碰撞中恢复
```

**为什么 IL 策略无法自纠正？**

- IL 只见过专家演示（全是成功的），**从未见过碰撞状态**
- 遇到碰撞时，策略处于 OOD 状态，只能盲目重复之前的动作
- 即使看到过碰撞恢复数据，IL 也只能"模仿"恢复动作，无法理解**为什么**要恢复

**为什么 WMPO 能学到自纠正？**

- 世界模型通过策略行为对齐，**能忠实模拟碰撞和失败场景**
- GRPO 在多条想象轨迹的对比中，**自然发现"退后重试"比"持续推"更有可能成功**
- 这是一个**涌现行为**——没有人显式教策略要这么做，它通过 RL 自己发现了

### 9.2 高效执行

论文分析了成功轨迹的平均长度（以 Base Policy 为 100%）：

| 策略 | 相对轨迹长度 |
|------|-------------|
| Base Policy | 100% |
| GRPO | ~95% |
| DPO | ~90% |
| **WMPO** | **~70%** |

WMPO 策略的轨迹显著更短，因为它：
- 很少在次优状态"卡住"（学到了纠正策略）
- 执行更流畅、更快到达目标
- 这也意味着更高的**任务完成效率**

---

## 10. 泛化能力与终身学习

### 10.1 泛化到新场景

论文设计了三种干扰场景来测试泛化能力：

| 干扰类型 | 任务 | 描述 |
|---------|------|------|
| **位置干扰** | Square | 杆的位置从固定变为随机 |
| **背景干扰** | StackThree | 桌面背景替换为灰色 |
| **纹理干扰** | ThreePieceAssembly | 红色底座替换为深色木底座 |

**结果**：

| Methods | Pos. Dis. | Bg. Dis. | Tex. Dis. | **Mean** |
|---------|-----------|----------|-----------|----------|
| Base Policy | 14.1 | 46.1 | 10.9 | 23.7 |
| GRPO | 15.6 | 47.7 | 10.9 | 24.7 |
| DPO | 16.4 | 34.4 | 7.8 | 19.5 |
| **WMPO** | **22.3** | **50.0** | **16.4** | **29.6** |

**关键洞察**：

- **DPO 泛化最差**：在背景和纹理变化下甚至低于 Base Policy。说明 DPO 依赖了虚假视觉线索（如特定背景颜色），而非真正的操控技能
- **GRPO ≈ Base Policy**：在线 RL 用真实轨迹训练，泛化与 IL 相当
- **WMPO 泛化最好**：完全在世界模型中训练的策略捕获了**更通用的操控策略**，因为世界模型生成的多样想象轨迹天然具有数据增强效果

### 10.2 终身学习（Lifelong Learning）

WMPO 支持迭代式持续改进：

```
迭代 1: 策略 π₀ → 收集 128 条轨迹 → WMPO 优化 → 策略 π₁
迭代 2: 策略 π₁ → 收集 128 条轨迹 → WMPO 优化 → 策略 π₂
迭代 3: 策略 π₂ → 收集 128 条轨迹 → WMPO 优化 → 策略 π₃
```

在 StackThree 任务上的结果：

- **WMPO**：每次迭代稳定提升，三轮后显著超过所有 baseline
- **DPO**：由于训练不稳定，无法实现迭代改进
- **IL（更多专家演示）**：428 条专家轨迹不如 WMPO 用 128 条策略自身数据的效果

**重要区别**：Base Policy 需要人类收集的专家轨迹，而 WMPO 只依赖**策略自身收集的轨迹**，更具可扩展性。

---

## 11. 真实机器人实验

### 11.1 设置

- **平台**：Cobot Mobile ALOHA
- **任务**：将方块插入杆（间隙仅 5mm）
- **基础策略**：200 条专家演示微调 OpenVLA-OFT → 成功率 53%
- **WMPO**：用基础策略收集 128 条轨迹 → 微调世界模型 → WMPO 优化
- **评估**：30 次试验平均

### 11.2 结果

| 方法 | 成功率 |
|------|--------|
| Base Policy | 53% |
| DPO | 60% |
| **WMPO** | **70%** |

WMPO 比基础策略提升 **+17pp**，比 DPO 高 **+10pp**。

### 11.3 世界模型的预测精度

论文展示了世界模型在真实场景中的预测能力：

- **成功轨迹**：世界模型能准确预测未来的成功演化（图 7）
- **失败轨迹**：世界模型能成功预测失败——当方块与杆未对准时，模型预测到方块无法插入（图 8）
- **失败案例**：极少数情况下，模型无法捕捉方块卡在杆上的微妙扰动（图 9），但这类情况在验证集上较为罕见

---

## 12. 与相关工作的对比

### 12.1 VLA RL 方法对比

| 方法 | RL 类型 | 训练环境 | On/Off-Policy | 样本效率 | 可扩展性 |
|------|---------|---------|:---:|---------|---------|
| **Human-in-the-loop RL** | PPO + 人类纠正 | 真实世界 | On | ★★☆ | ★☆☆ |
| **VLA-RL (仿真)** | PPO/GRPO | 仿真器 | On | ★★★ | ★★☆ |
| **GRAPE (DPO)** | DPO | 离线数据 | Off | ★★☆ | ★★★ |
| **SimpleVLA-RL** | GRPO | 仿真/真实 | On | ★★☆ | ★★☆ |
| **WMPO (本文)** | GRPO in WM | 世界模型 | **On** | ★★★ | ★★★ |

### 12.2 世界模型方法对比

| 方法 | 世界模型类型 | 空间 | RL 算法 | 适合 VLA？ |
|------|------------|------|---------|-----------|
| **DreamerV3** | RSSM 隐空间 | 隐空间 | Actor-Critic | ✗ 隐空间与 VLA 不兼容 |
| **DIAMOND** | 扩散模型 | 像素空间 | PPO/Gaussian | △ Gaussian 策略不适合 token 化 VLA |
| **Genie 3** | 大型视频生成 | 像素空间 | — | △ 未验证 RL 可行性 |
| **Cosmos (NVIDIA)** | 基础世界模型 | 像素空间 | — | △ 通用平台，未针对 VLA RL |
| **WMPO (本文)** | OpenSora + SDXL VAE | **像素空间** | **GRPO** | **✓ 首次验证可行性** |

### 12.3 与同系列工作对比

| 对比维度 | WMPO | BOOM | DreamTacVLA |
|---------|------|------|-------------|
| **核心思路** | 世界模型想象 + GRPO | 世界模型自举 + Off-policy RL | 世界模型 + Diffusion Policy |
| **世界模型空间** | 像素（视频生成） | 隐空间（RSSM） | 像素（扩散生成） |
| **RL 类型** | On-policy GRPO | Off-policy (SAC+对齐) | Diffusion Policy RL |
| **策略类型** | 离散化 VLA | 连续高斯 | Flow-matching |
| **适用场景** | VLA 精细操控 | 高维连续控制 | 触觉操控 |

---

## 13. 关键创新与贡献总结

### 13.1 方法论创新

| # | 创新 | 本质 | 影响 |
|---|------|------|------|
| 1 | **像素空间世界模型 for VLA RL** | 用视频生成模型（非隐空间）作为 RL 环境，保持与 VLA 预训练特征的兼容 | 首次验证了高保真像素世界模型可以支撑可扩展的 VLA RL |
| 2 | **策略行为对齐** | 用策略自身数据微调世界模型，使其能模拟失败场景 | 解决了世界模型的分布偏移问题，是 WMPO 能学到自纠正行为的关键 |
| 3 | **Noisy-Frame Conditioning** | 训练时对条件帧注入轻微噪声 | 使世界模型能稳定生成数百帧长轨迹而不崩溃 |
| 4 | **帧级 AdaLN 动作控制** | 在每个 Transformer 块的每个帧级别注入动作信号 | 实现精细的动作-帧对齐，避免动作影响"模糊化" |
| 5 | **世界模型内的 On-Policy GRPO** | 利用世界模型"从同一初始状态重复采样"的能力 | 首次在 VLA 领域实现 on-policy GRPO，显著优于 off-policy DPO |

### 13.2 实验贡献

| # | 发现 | 意义 |
|---|------|------|
| 1 | **涌现自纠正** | RL 在世界模型中学到了专家演示中不存在的恢复策略 |
| 2 | **DPO 泛化陷阱** | DPO 虽然在分布内略有提升，但泛化性反而下降 |
| 3 | **终身学习可行性** | WMPO 可通过迭代收集-优化持续改进，而 DPO 因训练不稳定无法实现 |
| 4 | **真实机器人验证** | 5mm 间隙精细操控任务上成功率从 53% 提升到 70% |

---

## 14. 局限性、未来方向与反思

### 14.1 论文承认的局限性

1. **离散化动作表示**：当前 WMPO 只支持离散化的动作 token 表示，尚未扩展到更强大的 flow-matching 策略（如 π₀）。论文提到未来计划结合 FlowGRPO 进行扩展。

2. **世界模型保真度上限**：在极少数情况下，世界模型无法捕捉微妙物理扰动（如方块卡在杆上的瞬间），这会引入噪声奖励信号。

3. **计算资源需求**：世界模型预训练需要 12M 步，微调需要 3M 步，总计在 32 × H100 上训练。完整的 checkpoint + 数据集约 900GB（364GiB + 530GiB）。

### 14.2 进一步的思考

1. **Sim-to-Real Gap 的世界模型**：当前实验在仿真中验证较多，真实世界仅一个任务。世界模型的仿真到真实迁移能力仍需更广泛验证。

2. **奖励模型的可靠性天花板**：虽然 F1 > 0.95 很高，但在更复杂、更长期的 manipulation 任务中，二值奖励可能不足以区分"好"与"更好"。未来可能需要更细粒度的奖励信号。

3. **长程任务的挑战**：当前任务最长 ~384 步，对于需要数百甚至数千步的长程任务，世界模型的累积误差和奖励模型的准确性都会成为瓶颈。

4. **多任务统一**：当前每个任务训练独立的世界模型和策略。是否可以训练一个通用世界模型服务多个任务的 VLA RL？这是一个重要的扩展方向。

---

## 15. 对"世界模型 + VLA RL"赛道的学习价值

### 15.1 技术路线的启示

WMPO 验证了一条之前未被充分探索的技术路线：

> **高保真像素世界模型 + on-policy RL = 无需真实交互的 VLA 自我改进**

这条路线的核心假设是：**世界模型的保真度足够高，可以替代真实环境进行策略优化**。WMPO 通过以下设计确保这一假设成立：

```
世界模型保真度保障体系：
                    OXE 大规模预训练
                         │  广泛的物理动力学知识
                         ▼
               策略行为对齐微调
                         │  适应策略的 (s,a) 分布 + 模拟失败
                         ▼
               Noisy-Frame Conditioning
                         │  抗误差累积 + 长程稳定
                         ▼
               帧级 AdaLN 动作控制
                         │  精细的动作-帧对齐
                         ▼
               高保真像素空间世界模型
                         │  生成的想象轨迹与 VLA 预训练特征兼容
                         ▼
               GRPO on-policy 优化
```

### 15.2 与本知识库其他工作的关联

| 工作 | 与 WMPO 的关系 |
|------|---------------|
| **BOOM** | 同为"世界模型 + RL"路线，但 BOOM 在隐空间操作、用 off-policy 对齐；WMPO 在像素空间操作、用 on-policy GRPO |
| **DreamTacVLA** | 同为像素空间世界模型 + Diffusion Policy，但 WMPO 用离散化 VLA + GRPO |
| **OmniVTA** | OmniVTA 用世界模型做预测+触觉反射控制，WMPO 用世界模型做 RL 训练环境——世界模型的用途不同 |
| **UniVTAC** | UniVTAC 提供仿真平台，可作为 WMPO 世界模型的训练数据来源之一 |

### 15.3 可迁移的设计模式

1. **"想象 + 对比"范式**：用世界模型从同一初始状态生成多条轨迹，通过对比学习——这条范式可迁移到任何需要 on-policy RL 但真实采样困难的场景。

2. **策略行为对齐**：不管用什么世界模型，都需要确保它能模拟策略的行为（而非仅模拟专家的行为）。这个原则是通用的。

3. **二值结果奖励**：避免复杂奖励工程的简洁方案——只判断"最终是否成功"。在稀疏奖励设定下，GRPO 比 value-based 方法更适合。

4. **像素空间生成 vs 隐空间**：WMPO 的实践表明，对于预训练在大规模图像上的 VLA 模型，像素空间世界模型比隐空间更合适，因为它保持了特征空间的兼容性。

---

## 16. 参数参考表

### 16.1 世界模型训练超参数

| 超参数 | 值 |
|--------|-----|
| Optimizer | AdamW (β₁=0.9, β₂=0.999) |
| Learning Rate | 1×10⁻⁴ |
| Batch Size | 128 |
| Gradient Clip | 0.1 |
| Pretrain Steps | 12,000,000 |
| Fine-tune Steps | 3,000,000 |
| EMA | 0.9999 |
| Weight Decay | 0.0 |
| Prediction Target | ε (噪声预测) |
| 条件帧数 $c$ | 4 |
| 动作块长度 $K$ | 8 |
| Noisy-Frame 步数 | 50/1000 |

### 16.2 GRPO 策略优化超参数

| 超参数 | 值 |
|--------|-----|
| Optimizer | AdamW (β₁=0.9, β₂=0.999) |
| Learning Rate | 5×10⁻⁶ |
| Training Batch Size | 64 |
| Group Size $G$ | 8 |
| Mini-batch Size | 128 |
| Clip Ratio $\epsilon_{\text{low}}$ | 0.20 |
| Clip Ratio $\epsilon_{\text{high}}$ | 0.28 |
| Temperature | 1.6 |
| KL Regularization | 无（移除，DAPO 风格） |

### 16.3 奖励模型超参数

| 超参数 | 值 |
|--------|-----|
| Backbone | VideoMAE-Base |
| 窗口长度 $L$ | 8 |
| 推理 Stride | 1 |
| 阈值搜索范围 | 0.3 ~ 1.0 |
| 阈值搜索步数 | 20 |
| 分类标签 | 2（成功/失败） |

### 16.4 基础 VLA 策略训练

| 超参数 | 值 |
|--------|-----|
| Base Model | OpenVLA-OFT |
| SFT GPU | 8 × H100 |
| Expert Demos / Task | 300 |
| 动作表示 | 离散化 (256 bins/dim) |
| Action Chunk Length | 8 |
| VLA Rollout + WM 训练 GPU | 32 × H100 |

### 16.5 计算资源

| 资源 | 用途 | 规模 |
|------|------|------|
| SFT (OpenVLA-OFT) | Base Policy 微调 | 8 × H100 |
| World Model Pretrain | OXE 预训练 | 32 × H100, 12M steps |
| World Model Fine-tune | 策略行为对齐 | 32 × H100, 3M steps |
| WMPO GRPO | 策略优化 | 32 × H100 |
| Checkpoint + Data | 存储 | ~900 GB (364GiB + 530GiB) |

---

## 参考文献

1. Zhu, F., Yan, Z., Hong, Z., Shou, Q., Ma, X., & Guo, S. (2025). WMPO: World Model-based Policy Optimization for Vision-Language-Action Models. *arXiv:2511.09515*.
2. Shao, Z., et al. (2024). DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. *arXiv:2402.03300*.
3. DeepSeek-AI, et al. (2025). DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. *arXiv:2501.12948*.
4. Yu, Q., et al. (2025). DAPO: An Open-Source LLM Reinforcement Learning System at Scale. *arXiv:2503.14476*.
5. Open X-Embodiment Collaboration. (2023). Open X-Embodiment: Robotic Learning Datasets and RT-X Models. *arXiv:2310.08864*.
6. Kim, M. J., Finn, C., & Liang, P. (2025). Fine-tuning Vision-Language-Action Models: Optimizing Speed and Success. *arXiv:2502.19645*.
7. Zheng, Z., et al. (2024). Open-Sora: Democratizing Efficient Video Production for All. *arXiv:2412.20404*.
8. Zhu, F., et al. (2025). IRASim: A Fine-grained World Model for Robot Manipulation. *arXiv:2406.14540*.
9. Tong, Z., et al. (2022). VideoMAE: Masked Autoencoders are Data-efficient Learners for Self-supervised Video Pre-training. *NeurIPS*.
10. Mandlekar, A., et al. (2023). MimicGen: A Data Generation System for Scalable Robot Learning using Human Demonstrations. *arXiv:2310.17596*.

---

> **解读说明**：本报告基于 arXiv 论文全文（2511.09515v1，含全部附录）、项目主页（wm-po.github.io）和 GitHub 官方源码（WM-PO/WMPO，212 stars）交叉验证撰写。代码库包含完整的训练代码、世界模型代码、奖励模型代码以及预训练 checkpoint 和训练数据，可复现性高。
