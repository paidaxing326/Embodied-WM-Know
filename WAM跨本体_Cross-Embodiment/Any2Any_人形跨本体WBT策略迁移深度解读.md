# Any2Any: 人形机器人 WBT 策略跨本体高效迁移深度解读

> **论文原文**: Any2Any: Efficient Cross-Embodiment Transfer for Humanoid Whole-Body Tracking
> **arxiv**: https://arxiv.org/abs/2605.23733
> **HTML 版本**: https://arxiv.org/html/2605.23733v1
> **PDF**: https://arxiv.org/pdf/2605.23733
> **提交时间**: 2026-05-22 (v1)
> **作者团队**: Ming Yang, Tao Yu (Project Lead, 通讯), Feng Li, Hua Chen (通讯) — 全部来自 **LimX Dynamics (松延动力)**
> **学科分类**: cs.RO (主), cs.AI
> **许可证**: arXiv.org perpetual non-exclusive license

---

## 0. TL;DR

1. **Any2Any 把"已经预训练好的 WBT 专家策略"在 ~1% 数据 + ~1% 算力下迁移到新人形本体**，不需要重新从零训练，也不会破坏源策略已经学到的 motor prior。
2. **方法分两步走**：先做"运动学对齐"（Kinematic Alignment）解决关节数/排列/拓扑差异，再做"动力学适配"（Dynamic Adaptation）用 LoRA 仅更新少量动力学敏感模块（5.26% 参数）。
3. **核心洞察**：现代 WBT 网络的 Reference Motion Encoder（参考运动编码器）携带的是与本体无关的运动先验，而 proprio 输入头 / action 输出头 / critic backbone 才是真正与本体动力学耦合的部分——所以 LoRA 应该只插这些位置。
4. **实战意义**：把跨本体 WBT 部署的成本从 "9k GPU hours + 100M motion frames"（Sonic 原始预训练量级）压缩到 "4×A100 + 1% 数据" 量级，对二级人形厂商（不是 NVIDIA、不是 Google）极其友好。
5. **开源情况**：⚠️ **截至研究时（2026-05-26），Any2Any 论文本身没有公开代码仓库**，但其用到的 Sonic 源策略由 NVIDIA 开源（GR00T-WholeBodyControl + HuggingFace `nvidia/GEAR-SONIC`），论文提到的 LimX Oli URDF 也在 LimX 公开下载页可获取。

---

## 一、论文定位与动机

### 1.1 它要解决的痛点

当前人形 WBT (Whole-Body Tracking) 已经走上"基础模型 (Behavior Foundation Model, BFM)"路线：

| 代表工作 | 模型规模 | 数据规模 | 算力 |
|---------|---------|---------|------|
| TWIST (Ze et al. 2025) | 中等 | - | 大规模并行仿真 |
| GMT (Chen et al. 2025) | 中等 | - | 大规模并行仿真 |
| **SONIC (Luo et al. 2025)** | **42M 参数** | **100M+ frames / 700+ 小时** | **9k GPU hours** |

**问题**：这些预训练好的 WBT 策略**与源机器人本体强耦合**——观察空间、动作空间、关节排列、reward 设计、domain randomization 都是为某一台具体机器人调出来的。换一台新的人形（即使形态相似）就用不了。

> 引用论文原文：*"Such policies are tightly coupled to the source robot's morphology, actuator configuration, and sensor interface, so even between similar humanoids the resulting structural and dynamical differences make direct deployment unreliable."*

而现有的"跨本体"方法主要走两条路：
- **多本体联合预训练**：morphology randomization、unified action space、expert routing（如 Octo、Open-X、GR00T N1）——需要海量多本体数据，从零训练贵到爆。
- **embodiment-aware 基座**：post-training/路由——同样需要先把多本体大模型搞出来。

LimX 的设定是 **post-training**：**已经有一台机器人的 WBT 专家了，怎么把它高效搬到下一台机器人**——这是任何二级人形厂商最现实的问题。

### 1.2 核心洞察：Kinematics-Dynamics 分离

论文提出一个看似简单但非常关键的观察：

> 源策略和目标策略的差异沿着**两个独立轴**展开：
> - **Kinematically (运动学差异)**：关节数、关节排列、连杆几何、observation/action layout 不同 → 是**结构性 / 表征性**问题
> - **Dynamically (动力学差异)**：质量分布、惯量、actuator response、接触行为 → 是**物理 / 参数性**问题

而现代 WBT 网络结构（TWIST、GMT、SONIC、CLOT、HoloMotion-1 等）**自然就是双流架构**：

```
现代 WBT 网络结构
├─ Reference Motion Encoder ─── 提取以运动学为主的特征 → 跨本体可复用
└─ Action Decoder ───────────── 与本体动力学紧耦合 → 需要适配
```

所以 LimX 的核心 hypothesis 是：

> **本体改变对 WBT 策略不同模块的影响是不均匀的**——全局运动技能（balance、肢体协调）可大量复用，仅本体敏感模块需要适配。

这直接催生了 Any2Any 的两步设计。

---

## 二、方法详解

### 2.1 整体架构

```
┌──────────────────────────────────────────────────────┐
│                     Any2Any 框架                      │
├──────────────────────────────────────────────────────┤
│                                                      │
│  目标机器人观测                                        │
│       │                                              │
│       ▼                                              │
│  ┌─────────────────────┐                             │
│  │ Kinematic Alignment │  Level 1: Layout 对齐        │
│  │   (运动学对齐)       │  Level 2: Joint-level 对齐  │
│  └─────────────────────┘     (S_r, D_r, J_r)         │
│       │ 源对齐后的统一表征                             │
│       ▼                                              │
│  ┌─────────────────────────────────────┐             │
│  │   Frozen Pretrained WBT Backbone   │             │
│  │   (源策略大部分参数不动)              │             │
│  │   ┌──────────────────────────────┐  │             │
│  │   │  + LoRA on selected modules  │  │ ← 仅 5.26%  │
│  │   │  (actor backbone, proprio,   │  │   参数      │
│  │   │   output, critic backbone)   │  │   可训练     │
│  │   └──────────────────────────────┘  │             │
│  └─────────────────────────────────────┘             │
│       │                                              │
│       ▼ 源对齐空间下的动作                             │
│  ┌─────────────────────┐                             │
│  │ 反向 Kinematic Map  │  Φ_r^+: 映回目标本体          │
│  └─────────────────────┘                             │
│       │                                              │
│       ▼ 目标机器人执行的动作                           │
└──────────────────────────────────────────────────────┘
```

### 2.2 问题形式化

把 WBT 建模为 MDP `M = (X, A, P, r, γ)`：

- 源机器人 `E_S` 上有预训练好的策略 `π_θS : X_S → A_S`（θ_S 是全部参数）
- 目标机器人 `E_T` 上 `X_S ≠ X_T`、`A_S ≠ A_T`（自由度、关节配置不同）
- **目标**：冻结 `θ_S`，学习一个小的目标特定调整 `Δθ_T`，满足 `|Δθ_T| ≪ |θ_S|`

最终的目标策略：
```
π_θT(a|s) = π_{θ_S ⊕ Δθ_T}(a|s)
```
其中 `⊕` 表示参数高效注入（LoRA / Adapter / Prefix Tuning 等）。

### 2.3 步骤一：Kinematic Alignment（两级运动学对齐）

#### Level 1: Observation Layout Alignment（观测布局对齐）

不同人形平台对 observation 向量的组织顺序不同，但语义可对应（base state、reference motion、proprio、action history）。这一级把目标机器人的观测 **重排成源策略期待的布局**。

#### Level 2: Joint-Level Kinematic Alignment（关节级运动学对齐）

这是论文的核心数学贡献之一。设源机器人 `r_0` 自由度 `T`，目标机器人 `r` 自由度 `N_r`，定义一对映射：

```
Φ_r  : ℝ^N_r → ℝ^T   (前向映射: target → source 空间)
Φ_r^+: ℝ^T → ℝ^N_r   (反向映射: source → target 空间)
满足 Φ_r^+ Φ_r = I_{N_r}
```

`Φ_r` 由三个结构化部件组成：

**(i) Sparse Scattering Matrix `S_r ∈ {0,1}^{T×N_r}`**

由关节对应关系 `π_r: {0..N_r-1} ↪ {0..T-1}` 诱导：

```
(S_r)_{ij} = 1[π_r(j) = i]
```

每列恰有一个 1，把目标关节值散布到源布局对应位置，无对应位置补零；目标多余关节（不在 π_r 域内）被丢弃。

**(ii) Hip Decoupling Matrix `D_r ∈ ℝ^{T×T}`**

源本体（如 Sonic 训练在 G1 上）有一个**倾斜的 hip-pitch 轴**，导致 hip 坐标耦合。`D_r` 初始化为单位阵，仅左右 hip 子矩阵替换为：

```
H_{L/R} = ⎡ cos α    0 ⎤
          ⎣ ∓sin α   1 ⎦
```

左右脚符号相反（镜像 hip 轴朝向）。这一步把 hip 坐标转换到源对齐约定。

**(iii) Parallel Joint Coupling 闭环修正 `J_r ∈ ℝ^{T×T}`**

某些目标机器人有闭环机构（平行四边形踝关节、闭环腰）——actuated joint 值和源策略期望的串联链 joint 值不一致。`J_r` 是闭环 Jacobian 修正。

**最终对齐映射**：
```
Φ_r   = J_r · D_r^{-1} · S_r
Φ_r^+ = S_r^T · D_r · J_r^{-1}
```

观测端（包括 reference motion、joint position observation、action history）：`q̃_r = Φ_r · q_r`
动作端：`a_r = Φ_r^+ · ã`（把源对齐空间的策略输出转换回目标 actuated joint 空间执行）

> 这一步**完全是 hand-crafted、非学习的**。它把"几何/拓扑差异"全部吸收到对齐模块里，让冻结的源策略始终在一个稳定的关节语义空间上工作。

### 2.4 步骤二：Dynamic Adaptation（动力学适配）

运动学对齐之后，剩下的 gap 是**动力学**层面的。论文从机器人动力学方程出发推导：

```
M(q)q̈ + C(q,q̇)q̇ + G(q) = τ + τ_ext
```

源机器人和目标机器人的 (M, C, G) 不同，加上 actuator 特性、关节摩擦、接触行为差异，导致同一目标动作 `(q, q̇, q̈)` 需要不同 generalized force。把所有本体特定物理量收进 `η_e`，则 cross-embodiment dynamics gap：

```
Δη = η_T − η_S
```

**关键论证**：对于拓扑相似、规模相近的人形机器人，`Δη` 在结构上是**低维的**——刚体部分 (ΔM, ΔC, ΔG) 完全由每连杆的惯性差参数化，规模随连杆数而非整个 backbone θ_S 增长。

→ 这个低维残差**自然适合用 LoRA 这种低秩补丁来吸收**。

**LoRA 公式**：对策略中每个被适配的线性投影 `W ∈ ℝ^{d_out × d_in}`：
```
W' = W + BA
A ∈ ℝ^{k × d_in},  B ∈ ℝ^{d_out × k}
k ≪ min(d_in, d_out)
```

仅 `{A, B}` 训练，`W` 与源共享冻结。秩 `k` 控制适配容量。

### 2.5 训练流程

- **算法**: PPO (Proximal Policy Optimization)
- **仿真平台**: Isaac Lab
- **关键设计**: 为了隔离 cross-embodiment adaptation 的效果，**保持 action space、observation、reward、PPO 超参、reference motion sampling、domain randomization 完全与源预训练一致**——只有 kinematic alignment 和 dynamic adaptation 是新加的。

这是非常 clean 的对照实验设计。

---

## 三、实验设置

### 3.1 Cross-Embodiment Benchmark（4 台人形 × 2 套源策略 × 5 个迁移对）

**参与的人形机器人**：

| 机器人 | 厂商 | 自由度 | 关键特征 |
|--------|------|--------|---------|
| **LimX Oli** | LimX Dynamics | **31 DoF** | LimX 全尺寸人形，Any2Any 自训源 |
| **LimX Luna** | LimX Dynamics | **27 DoF** | LimX 另一全尺寸人形（产品页未公开发布） |
| **Unitree G1** | 宇树 | (~29 DoF) | Sonic 原始预训练平台 |
| **Unitree H1** | 宇树 | - | Unitree 第一代全尺寸人形 |

**源策略 (Source WBT Backbones)**：

| 源策略 | 来源 | 训练数据 | Backbone |
|--------|------|---------|---------|
| **Oli-WBT** | LimX 自训（论文作者自己训的） | LimX Oli 上 500+ 小时 motion data | MLP + Transformer 两个版本 |
| **Sonic** | NVIDIA 开源 | Unitree G1 上 100M+ frames | Robot Motion Encoder + FSQ + dynamics decoder |

**5 个迁移实例**：

| 迁移对 | 源 | 目标 | 含义 |
|--------|----|----|------|
| OliWBT2G1 | Oli-WBT | Unitree G1 | 自训源 → 跨厂商 |
| OliWBT2H1 | Oli-WBT | Unitree H1 | 自训源 → 跨厂商 |
| OliWBT2Luna | Oli-WBT | LimX Luna | 自训源 → 同厂商不同型 |
| Sonic2Oli | Sonic | LimX Oli | 外部开源源 → 自家机器 |
| Sonic2Luna | Sonic | LimX Luna | 外部开源源 → 自家机器 |

**统一数据 pipeline**：所有迁移实验都用 **AMASS 数据集**（人体 motion capture），通过 **GMR (General Motion Retargeting)** 重定向到每个目标机器人。

**算力**：所有 adaptation 都用 **4×NVIDIA A100**（远低于 Sonic 原始预训练 9k GPU hours 的量级）。

### 3.2 Baselines

| Baseline | 描述 |
|----------|------|
| **Specialist (Training from Scratch)** | 目标本体上 PPO 从零训练，无 prior，标准人形 RL pipeline |
| **Full Fine-Tuning (w/ alignment)** | 用 kinematic alignment，但全参数微调 |
| **Full Fine-Tuning (w/o alignment)** | 直接全参数微调，不做对齐 |
| **LoRA / Adapter / Prefix-Tuning** | 不同 PEFT 方法对比 |

### 3.3 Evaluation Metrics

**训练阶段**：
- Tracking joint position reward / tracking body position reward
- 收敛速度 + 最终 reward 值

**部署阶段（MuJoCo sim-to-sim）**：
- Success rate
- MPJPE (Mean Per-Joint Position Error)
- Base position error / orientation error in world frame
- Action velocity / acceleration magnitude（衡量平滑度，间接指示 sim-to-real 可部署性）

---

## 四、关键实验结果

### 4.1 5 个迁移对全面胜出

> Q1: Can Any2Any successfully transfer diverse pretrained WBT policies to novel humanoid platforms?

**结论**: ✅ 是的，5/5 全部成功。

- **Sonic2Oli / Sonic2Luna**：LoRA 仅插入 actor dynamics decoder + critic 网络，FSQ 模块和其他预训练组件冻结。total reward 早期快速提升、收敛到比 from-scratch 更高的最终值，anchor-position tracking 和 relative-body-position tracking 都改善。
- **OliWBT2G1 / OliWBT2H1 / OliWBT2Luna**：radar plot 显示 joint-level / body-level metrics 大部分都低于 Specialist baseline；reward 收敛更快、最终 reward 更高或相当。
- **MuJoCo rollout**：在三个目标机器人上稳定执行 walking、running、manipulation、squatting、bending 等多种动作。

### 4.2 架构组件消融（Q2）

#### 4.2.1 Kinematic Alignment 是必需的

| 设置 | 收敛速度 | 最终 reward |
|------|---------|------------|
| Training from scratch | 慢 | 低 |
| Full FT (w/o alignment) | 中 | 受限 |
| **Full FT (w/ alignment)** | **快** | **高** |
| **Any2Any (LoRA + alignment)** | **快** | **接近 Full FT** |

> *"Full fine-tuning without alignment improves over scratch, but its convergence and final reward are still limited. This suggests that the pretrained WBT prior cannot be effectively reused when the target robot's observations and actions are interpreted under inconsistent joint semantics."*

#### 4.2.2 LoRA 是最佳 PEFT 方案

**OliWBT2Luna 上对比**（Transformer + MLP 双 backbone）：

| 方法 | Trainable Params | FPS | Collection Cost | Learning Cost | Joint Reward | Total Reward |
|------|-----------------:|----:|----------------:|--------------:|-------------:|-------------:|
| Full FT (w/ align) | **100.00%** | 34.8k | 21.10 | 8.82 | 1.64 | 22.67 |
| **Any2Any (LoRA)** | **5.26%** | **111.7k** | **7.51** | **7.60** | **1.63** | **22.58** |

**关键数字**：参数量降到 **5.26%**，FPS 提升 **3.2×**（34.8k → 111.7k），数据收集成本降到 **35.6%**（21.10 → 7.51），而 joint reward 和 total reward 几乎不变。

PEFT 方法对比：
- **LoRA**: 收敛最快、最稳定、最终 reward 最高 ✅
- **Adapter**: 收敛慢于 LoRA，最终 reward 低（bottleneck 模块在闭环 WBT 中难优化）
- **Prefix Tuning**: 最不稳定，reward 要么饱和在低值要么训练时退化（仅修改 input conditioning 不足以补偿 embodiment-level dynamics mismatch）

#### 4.2.3 LoRA 注入位置消融（最关键发现之一）

论文测试了 9 种 LoRA 注入组合（S1-S9），覆盖：
- Actor: backbone, reference input projection, proprio input projection, action output projection
- Critic: backbone, input/output projection

**最佳组合 S7**：
```
LoRA 插入位置 = {
    actor backbone,
    actor proprioception input projection,
    actor output projection,
    critic backbone
}
完全冻结 = {
    actor reference input projection (!)
    其他模块
}
```

**核心发现**：
> *"The dominant residual mismatch after kinematic alignment lies in the dynamics-aware control pathway: the policy needs to reinterpret the target robot's proprioceptive state and recent action history, and also adjust how the shared motion representation is decoded into target-specific joint commands."*

也就是说：
- **Reference motion stream → 跨本体 agnostic** → 保持冻结
- **Proprioception stream / action output head / critic backbone → 跟目标机器人质量分布、actuator response、接触行为强耦合** → 需要适配

这与论文一开始的 hypothesis 完全吻合，并且**给后续做 cross-embodiment WBT 迁移的人提供了非常具体的工程指引**。

### 4.3 数据 & 算力效率（Q3）

#### 4.3.1 数据规模消融（OliWBT2Luna）

| Frames | Specialist Base Pos. (cm) | **Any2Any Base Pos. (cm)** | 改善 |
|-------:|--------------------------:|---------------------------:|----:|
| 15M | 42.1 | **30.5** | -27.6% |
| 7.5M | 60.3 | **44.1** | -26.9% |
| 1.5M | 71.5 | **38.2** | -46.6% |
| 0.4M | 79.5 | **51.9** | -34.7% |
| **0.04M** | **100.5** | **58.4** | **-41.9%** |

**核心信号**：**数据越少，Any2Any 优势越明显**——0.04M frames 时 Specialist 几乎崩了（100.5cm 误差），Any2Any 还能撑住 58.4cm。

> *"The pretrained WBT prior provides reusable balance and coordination knowledge, allowing the target policy to learn embodiment-specific corrections from much less data instead of relearning whole-body control from scratch."*

#### 4.3.2 算力规模消融（不同 GPU 设置 → 不同 sampling rate）

| GPU 设置 | Sampling Rate |
|---------|---------------|
| 1× RTX 4090 | 16K |
| 2× RTX 4090 | 32K |
| 4× RTX 4090 | 64K |
| 4× A100 | 80K |

**Sampling rate 从 80K 降到 16K 时**：
- Specialist：joint reward 掉 **18.6%**，episode reward 掉 **16.4%**
- **Any2Any：joint reward 仅掉 10.5%，episode reward 仅掉 8.9%**

**结论**：Specialist 高度依赖大规模并行采样，Any2Any 对算力鲁棒得多——预训练先验缩小了优化搜索空间，目标机器人只需学**残差**而不是从零发现 WBT。

### 4.4 论文的核心数字总结

| 维度 | Any2Any vs from-scratch |
|------|------------------------|
| 算力 | **~1%** |
| 数据 | **~1%** |
| 可训练参数 | **5.26%** |
| FPS（训练吞吐） | **3.2×** |
| 数据收集成本 | **0.36×** |
| 0.04M frames 下 base pos error | **0.58×**（58.4 vs 100.5cm） |
| 总 reward | **基本持平**（22.58 vs 22.67） |

---

## 五、开源情况分析

### 5.1 Any2Any 论文本身

| 维度 | 状态 | 说明 |
|------|------|------|
| 论文 PDF/HTML | ✅ 公开 | arxiv 2605.23733 |
| **代码仓库** | ❌ **未发现公开仓库** | 截至 2026-05-26 |
| **项目主页** | ❌ **未发现** | 论文内未给出 project page URL |
| **预训练权重** | ❌ **未公开** | Oli-WBT 是 LimX 自训的内部资产 |
| **训练数据** | 部分 | AMASS 是公开的，但 retargeted 后版本未公开 |
| **HuggingFace** | ❌ 未发现相关条目 | |
| **LimX Dynamics 官方 GitHub** | ✅ 存在 | 但只有 tron1/humanoid-rl 部署工具，无 Any2Any 相关条目 |

**搜索过的渠道**：
- arXiv 论文全文（无任何 GitHub/website URL，仅引用 LimX 产品页和 Unitree 产品页）
- GitHub `limxdynamics` 组织所有 25 个仓库（最相关的是 `tron1-rl-isaaclab`、`humanoid-rl-deploy-*`，均为部署工具，无 Any2Any 相关）
- GitHub repo search: `any2any`, `any2any+humanoid+cross-embodiment` → 0 results
- Tao Yu (Project Lead) 个人主页 (tao-yu.github.io) → 未公开 Any2Any
- Web search: `Any2Any LimX Dynamics github` → 无结果

**判断**：论文是 **2026-05-22 刚提交的全新 work（4 天前）**，目前应该还在等待开源准备阶段，或者 LimX 决定保留为商业核心资产。

### 5.2 论文使用的外部资产（间接开源情况）

| 资产 | 状态 | 链接 |
|------|------|------|
| **Sonic 源策略** (Luo et al. 2025) | ✅ 完全开源 | [NVlabs/GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl), HF: `nvidia/GEAR-SONIC` |
| **AMASS 动作数据集** | ✅ 公开 | https://amass.is.tue.mpg.de |
| **GMR (General Motion Retargeting)** | ✅ 开源 | https://github.com/YanjieZe/GMR |
| **Isaac Lab** | ✅ 开源 | NVIDIA Omniverse |
| **LimX Oli URDF / robot description** | ✅ 公开 | https://github.com/limxdynamics/robot-description |
| **LimX Oli SDK** | ✅ 公开 | https://github.com/limxdynamics/limxsdk-lowlevel |
| **Unitree G1 / H1 URDF** | ✅ 公开 | Unitree GitHub |

**重要意义**：即使 Any2Any 本体不开源，**复现路线是清晰的**——
1. 拿 Sonic 当源策略（NVIDIA 已开源）
2. 拿 AMASS + GMR 做 retargeting
3. 自己写 Φ_r = J_r D_r^-1 S_r 这套对齐矩阵（论文写得很清楚）
4. 在 Isaac Lab 里用 PEFT (Hugging Face PEFT 库或 PyTorch LoRA) 套上去
5. PPO 训练

社区如果决心复现，**理论上 1-2 个月内能跑出 Sonic2Oli / Sonic2G1 这条线**（前提是有 G1 或 Oli 的 sim 环境）。

### 5.3 开源程度评分

```
开源程度评分: ★★☆☆☆ (2/5)

✅ 可获得:
  - 论文完整方法描述（数学公式都列出来了）
  - 源策略 Sonic 已开源
  - 所需的所有外部数据集和工具

❌ 不可获得:
  - Any2Any 实现代码
  - Oli-WBT 自训权重（论文核心实验之一的源）
  - Hyperparameter 配置（PPO 超参与源预训练一致，但具体数值未给）
  - retargeted AMASS 数据
  - LoRA rank k、Φ_r 中 hip 倾角 α 等具体数值
```

---

## 六、技术深挖与思考

### 6.1 为什么这套方法 work？三个层次的解释

**(a) 表征层面**：
- Reference Motion 本质是"运动学描述"（关节位置、速度），跨本体语义一致
- Proprioception 包含 "我现在的状态"——直接受动力学影响
- 论文用对齐矩阵把语义结构对齐 → 让源策略以为自己还在源机器人上工作

**(b) 优化层面**：
- 从零训练需要在巨大参数空间里搜（θ_S 整个 10M+ 参数）
- 有先验后只需在 LoRA 低秩子空间搜（5.26% × rank）
- 预训练先验提供了"良好的初始化分布"，避免 RL 中常见的 mode collapse

**(c) 物理层面**：
- 同拓扑、同 scale 的人形之间，dynamics gap (ΔM, ΔC, ΔG) 维度由连杆数决定
- 与策略网络规模无关（10M 参数 vs 几百维物理参数）
- 所以用低秩补丁 (LoRA k 维) 吸收物理残差是 **scale-matched** 的

### 6.2 与 Sonic 论文的关系

Sonic（同样在 2025 年出来）的 motto 是 "supersizing"——把 WBT 当 foundation model，往规模上推（1.2M → 42M 参数，100M+ frames，9k GPU hours）。

Any2Any 是 Sonic 的**对偶论文**：Sonic 解决"从零训练大模型"，Any2Any 解决"已有大模型怎么用到下一台"。

> 你可以理解为：Sonic 是 GPT-3，Any2Any 是 LoRA-for-WBT。

### 6.3 与本对话之前讨论的 LeRobot / Pi0.5 / LingBot-VA 的关系

| 维度 | LeRobot | Pi0.5 / LingBot-VA / GigaWorld | **Any2Any** |
|------|---------|------------------------------|-------------|
| **任务层** | 操作 (manipulation) | 操作 (manipulation) | **全身控制 (locomotion + WBT)** |
| **算法层** | ACT/Diffusion/VLA | VLA / WAM | **PPO + LoRA** (RL post-training) |
| **学习范式** | 模仿学习为主 | 模仿 + 部分 RL | **强化学习** |
| **关注点** | 数据/平台/部署生态 | 模型架构 | **本体迁移效率** |
| **代码开源** | ★★★★★ | ★★★☆☆ ~ ★★★★☆ | **★★☆☆☆** |

Any2Any 跟你前面研究的 VLA/WAM 路线是**互补**的——VLA/WAM 解决"上半身做什么"，Any2Any 解决"下半身怎么稳/全身怎么协调"。在真实人形机器人系统里这两类策略往往是**层级组合**的（高层 VLA 输出 reference motion → 低层 WBT 跟踪）。

### 6.4 局限性

1. **仍然依赖 sim-to-sim 评估**：论文实验主要在 Isaac Lab（训练）+ MuJoCo（评估）做，**没有大规模 real-world deployment 数据**——只在 conclusion 说"deploy the resulting policies on real hardware across multiple downstream tasks"，但 metrics 没在 real robot 上测。
2. **需要源-目标 topology 相似**：论文明确假设 "humanoids of similar topology and scale"——双足人形之间还行，但人形 → 四足、人形 → 双臂工厂机器人就不 work。
3. **Kinematic alignment 是 hand-crafted**：α、π_r 这些都需要人工配，对每对新机器人都要做一次。
4. **没解决 reward / domain randomization 适配**：论文刻意让目标 reward 和 source 一致，但实战中目标机器人可能需要不同的 reward shaping。
5. **闭源**：复现要从论文公式重新写起。

### 6.5 论文的工程价值

如果你是某个二级人形厂商的算法 lead，Any2Any 给了你一个非常实用的 playbook：

```
Step 1: 拿 NVIDIA 开源的 Sonic 当源策略 (G1 上预训练好)
Step 2: 写一份你自己机器人的 URDF + AMASS retargeting
Step 3: 实现 Φ_r = J_r D_r^-1 S_r （主要工程量在 hip 倾角和闭环 Jacobian）
Step 4: 在 actor backbone + actor proprio in/out + critic backbone 插 LoRA
Step 5: 在 4×A100 上跑 PPO（用与 Sonic 相同的超参）
Step 6: ~1% 数据/算力下你应该能拿到一个能跑的 WBT 策略
```

这是 **CTO/技术总监级别的工程优化**，不是研究突破——但对国内人形产业（一堆腰部公司想做 WBT 又没 NVIDIA 量级资源）是真正的礼物。

---

## 七、对具身智能 / 世界模型路线的启示

### 7.1 PEFT 的具身落地

LoRA 在 NLP / 视觉领域早已成熟，但 **闭环控制（closed-loop control）** 上一直缺好的范式。Any2Any 是少有的系统性证明：

- LoRA 在 PPO closed-loop control 上**稳定可用**（Adapter / Prefix 都不行）
- 在哪里插 LoRA **极大影响**性能（不是简单全部插）
- 物理直觉（kinematics vs dynamics）能直接指导 PEFT 设计

这对未来所有的"VLA/WAM 微调到具体机器人"任务都有启发——**先做 representation alignment，再做 dynamics adaptation**，不要全参微调。

### 7.2 WBT 作为 Behavior Foundation Model

论文实际上验证了一个更大的命题：

> **预训练好的 WBT 策略中，相当一部分 prior 是 embodiment-agnostic 的**——尤其是 reference motion encoder。

这意味着以后 WBT 可以走 NLP/CV 路线：先有几个 "Foundation WBT"（Sonic 是 NVIDIA 的，未来可能有 Pi 的、Google 的），下游每家厂用 LoRA 适配自己机器人。

### 7.3 与 GR00T N1.5 / Pi0.5 / SmolVLA 的合流

注意 Any2Any 用的源策略 Sonic 是 NVIDIA **GR00T-WholeBodyControl** 的一部分——同一仓库里包含 GR00T N1.5 / N1.6 的 WBC 模块。这暗示着：

```
未来的人形栈可能长这样：
  上层 (高级语义、操作): Pi0.5 / GR00T N1.6 / SmolVLA
  下层 (全身控制、运动跟踪): Sonic (源) → Any2Any (跨本体适配)
```

LimX 自己也在做这条线（看 tao-yu.github.io 也有相关 publication），LingBot-VA / GigaWorld 是国产对应栈。

---

## 八、关键参考链接

| 资源 | 链接 |
|------|------|
| **Any2Any 论文 (arxiv)** | https://arxiv.org/abs/2605.23733 |
| **Any2Any HTML 版本** | https://arxiv.org/html/2605.23733v1 |
| **Any2Any PDF** | https://arxiv.org/pdf/2605.23733 |
| **LimX Dynamics 官网** | https://www.limxdynamics.com/en/ |
| **LimX Oli 产品页** | https://www.limxdynamics.com/en/products/oli |
| **LimX Dynamics GitHub** | https://github.com/limxdynamics |
| **LimX 机器人描述文件** | https://github.com/limxdynamics/robot-description |
| **LimX 低层 SDK** | https://github.com/limxdynamics/limxsdk-lowlevel |
| **LimX Tron1 Isaac Lab 模板** | https://github.com/limxdynamics/tron1-rl-isaaclab |
| **Tao Yu 个人主页** | https://tao-yu.github.io/ |
| **Sonic 论文 (源策略)** | https://arxiv.org/abs/2511.07820 |
| **Sonic 项目主页** | https://nvlabs.github.io/GEAR-SONIC |
| **GR00T-WholeBodyControl 代码** | https://github.com/NVlabs/GR00T-WholeBodyControl |
| **GEAR-SONIC HuggingFace** | https://huggingface.co/nvidia/GEAR-SONIC |
| **GMR (Motion Retargeting)** | https://github.com/YanjieZe/GMR |
| **GMR 论文** | https://arxiv.org/abs/2510.02252 |
| **AMASS 数据集** | https://amass.is.tue.mpg.de |
| **Unitree G1 产品页** | https://www.unitree.com/g1 |
| **Unitree H1 产品页** | https://www.unitree.com/h1 |

---

## 九、复现路线建议（如果你想动手做）

```
Phase 0: 准备 (1-2 周)
  - 跑通 Isaac Lab 基础环境
  - 跑通 NVlabs/GR00T-WholeBodyControl 的 Sonic 推理
  - 下载 AMASS + 跑通 GMR retargeting 到 G1/H1

Phase 1: 基线复现 (2-3 周)
  - 在 G1 上验证 Sonic 推理可用
  - 在某个目标机器人（如 H1 或自有平台）上跑 Specialist baseline (PPO from scratch)

Phase 2: Kinematic Alignment 实现 (1-2 周)
  - 写出 π_r (joint correspondence map)
  - 实现 S_r (sparse scattering matrix)
  - 测量并实现 D_r (hip decoupling，需要源/目标 hip 几何参数)
  - 如果目标有闭环关节，实现 J_r

Phase 3: LoRA 注入 (1 周)
  - 用 PEFT 库或自己写 LoRA wrapper
  - 按 S7 配置注入
  - 调整 rank k (论文未给具体数值，建议从 8 / 16 起)

Phase 4: 训练与评估 (2-3 周)
  - 4×A100 / 4×4090 上 PPO 训练
  - sim-to-sim 在 MuJoCo 评估 MPJPE / base pos error
  - 与 Specialist 对比

总计: ~2-3 个月，1-2 人
```

---

## 十、与同类 cross-embodiment 工作的对比

| 工作 | 路线 | 数据需求 | 算力需求 | 跨本体粒度 | 开源 |
|------|------|---------|---------|-----------|------|
| **Any2Any** | **post-training PEFT** | **极低 (~1%)** | **极低 (~1%)** | 同拓扑人形 | ❌ |
| Octo (UC Berkeley) | 多本体联合训练 | 极高 | 极高 | 跨多种机器人 | ✅ |
| Open-X / RT-X | 大规模多本体预训练 | 极高 | 极高 | 任意机器人 | ✅ |
| GR00T N1 | embodiment-aware backbone | 高 | 高 | 多人形 | ✅ |
| H-Zero (清华等) | cross-humanoid pretraining | 中 | 中 | 人形 | - |
| LocoFormer (CMU) | long-context adaptation | 中 | 中 | locomotion | - |
| MOSAIC (浙大等) | residual adaptation (sim2real) | 中 | 中 | 人形 | - |

**Any2Any 的独特定位**：它是**唯一一个把"已经预训练好的 WBT 专家"作为输入、追求最小 adaptation 成本**的工作。其他 cross-embodiment 方法都是从更大规模的预训练入手，资源门槛高得多。

---

## 十一、一句话总结

**Any2Any 是 LoRA 在人形 WBT 跨本体迁移上的标准答案——把一个已经训练好的 WBT 策略以 1% 数据 + 1% 算力 + 5.26% 参数搬到新机器人，关键在"对齐运动学 + 适配动力学"两步走，且代码暂未开源但方法已完整公开。**

---

## 附录：论文中关键数学公式速查

```
1. 目标策略形式
   π_θT(a|s) = π_{θ_S ⊕ Δθ_T}(a|s)         |Δθ_T| ≪ |θ_S|

2. 运动学对齐 (Joint-level)
   Φ_r   = J_r · D_r^{-1} · S_r              ℝ^{N_r} → ℝ^T
   Φ_r^+ = S_r^T · D_r · J_r^{-1}            ℝ^T → ℝ^{N_r}
   Φ_r^+ Φ_r = I_{N_r}                       (左逆性质)

3. Hip Decoupling Block
   H_{L/R} = ⎡ cos α    0 ⎤
             ⎣ ∓sin α   1 ⎦                 (左右镜像)

4. 动力学 gap 推导
   Δτ(q,q̇,q̈) = ΔM(q)q̈ + ΔC(q,q̇)q̇ + ΔG(q)
   Δη = η_T − η_S                            (本体物理参数差)

5. LoRA 适配
   W' = W + BA
   A ∈ ℝ^{k × d_in},  B ∈ ℝ^{d_out × k}
   k ≪ min(d_in, d_out)

6. 训练目标
   max_{Δθ_T} 𝔼_{π_θT} [Σ γ^t r(s_t, a_t)]   (PPO 标准 expected return)
```

---

> 报告生成日期: 2026-05-26
> 论文版本: arXiv 2605.23733v1 (2026-05-22 提交)
> ⚠️ Any2Any 是非常新的工作，**未来如果作者团队在 GitHub / 项目主页公开代码，建议把开源情况一节更新**。
