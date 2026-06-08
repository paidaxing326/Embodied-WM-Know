# DreamTacVLA: 触觉世界模型驱动的视觉-语言-动作框架深度解读

> **论文**: arXiv:2512.23864v3 (2025/2026)
> **机构**: Northwestern University Department of Computer Science
> **代码**: https://github.com/michaelyeah7/learning-to-feel-the-future
> **数据集**: HuggingFace `michaelyeah7/dobot_xtrainer_tac_1107`
> **许可**: MIT License（V-JEPA 权重为 CC BY-NC 4.0）

---

## 一、核心问题与研究动机

### 1.1 VLA 的"接触盲区"

Vision-Language-Action (VLA) 模型通过将互联网规模的知识映射到机器人控制，展示了强大的泛化能力。从 RT-1/RT-2 到 π₀/π₀.5，VLA 已成为通用机器人控制的主流范式。然而，**所有这些模型本质上都是"接触盲"的**——它们无法感知物理接触。

在接触密集的操控任务中，这一缺陷尤为致命：

| 场景 | 视觉的局限 | 触觉的不可替代性 |
|------|-----------|----------------|
| USB 插入 | 亚毫米容差，深度精度不足 | 直接感知接触力与对齐状态 |
| 齿轮装配 | 视角遮挡，无法判断是否对准 | 感知插入阻力和微滑移 |
| 工具稳定 | 无法判断接触力分布 | 实时监测力的平衡状态 |
| 钥匙操作 | 遮挡严重 | 感知旋转力矩和卡扣状态 |

### 1.2 现有触觉方法的不足

已有工作尝试将触觉引入机器人策略，但存在关键局限：

1. **低维力/力矩信号**：如六维力传感器，信息稀疏且模糊，无法回答"接触发生在哪里""接触面是什么形状"
2. **简单拼接**：直接将触觉信号拼接到 VLA 输入中，模型倾向于忽略触觉信息（因为视觉-语言骨干网络从未在触觉数据上预训练）
3. **缺乏时序建模**：无法预测接触状态的演化，只能被动响应

### 1.3 DreamTacVLA 的核心洞察

> **让机器人学会"感受未来"——通过预测候选动作的触觉后果，来精炼动作决策。**

这一洞察基于两个关键观察：
1. **触觉图像结构更简单**：相比全 RGB 观察，触觉图像结构更受约束、动态更可预测，适合作为世界模型的预测目标
2. **预测驱动使用**：如果模型需要预测触觉未来，它就**必须**学会使用触觉信息，而不能像简单拼接那样忽略它

---

## 二、方法架构全景

### 2.1 总体架构

DreamTacVLA 由三个核心模块组成：

```
┌─────────────────────────────────────────────────────────────────┐
│                      DreamTacVLA 架构                            │
│                                                                  │
│  ┌─────────────────────┐                                         │
│  │  多模态编码器 E_ψ     │  ← CLIP ViT (RGB) + V-JEPA2 (触觉)     │
│  │  Macro / Local / Micro│                                        │
│  └──────────┬──────────┘                                         │
│             │ H(t)_align                                         │
│             ▼                                                    │
│  ┌─────────────────────┐     ┌──────────────────┐               │
│  │  策略网络 π_θ         │────▶│ 触觉世界模型 W_φ  │               │
│  │  (Action Expert)     │     │ (V-JEPA2 冻结)   │               │
│  │                      │◀────│ + 残差适配器      │               │
│  └─────────────────────┘     └──────────────────┘               │
│    a_draft    a_final           H_dream                          │
│                                                                  │
│  ┌─────────────────────┐                                        │
│  │  预测 MLP F_η        │  ← 轻量级未来状态预测器                   │
│  └─────────────────────┘                                        │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 三级感知层次

DreamTacVLA 建立了**三级视觉层次**，将不同尺度的感知融合到统一潜在空间：

```
┌─────────────────────────────────────────────────┐
│              三级视觉层次                          │
├─────────────┬──────────────┬────────────────────┤
│   Macro     │    Local     │     Micro          │
│  第三人称视觉 │  腕部相机     │   触觉传感器        │
│             │              │                    │
│ · 全局任务   │ · 末端执行器  │ · 指尖接触细节      │
│   上下文     │   视觉引导    │ · 滑移/插入力       │
│ · 场景布局   │ · 近距离观测  │ · 纹理/形变        │
│ · 物体定位   │ · 粗对齐     │ · 精密力控制        │
├─────────────┼──────────────┼────────────────────┤
│ CLIP ViT    │ CLIP ViT     │ V-JEPA2 ViT-L/G   │
│ (冻结)      │ (冻结)       │ (冻结+适配器)       │
└─────────────┴──────────────┴────────────────────┘
```

**关键设计**：触觉图像被视为**"微型视觉"（Micro-vision）**，而非完全独立的模态。这一设计哲学使得触觉信息能以视觉编码器可理解的形式融入多模态表征。

---

## 三、核心技术一：分层空间对齐（HSA）

### 3.1 动机

视觉和触觉信号在**形式和语义上**存在根本差异：
- 视觉：大视角、彩色 RGB、场景级信息
- 触觉：小视角、凝胶变形图、接触级信息

如果不在空间维度上建立显式对应，模型很难学会"我看到的地方就是我摸到的地方"。

### 3.2 HSA 损失的数学形式

#### 步骤 1：触觉传感器投影

利用机器人正运动学和标定的相机参数，将触觉传感器的 3D 位姿投影到两个相机视图中：

$$P^{(t)}_{sensor} \in SE(3) \xrightarrow{\text{FK+标定}} B^{(t)}_w, B^{(t)}_{tp}$$

其中 $B^{(t)}_w$ 是腕部视图中的 2D 边界框，$B^{(t)}_{tp}$ 是第三人称视图中的 2D 边界框。

#### 步骤 2：特征提取与池化

从 LLM 中间层提取特征 token，计算三个平均池化向量：

| 向量 | 定义 | 含义 |
|------|------|------|
| $h_\tau$ | 触觉 token $Z^{(t)}_\tau$ 的平均池化 | "机器人感觉到了什么" |
| $h_w$ | 腕部图中落入 $B^{(t)}_w$ 的 token 平均池化 | "在腕部视角中对应位置看到了什么" |
| $h_{tp}$ | 第三人称图中落入 $B^{(t)}_{tp}$ 的 token 平均池化 | "在全局视角中对应位置看到了什么" |

#### 步骤 3：InfoNCE 对比损失

**触觉-腕部对齐**：

$$L_{HSA\text{-}W} = -\log \frac{\exp(h_\tau \cdot h_w / \kappa)}{\exp(h_\tau \cdot h_w / \kappa) + \sum_{i=1}^{N_k} \exp(h_\tau \cdot h^{neg}_{w,i} / \kappa)}$$

**触觉-第三人称对齐**：类似形式 $L_{HSA\text{-}TP}$

**总 HSA 损失**：

$$L_{HSA} = L_{HSA\text{-}W} + L_{HSA\text{-}TP}$$

### 3.3 HSA 的工程细节

**边界框到 Patch Token 的映射**：

CLIP ViT-L/14 在 224×224 输入上的 patch grid 为 16×16。将像素级边界框 $[x_0, x_1] \times [y_0, y_1]$ 映射到 patch 索引：

$$i \in \left[\lfloor\frac{x_0}{14}\rfloor, \lfloor\frac{x_1}{14}\rfloor\right], \quad j \in \left[\lfloor\frac{y_0}{14}\rfloor, \lfloor\frac{y_1}{14}\rfloor\right]$$

**负样本策略**：
- **图像内负样本**：边界框外的 patch（硬负样本）
- **批次内负样本**：其他样本的 patch（多样性负样本）

**温度参数**：$\kappa = 0.07$，固定不变。

**可见性感知加权**：当触觉传感器在第三人称视图中被遮挡时：

$$L_{HSA} = L_{HSA\text{-}W} + \alpha(t) \cdot L_{HSA\text{-}TP}$$

其中 $\alpha(t) \in [0, 1]$ 根据投影有效性动态调整。

### 3.4 HSA 的关键意义

HSA 不仅是一个损失函数，更是一种**空间锚定机制**。它强制模型建立"我看到的位置 = 我摸到的位置"这一对应关系，这是后续触觉世界模型能正确工作的前提。

消融实验证明：
- **去除 HSA**（仅保留 Dream）：性能从 84.1% 降至 71.0%，说明空间接地无法被隐式学习
- **仅保留 HSA**（无 Dream）：性能降至 54.7%，说明对齐了空间但没有时序预测能力

---

## 四、核心技术二：触觉世界模型

### 4.1 设计哲学

传统世界模型预测完整 RGB 观察，计算昂贵且不稳定。DreamTacVLA 转而**只预测触觉图像的未来状态**，因为：

1. 触觉图像结构更简单、更受约束
2. 触觉动态更有信息量（直接反映接触物理）
3. 计算开销更低

### 4.2 V-JEPA2 作为触觉特征提取器

世界模型基于 **V-JEPA2 ViT-L/ViT-G**（Meta 发布的自监督视频模型）：

**预训练方式**：教师-学生架构
- **教师编码器**：接收完整触觉序列，生成目标嵌入（EMA 更新）
- **学生预测器**：接收时序遮蔽的触觉序列，预测被遮蔽 patch 的潜在表示

**预测损失**：

$$L_W = \sum_{t \in \mathcal{M}} \|z^{teacher}(t) - \hat{z}^{student}(t)\|_2^2$$

仅在潜空间计算，不要求像素级重建。

### 4.3 残差适配器架构

冻结的 V-JEPA2 输出 1024 维 patch 嵌入，通过轻量级适配器进行任务特定适应：

$$z' = z + \alpha \cdot \text{MLP}(\text{LayerNorm}(z))$$

| 参数 | 值 |
|------|---|
| MLP 层数 | 3 |
| 隐藏维度 | 512 |
| 激活函数 | GELU |
| Dropout | 0.1 |
| 残差缩放 α 初始化 | 0.1 |
| 聚合方式 | 学习式注意力池化（8头） |
| 额外可训练参数 | **5.5M**（仅为冻结 ViT-L 的 **1.8%**） |

适配器处理所有 196 个 patch token（非仅 CLS token），通过学习式查询 token 的注意力池化聚合成单一向量。

### 4.4 未来状态预测（Forecasting MLP）

轻量级预测网络 $F_\eta$ 接受两个输入——当前触觉嵌入和候选动作——预测未来触觉状态：

$$H^{(t+N)}_{dream} = F_\eta(z^{(t)}_\tau, a^{(t)}_{draft})$$

其中 $N$ 是预测的时间范围。训练时，预测目标为冻结 V-JEPA2 编码器对未来真实触觉帧的嵌入：

$$L_W = \|\hat{z}^{(t+N)}_\tau - z^{(t+N)}_\tau\|_2^2$$

**关键设计**：
- 世界模型 $W_\phi$ 始终**冻结**，仅提供稳定的特征提取
- 预测 MLP $F_\eta$ 从头训练
- 不需要显式的奖励函数或 MPC 规划器

---

## 五、核心技术三：Think–Dream–Act 推理管线

### 5.1 推理流程

每个策略步骤分三个阶段执行：

```
┌───────────────────────────────────────────────────────┐
│              Think – Dream – Act 推理管线               │
│                                                        │
│  ┌───────┐       ┌──────────┐       ┌──────────┐     │
│  │ THINK │──────▶│  DREAM   │──────▶│   ACT    │     │
│  │       │       │          │       │          │     │
│  │当前状态│       │触觉世界   │       │精炼后的   │     │
│  │+空预测 │       │模型预测   │       │最终动作   │     │
│  │       │       │触觉未来   │       │          │     │
│  │a_draft│       │H_dream   │       │ a_final  │     │
│  └───────┘       └──────────┘       └──────────┘     │
│                                                        │
│  输入: H_align(t) + H_null   输入: z_τ(t) + a_draft    │
│                                输出: H_dream(t+N)       │
│  输入: H_align(t) + H_dream                           │
│  输出: a_final                                        │
└───────────────────────────────────────────────────────┘
```

**THINK**：策略 π_θ 基于当前对齐状态和空预测（zero tensor）生成候选动作 $a^{(t)}_{draft}$

**DREAM**：预测 MLP 利用当前触觉嵌入和候选动作预测未来触觉状态 $H^{(t+N)}_{dream}$

**ACT**：策略将预测的未来触觉状态与当前状态结合，输出精炼后的最终动作 $a^{(t)}_{final}$

### 5.2 与传统世界模型的对比

| 维度 | 传统世界模型 | DreamTacVLA |
|------|------------|-------------|
| 预测目标 | 完整 RGB 观察 | 触觉潜嵌入 |
| 规划方式 | MPC + 奖励函数 | 无需显式规划器 |
| 动作生成 | 搜索/优化 | 端到端策略网络 |
| 计算开销 | 高（多次前向传播） | 低（两次策略前向） |
| 部署复杂度 | 需要奖励设计 | 开箱即用 |

---

## 六、两阶段训练流程

### 6.1 Stage 1：编码器与基础策略预训练

**目标**：训练多模态编码器理解触觉传感器在视觉世界中的位置，并建立基础策略。

**损失函数**：

$$L_{Stage1} = L_{action} + \lambda_{HSA} \cdot L_{HSA}$$

**动作损失**（行为克隆）：

$$L_{action} = \frac{1}{H}\sum_{j=0}^{H-1} \|\hat{a}^{(t)}_j - a^{(t)}_j\|_1$$

- H = 45 步动作预测范围
- ℓ₁ 损失（对异常值更鲁棒）

**关键细节**：此阶段没有训练好的世界模型，用零张量替代 $H^{(t+N)}_{dream}$。

### 6.2 Stage 2：世界模型精调

**目标**：激活触觉世界模型，使策略学会基于预测的触觉未来来精炼动作。

**损失函数**：

$$L_{Stage2} = \lambda_{action} \cdot L_{action} + \lambda_{HSA} \cdot L_{HSA} + \lambda_W \cdot L_W$$

新增的潜空间预测损失 $L_W$ 驱动世界模型学习。

**训练配置**：

| 组件 | 状态 |
|------|------|
| $E_\psi$（编码器） | 微调 |
| $\pi_\theta$（策略） | 微调 |
| $F_\eta$（预测 MLP） | 从头训练 |
| $W_\phi$（世界模型） | **冻结** |

### 6.3 完整训练算法（Algorithm 1）

```
Require: 数据集 D₁, D₂; 模块 E_ψ, π_θ, W_φ, F_η
Require: 超参 S₁, S₂, λ_HSA, λ_W, λ_action

Initialize E_ψ, π_θ
Set H_null = zeros

// Stage 1: HSA 对齐 + 基础策略
for s ← 1 to S₁ do
    Sample batch from D₁
    H_align ← E_ψ(language, images, tactile, state)
    L_HSA ← HSA_InfoNCE(H_align, bounding_boxes)
    â_draft ← π_θ(H_align, H_null)
    L_action ← L_diffusion(â, a)
    Backprop L_action + λ_HSA · L_HSA
    Update ψ, θ
end

// Stage 2: 加入世界模型精调
Load and freeze W_φ
Initialize F_η (trainable)
for s ← 1 to S₂ do
    Sample batch from D₂
    H_align ← E_ψ(...)
    // THINK
    a_draft ← stopgrad(π_θ(H_align, H_null))
    // DREAM
    z_τ ← W_φ(I_τ)                           // 冻结编码器
    ẑ_τ(t+N) ← F_η(z_τ, a_draft)             // 预测未来
    z_τ(t+N) ← stopgrad(W_φ(I_τ(t+N)))       // 目标
    L_W ← ‖ẑ_τ(t+N) - z_τ(t+N)‖₂²           // 预测损失
    // ACT
    a_final ← π_θ(H_align, ẑ_τ(t+N))
    L_action ← L_diffusion(a_final, a)
    Backprop λ_action·L_action + λ_HSA·L_HSA + λ_W·L_W
    Update ψ, θ, η  // φ 保持冻结
end
```

**关键设计决策**：
- `a_draft` 使用 `stopgrad`——候选动作不接收梯度，避免策略和世界模型之间的梯度冲突
- 目标嵌入 `z_τ(t+N)` 也使用 `stopgrad`——防止梯度回流到冻结的世界模型

---

## 七、仿真管线与数据集

### 7.1 IsaacSim 仿真环境

| 配置项 | 值 |
|--------|---|
| 物理时间步 | 1/60s |
| 求解器子步数 | 2 |
| 控制频率 | 30 Hz |
| 最大 Episode 长度 | 300 步 |
| 并行环境 | 1024 / GPU (24GB) |
| 传感器 | TPV RGB + Wrist RGB + 双触觉 |

### 7.2 自动化专家演示（cuRobo）

专家演示**不是**通过人类遥操作采集，而是使用 cuRobo 自动生成：

| 方面 | 规格 |
|------|------|
| 规划器 | cuRobo (批量化) |
| 批量大小 | 256 查询 |
| 规划范围 | 32 路径点 (minimum-jerk) |
| 碰撞检测 | 机器人 + 物体 + 桌面 |
| IK 容差 | 2mm 位置 / 2° 旋转 |
| 动作格式 | 6D 末端 Δ-位姿 + 夹爪 |

**优势**：
- 亚毫米精度的稳定接触行为，人类遥操作几乎不可能达到
- 低噪声、一致的演示质量
- 可大规模并行生成

### 7.3 混合触觉数据集

| 指标 | 值 |
|------|---|
| 总触觉帧数 | **200 万** |
| 操控任务 | 4 类 |
| 物体数量 | 9 个 |
| 仿真/真实比例 | 80% / 20% |
| 数据格式 | HuggingFace 数据集 |

**数据构成**：
- 仿真数据：IsaacSim + TacEx + Taxim 风格触觉仿真
- 真实数据：Dobot X-Trainer 平台 + GelSight 传感器采集
- 仅保留成功 episode 作为演示
- 所有模态严格步级同步

---

## 八、四大评估任务

### 8.1 任务描述

| 任务 | 成功率 | 难度 | 关键挑战 |
|------|:------:|:----:|---------|
| **Peg-in-Hole** | **95.0%** | ★★★ | 经典高精度插入，端口部分遮挡 |
| **USB Insert** | **85.7%** | ★★★★ | USB-A 插头插入，亚毫米容差 |
| **Gear Assembly** | **81.1%** | ★★★★ | 齿轮滑入轴，需对准齿孔 |
| **Tool Stabilize** | **74.6%** | ★★★ | 立方体顶点支撑薄圆柱，维持稳定 |

### 8.2 任务选择的系统性

四个任务按**不同维度**考验触觉感知能力：

```
                    容差精度
                       ↑
            Tool       │       Peg-in-Hole
           Stabilize   │       USB Insert
                       │
                       │       Gear Assembly
                       │
    ┌──────────────────┼──────────────────┐
    │                  │                  │
    │     接触持续时间  │→                 │
    │      短 ←───────┼──────── 长       │
    │                  │                  │
```

- **Peg-in-Hole**：中等容差 + 短接触 → 触觉辅助快速对齐
- **USB Insert**：极小容差 + 中等接触 → 触觉引导精密插入
- **Gear Assembly**：小容差 + 长接触 → 持续触觉反馈调整
- **Tool Stabilize**：中等容差 + 持续接触 → 触觉平衡控制

---

## 九、实验结果深度分析

### 9.1 真实世界主实验

| 方法 | Peg-in-Hole | USB Insert | Gear Assembly | Tool Stabilize | **平均** |
|------|:-----------:|:----------:|:-------------:|:--------------:|:--------:|
| ACT | 35.2 ± 0.7 | 62.6 ± 0.5 | 22.4 ± 0.8 | 19.3 ± 0.6 | 34.9% |
| Diffusion Policy | 35.5 ± 0.9 | 56.3 ± 0.8 | 33.1 ± 0.7 | 30.4 ± 0.9 | 38.8% |
| π₀ | 48.7 ± 1.0 | 59.4 ± 0.9 | 45.2 ± 1.1 | 41.0 ± 0.8 | 48.6% |
| ACT (w/ Tactile) | 45.6 ± 0.3 | 63.1 ± 0.4 | 40.2 ± 0.2 | 39.1 ± 0.7 | 47.0% |
| | | | | | |
| Ours (HSA-Only) | 60.8 ± 0.9 | 63.7 ± 0.8 | 51.5 ± 1.0 | 42.9 ± 0.7 | 54.7% |
| Ours (Dream-Only) | 75.4 ± 0.8 | 75.2 ± 0.7 | 64.9 ± 0.6 | 68.5 ± 0.9 | 71.0% |
| **Ours (Full)** | **95.0 ± 0.2** | **85.7 ± 0.6** | **81.1 ± 0.4** | **74.6 ± 0.5** | **84.1%** |

> 100 次试验/任务，均值 ± 标准差（3 次运行）

### 9.2 逐任务深度分析

#### Peg-in-Hole（95.0%）—— 最亮眼的表现

- 基线 ACT 仅 35.2%，DreamTacVLA 提升 **+59.8%**
- 即使是 π₀（48.7%）也远不及
- **原因**：初始抓取偏差和腕部方向的微小变化，对视觉基线是致命的，但触觉世界模型能预测插入后果并进行残余修正

#### USB Insert（85.7%）—— 亚毫米精度挑战

- 视觉完全无法解决的场景（容差 < 0.5mm）
- ACT + 触觉（63.1%）已有提升，但远不及 DreamTacVLA（85.7%）
- Think-Dream-Act 机制在此任务中最为关键：末端执行器接近端口时执行受控残余调整

#### Gear Assembly（81.1%）—— 多自由度对齐

- 需要同时满足旋转和平移对齐
- 视觉基线极度挣扎（ACT 22.4%），因为遮挡使视觉无法判断齿是否对准
- HSA 带来显著提升（51.5%），但 Dream 进一步推高至 81.1%

#### Tool Stabilization（74.6%）—— 动态平衡

- 需要持续的触觉反馈来维持平衡
- Diffusion Policy 的表现（30.4%）揭示其在接触处理上的时间不一致性
- DreamTacVLA 通过预测触觉未来，能提前预判失稳并做出修正

### 9.3 仿真侧结果（对照实验）

| 模型 | Peg-in-Hole | USB Insert | Gear Assembly | Tool Stabilize |
|------|:-----------:|:----------:|:-------------:|:--------------:|
| ACT | 72.4 | 78.1 | 65.3 | 61.7 |
| Diffusion Policy | 75.6 | 80.2 | 69.4 | 66.1 |
| π₀ | 83.5 | 85.9 | 77.2 | 74.6 |
| HSA-Only | 89.1 | 90.4 | 84.6 | 82.7 |
| Dream-Only | 93.8 | 94.6 | 88.9 | 90.1 |
| **Full** | **98.6** | **97.9** | **95.2** | **94.7** |

**关键洞察**：仿真侧的相对性能排序与真实世界完全一致，证明 HSA + Dream 的收益不是由真实世界噪声驱动的，而是方法本身的内在优势。

### 9.4 消融实验

#### HSA 与世界模型的互补性

| 配置 | 平均成功率 | 说明 |
|------|:---------:|------|
| HSA + Dream（完整模型） | **84.1%** | 最佳性能 |
| Dream-Only（无 HSA） | 71.0% | 缺少空间锚定，策略无法精确定位触觉信号 |
| HSA-Only（无 Dream） | 54.7% | 有空间对齐但缺乏时序预测能力 |
| ACT + Tactile（无任何创新） | 47.0% | 证明简单拼接触觉不够 |
| **完整 vs HSA-Only** | **+29.4%** | 世界模型的巨大贡献 |
| **完整 vs Dream-Only** | **+13.1%** | HSA 的显著增益 |
| **完整 vs 两者缺一的平均增益** | **+22.3%** | 两者协同效应 |

**结论**：空间接地（HSA）和时序想象（Dream）**缺一不可**，可靠插入行为仅在两者结合时才涌现。

#### 世界模型预测目标消融

| 预测目标 | 效果 |
|---------|------|
| 仅预测未来视觉 | 微小提升 |
| 仅预测未来触觉 | 显著提升（最关键） |
| 预测触觉 + 视觉 | **最佳**——学到更一致的跨模态物理模型 |

**结论**：触觉预测是核心贡献，但跨模态联合预测能进一步提升。

#### 数据规模消融

- 20% → 60% 数据：性能快速增长
- 60% → 80%：增长放缓但持续
- 80% → 100%：边际收益递减但仍未完全饱和
- **暗示**：进一步扩大触觉数据可能带来额外收益

---

## 十、代码架构分析

### 10.1 仓库结构

```
learning-to-feel-the-future/
├── ModelTrain/                 # 核心训练代码
│   ├── model_train.py          # 主训练入口
│   ├── constants.py            # 任务配置（路径、episode长度、相机名称）
│   └── (策略模型、HSA损失、世界模型等实现)
├── dobot_control/              # 机器人控制 + 触觉特征提取
├── experiments/                # 推理 / 控制 / 启动节点
│   └── run_inference.py        # 推理入口
├── scripts/                    # 数据采集工具
│   ├── 4_collect2train_data.py # 转换采集数据为训练格式
│   └── 6_dataset_count.py      # 数据集统计
├── examples/                   # 最小使用示例
├── third_party/                # 内置依赖
│   ├── DynamixelSDK/           # 舵机控制 SDK
│   └── Feetech/                # 飞特舵机库
├── robomimic-r2d2/             # 内置 robomimic 分支
├── jepa_ckpt/                  # V-JEPA 预训练权重目录
├── docs/figures/               # README 图表
├── train_peg.sh                # Peg-in-Hole 训练脚本
├── compare_jepa_embeddings.sh  # V-JEPA 嵌入对比脚本
├── visualize_adapter_embeddings.py  # 适配器嵌入可视化
└── requirements.txt            # Python 依赖
```

### 10.2 策略类继承体系

代码中实现了四个策略类，形成渐进式继承链：

```
ACTPolicy (基类)
  │  · ResNet-18 骨干（RGB only）
  │  · Transformer CVAE 解码器
  │  · L1 动作损失 + KL 散度
  │
  ├── ACTJEPA (触觉集成)
  │     · ResNet-18 (RGB) + V-JEPA ViT (触觉，冻结)
  │     · ViTG (1408-dim) 或 ViTL (1024-dim) 编码器
  │     · 触觉与视觉流分别处理后拼接
  │     │
  │     └── ACTJEPAAdapter (残差适配器) ← 主力策略类
  │           · 冻结 V-JEPA ViT + patch 级残差适配器
  │           · 适配器处理所有 196 个 patch token（非仅 CLS）
  │           · 学习式注意力池化聚合特征
  │           · 保持预训练知识的同时实现任务特定适应
  │
  └── ACTPolicyWithHSA (HSA 对齐)
        · 扩展基策略，加入 HSA 对比损失
        · 正运动学 3D→2D 投影实现空间对齐
        · 可与 ACTJEPAAdapter 组合使用
```

**核心策略类 `ACTJEPAAdapter`** 的完整组件：

```
ACTJEPAAdapter
├── CLIP ViT-B/16 (冻结)           → RGB 编码 (TPV + Wrist)
├── V-JEPA2 ViT-L (冻结+适配器)     → 触觉图像编码
│   └── PatchResidualAdapter
│       ├── LayerNorm → 3层MLP(GELU+Dropout) → 残差连接
│       ├── 可学习缩放因子 α (初始化 0.1)
│       └── 注意力池化 (8头, 学习式查询token)
├── CVAE Encoder (4层 Transformer)  → 训练时潜在变量
├── Action Decoder (7层 Transformer) → 45步动作预测
├── HSA Loss Module (`dobot_control/hsa_loss.py`)
│   ├── 触觉特征: mean-pool 所有 patch tokens
│   ├── 腕部特征: 夹爪感知偏移 (gripper-aware offset)
│   └── 第三人称特征: 正运动学投影 → 边界框 → 特征选择
├── Forecasting MLP                → 未来触觉状态预测
└── CLIP Text Encoder (冻结)       → 语言条件编码
```

### 10.3 关键源码模块

| 文件路径 | 功能 |
|---------|------|
| `ModelTrain/model_train.py` | 主训练入口，解析参数并启动训练循环 |
| `ModelTrain/constants.py` | 任务配置：数据路径、episode 长度、相机名称、触觉传感器参数 |
| `ModelTrain/module/policy*.py` | 四种策略类实现（ACT → ACTJEPA → ACTJEPAAdapter → ACTPolicyWithHSA） |
| `ModelTrain/module/train_module.py` | 训练循环逻辑，Stage 1/2 切换 |
| `ModelTrain/detr/` | DETR-based Transformer 架构实现 |
| `dobot_control/hsa_loss.py` | HSA 对比损失的完整实现 |
| `dobot_control/tactile_feature_extraction.py` | 多模态特征提取（触觉+视觉+语言） |
| `dobot_control/env.py` | Dobot 机器人环境接口 |
| `experiments/run_inference.py` | 实时推理脚本，Think-Dream-Act 执行 |

### 10.4 夹爪感知偏移（Gripper-Aware Offset）

HSA 实现中的一个精巧设计：触觉传感器的 2D 位置不是固定的，而是根据夹爪开合宽度动态计算：

1. 从夹爪关节角度计算触觉传感器 3D 位置
2. 使用针孔相机模型投影到 2D 图像平面
3. 选择对应位置的 patch token 作为腕部特征

这使得 HSA 能适应抓取不同大小物体时夹爪宽度变化导致的传感器位置偏移。

### 10.5 训练入口

```bash
# Peg-in-Hole 任务完整训练命令
python ModelTrain/model_train.py \
    --policy_class ACTJEPAAdapter \      # 策略类：ACT + V-JEPA 适配器
    --task_name dobot_peginhole_tac_1107 \  # 任务名（在 constants.py 中定义）
    --ckpt_dir ckpt/my_experiment \         # 检查点输出目录
    --vit_ckpt_path jepa_ckpt/vitl_peg_e150.pt \  # V-JEPA 触觉骨干权重
    --vit_model vitl \                      # V-JEPA 模型规模 (vitl/vitg)
    --clip_model ViT-B-16 --freeze_clip \  # CLIP RGB 编码器（冻结）
    --enable_text --text_prompt "Insert the peg into the hole" \  # 语言条件
    --enable_hsa --hsa_weight 1.0 \         # 启用 HSA 损失
    --num_steps 20000 --batch_size 16 --lr 1e-5  # 训练超参
```

### 10.3 关键命令行参数

| 分组 | 参数 | 用途 |
|------|------|------|
| 必需 | `--task_name` | constants.py 中定义的任务条目 |
| 必需 | `--ckpt_dir` | 检查点输出目录 |
| 必需 | `--vit_ckpt_path` | V-JEPA 触觉骨干权重路径 |
| 必需 | `--vit_model` | V-JEPA 模型规模 (`vitl` 或 `vitg`) |
| CLIP | `--clip_model` | RGB 编码器变体 |
| CLIP | `--freeze_clip` | 冻结 CLIP（推荐） |
| 文本 | `--enable_text` | 启用语言条件 |
| 文本 | `--text_prompt` | 语言提示词 |
| HSA | `--enable_hsa` | 启用分层空间对齐损失 |
| HSA | `--hsa_weight` | HSA 损失权重 |

### 10.4 策略类架构：ACTJEPAAdapter

核心策略类 `ACTJEPAAdapter` 融合了多个组件：

```
ACTJEPAAdapter
├── CLIP ViT-B/16 (冻结)        → RGB 编码 (TPV + Wrist)
├── V-JEPA2 ViT-L (冻结+适配器)  → 触觉图像编码
├── CVAE Encoder (4层 Transformer) → 训练时潜在变量
├── Action Decoder (7层 Transformer) → 45步动作预测
├── HSA Loss Module               → 分层空间对齐
├── Forecasting MLP               → 未来触觉状态预测
└── Residual Adapter (3层 MLP)    → V-JEPA2 输出适配
```

### 10.5 推理部署

```bash
python experiments/run_inference.py \
    --ckpt_dir ckpt/my_experiment \
    --task_name dobot_peginhole_tac_1107
```

推理时 CVAE 编码器被旁路，潜在变量设为零（先验均值）。动作以 45 步 chunk 预测，使用指数加权时间聚合平滑执行：

$$a_t = \sum_{k=0}^{K} w_k \cdot a_t^{(k)}, \quad w_k \propto \exp(-\lambda k)$$

### 10.6 可视化工具

```bash
# 对比不同 V-JEPA 检查点的适配器嵌入热力图
bash compare_jepa_embeddings.sh

# 单检查点嵌入可视化
python visualize_adapter_embeddings.py
```

### 10.7 环境依赖

| 依赖 | 说明 |
|------|------|
| Python 3.10 | 运行环境 |
| PyTorch (CUDA) | 深度学习框架 |
| CLIP / open_clip | RGB 编码器 |
| V-JEPA / V-JEPA 2 (Meta) | 触觉世界模型骨干 |
| robomimic (内置分支) | 行为克隆框架 |
| DynamixelSDK / Feetech (内置) | 舵机控制 |
| IsaacSim + TacEx | 仿真环境 |

---

## 十一、模型超参数完整清单

| 超参数 | 值 |
|--------|---|
| **Transformer** | |
| 隐藏维度 | 512 |
| 前馈维度 | 3200 |
| 编码器层数 | 4 |
| 解码器层数 | 7 |
| 注意力头数 | 8 |
| Dropout | 0.1 |
| CVAE 潜在维度 | 32 |
| 动作 chunk 大小 | 45 |
| **视觉编码器** | |
| RGB 骨干 | ResNet-18 (冻结) |
| RGB 输入分辨率 | 640×480 |
| 触觉编码器 | V-JEPA2 ViT-L (冻结) |
| 触觉嵌入维度 | 1024 |
| 触觉输入分辨率 | 224×224 |
| **残差适配器** | |
| 隐藏维度 | 512 |
| 层数 | 3 |
| Dropout | 0.1 |
| 残差缩放 α 初始化 | 0.1 |
| 聚合方式 | 注意力池化 |
| **HSA 损失** | |
| 温度 κ | 0.07 |
| 特征提取器 | ViT-B/16 |
| HSA 权重 | 1.0 |
| **训练** | |
| 优化器 | AdamW |
| 学习率 | 1×10⁻⁵ |
| 骨干学习率 | 1×10⁻⁵ |
| 权重衰减 | 1×10⁻⁴ |
| Batch Size | 16 |
| KL 权重 | 10 |
| 训练步数 | 10,000–20,000 |
| 单 GPU 训练时间 | 8–12 小时 |

---

## 十二、硬件平台

### 12.1 真实世界平台

| 组件 | 规格 |
|------|------|
| 机器人 | Dobot X-Trainer 双臂平台 |
| 触觉传感器 | GelSight Digit（双指，高分辨率） |
| 腕部相机 | RealSense D405 × 2 |
| 第三人称相机 | RealSense（全局视角） |
| 控制频率 | 50 Hz |

### 12.2 仿真数字孪生

- 基于 IsaacSim 的 Dobot X-Trainer 数字孪生
- 共享相机外参（仿真与真实一致）
- TacEx + Taxim 风格触觉渲染

### 12.3 相机标定参数

**外参（基座到第三人称相机）**：

$$T_b^c = \begin{bmatrix} 1.0 & 0.0 & 0.0 & 0.7 \\ 0.0 & -1.0 & 0.0 & -0.49 \\ 0.0 & 0.0 & 1.0 & 1.14 \\ 0.0 & 0.0 & 0.0 & 1.0 \end{bmatrix}$$

**内参矩阵**：

$$K = \begin{bmatrix} 647.0 & 0.0 & 653.0 \\ 0.0 & 644.0 & 364.0 \\ 0.0 & 0.0 & 1.0 \end{bmatrix}$$

**DH 参数（Dobot Nova 2）**：

| 关节 | θ | d (m) | a (m) | α |
|------|---|--------|--------|---|
| 1 | q₀ | 0.2234 | 0 | π/2 |
| 2 | q₁ | -π/2 | -0.280 | 0 |
| 3 | q₂ | 0 | -0.225 | 0 |
| 4 | q₃ | -π/2 | 0.1175 | π/2 |
| 5 | q₄ | 0 | 0.120 | -π/2 |
| 6 | q₅ | 0 | 0.088 | 0 |

---

## 十三、核心创新点与贡献总结

### 13.1 技术贡献

| # | 贡献 | 意义 |
|---|------|------|
| 1 | **HSA 损失** | 首次实现触觉-视觉的多尺度空间对齐，使触觉信号"知道自己在哪" |
| 2 | **触觉世界模型** | 将 V-JEPA2 用于触觉时序预测，建立隐式接触物理引擎 |
| 3 | **Think-Dream-Act 范式** | 无需显式规划器或奖励函数，通过"想象触觉未来"精炼动作 |
| 4 | **混合大规模数据集** | 2M 触觉帧，仿真+真实混合，缓解触觉数据稀缺 |
| 5 | **残差适配器** | 仅 1.8% 额外参数高效适配冻结 V-JEPA2 |

### 13.2 方法论贡献

1. **触觉 = 微型视觉**：将触觉图像重新定义为第三种视觉尺度，而非独立模态
2. **预测驱动使用**：通过要求模型预测触觉未来，强制其学会使用触觉信息
3. **无规划器的世界模型**：将世界模型从 MPC 框架中解放出来，端到端训练
4. **仿真即数据工厂**：80% 仿真数据 + 20% 真实数据的混合策略平衡了规模和真实性

---

## 十四、局限性与未来方向

### 14.1 当前局限

| 局限 | 具体表现 |
|------|---------|
| **推理开销** | Think-Dream-Act 需要两次策略前向传播，推理速度约为单次策略的 2 倍 |
| **触觉传感器限制** | 仅验证了 GelSight 类光学传感器，其他类型传感器（力/力矩、电容式）未测试 |
| **任务多样性** | 仅在 4 个任务上评估，未涉及可变形物体、多步装配等更复杂场景 |
| **单臂数** | 基于 Dobot X-Trainer 双臂平台但实验中主要使用单臂 |
| **世界模型冻结** | V-JEPA2 在策略训练期间完全冻结，可能限制了触觉表征的进一步优化 |

### 14.2 未来方向

1. **策略蒸馏**：将 Think-Dream-Act 精炼为单次前向推理，消除推理开销
2. **自适应 Dreaming**：仅在接触密集阶段启用世界模型预测，其余阶段跳过
3. **更大规模触觉预训练**：利用更多样化的触觉数据提升世界模型的泛化能力
4. **跨传感器迁移**：扩展到更多触觉传感器类型
5. **开放式操控**：从固定任务集扩展到语言指导的零样本操控
6. **与 LLM 集成**：将触觉世界模型与大语言模型结合，实现触觉感知的语义理解

---

## 十五、与 UniVTAC 的系统性对比

| 维度 | UniVTAC | DreamTacVLA |
|------|---------|-------------|
| **定位** | 仿真基础设施 + 表征学习 | 策略范式 + VLA 集成 |
| **层级** | 数据层 → 表征层 → 评估层 | 感知层 → 预测层 → 决策层 |
| **编码器** | ResNet-18 + 三路径监督 | V-JEPA2 ViT-L + HSA 对齐 |
| **世界模型** | ❌ | ✅ 触觉未来预测 |
| **传感器** | 3 种 (GelSight/ViTai/Xense) | 1 种 (GelSight) |
| **训练范式** | 预训练编码器 → 下游策略 | 两阶段端到端训练 |
| **数据规模** | ~200k 帧 | 2M 帧 (10×) |
| **Sim-to-Real** | +25% | +47% |
| **开源完整度** | 完整平台代码 + 数据 | 完整训练代码 + 数据 |
| **互补角色** | 提供**数据与表征基础设施** | 展示**策略进阶上限** |

> **核心洞察**：UniVTAC 回答了"如何获取触觉数据和学习触觉表征"，DreamTacVLA 回答了"如何让策略真正利用触觉信息"。两者共同指向触觉操控的终极目标：**让机器人像人类一样，既能感知接触，也能预判接触后果**。

---

## 参考文献

1. Ye, G., Zhang, Z., et al. "Learning to Feel the Future: DreamTacVLA for Contact-Rich Manipulation." arXiv:2512.23864, 2025.
2. Chen, B., Wan, W., et al. "UniVTAC: A Unified Simulation Platform for Visuo-Tactile Manipulation Data Generation, Learning, and Benchmarking." arXiv:2602.10093, 2026.
3. Zhao, T. Z., et al. "Learning Fine-grained Bimanual Manipulation with Low-cost Hardware." arXiv:2304.13705 (ACT), 2023.
4. Assran, M., et al. "V-JEPA 2: Self-supervised Video Models Enable Understanding, Prediction and Planning." arXiv:2506.09985, 2025.
5. Radford, A., et al. "Learning Transferable Visual Models from Natural Language Supervision." ICML (CLIP), 2021.
6. Chi, C., et al. "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion." IJRR, 2025.
7. Black, K., et al. "π₀: A Vision-Language-Action Flow Model for General Robot Control." arXiv:2410.24164, 2024.
8. Nguyen, D. H., et al. "TacEx: GelSight Tactile Simulation in Isaac Sim." arXiv:2411.04776, 2024.

---

*本文档基于 DreamTacVLA 论文全文（arXiv:2512.23864v3，含附录）、项目网站（michaelyeah7.github.io）、GitHub 仓库源码及 README 综合撰写。*
