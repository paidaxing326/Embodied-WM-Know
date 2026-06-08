# LeWorldModel (LeWM)：稳定端到端联合嵌入预测架构深度解读

> **论文**: LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels
> **作者**: Lucas Maes*, Quentin Le Lidec* (NYU), Damien Scieur (Mila/Samsung SAIL), Yann LeCun (NYU), Randall Balestriero (Brown)
> **机构**: Mila & Université de Montréal, New York University, Samsung SAIL, Brown University
> **论文链接**: arxiv.org/abs/2603.19312 | **代码**: github.com/lucas-maes/le-wm
> **发表时间**: 2026年3月

---

## 目录

1. [背景与动机](#1-背景与动机)
2. [核心架构详解](#2-核心架构详解)
3. [SIGReg：防坍塌正则化的数学基础](#3-sigreg防坍塌正则化的数学基础)
4. [训练目标与损失函数](#4-训练目标与损失函数)
5. [数据管线与训练流程](#5-数据管线与训练流程)
6. [潜在空间规划：CEM-MPC](#6-潜在空间规划cem-mpc)
7. [物理理解评估体系](#7-物理理解评估体系)
8. [源码架构全景](#8-源码架构全景)
9. [关键创新与贡献总结](#9-关键创新与贡献总结)
10. [与相关工作的对比](#10-与相关工作的对比)
11. [局限性、未来方向与反思](#11-局限性未来方向与反思)
12. [参数参考表](#12-参数参考表)

---

## 1. 背景与动机

### 1.1 JEPA 的坍塌困境

联合嵌入预测架构（JEPA）[LeCun 2022] 是学习世界模型的理想框架——它在紧凑的潜在空间中预测未来状态，避免在像素空间中建模一切无关细节。然而，**表征坍塌**（representation collapse）是 JEPA 的核心挑战：编码器将所有输入映射到近乎相同的表示，从而平凡满足时间预测目标，导致表示完全不可用。

现有方法依赖各种启发式来避免坍塌：
- **EMA + Stop-Gradient**（I-JEPA, V-JEPA）：对目标编码器使用指数移动平均和梯度截断，理论理解有限，不对应任何良定义目标函数的最小化
- **预训练编码器冻结**（DINO-WM）：冻结 DINOv2 编码器避免坍塌，但放弃了端到端学习，限制了表示表达能力
- **VICReg 多项正则化**（PLDM）：使用七项损失（pred + var + cov + time-sim + time-var + time-cov + IDM），六个可调超参，训练不稳定

### 1.2 LeWM 的核心主张

LeWorldModel (LeWM) 的突破在于：**首次实现纯像素输入的端到端稳定 JEPA 训练，仅需两项损失、一个可调超参**。

具体来说：
- 损失函数仅有 $\mathcal{L}_{\text{pred}}$（预测损失）+ $\lambda \cdot \text{SIGReg}$（防坍塌正则化）
- 无需 Stop-Gradient、EMA、预训练编码器、辅助监督
- 梯度通过所有组件反向传播，所有参数联合端到端优化
- $\lambda$ 是唯一需要调的超参，可用二分搜索在 $\mathcal{O}(\log n)$ 内高效优化（对比 PLDM 的 $\mathcal{O}(n^6)$）
- ~15M 参数，单 GPU 数小时即可训练完成
- 规划速度比基于基础模型的 DINO-WM 快 **48倍**

---

## 2. 核心架构详解

### 2.1 整体架构概览

LeWM 由四个核心组件构成：

```
观测 o_t → [ViT Encoder] → CLS token → [Projector MLP] → 嵌入 z_t
                                                    ↘
动作 a_t → [Action Encoder] → act_emb              → [ARPredictor (AdaLN Transformer)] → 预测嵌入 z_{t+1}
                                                                          ↘
                                                              [Pred Projector MLP] → ẑ_{t+1}
```

**[源码验证]** `jepa.py:11-55` 完整实现 JEPA 类，包含 encoder、predictor、action_encoder、projector、pred_proj 五个子模块。

### 2.2 编码器（Encoder）：ViT-Tiny

编码器采用 Vision Transformer Tiny (ViT-Tiny)：

| 参数 | 值 |
|---|---|
| Patch Size | 14 |
| 图像尺寸 | 224×224 |
| 层数 | 12 |
| 注意力头数 | 3 |
| 隐藏维度 | 192 |
| 参数量 | ~5M |

**关键设计决策**：

1. **CLS Token 提取**：从 ViT 最后层取 `[CLS]` token 作为帧级嵌入（`jepa.py:38`: `output.last_hidden_state[:, 0]`）
2. **投影层必要性**：ViT 最后层使用 LayerNorm，这会阻碍 SIGReg 的优化效果。因此需要一个 Projector MLP 将 CLS token 映射到新的表示空间
3. **Projector 架构**：1层 MLP + BatchNorm1d（`module.py:217-241` MLP 类，`train.py:104-109` 使用 `norm_fn=torch.nn.BatchNorm1d`）
4. **不使用预训练**：`pretrained=False, use_mask_token=False`（`train.py:82-88`）
5. **ImageNet 标准化**：像素输入使用 ImageNet 均值/方差标准化（`utils.py:8-11`）

**[源码验证]** 编码器通过 `spt.backbone.utils.vit_hf` 创建，配置来自 `config/train/lewm.yaml` 中 `encoder_scale: tiny, patch_size: 14, img_size: 224`。

### 2.3 动作编码器（Action Encoder）：Embedder

动作编码器是一个轻量级 MLP 嵌入模块：

```python
# module.py:189-214
class Embedder(nn.Module):
    def __init__(self, input_dim=10, smoothed_dim=10, emb_dim=10, mlp_scale=4):
        self.patch_embed = nn.Conv1d(input_dim, smoothed_dim, kernel_size=1, stride=1)
        self.embed = nn.Sequential(
            nn.Linear(smoothed_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )
```

**设计要点**：
- 输入维度 = `frameskip × action_dim`（默认 5 × 2 = 10 for PushT）
- 使用 Conv1d(kernel_size=1) 先将动作维度映射到 `smoothed_dim`
- 然后通过 SiLU-MLP 嵌入到 `emb_dim`（默认 192）
- 输出形状 `(B, T, emb_dim)` 与潜在嵌入同维度

**[源码验证]** `train.py:102`: `action_encoder = Embedder(input_dim=effective_act_dim, emb_dim=embed_dim)`，`effective_act_dim = cfg.data.dataset.frameskip * cfg.wm.action_dim`。

### 2.4 预测器（Predictor）：ARPredictor

预测器是带 AdaLN-zero 条件化的因果 Transformer：

```python
# module.py:244-285
class ARPredictor(nn.Module):
    def __init__(self, num_frames, depth, heads, mlp_dim, input_dim, hidden_dim,
                 output_dim=None, dim_head=64, dropout=0.0, emb_dropout=0.0):
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, input_dim))
        self.transformer = Transformer(..., block_class=ConditionalBlock)
```

| 参数 | 值 |
|---|---|
| 层数 (depth) | 6 |
| 注意力头数 | 16 |
| dim_head | 64 |
| mlp_dim | 2048 |
| Dropout | 0.1 |
| 参数量 | ~10M |

**核心机制：AdaLN-zero 条件化**

ConditionalBlock 实现了自适应 Layer Normalization：

```python
# module.py:88-111
class ConditionalBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)
        )
        # 关键：零初始化
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x
```

**AdaLN-zero 的关键设计**：
1. 从条件信号 c（动作嵌入）生成 6 个调制参数：shift、scale、gate 各两个（分别用于 attention 和 MLP）
2. `modulate(x, shift, scale) = x * (1 + scale) + shift` —— 这就是 AdaLN
3. **零初始化至关重要**：weight=0, bias=0 → 初始时 gate_msa=gate_mlp=0，shift=scale=0 → 模块等效于 `x = x + 0 * attn(LN(x)) + 0 * mlp(LN(x)) = x`。动作条件化对训练的影响从零开始渐进增长，防止训练初期不稳定
4. causal masking 确保自回归预测不窥视未来嵌入（`module.py:75-85` `is_causal=causal`）

**[源码验证]** 配置来自 `config/train/lewm.yaml`: `predictor: depth=6, heads=16, mlp_dim=2048, dim_head=64, dropout=0.1`。

### 2.5 预测投影器（Pred Projector）

预测器输出经过另一个 Projector MLP 映射回嵌入空间：

```python
# train.py:111-116
predictor_proj = MLP(
    input_dim=hidden_dim,      # 192 (ViT hidden_size)
    output_dim=embed_dim,       # 192
    hidden_dim=2048,
    norm_fn=torch.nn.BatchNorm1d,
)
```

**设计逻辑**：与编码器的 Projector 结构相同（MLP + BatchNorm1d），确保预测嵌入与真实嵌入在同一空间中可比较。

### 2.6 嵌入维度设计

论文与代码中存在两层维度映射：

- `hidden_dim` = ViT 的隐藏维度（ViT-Tiny = 192）
- `embed_dim` = SIGReg 操作的实际嵌入维度（默认等于 hidden_dim = 192）

**[源码验证]** `train.py:91`: `embed_dim = cfg.wm.get("embed_dim", hidden_dim)`，配置 `wm.embed_dim: 192`。

---

## 3. SIGReg：防坍塌正则化的数学基础

### 3.1 核心思想

SIGReg（Sketch Isotropic Gaussian Regularizer）源自 Balestriero & LeCun 2025 的 LeJEPA 工作。其核心思想是：**将高维嵌入分布推向标准各向同性高斯分布 $\mathcal{N}(0, \mathbf{I})$**。

### 3.2 Cramér-Wold 定理 + Epps-Pulley 检验

高维正态性检验的挑战是经典统计检验设计为一元数据，不能可靠地随维度扩展。SIGReg 利用两个统计组件优雅地绕过：

**第一步：随机投影降维**

沿 $M$ 个随机单位方向 $\mathbf{u}^{(m)} \in \mathbb{S}^{D-1}$ 投射嵌入 $\mathbf{Z}$：

$$\mathbf{h}^{(m)} = \mathbf{Z}\mathbf{u}^{(m)}$$

**第二步：一元 Epps-Pulley 检验统计量**

对每个投影方向计算一元正态性检验：

$$T^{(m)} = \int_{-\infty}^{\infty} w(t) \left|\phi_N(t; \mathbf{h}^{(m)}) - \phi_0(t)\right|^2 dt$$

其中：
- $\phi_N(t; \mathbf{h}) = \frac{1}{N}\sum_{n=1}^{N} e^{ith_n}$ 是经验特征函数（ECF）
- $\phi_0(t) = e^{-t^2/2}$ 是标准正态分布的特征函数
- $w(t) = e^{-t^2/2}$ 是权重函数

**第三步：聚合**

$$\text{SIGReg}(\mathbf{Z}) = \frac{1}{M}\sum_{m=1}^{M} T^{(m)}$$

**Cramér-Wold 保证**：在 $M \to \infty$ 的极限下：

$$\text{SIGReg}(\mathbf{Z}) \to 0 \iff \mathbb{P}_\mathbf{Z} \to \mathcal{N}(0, \mathbf{I})$$

即：匹配所有一维边际分布等价于匹配联合分布。

### 3.3 源码实现详解

```python
# module.py:10-36
class SIGReg(torch.nn.Module):
    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)  # 梯形积分节点
        dt = 3 / (knots - 1)
        weights = torch.full((knets,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt  # 梯形法两端权重
        window = torch.exp(-t.square() / 2.0)  # 权重窗口 w(t) = exp(-t²/2)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)     # φ_0(t) = exp(-t²/2)
        self.register_buffer("weights", weights * window)  # 组合权重

    def forward(self, proj):
        # proj: (T, B, D) — 时间步 × 批次 × 嵌入维度
        # 步骤1：采样随机投影方向
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))  # 归一化为单位向量
        # 步骤2：投影并计算 ECF
        x_t = (proj @ A).unsqueeze(-1) * self.t  # (T, B, M, knots)
        # 步骤3：计算 Epps-Pulley 统计量
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()  # 对投影和时间取平均
```

**实现要点**：
1. `knots=17`：梯形积分节点数，均匀分布在 $[0, 3]$
2. `num_proj=1024`：随机投影方向数（$M=1024$）
3. 投影方向 A 每次前向传播重新随机采样（不缓存），保证统计多样性
4. 输入形状 `(T, B, D)` — SIGReg 在每个时间步独立应用（step-wise），不在时间维度上操作
5. `proj.size(-2)` = B（批次大小），作为统计量的缩放因子

**[源码验证]** 配置 `loss.sigreg.kwargs: knots=17, num_proj=1024`, `loss.sigreg.weight: 0.09`（接近论文推荐的 $\lambda=0.1$）。

### 3.4 SIGReg vs. VICReg 的本质区别

| 特性 | SIGReg | VICReg |
|---|---|---|
| 目标分布 | 各向同性高斯 $\mathcal{N}(0, \mathbf{I})$ | 无明确目标，仅约束方差>阈值、协方差→0 |
| 数学保证 | Cramér-Wold 渐近收敛保证 | 无形式化收敛保证 |
| 超参数量 | 1（$\lambda$） | 3+（var_weight, cov_weight, sim_weight...） |
| 高维处理 | 随机投影降维 → 一元检验 | 直接在 D 维空间操作协方差矩阵 |
| 计算复杂度 | $\mathcal{O}(M \cdot N \cdot D)$ | $\mathcal{O}(D^2)$ |

---

## 4. 训练目标与损失函数

### 4.1 LeWM 两项损失

$$\mathcal{L}_{\text{LeWM}} = \mathcal{L}_{\text{pred}} + \lambda \cdot \text{SIGReg}(\mathbf{Z})$$

**预测损失**（teacher-forcing）：

$$\mathcal{L}_{\text{pred}} = \|\hat{\mathbf{z}}_{t+1} - \mathbf{z}_{t+1}\|_2^2, \quad \hat{\mathbf{z}}_{t+1} = \text{pred}_\phi(\mathbf{z}_t, \mathbf{a}_t)$$

**SIGReg 正则化**：在嵌入张量 $\mathbf{Z} \in \mathbb{R}^{N \times B \times D}$ 上应用，每个时间步独立。

### 4.2 源码实现

```python
# train.py:18-46 — lejepa_forward 函数
def lejepa_forward(self, batch, stage, cfg):
    ctx_len = cfg.wm.history_size      # 3 (默认)
    n_preds = cfg.wm.num_preds          # 1 (默认)
    lambd = cfg.loss.sigreg.weight      # 0.09

    batch["action"] = torch.nan_to_num(batch["action"], 0.0)  # NaN→0（序列边界处）
    output = self.model.encode(batch)

    emb = output["emb"]       # (B, T, D)
    act_emb = output["act_emb"]

    ctx_emb = emb[:, :ctx_len]       # 上下文嵌入
    ctx_act = act_emb[:, :ctx_len]   # 上下文动作嵌入

    tgt_emb = emb[:, n_preds:]       # 目标嵌入（预测目标）
    pred_emb = self.model.predict(ctx_emb, ctx_act)  # 预测嵌入

    # LeWM 两项损失
    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))  # (T, B, D)
    output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]
```

**关键细节**：
1. `history_size=3`：预测器使用前3帧嵌入作为上下文
2. `num_preds=1`：仅预测下1步嵌入（单步预测）
3. SIGReg 输入形状 `emb.transpose(0,1)` → `(T, B, D)`，在每个时间步独立应用
4. NaN 处理：序列边界处动作可能为 NaN，替换为 0

### 4.3 对比 PLDM 的七项损失

PLDM 的损失函数：

$$\mathcal{L}_{\text{PLDM}} = \mathcal{L}_{\text{pred}} + \alpha\mathcal{L}_{\text{var}} + \beta\mathcal{L}_{\text{cov}} + \gamma\mathcal{L}_{\text{time-sim}} + \zeta\mathcal{L}_{\text{time-var}} + \nu\mathcal{L}_{\text{time-cov}} + \mu\mathcal{L}_{\text{IDM}}$$

包含：
- $\mathcal{L}_{\text{var}}$：跨批次方差约束（每维度方差 > 1）
- $\mathcal{L}_{\text{cov}}$：跨特征协方差去相关
- $\mathcal{L}_{\text{time-sim}}$：时间相似性（相邻嵌入接近）
- $\mathcal{L}_{\text{time-var}}$：跨时间方差约束
- $\mathcal{L}_{\text{time-cov}}$：跨时间协方差去相关
- $\mathcal{L}_{\text{IDM}}$：逆动力学模型损失

**六个可调超参** $(\alpha, \beta, \gamma, \zeta, \nu, \mu)$ → 搜索空间 $\mathcal{O}(n^6)$

**LeWM 仅一个** $\lambda$ → 搜索空间 $\mathcal{O}(\log n)$（二分搜索）

### 4.4 训练曲线特征

论文 Fig.18-19 展示了 LeWM 与 PLDM 的训练曲线对比：

- **LeWM**：两项损失呈现平滑单调收敛——预测损失稳步下降，SIGReg 在训练初期快速下降后趋于平稳（潜在分布快速逼近各向同性高斯）
- **PLDM**：七项损失显示噪声和非单调行为——多个正则化项的竞争梯度导致训练不稳定

---

## 5. 数据管线与训练流程

### 5.1 数据格式

数据使用 HDF5 格式存储，通过 `stable_worldmodel.data.HDF5Dataset` 加载：

```python
# train.py:54
dataset = swm.data.HDF5Dataset(**cfg.data.dataset, transform=None)
```

每个数据集包含的列：
- `pixels`：RGB 像素观测序列
- `action`：动作向量序列
- `proprio`：本体感知信息（可选）
- `state`：状态信息（可选）
- `observation`：观测向量（OGBench 环境）
- `episode_idx` / `step_idx`：轨迹索引

### 5.2 数据预处理管线

```python
# train.py:55-68
transforms = [get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size)]

for col in cfg.data.dataset.keys_to_load:
    if col.startswith("pixels"):
        continue
    normalizer = get_column_normalizer(dataset, col, col)  # 标准化非像素列
    transforms.append(normalizer)
    setattr(cfg.wm, f"{col}_dim", dataset.get_dim(col))  # 记录维度信息

transform = spt.data.transforms.Compose(*transforms)
dataset.transform = transform
```

**[源码验证]** `utils.py:8-11` ImageNet 标准化 + Resize(224)；`utils.py:14-27` 对非像素列计算均值/标准差并标准化。

### 5.3 子轨迹构建

配置中 `num_steps = wm.num_preds + wm.history_size = 1 + 3 = 4`，`frameskip = 5`。

这意味着每个训练样本包含 4 帧（3帧上下文 + 1帧预测目标），帧间跳跃 5 步环境时间步。4帧 × 5帧跳 = 覆盖 20 个环境时间步。

### 5.4 四个评估环境

| 环境 | 类型 | 数据量 | 平均轨迹长度 | 动作维度 | 训练轮数 |
|---|---|---|---|---|---|
| **TwoRoom** | 2D 导航 | 10,000 episodes | 92 步 | 2D 连续 | 10 epochs |
| **PushT** | 2D 操控 | 20,000 expert episodes | 196 步 | 2D 连续 | 10 epochs |
| **OGBench-Cube** | 3D 机械臂操控 | 10,000 episodes | 200 步 | 7D 连续 | 10 epochs |
| **Reacher** | 2D 机械臂到达 | 10,000 episodes | 200 步 | 2D 连续 | 10 epochs |

### 5.5 训练配置

| 参数 | 值 |
|---|---|
| Batch Size | 128 |
| 最大 Epochs | 100 |
| 学习率 | 5e-5 |
| 优化器 | AdamW |
| Weight Decay | 1e-3 |
| Gradient Clip | 1.0 |
| Precision | bf16 |
| Scheduler | LinearWarmupCosineAnnealingLR |
| Train/Val Split | 0.9 / 0.1 |
| Seed | 3072 |

**[源码验证]** 全部来自 `config/train/lewm.yaml`。

---

## 6. 潜在空间规划：CEM-MPC

### 6.1 规划流程

推理时，LeWM 在潜在空间中进行轨迹优化：

1. **编码初始观测**：$\hat{\mathbf{z}}_1 = \text{enc}_\theta(\mathbf{o}_1)$
2. **编码目标观测**：$\mathbf{z}_g = \text{enc}_\theta(\mathbf{o}_g)$
3. **自回归展开**：$\hat{\mathbf{z}}_{t+1} = \text{pred}_\phi(\hat{\mathbf{z}}_t, \mathbf{a}_t)$，沿规划视界 H 展开
4. **终端代价**：$\mathcal{C}(\hat{\mathbf{z}}_H) = \|\hat{\mathbf{z}}_H - \mathbf{z}_g\|_2^2$
5. **CEM 优化**：最小化代价，找到最优动作序列

### 6.2 Rollout 实现详解

```python
# jepa.py:61-110 — rollout 方法
def rollout(self, info, action_sequence, history_size: int = 3):
    B, S, T = action_sequence.shape[:3]  # B=批次, S=候选数, T=时间
    act_0, act_future = torch.split(action_sequence, [H, T - H], dim=2)

    # 编码初始帧
    _init = self.encode(_init)
    emb = _init["emb"].unsqueeze(1).expand(B, S, -1, -1)

    # 展平 batch × sample 维度
    emb = rearrange(emb, "b s ... -> (b s) ...").clone()
    act = rearrange(act_0, "b s ... -> (b s) ...")
    act_future = rearrange(act_future, "b s ... -> (b s) ...")

    # 自回归展开
    HS = history_size  # 3
    for t in range(n_steps):
        act_emb = self.action_encoder(act)
        emb_trunc = emb[:, -HS:]      # 取最近 HS 帧嵌入
        act_trunc = act_emb[:, -HS:]   # 取最近 HS 帧动作嵌入
        pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]  # 仅取最后预测
        emb = torch.cat([emb, pred_emb], dim=1)
        next_act = act_future[:, t:t+1, :]
        act = torch.cat([act, next_act], dim=1)

    # 预测最终状态
    act_emb = self.action_encoder(act)
    emb_trunc = emb[:, -HS:]
    act_trunc = act_emb[:, -HS:]
    pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]
    emb = torch.cat([emb, pred_emb], dim=1)
```

**关键设计**：
- `history_size=3`：每次预测只使用最近3帧的嵌入和动作嵌入作为输入，而非全部历史
- `[:, -1:]`：预测器输出整个序列的预测，但仅取最后一步（因果 Transformer 的自然行为）
- 批次 × 候选维度展平：`(B, S, ...) → (B*S, ...)` 以并行处理所有候选动作序列

### 6.3 CEM 优化器

交叉熵方法（CEM）是采样式零阶优化算法：

| 参数 | PushT | 其他环境 |
|---|---|---|
| 每迭代采样数 | 300 | 300 |
| 优化迭代数 | 30 | 10 |
| Top-k (elites) | 30 | 30 |
| 初始方差 | 1.0 | 1.0 |

**CEM 流程**：
1. 初始化采样分布 $\mu_0=0, \Sigma_0=\mathbf{I}$
2. 每迭代采样 N=300 个候选动作序列
3. 用世界模型展开每个候选，计算终端代价
4. 选择代价最低的 K=30 个候选（elites）
5. 用 elites 更新采样分布参数
6. 重复至收敛

**MPC 递减视界**：规划视界 H=5，但整个优化动作序列都执行后才重新规划（receding_horizon=5），对应 5 × frameskip=25 个环境时间步。

### 6.4 规划速度对比

论文核心发现：

| 方法 | 编码 token 数 | PushT 规划时间 |
|---|---|---|
| DINO-WM (DINOv2-ViT-G) | ~260K tokens | ~50s |
| LeWM (ViT-Tiny) | ~1.3K tokens | ~1s |

LeWM 快 **48倍**，因为：
1. ViT-Tiny 编码器产生极少的 token（1个CLS vs DINOv2 的数百个patch token）
2. 潜在空间极紧凑（192维 vs DINOv2 的数百/数千维）
3. 预测器仅处理3帧 × 192维的序列

---

## 7. 物理理解评估体系

### 7.1 物理量探针（Probing）

论文通过线性探针和非线性探针（MLP）评估潜在空间中可恢复的物理量：

**PushT 环境结果**：

| 物理量 | 模型 | Linear MSE↓ | Linear r↑ | MLP MSE↓ | MLP r↑ |
|---|---|---|---|---|---|
| Agent Location | LeWM | **0.052** | 0.974 | **0.004** | 0.998 |
| Agent Location | PLDM | 0.090 | 0.955 | 0.014 | 0.993 |
| Agent Location | DINO-WM | 1.888 | **0.977** | **0.003** | **0.999** |
| Block Location | LeWM | 0.029 | 0.986 | **0.001** | **0.999** |
| Block Location | PLDM | 0.122 | 0.938 | 0.011 | 0.994 |
| Block Location | DINO-WM | **0.006** | **0.997** | 0.002 | **0.999** |
| Block Angle | LeWM | 0.187 | 0.902 | 0.021 | 0.990 |
| Block Angle | PLDM | 0.446 | 0.745 | 0.056 | 0.972 |
| Block Angle | DINO-WM | **0.050** | **0.979** | **0.009** | **0.995** |

**解读**：
- LeWM **在 Linear 探针上大幅优于 PLDM**（Agent Location MSE: 0.052 vs 0.090）
- DINO-WM 在部分属性上略优，但这是因为它在 ~124M 图像上预训练的结果
- 所有方法在 Block Angle/Quaternion 上表现较弱——细粒度旋转信息在紧凑潜在空间中难以编码

### 7.2 违反预期框架（Violation-of-Expectation）

VoE 评估检测物理不可能事件的能力：

- **视觉扰动**：物体颜色突变（不违反物理连续性）
- **物理扰动**：物体瞬移到随机位置（违反物理连续性）

**核心发现**：
- LeWM 对物理扰动（瞬移）的惊讶度显著高于未扰动轨迹（paired t-test, p < 0.01）
- 对颜色扰动的惊讶度增加较弱且不显著——模型更敏感于物理违反而非视觉变化
- 这个结果在三个环境（TwoRoom, PushT, OGBench-Cube）上一致

### 7.3 时间潜在路径直线化（Temporal Straightening）

一个意外发现：LeWM 的潜在轨迹随训练逐渐变得"更直"——连续速度向量的余弦相似度趋近 1。

$$\mathcal{S}_{\text{straight}} = \frac{1}{B(T-2)}\sum_{i=1}^{B}\sum_{t=1}^{T-2}\frac{\langle\mathbf{v}_t^{(i)}, \mathbf{v}_{t+1}^{(i)}\rangle}{\|\mathbf{v}_t^{(i)}\| \|\mathbf{v}_{t+1}^{(i)}\|}$$

**有趣之处**：
- SIGReg 在每个时间步独立应用，不在时间维度操作
- 这允许编码器趋向一种"时间坍塌"——连续嵌入沿越来越直的路径演化
- LeWM 的时间直线化程度 **高于 PLDM**，尽管 PLDM 有显式的时间平滑正则化项 $\mathcal{L}_{\text{time-sim}}$

### 7.4 解码器可视化

论文训练了一个轻量 Transformer 解码器（仅用于诊断）来验证潜在嵌入的信息含量：

- 解码器架构：跨注意力层 + learnable query tokens（每个 patch 一个 query）
- 196 个 query tokens（224/16)^2 对应 196 个 16×16×3 patch
- **关键发现**：尽管训练中从不使用重建损失，解码器能恢复视觉场景——证明 192 维 CLS token 嵌入保留了足够物理状态信息
- 训练初期解码图像仅显示"慢特征"（全局布局），后期逐步恢复细节

---

## 8. 源码架构全景

### 8.1 项目文件结构

```
le-wm/
├── jepa.py          # JEPA 核心模型（encode, predict, rollout, criterion, get_cost）
├── module.py         # 所有子模块（SIGReg, ARPredictor, ConditionalBlock, Attention, MLP, Embedder）
├── train.py          # 训练脚本（数据加载、模型构建、损失计算、Lightning训练循环）
├── eval.py           # 评估脚本（环境创建、MPC规划、结果记录）
├── utils.py          # 工具函数（图像预处理、列标准化、checkpoint回调）
├── config/
│   ├── train/
│   │   ├── lewm.yaml           # 主训练配置
│   │   ├── data/               # 环境特定数据配置
│   │   │   ├── pusht.yaml
│   │   │   ├── dmc.yaml        # Reacher
│   │   │   ├── tworoom.yaml
│   │   │   └b.yaml
│   │   └── launcher/
│   │       └local.yaml
│   └── eval/
│       ├── pusht.yaml
│       ├── cube.yaml
│       ├── tworoom.yaml
│       ├── reacher.yaml
│       └── solver/
│           ├── cem.yaml         # CEM 规划器配置
│           └b.yaml        # Adam 规划器配置（可选）
│       └── launcher/
│           └local.yaml
```

### 8.2 核心数据流

**训练流程**：

```
HDF5Dataset → DataLoader → lejepa_forward()
                              │
                              ├─ self.model.encode(batch)
                              │    ├─ ViT Encoder → CLS token
                              │    ├─ Projector MLP → emb (B, T, D)
                              │    └─ ActionEncoder → act_emb
                              │
                              ├─ ctx_emb = emb[:, :3]  # 上下文
                              ├─ tgt_emb = emb[:, 1:]  # 目标
                              ├─ pred_emb = self.model.predict(ctx_emb, ctx_act)
                              │
                              ├─ pred_loss = MSE(pred_emb, tgt_emb)
                              ├─ sigreg_loss = SIGReg(emb.transpose(0,1))
                              └─ loss = pred_loss + 0.09 * sigreg_loss
```

**规划流程**：

```
AutoCostModel → WorldModelPolicy → CEMSolver
     │                │                │
     │                │                ├─ 采样 N=300 候选动作序列
     │                │                ├─ 对每个候选调用 model.get_cost()
     │                │                │    ├─ encode(goal) → goal_emb
     │                │                │    ├─ rollout(init, action_candidates)
     │                │                │    │    ├─ encode(initial) → init_emb
     │                │                │    │    ├─ 自回归展开 H=5 步
     │                │                │    │    └─ 收集所有预测嵌入
     │                │                │    └─ criterion(pred_emb, goal_emb) → cost
     │                │                ├─ 选择 top-30 elites
     │                │                └─ 更新采样分布
     │                └─ 执行最优动作序列，重新规划
```

### 8.3 关键代码量统计

| 文件 | 行数 | 核心职责 |
|---|---|---|
| jepa.py | 154 | JEPA 主类（编码、预测、展开、代价计算） |
| module.py | 285 | SIGReg + Transformer组件 + Embedder + MLP |
| train.py | 183 | 训练脚本（数据加载+模型构建+损失+Lightning） |
| eval.py | 171 | 评估脚本（环境+规划+结果） |
| utils.py | 55 | 图像预处理+标准化+checkpoint回调 |
| **总计** | **~848** | **极度精简——整个核心贡献仅需 <1000 行代码** |

### 8.4 依赖框架

- **stable-pretraining** (spt)：训练基础设施（ViT backbone、数据变换、Lightning wrapper）
- **stable-worldmodel** (swm)：评估基础设施（HDF5Dataset、World环境、CEM solver、AutoCostModel）
- **lightning**：训练循环管理
- **hydra**：配置管理
- **einops**：张量形状操作

**论文明确指出**：这两个框架（stable-pretraining + stable-worldmodel）将仓库缩减到仅核心贡献——模型架构和训练目标。

---

## 9. 关键创新与贡献总结

### 9.1 方法层面

1. **SIGReg 防坍塌**：首次在 JEPA 世界模型中使用各向同性高斯正则化，提供 Cramér-Wold 渐近收敛保证，无需 EMA/Stop-Gradient/预训练
2. **超参极简化**：从 PLDM 的 6 个可调超参降至 1 个（$\lambda$），二分搜索从 $\mathcal{O}(n^6)$ 降至 $\mathcal{O}(\log n)$
3. **端到端稳定训练**：所有参数联合优化，梯度完整回传，训练曲线平滑单调
4. **零初始化 AdaLN**：动作条件化从零影响渐进增长，避免训练初期不稳定

### 9.2 效率层面

1. **48倍规划加速**：ViT-Tiny (192维CLS) vs DINOv2-ViT-G (~260K tokens)
2. **单 GPU 训练**：~15M 参数，数小时完成
3. **极简代码**：核心贡献 <1000 行 Python

### 9.3 理解层面

1. **潜在空间编码物理结构**：线性探针可恢复位置信息，非线性探针进一步恢复角度
2. **违反预期检测**：可靠检测物理不可能事件（瞬移），区分物理违反与视觉变化
3. **时间直线化涌现**：SIGReg 不约束时间维度，但潜在轨迹自然趋于直线——一种有用的隐式偏好

---

## 10. 与相关工作的对比

### 10.1 JEPA 方法谱系

| 方法 | 训练范式 | 防坍塌策略 | 损失项数 | 可调超参 | 端到端 | 编码器 |
|---|---|---|---|---|---|---|
| **I-JEPA** | SSL | EMA + SG | 1 | 0 (EMA固定) | 否 | ViT |
| **V-JEPA** | SSL | EMA + SG | 1 | 0 | 否 | ViT |
| **DINO-WM** | 世界模型 | 冻结预训练 | 1 | 0 | **否** | DINOv2-ViT-G (冻结) |
| **PLDM** | 世界模型 | VICReg 7项 | 7 | **6** | 是 | ViT (联合训练) |
| **LeJEPA** | SSL | SIGReg | 2 | 1 | 是 | ViT |
| **LeWM** | 世界模型 | **SIGReg** | **2** | **1** | **是** | **ViT-Tiny (联合训练)** |

### 10.2 规划性能对比

| 环境 | LeWM | PLDM | DINO-WM | GCBC | GCIQL | GCIVL |
|---|---|---|---|---|---|---|
| **PushT SR↑** | **96.0±2.83** | 78.0±5.0 | 92.0±1.63 | — | — | — |
| **PushT (DINO-WM+proprio)** | **96.0** (pixels-only) | — | ~92 | — | — | — |
| **Reacher** | 竞争力 | 较弱 | 竞争力 | — | — | — |
| **OGBench-Cube** | 竞争力 | 较弱 | 略优 | — | — | — |
| **TwoRoom** | 较弱 | 优 | 优 | — | — | — |

**LeWM 在 PushT 上比 PLDM 高 18% 成功率**，甚至超过带 proprio 的 DINO-WM。

### 10.3 与 PointWorld 的对比

| 特性 | LeWM | PointWorld |
|---|---|---|
| 表示空间 | 2D潜在空间（192维CLS） | 3D点流空间 |
| 输入模态 | 像素 | 像素+深度+机器人URDF |
| 编码器 | ViT-Tiny (~5M) | DINOv3-ViT-L16 (冻结) |
| 预测目标 | 下一帧嵌入 | 3D点流（位置+速度） |
| 不确定性 | 无（SIGReg隐式） | 异方差不确定性（per-point log_var） |
| 规划方法 | CEM-MPC (潜在空间) | 未见公开规划代码 |
| 数据格式 | HDF5 (简单) | WebDataset tar shards (复杂) |
| 参数量 | ~15M | ~数百M (PTv3-Large) |
| 训练资源 | 单GPU数小时 | 多GPU长时间 |
| 核心创新 | SIGReg防坍塌 | 3D点流+FiLM+异方差不确定性 |
| 防坍塌 | SIGReg正则化 | 不需要（3D表示天然不坍塌） |

---

## 11. 局限性、未来方向与反思

### 11.1 论文承认的局限

1. **短视界规划**：当前潜在世界模型局限于短视界推理。层次化世界建模是解决长视界推理的方向
2. **SIGReg 在低复杂度环境的问题**：TwoRoom 环境中 LeWM 表现较差——低内在维度环境在高维潜在空间中匹配各向同性高斯先验困难
3. **依赖动作标签**：需要动作标注来预测未来状态，获取成本高。逆向动力学建模可能减少对显式动作标注的需求
4. **数据多样性要求**：需要充分覆盖环境动力学的高质量离线数据集

### 11.2 额外观察与反思

1. **ViT-Tiny 的表示能力限制**：~5M 参数的 ViT-Tiny 在视觉复杂环境（如 3D OGBench-Cube）中编码器训练更困难——DINO-WM 在该环境略优可能源于 DINOv2 的丰富视觉先验
2. **Frame-skip 的权衡**：frame_skip=5 使计算高效但损失帧间精细动力学信息——不适合需要精细时序控制的任务
3. **CEM 的局限**：CEM 在高维动作空间中受维度诅咒影响，可能不适合多自由度机器人
4. **重建损失有害的启示**：论文消融实验表明加入重建损失反而降低性能（96→86 SR）——JEPA 预测目标已捕获规划所需信息，重建损失鼓励编码不相关视觉细节
5. **Dropout 的关键性**：预测器 dropout=0.1 从 78→96 SR（vs dropout=0 的 78 SR）——适度 dropout 是必要的正则化
6. **SIGReg 的隐式时间偏好**：时间直线化作为涌现现象而非显式设计，这提示 SIGReg 的 step-wise 应用可能无意中创造了对时间结构有益的隐式约束

### 11.3 未来研究方向

1. **层次化世界模型**：高层次规划 + 低层次执行的分层架构
2. **大规模视频预训练**：在多样化自然视频上预训练编码器，减少对领域特定数据的依赖
3. **逆向动力学建模**：学习动作表示而非依赖显式动作标注
4. **SIGReg 自适应**：根据环境复杂度调整高斯先验的维度或形状
5. **替代编码器架构**：论文已验证 ResNet-18 可达竞争力（94 vs 96 SR），未来可探索 ConvNeXt、Swin Transformer 等

---

## 12. 参数参考表

### 12.1 模型架构参数

| 参数 | 值 | 来源 |
|---|---|---|
| encoder_scale | tiny | config/train/lewm.yaml |
| patch_size | 14 | config/train/lewm.yaml |
| img_size | 224 | config/train/lewm.yaml |
| ViT hidden_dim | 192 | ViT-Tiny default |
| embed_dim | 192 | config/train/lewm.yaml (wm.embed_dim) |
| predictor depth | 6 | config/train/lewm.yaml |
| predictor heads | 16 | config/train/lewm.yaml |
| predictor dim_head | 64 | config/train/lewm.yaml |
| predictor mlp_dim | 2048 | config/train/lewm.yaml |
| predictor dropout | 0.1 | config/train/lewm.yaml |
| AdaLN modulation | SiLU → Linear(dim, 6*dim) | module.py:98-103 |
| AdaLN init | weight=0, bias=0 | module.py:102-103 |
| history_size | 3 | config/train/lewm.yaml |
| num_preds | 1 | config/train/lewm.yaml |
| action input_dim | frameskip × action_dim | train.py:92 |
| Embedder mlp_scale | 4 | module.py:196 default |
| Projector hidden_dim | 2048 | train.py:108 |
| Projector norm | BatchNorm1d | train.py:109 |
| SIGReg knots | 17 | config/train/lewm.yaml |
| SIGReg num_proj | 1024 | config/train/lewm.yaml |
| SIGReg λ (weight) | 0.09 | config/train/lewm.yaml |
| 总参数量 | ~15M | 论文正文 |

### 12.2 训练超参

| 参数 | 值 | 来源 |
|---|---|---|
| batch_size | 128 | config/train/lewm.yaml |
| max_epochs | 100 | config/train/lewm.yaml |
| lr | 5e-5 | config/train/lewm.yaml |
| optimizer | AdamW | config/train/lewm.yaml |
| weight_decay | 1e-3 | config/train/lewm.yaml |
| gradient_clip | 1.0 | config/train/lewm.yaml |
| precision | bf16 | config/train/lewm.yaml |
| scheduler | LinearWarmupCosineAnnealingLR | train.py:132 |
| train_split | 0.9 | config/train/lewm.yaml |
| seed | 3072 | config/train/lewm.yaml |
| frameskip | 5 | 各 data config |
| num_workers | 6 | config/train/lewm.yaml |

### 12.3 规划超参

| 参数 | PushT值 | 其他环境值 | 来源 |
|---|---|---|---|
| CEM num_samples | 300 | 300 | config/eval/solver/cem.yaml |
| CEM n_steps | 30 | ~10 | 论文 App.D |
| CEM topk | 30 | 30 | config/eval/solver/cem.yaml |
| CEM var_scale | 1.0 | 1.0 | config/eval/solver/cem.yaml |
| horizon | 5 | 5 | 各 eval config |
| receding_horizon | 5 | 5 | 各 eval config |
| action_block | 5 | 5 | 各 eval config (= frameskip) |
| eval_num_eval | 50 | 50 | 各 eval config |
| goal_offset_steps | 25 | 25 | 各 eval config |
| eval_budget | 50 | 50 | 各 eval config |
| img_size (eval) | 224 | 224 | 各 eval config |

---

## 附录：论文核心伪代码

```python
# 论文 Alg.5 — LeWM 训练伪代码（源码验证版）
def LeWorldModel(obs, actions, lambd=0.09):
    """
    obs: (B, T, C, H, W) 原始像素序列
    actions: (B, T, A) 动作序列
    lambd: SIGReg 损失权重
    """
    emb = encoder(obs)           # (B, T, D) — ViT + Projector
    act_emb = action_encoder(actions)  # (B, T, D)

    # 上下文 + 预测
    ctx_emb = emb[:, :3]         # 最近3帧嵌入
    ctx_act = act_emb[:, :3]     # 最近3帧动作嵌入
    pred_emb = predictor(ctx_emb, ctx_act)  # 自回归预测
    pred_emb = pred_proj(pred_emb)          # 预测投影

    tgt_emb = emb[:, 1:]         # 目标嵌入

    # 下一嵌入预测损失
    pred_loss = F.mse_loss(pred_emb, tgt_emb)

    # SIGReg 防坍塌正则化（每个时间步独立）
    sigreg_loss = SIGReg(emb.transpose(0, 1))  # (T, B, D)

    return pred_loss + lambd * sigreg_loss
```

---

> **报告撰写说明**: 本报告基于论文原文（arxiv HTML版本全文提取）与源码（github.com/lucas-maes/le-wm 全部核心文件逐行阅读）交叉验证完成。所有关键结论均标注 `[源码验证]` 或明确标注论文出处。源码总计 ~848 行，极度精简，与论文"仅核心贡献"的声明完全吻合。