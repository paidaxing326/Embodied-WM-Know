# γ-World (Gamma-World): NVIDIA 多智能体生成式世界模型深度解读

> **论文原文**: Gamma-World: Generative Multi-Agent World Modeling Beyond Two Players
> **arxiv**: https://arxiv.org/abs/2605.28816
> **项目页**: https://research.nvidia.com/labs/sil/projects/gamma-world/
> **开源代码**: https://github.com/nv-tlabs/Gamma-World （**目前仅 README + assets，代码尚未发布**）
> **宣发微信公众号文章**: https://mp.weixin.qq.com/s/KNZbAE4IgihPUp8_5k8JJg
> **发布时间**: 2026-05-28
> **作者团队**: Fangfu Liu, Kai He, Tianchang Shen, Tianshi Cao, Sanja Fidler, Yueqi Duan, Jun Gao, Igor Gilitschenski, Zian Wang, Xuanchi Ren
> **机构**: NVIDIA + Tsinghua University + University of Toronto + Vector Institute
> **许可证**: Apache-2.0（计划，待代码发布时确认）

---

## 0. TL;DR

1. **γ-World 解决的是"多智能体共享世界生成"问题**——之前的交互式世界模型（Genie、Oasis、Vid2World 等）几乎全是单智能体设定，γ-World 首次把世界模型扩展到 N 个独立可控、置换对称的 agent 共享同一虚拟环境。
2. **两个核心创新**：
   - **Simplex Rotary Agent Encoding (SRAE)**：把 agent 编码为单纯形（regular simplex）顶点的旋转角，参数为 0、所有 agent 等距、天然置换对称——这是它"训 2 人推 4 人"零样本泛化的根本原因
   - **Sparse Hub Attention (SHA)**：用少量可学习 hub token 中转跨 agent 通信，把交互复杂度从 O(N²) 降到 O(N)
3. **效果**：基于 Wan2.2-5B 蒸馏的因果学生模型实现 24 FPS 实时流式推理，从 2 智能体训练数据零样本扩展到 4 智能体共享世界，覆盖虚拟游戏 + 真实多机器人协作两类场景。

---

## 一、论文定位与动机

### 1.1 要解决的核心问题

**单智能体世界模型已经成熟**——Genie、Genie-2、Oasis、Vid2World、Lyra 等都展示了"用户输入动作 → 模型生成下一帧"的闭环。但真实世界本质上是**多智能体共享空间**：

- 多人游戏：多个玩家在同一个 Minecraft/CS 地图中同时操作
- 多机器人协作：仓库 AGV 群、双臂协同、人机共融
- 真实社会场景：路上多个司机、街上多个行人

把单智能体世界模型简单堆叠（每人一个独立模型）会立刻崩溃：**世界状态不一致**——A 玩家眼中的 B 玩家位置和 B 玩家眼中的自己位置无法对齐。

γ-World 的目标：**一个模型，同时生成 N 个 agent 的视角，保持共享世界一致性，且对 N 可扩展**。

### 1.2 三条硬约束

任何"原生多智能体"世界模型必须满足：

| 约束 | 含义 | 朴素方案的失败 |
|------|------|---------------|
| **独立可控 (Independently Controllable)** | 每个 agent 接收自己的动作，不被他人替代 | 把动作拼接成大向量会丢失 agent 身份 |
| **置换对称 (Permutation-Symmetric)** | 交换 agent 1 和 agent 2 的输入，输出对应交换 | 给每个 agent 学一个 ID embedding 会破坏对称性，且固定槽位数量 |
| **可扩展推理 (Scalable Inference)** | N 增加时计算量不爆炸 | 全连接跨 agent 注意力是 O(N²) |

γ-World 的两个创新刚好对应解决这三条约束：SRAE 解决前两条，SHA 解决第三条。

### 1.3 在世界模型谱系中的位置

```
单智能体世界模型路线
├─ Genie / Genie-2 (DeepMind)        — 单智能体，2D 平台游戏
├─ Oasis (Decart)                    — 单智能体，Minecraft，实时
├─ Vid2World                         — 视频扩散转世界模型
├─ Lyra / Lyra2 (NVIDIA SIL)        — 可探索 3D 世界
└─ Wan2.2-VA / GigaWorld-Policy     — 单 agent VLA + WAM

多智能体路线
└─ γ-World (NVIDIA SIL, 2026-05)    — 首个原生多智能体生成式世界模型
```

γ-World 是 NVIDIA Spatial Intelligence Lab (SIL) 继 Lyra 之后又一力作，与 Lyra 的"3D 探索"形成互补：Lyra 解决"一个 agent 探索丰富 3D 空间"，γ-World 解决"多个 agent 共享同一空间"。

---

## 二、核心方法

### 2.1 整体架构

```
输入 (per agent i, ∀ i ∈ {1,...,N}):
  • 历史观测帧 o^i_{1:t}      （每个 agent 的第一人称视角）
  • 历史动作    a^i_{1:t}      （键盘/手柄/关节指令）
  ↓
共享视觉编码器 (frozen Wan2.2 VAE)
  → 每个 agent 的视觉 token 流 V^i
共享动作编码器 (lightweight MLP)
  → 每个 agent 的动作 token 流 A^i
  ↓
 Multi-Agent DiT (causal)
 ├─ 时间维度: 3D RoPE (T, H, W) — 帧/空间位置编码
 ├─ Agent 维度: Simplex Rotary Agent Encoding — 关键创新 1
 └─ 跨 agent 通信: Sparse Hub Attention — 关键创新 2
  ↓
输出: 多 agent 同步未来帧 ô^i_{t+1:T}
  ↓
推理: Block-causal student + KV cache → 24 FPS streaming
```

**底座选择**：基于 Wan2.2-5B 视频扩散模型微调（与 LingBot-VA、GigaWorld-Policy 同源）。

### 2.2 Simplex Rotary Agent Encoding (SRAE) ★

这是论文最优雅的部分。

#### 2.2.1 问题：如何给 agent 一个身份？

朴素思路有三种，全都有缺陷：

| 方案 | 问题 |
|------|------|
| 学习 per-agent embedding `e_i` | 破坏置换对称，且固定 agent 数量 |
| 给 agent 标量索引 (1, 2, 3, ...) | 引入伪序关系，agent 不再等价 |
| 完全不区分 agent | 失去 independently controllable 性质 |

#### 2.2.2 SRAE 的核心思路

**把 N 个 agent 放在 N-1 维空间中正单纯形（regular simplex）的顶点上**，作为旋转角度。

回顾正单纯形的关键性质：
- 2 个顶点构成 1D 中的两个点（线段）
- 3 个顶点构成 2D 中的等边三角形
- 4 个顶点构成 3D 中的正四面体
- N 个顶点 ⇒ 任意两点对距离相等

具体构造（论文 Appendix A 给出严格证明）：

第 i 个 agent 的方向向量为：

$$
v_i = e_i - \frac{1}{N}\sum_{j=1}^{N} e_j
$$

其中 `e_i` 是 N 维标准基。然后归一化并嵌入到旋转角。论文证明：

$$
\|v_i - v_j\|^2 = 2 \quad \forall i \neq j
$$

**所有 agent 两两距离相同 = 完美置换对称**。

#### 2.2.3 与 RoPE 的统一

3D RoPE 已经为视频提供了 (T, H, W) 三维位置编码。SRAE 把 agent 维度作为**第四维**叠加进 RoPE：

$$
\text{Token}^i_{t,h,w} \xrightarrow{\text{RoPE}} R_T(t) R_H(h) R_W(w) R_A(v_i) \cdot x
$$

其中 `R_A(v_i)` 是基于单纯形顶点 `v_i` 的旋转矩阵。**参数量为 0**（纯几何构造）。

#### 2.2.4 为什么能"训 2 人推 4 人"？

关键在于：**单纯形几何对所有 N 都自洽**。训练时使用 N=2（线段两端点），推理时换成 N=4（正四面体顶点），模型的 attention 计算公式不变——因为 attention 只关心**相对相位差**，而单纯形性质保证任意两个顶点的相对几何关系都是等距的。

这个性质非常类似 RoPE 在序列长度外推上的优势——SRAE 把外推从"序列长度"扩展到了"agent 数量"。

### 2.3 Sparse Hub Attention (SHA)

#### 2.3.1 朴素方案：Dense Cross-Agent Attention

每个 agent 的 token 与所有其他 agent 的所有 token 全连接 attention：

$$
\text{Cost} = O(N^2 \cdot L^2)
$$

其中 N 是 agent 数，L 是每个 agent 的 token 数。N=4 时已经是 16× 复杂度，N=8 几乎不可承受。

#### 2.3.2 SHA 的设计

引入 **K 个可学习的 hub token**（论文中 K 远小于 N×L）。Attention 模式：

```
Agent_i's token  ←→  Agent_i's other tokens  (intra-agent self-attn)
Agent_i's token  ←→  Hub tokens              (agent → hub)
Hub tokens       ←→  Hub tokens              (hub self-attn, optional)
Hub tokens       ←→  All agent tokens        (hub → broadcast)
```

效果上：
- 每个 agent 只需 attend 自己的 stream + K 个 hub token → O(L²) + O(L·K)
- 跨 agent 信息**通过 hub 中转**：agent_i 的信息 → hub → agent_j
- 总复杂度：**O(N·(L² + L·K)) = O(N) （视为 K, L 常数）**

#### 2.3.3 直觉理解

这个设计很像**信息瓶颈 + 公告板**：
- Hub token 是"公告板"，所有 agent 把摘要写上去
- 每个 agent 只读公告板和自己的笔记，不直接 peek 其他 agent
- 学习目标驱动 hub 提取**对所有 agent 都有用的全局世界状态**（物理一致性、共享物体位置等）

类比已有工作：
- Set Transformer 的 inducing points
- Perceiver IO 的 latent array
- Cross-attention 的 query bottleneck

但 γ-World 是**首次将这个范式应用到多智能体世界模型**。

#### 2.3.4 实测加速

论文 Figure（`assets/sparse-hub-timing.png`）显示：随 N 从 2 增加到 8+，dense attention 延迟和 FLOPs 二次爆炸，SHA 保持线性增长。在 N=4 时 SHA 已经显著快于 dense；N=8 时差距进一步扩大。

### 2.4 Block-Causal 流式推理

#### 2.4.1 训练 vs 推理的不对称

- **训练**：full-context bidirectional diffusion（教师），看到完整的过去 + 未来 token，质量高但不能流式
- **推理**：需要因果 + 实时——给定当前帧和动作立刻产出下一帧

#### 2.4.2 蒸馏方案

1. 先训练 full-context teacher（双向 attention，看完整 chunk）
2. 蒸馏到 **block-causal student**：
   - 时间维度上分块（block），块内双向、块间因果
   - 每个新块只能看到过去块的 KV cache
   - 与 LLM 推理类似的 prefix caching 思想
3. 推理时：
   - 视觉 token KV cache + Hub token KV cache 都缓存
   - 每个新块只需计算自己 + cross-attend cache
   - 实测 **24 FPS** 实时流式

这个流程和 LingBot-World、Vid2World 的 "video diffusion → autoregressive" 蒸馏思路一脉相承，但 γ-World 加上了 **multi-agent KV cache 维度**——hub token 作为公共缓存，所有 agent 共享。

---

## 三、训练设置

### 3.1 数据

论文未公开完整数据细节，能从论文和项目页推断：

| 数据类型 | 规模 | 来源 |
|---------|------|------|
| 多人游戏视频 + 动作日志 | 大规模（具体未公开） | 内部采集 + 公开数据集 |
| 真实多机器人协作 | 较小规模 | NVIDIA 内部机器人实验 |
| 训练 agent 数 | **N=2** | 仅用 2 人数据训练 |
| 测试 agent 数 | **N=2 ~ 4** | 4 人为零样本泛化 |

**关键信息**：γ-World 完全用 2 智能体数据训练，4 智能体场景是**完全零样本**——这正是 SRAE 单纯形几何对称性的胜利。

### 3.2 训练流程

```
Stage 1: Multi-Agent DiT Pretraining
  • 底座：Wan2.2-5B 视频扩散
  • 输入：N=2 同步多视角观测 + 动作
  • 目标：full-context flow matching loss
  • 时长：未公开（推测多卡数百 H100 小时级别）

Stage 2: Block-Causal Distillation
  • 教师：Stage 1 的 full-context 模型
  • 学生：相同架构但 attention mask 改为 block-causal
  • 损失：teacher-student feature distillation + diffusion loss
  • 输出：可 KV cache 的流式模型

Stage 3 (Inference): Streaming Rollout
  • 实时接收每个 agent 的动作
  • 维护视觉 KV cache + hub KV cache
  • 输出：24 FPS 多 agent 同步未来帧
```

### 3.3 损失函数

主体是 flow matching 损失（与 Wan2.2 一致）：

$$
\mathcal{L} = \mathbb{E}_{x, t, \epsilon} \| v_\theta(x_t, t, c) - (x_1 - x_0) \|^2
$$

其中 `c` 包含历史观测、所有 agent 动作、agent 身份编码（SRAE）。

蒸馏阶段加上学生-教师特征对齐损失。

---

## 四、实验结果

### 4.1 主要展示场景

论文与项目页给出三大类定性结果：

#### A. 两智能体交互
- 多人虚拟游戏环境（类 Minecraft 设定）
- 两个独立可控的 agent，各自第一人称视角
- 验证：互相可见、动作互不干扰、世界状态一致

#### B. 四智能体零样本泛化
- **训练时只见过 N=2**
- 推理时直接处理 N=4，无需任何额外训练或微调
- 这是论文最重要的卖点——证明 SRAE 对称性的 generalization 能力
- 项目页 `four-agent-generalization.png` 展示 4 个第一人称视角同时连贯

#### C. 真实世界多机器人协作
- 从虚拟游戏延伸到**真实机器人**
- 多个机械臂/AGV 在共享空间中协作
- 项目页 `robotics-coordination.png` 展示真实机器人场景
- 这一部分把 γ-World 从"游戏世界模型"提升到"通用具身世界模型"

### 4.2 量化指标（推断）

论文给出的核心数字：

| 指标 | 数值 | 备注 |
|------|------|------|
| 推理帧率 | **24 FPS** | 蒸馏后的因果学生 |
| 训练 agent 数 | 2 | |
| 零样本扩展 agent 数 | 4 | 无需重训 |
| 跨 agent 复杂度 | O(N²) → **O(N)** | SHA 带来的 |
| 模型规模 | 基于 Wan2.2-**5B** | |

### 4.3 消融实验（Ablation）

论文 Appendix（Page 14-16）给出核心消融：

| 消融项 | 实验设置 | 结论 |
|--------|---------|------|
| Agent encoding | SRAE vs Learned Embedding vs Index | SRAE 在 N=4 泛化上显著优于其他 |
| Cross-agent attention | SHA vs Dense vs No-cross | SHA 接近 dense 质量但 4× 更快 |
| Number of hub tokens | K = 4, 16, 64 | 中等 K 即可，过多无收益 |
| Block size | 1, 4, 16 frame block | 4 帧块在质量与延迟间最优 |

---

## 五、与同类工作对比

| 工作 | 多智能体支持 | 置换对称 | 跨 agent 复杂度 | 实时 | 真实机器人 |
|------|-------------|---------|----------------|------|-----------|
| **γ-World** | **✅ 原生多 agent** | **✅ Simplex** | **O(N)** | **24 FPS** | ✅ |
| Genie / Genie-2 | ❌ 单 agent | N/A | N/A | ✅ | ❌ |
| Oasis | ❌ 单 agent | N/A | N/A | ✅ | ❌ |
| Vid2World | ❌ 单 agent | N/A | N/A | ⚠️ | ⚠️ |
| Lyra / Lyra2 | ❌ 单 agent (3D 探索) | N/A | N/A | ✅ | ❌ |
| Cosmos (NVIDIA) | ⚠️ 通过条件支持多对象 | ❌ | O(N²) | ❌ | ✅ |
| GigaWorld-Policy | ❌ 单 agent (action-centered) | N/A | N/A | ⚠️ | ✅ |
| LingBot-World | ❌ 单 agent | N/A | N/A | ⚠️ | ✅ |

**γ-World 的独特定位**：在世界模型这个赛道里，**第一个把"多智能体共享空间"作为一等公民**的模型，且同时具备置换对称、O(N) 扩展、实时、跨域（虚拟+真实）四个特性。

---

## 六、开源程度深度分析

### 6.1 当前公开状态（截至 2026-05-29）

| 维度 | 状态 | 详情 |
|------|------|------|
| 论文原文 (PDF) | ✅ 公开 | arxiv 2605.28816 |
| 项目主页 | ✅ 公开 | NVIDIA SIL 项目页 |
| 演示视频 | ✅ 公开 | 项目页 + GitHub assets |
| 定性结果图 | ✅ 公开 | 4 张大图 + teaser |
| 方法概览图 | ✅ 公开 | `assets/method-overview.png` |
| **推理代码** | ❌ **未发布** | README 标注 "Coming Soon" |
| **蒸馏 streaming checkpoint** | ❌ **未发布** | README 标注 "Coming Soon" |
| **训练代码** | ❌ **未发布** | README 标注 "Planned, future update" |
| **数据预处理工具** | ❌ **未发布** | README 标注 "Planned, future update" |
| **预训练权重** | ❌ **未发布** | 无 |
| **训练数据集** | ❌ **未公开** | 无 |
| 许可证 | ⚠️ 计划 Apache-2.0 | "Final license terms will be confirmed at the code release" |

### 6.2 Repo 实际内容核查

```
/Users/zhc/Desktop/项目库/Gamma-World/
├── README.md          # 项目介绍
└── assets/            # 论文 PDF + 演示图/视频
    ├── gamma-world.pdf       (4.8MB 论文)
    ├── gamma-world-video.mp4 (10.4MB 演示视频)
    ├── teaser.png
    ├── two-agent-interaction.png
    ├── four-agent-generalization.png
    ├── robotics-coordination.png
    ├── method-overview.png
    └── sparse-hub-timing.png
```

**结论**：当前 repo 是一个"论文+宣传 repo"，**完全没有任何源代码**。

### 6.3 开源程度评分

```
开源程度评分: ★☆☆☆☆ (1/5) — 当前阶段

✅ 已公开:
  - 论文 PDF（方法描述完整）
  - 演示视频和定性结果

❌ 未公开:
  - 任何代码（训练/推理/数据处理）
  - 任何权重（教师/学生/蒸馏）
  - 训练数据
  - 完整训练配方
  - 评估脚本

待发布后可能升至:
  ★★★☆☆ (3/5) — 如果只发推理代码 + 蒸馏权重
  ★★★★☆ (4/5) — 如果加上训练代码
  ★★★★★ (5/5) — 如果加上完整训练数据（极不可能）
```

### 6.4 复现可行性

即使代码不发布，论文方法描述足够清晰，理论上可以基于已开源的 Wan2.2-5B 复现：

| 复现要素 | 难度 | 备注 |
|---------|------|------|
| SRAE 实现 | 低 | 纯数学构造，可直接基于现有 RoPE 代码扩展 |
| SHA 实现 | 中 | Hub token 设计需要 attention mask 工程 |
| Block-causal 蒸馏 | 高 | 需要先训完 teacher 才能蒸馏 |
| 多 agent 数据采集 | **极高** | 同步多视角 + 动作日志的多人游戏数据采集是工程难点 |
| 训练算力 | 高 | 5B 模型 + 多视角输入，估计需要 64+ H100 数周 |

实际上 **数据是最大壁垒**——同步多人游戏数据需要修改游戏引擎记录每个玩家的视角和动作。

---

## 七、技术深挖

### 7.1 SRAE 为什么是"参数为 0"？

很多人会问：既然单纯形顶点向量是固定的几何构造，能不能用 learned positional embedding 替代？

不能，原因有三：

1. **置换对称性是几何属性**：单纯形顶点之间的等距性是几何事实，learned embedding 不可能精确等距
2. **泛化到任意 N**：N=2 训练得到的 learned embedding 不可能正确处理 N=4——而单纯形从 2 顶点扩展到 4 顶点是几何上自然的
3. **参数 0 = 无过拟合风险**：不需要任何额外训练数据来"学" agent 编码

这本质上是 RoPE 哲学的延续：**位置/身份信息用几何构造，不用学习**。

### 7.2 SHA 的 Hub Token 学什么？

这是论文没有明说但值得思考的问题。从训练动力学推断：

- Hub token 是 cross-agent 信息的瓶颈，必须编码"对所有 agent 都有用的全局信息"
- 训练目标驱动 hub 提取**共享世界状态**：
  - 共享物体的位置和状态
  - 物理约束（碰撞、阻挡）
  - 高层场景语义（这是个客厅、这有把椅子）
- agent 自己的私有信息（自己的视角、自己的动作）留在自己的 stream

这种**信息分工**让 hub 自然成为"世界状态摘要"。如果未来开放可视化，hub token 的 attention map 应该会高亮所有 agent 共同看到的物体。

### 7.3 与 GigaWorld-Policy / LingBot-VA 的关系

三者都基于 Wan2.2-5B，但解决的问题完全不同：

| 模型 | 解决什么问题 | Wan2.2 改造方式 |
|------|------------|----------------|
| GigaWorld-Policy | action-centered 单 agent WAM | 增加 action embedder + action head |
| LingBot-VA | 单 agent 因果 video+action 联合建模 | 三维 ID mask + 共享 backbone |
| **γ-World** | **多 agent 共享世界** | **SRAE + SHA 两个新机制** |

γ-World 的贡献是正交的——它的 SRAE 和 SHA 完全可以叠加到 GigaWorld-Policy 或 LingBot-VA 上得到"多智能体 VLA"。

### 7.4 局限性

论文 Appendix 中作者诚实列出的局限：

1. **训练数据规模有限**：多人游戏数据收集成本高，可能限制了泛化广度
2. **动作空间假设**：假设所有 agent 共享同一动作空间（同质 agent），异质 agent（一个机器人 + 一个无人机）尚未验证
3. **长时序一致性**：超长 rollout（数分钟）的世界一致性可能仍有 drift
4. **物理保真度**：真实机器人场景下的物理交互精度未充分验证
5. **当前 N 上限**：虽然方法是 O(N)，但实际 KV cache 显存随 N 线性增长，N=10+ 在消费级 GPU 上仍困难

---

## 八、对世界模型与具身路线的启示

### 8.1 多智能体是世界模型的下一个 frontier

γ-World 之前，世界模型几乎全在做"单智能体 + 丰富环境"，γ-World 把"多智能体共享空间"立成新的研究方向。这与现实需求高度匹配：

- 智能驾驶：自车 + 周围 N 辆车 + 行人（典型多 agent）
- 仓库机器人：N 个 AGV 协同（典型多 agent）
- 人机共融：机器人 + 人类共享空间（典型多 agent）

预期 2026-2027 年会有大量后续工作填补：
- 异质多 agent（不同机器人类型混合）
- 大规模多 agent（N > 10）
- 多 agent + 长视频生成
- 多 agent + 真实世界 fine-tuning

### 8.2 几何对称性 > 学习身份

SRAE 的胜利重申了一个深层原理：**已知的对称性应该用几何构造编码，而不是从数据中学习**。
- 平移对称 → CNN
- 旋转对称 → 等变网络
- 序列位置 → RoPE
- agent 置换对称 → SRAE

这条原则适用于任何具有置换对称性的多实体系统：
- 多目标检测中的 query
- 多人姿态估计
- 多机器人编队

### 8.3 信息瓶颈型注意力是 O(N) 的标配

从 Set Transformer → Perceiver → Mamba → SHA，趋势越来越明显：**面对大 N，用少量 latent 中转通信**。γ-World 的 SHA 是这个范式在多智能体世界模型中的具体落地。

### 8.4 对国内具身路线的参考价值

国内 GigaWorld、LingBot、千寻、智元、银河 等团队大部分还在做单 agent VLA。γ-World 提供了一个清晰的下一步：

- 短期：把 SRAE/SHA 集成到现有单 agent VLA，得到多机器人协作 VLA
- 中期：构建多机器人共享数据集（这是真正的稀缺资源）
- 长期：训练原生多 agent 世界模型，作为大规模仿真训练场

---

## 九、关键参考链接

| 资源 | 链接 |
|------|------|
| 论文 (arxiv) | https://arxiv.org/abs/2605.28816 |
| 论文 PDF (项目下载) | https://arxiv.org/pdf/2605.28816 |
| 项目主页 | https://research.nvidia.com/labs/sil/projects/gamma-world/ |
| GitHub (待发布代码) | https://github.com/nv-tlabs/Gamma-World |
| HuggingFace 论文页 | https://huggingface.co/papers/2605.28816 |
| 第一作者主页 (Fangfu Liu) | https://liuff19.github.io/ |
| NVIDIA SIL Lab | https://research.nvidia.com/labs/sil/ |
| 相关：Lyra2 (同 Lab) | https://research.nvidia.com/labs/sil/projects/lyra2/ |
| 相关：Cosmos | https://arxiv.org/abs/2501.03575 |

---

## 十、附录：核心公式速查

### Simplex Rotary Agent Encoding

构造 N 个等距方向向量（agent i 的身份）：

$$
v_i = e_i - \frac{1}{N}\sum_{j=1}^{N} e_j, \quad \|v_i - v_j\|^2 = 2
$$

嵌入到 RoPE 第四维：

$$
\text{RoPE}_{4D}(x; t, h, w, i) = R_T(t) R_H(h) R_W(w) R_A(v_i) \cdot x
$$

### Sparse Hub Attention

给定 agent token `X^i ∈ R^{L×d}`，hub token `H ∈ R^{K×d}`：

$$
\begin{aligned}
\tilde{X}^i &= \text{SelfAttn}(X^i) + \text{CrossAttn}(X^i, H) \\
\tilde{H} &= \text{SelfAttn}(H) + \sum_i \text{CrossAttn}(H, X^i)
\end{aligned}
$$

复杂度：O(N · L²) + O(N · L · K) + O(K²)，**线性于 N**。

### Block-Causal 推理

对于第 b 个时间块：

$$
p(x_b | x_{<b}) = p_\theta(x_b | \text{KVCache}(x_{<b}), \text{HubCache}(h_{<b}), a_b^{1:N})
$$

每个块内双向 attention，块间因果，KV cache 跨块复用。

---

## 十一、总结

γ-World 用一个非常简洁的几何洞察（单纯形顶点等距）+ 一个工程友好的 attention 设计（hub token 中转），把世界模型从单智能体推进到原生多智能体。它的两个核心贡献——SRAE 和 SHA——正交、可移植、原理清晰，足以成为后续多智能体世界模型的事实标准。

**对入门者**：这是一篇"读懂了就能立刻动手"的论文——SRAE 实现是几行代码，SHA 是注意力 mask 工程。不需要等代码发布也能开始复现。

**对从业者**：γ-World 暗示了世界模型赛道的下一个增长曲线——多智能体共享空间。早入场者有显著优势。

**遗憾**：代码至今未发布，预训练权重也无消息。NVIDIA SIL Lab 历史上的"Coming Soon"通常会兑现（参考 Lyra），但时间窗口可能数月。
