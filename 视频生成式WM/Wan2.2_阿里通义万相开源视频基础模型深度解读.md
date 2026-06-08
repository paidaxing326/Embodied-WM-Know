# Wan2.2：阿里通义万相开源视频生成基座全景剖析

> **论文标题**：Wan: Open and Advanced Large-Scale Video Generative Models
> **arXiv**：2503.20314（v1：2025-03-26，v2：2025-04-19，共 60 页）
> **机构**：阿里巴巴通义万相（Wan Team, Alibaba Group）
> **Github**：https://github.com/Wan-Video/Wan2.2（Apache-2.0）
> **HuggingFace**：https://huggingface.co/Wan-AI
> **发布节奏**：Wan2.1（2025-02）→ Wan2.2（2025-07-28）→ Wan2.2-S2V（2025-08-26）→ Wan2.2-Animate（2025-09-19）
> **核心定位**：**开源视频生成事实标准**——业内（LingBot-World、LingBot-VA、X-WAM、HuMo、Helios、LightX2V、Wan2GP、DiffSynth 等）几乎所有二次开发的视频世界模型都把它当底座
>
> **本报告阅读前提**：已读完 Wan 技术报告 60 页 PDF 全文 + Wan2.2 仓库完整源码（`wan/modules/model.py` / `wan/modules/vae2_1.py` / `wan/modules/vae2_2.py` / `wan/text2video.py` / `wan/image2video.py` / `wan/textimage2video.py` / `wan/configs/*.py` 等）+ 五大模型权重卡片 + 官方 README 文档 507 行

---

## 目录

1. 项目概览与发展脉络
2. 四大核心创新（MoE / Wan-VAE / 高压缩 TI2V / 级联能力）
3. 五大模型变体逐个拆解
4. 架构与训练深度剖析（基于论文 + 源码）
5. 基础设施与性能优化
6. 评测与 SOTA 对比
7. 开源完整度审计（训练 / 数据 / 推理 / 权重 / 论文）
8. 为什么全行业都基于它搞二次开发
9. 硬件门槛与可玩性
10. 给学习者/研究者/创业者的三类建议

---

## 一、项目概览与发展脉络

### 1.1 时间线

```
2025-02-14  Wan2.1 发布（1.3B + 14B，开源史上第一次覆盖消费级 GPU 的 720P T2V）
2025-03-26  arXiv 2503.20314 v1（60 页技术报告，Wan 家族总述）
2025-04-19  v2 更新
2025-07-28  Wan2.2 发布：T2V-A14B + I2V-A14B + TI2V-5B 三件套同步开源
2025-07-28  ComfyUI / Diffusers 官方集成
2025-08-26  Wan2.2-S2V-14B（音频驱动视频生成）
2025-09-19  Wan2.2-Animate-14B（人物动画与替换）
2025-11-13  Animate 正式进入 Diffusers
```

这条发布节奏本身就很关键——**阿里把视频基座做成了像 Llama 那样的持续演进开源产品线**，不是一次性 dump 代码跑路。

### 1.2 模型总览

| 模型 | 参数量 | 任务 | 分辨率/帧率 | 权重开源 | 代码开源 |
|------|:------:|------|:-----------:|:--------:|:--------:|
| Wan2.2-T2V-A14B | 27B (14B active) | 文生视频 | 480P/720P @ 16fps | ✅ | ✅ |
| Wan2.2-I2V-A14B | 27B (14B active) | 图生视频 | 480P/720P @ 16fps | ✅ | ✅ |
| Wan2.2-TI2V-5B | 5B dense | 文/图生视频 | 720P @ **24fps** | ✅ | ✅ |
| Wan2.2-S2V-14B | 14B | 音频→视频 | 480P/720P | ✅ | ✅ |
| Wan2.2-Animate-14B | 14B | 人物动画/替换 | 720P @ 30fps | ✅ | ✅ |

**所有模型 Apache-2.0，所有权重都在 HuggingFace 和 ModelScope 双发**。这是这个项目最大的差异化——**不只开源权重，而是覆盖 T2V / I2V / TI2V / S2V / Animate 五大下游的完整能力矩阵**。

### 1.3 和 Wan2.1 的关系

Wan2.1 是 Wan 家族的底层基础，Wan2.2 在此基础上做三件事：

1. **把 14B dense 升级为 27B MoE（14B × 2 专家）** —— 总参数翻倍，激活参数不变
2. **升级训练数据规模** —— 图像 +65.6%，视频 +83.2%
3. **新增 TI2V-5B 消费级模型** —— 配 Wan2.2-VAE（64× 压缩率），4090 就能跑 720P @ 24fps

Wan2.2 的仓库 commits 非常清楚地显示：**VAE 2.1 和 VAE 2.2 在代码里共存**，VAE 2.1 给 14B MoE 用（兼容 Wan2.1 权重），VAE 2.2 给 5B 独享（高压缩）。

---

## 二、四大核心创新

论文把 Wan 的贡献列为四点，但 2.2 上真正有技术差异化的只有前三个：

### 2.1 创新一：时序-空间 3D Causal VAE（Wan-VAE）

这是 **Wan 最被低估的创新**，也是后续所有二次开发项目继承的最重要资产。

**设计挑战**：
1. 视频有时间+空间双维度，VAE 必须捕捉时空依赖
2. 高维视频带来内存爆炸
3. 因果时序（未来不能影响过去）必须保证，否则流式推理做不了

**Wan-VAE 的解法**（论文 Sec 4.1）：

| 维度 | Wan2.1-VAE | Wan2.2-VAE（新） |
|------|:----------:|:----------------:|
| 压缩比 | 4×8×8（= 256×） | **4×16×16（= 1024×）** |
| 潜变量通道 | 16 | **48** |
| 模型参数 | 127M | ~300M |
| 用于 | T2V/I2V/S2V/Animate-14B | **TI2V-5B 独享** |

**VAE 2.2 为什么能把压缩比提到 1024×还保持重建质量？** 三个关键技巧（论文 Sec 4.1.1，代码 `vae2_2.py` 可验证）：

1. **第一帧只做空间压缩** —— 保留图像信息（MagViT-v2 思路）
2. **所有 GroupNorm 换成 RMSNorm** —— 保证时序因果（GroupNorm 会让未来帧影响过去统计量）
3. **空间上采样层通道减半** —— 推理显存降 33%

**Feature Cache 机制**（论文 Sec 4.1.3，代码 `vae2_2.py::Resample.forward`）：

```python
# 视频分成 1 + T/4 个 chunk（与 latent 数量对齐）
# 每次 encode/decode 只处理一个 chunk
# 因果卷积的 padding 用上一 chunk 的尾部 feature 填充（"cache padding"）
# 从而做到"逐 chunk 处理任意长视频"，显存不会爆
```

这套缓存设计让 Wan-VAE 能**编码/解码任意长视频**，是流式视频生成的技术底座。LingBot-World-Fast、Helios、Streamer 等项目全都复用了这个机制。

### 2.2 创新二：MoE 双专家架构

**核心洞察**：扩散模型在不同 timestep 学的东西本质不同——
- **高噪声 step**（t 大）：画面是纯噪声，模型学的是**全局布局、物体位置、场景结构**
- **低噪声 step**（t 小）：画面已接近成品，模型学的是**高频纹理、细节打磨**

一个 dense 网络要同时做这两件事，参数利用效率低。Wan2.2 的答案是**把它们拆成两个专家**。

**架构细节**（README + 论文 Sec 4.2）：

```
Wan2.2-A14B (T2V / I2V):
├── high_noise_model    ~14B   负责扩散早期（t ≥ boundary × T）
└── low_noise_model     ~14B   负责扩散晚期（t < boundary × T）
```

- **总参数**：~27B
- **每步激活**：14B（只有一个专家在计算）
- **切换阈值**：基于 **SNR（信噪比）**，具体 `boundary = SNR_min 的一半对应的 timestep`
- **T2V boundary**：0.875（低于 0.875×1000 = 875 时切换到 low noise）
- **I2V boundary**：0.900（I2V 对细节更敏感，切换得晚一点）

**源码实现**（`wan/text2video.py::_prepare_model_for_timestep`，line 186-201）：

```python
if t.item() >= boundary:
    required_model_name = 'high_noise_model'
    offload_model_name = 'low_noise_model'
else:
    required_model_name = 'low_noise_model'
    offload_model_name = 'high_noise_model'
# 需要切换时把非激活专家 offload 到 CPU
if offload_model or self.init_on_cpu:
    if offload 的那个在 GPU 上: getattr(self, offload_model_name).to('cpu')
    if 要激活的那个在 CPU 上:  getattr(self, required_model_name).to(self.device)
return getattr(self, required_model_name)
```

这是**最朴素但最有效的 MoE 实现**——每一步只有一个 14B 模型在 GPU 上跑，推理显存几乎等同 dense 14B。

**有效性验证**（README 的 MoE 消融）：

| 配置 | 验证损失 |
|------|:-------:|
| Wan2.1（dense，无 MoE） | baseline |
| Wan2.1 + Wan2.2 High-Noise Expert | 下降 |
| Wan2.1 + Wan2.2 Low-Noise Expert | 下降 |
| **Wan2.2 (MoE 完整版)** | **最低** |

双专家分工让总能力真的上去了，不是伪提升。这个设计后来被几乎所有 Wan 衍生项目继承（LingBot-World、X-WAM 都用了双专家）。

### 2.3 创新三：高压缩 TI2V-5B

**问题**：A14B 模型再强，消费级硬件跑不起来——就算激活 14B，一个 80GB H100 也才刚好够单卡推理。要上 4090（24GB），只能走小模型路线。

**Wan 的小模型方案**：不是把 A14B 蒸馏成小号，而是**从头设计一个 5B dense 模型 + 高压缩 VAE 的组合**。

**TI2V-5B 的关键数字**（`wan_ti2v_5B.py` + 论文）：

```
dim          = 3072    # hidden dim（比 A14B 的 5120 小）
ffn_dim      = 14336   # FFN dim
num_heads    = 24
num_layers   = 30      # 30 层（A14B 是 40 层）
vae_stride   = (4, 16, 16)  # 使用 Wan2.2-VAE
patch_size   = (1, 2, 2)    # 再做 patchify
sample_fps   = 24            # 注意是 24fps，A14B 是 16fps
sample_steps = 50
frame_num    = 121           # 5 秒 × 24fps + 1
```

**总压缩率**：VAE 的 4×16×16 + patchify 的 1×2×2 = **4×32×32 = 4096×**

**5-second 720P 视频 token 数估算**：
```
121 frames × 720 × 1280 / 4096 = ~27,000 tokens
```

这样的 token 数让 5B 模型在单张 RTX 4090（24GB）上跑得动——README 明确说 "24GB VRAM (e.g., RTX 4090)" + "720P video in under 9 minutes"。

**为什么 TI2V-5B 意义重大？** 它是**第一个能在消费级显卡上跑 720P@24fps 的视频模型**。这直接开启了社区二次开发的爆发——没这个，Wan 就只能服务有 H100 集群的大厂。

### 2.4 创新四：级联下游能力（不是单一模型）

论文 Sec 5 花了 22 页讲 8 个下游任务：
- I2V（图生视频）
- Unified Video Editing（VACE）
- Text-to-Image（图像生成）
- Video Personalization（视频换人）
- Camera Motion Control（相机运动控制）
- Real-time Video Generation（实时生成，Streamer）
- Audio Generation（音频生成）
- First-last frame transformation（首尾帧转换）

这说明 Wan 不是个单纯的视频生成模型，而是**一个基础视频模型族**——每个下游任务都用 Wan 作为 backbone，但套的是不同的 adapter / 训练数据 / 蒸馏策略。这也解释了为什么后续 LingBot-World、LingBot-VA、X-WAM 这些项目都直接从 Wan 出发——Wan 已经验证了"一个基座支撑八种下游"的可行性。

---

## 三、五大模型变体拆解

### 3.1 T2V-A14B / I2V-A14B（旗舰 MoE）

**共享架构**（两份 config 几乎一致）：

```python
dim          = 5120
ffn_dim      = 13824
num_heads    = 40
num_layers   = 40      # 40 层 DiT block
patch_size   = (1, 2, 2)
vae_stride   = (4, 8, 8)      # 使用 Wan2.1-VAE
qk_norm      = True           # Query/Key RMSNorm
cross_attn_norm = True
sample_shift = 5.0 (I2V) / 12.0 (T2V)    # T2V shift 更大
sample_steps = 40                          # 两者一致
boundary     = 0.875 (T2V) / 0.900 (I2V)
sample_guide_scale = (3.0, 4.0) / (3.5, 3.5)
```

**差异点**：
- T2V 的 CFG scale 高噪声和低噪声不同（3.0 vs 4.0）—— 低噪声时更需要跟 prompt
- I2V 的 boundary 更大（0.900）—— 因为 image 已经提供了强先验，高噪声专家可以干更久

**Text encoder**：umT5-XXL（11B 参数量，bfloat16）。论文 Sec 4.2.1 说选 umT5 的三个原因：
1. 支持中英双语（Flan T5 仅英文）
2. 双向注意力 outperform 单向 LLM 在 composition 任务上
3. 同参数量下收敛更快

### 3.2 TI2V-5B（消费级代表作）

前面已详细展开，这里补一个关键细节：**TI2V-5B 是 dense 不是 MoE**。它不分 high_noise / low_noise expert，只有一份权重（`wan_ti2v_5B.py` 里根本没有 `boundary` 字段）。这是因为：
- 5B 参数量 dense 就够灵活，不需要 MoE
- MoE 在小模型上反而会因专家切换代价吃掉增益
- 目标用户是消费级硬件，进一步切换显存不值得

### 3.3 S2V-14B（音频驱动视频）

**config 里独特的字段**（`wan_s2v_14B.py`）：

```python
wav2vec                  = "wav2vec2-large-xlsr-53-english"  # 音频 encoder
transformer.enable_adain = True                              # 音频注入用 AdaIN
transformer.adain_mode   = "attn_norm"
transformer.audio_inject_layers = [0, 4, 8, 12, 16, 20, 24, 27, 30, 33, 36, 39]  # 12 层注入
transformer.audio_dim = 1024
transformer.motion_frames = 73          # motion 条件帧数
transformer.enable_framepack = True     # 启用 frame packing
transformer.framepack_drop_mode = 'padd'
```

**技术路径**：
- 用 wav2vec 提取音频特征（1024 维）
- 在 40 层 DiT 的**每 3-4 层（共 12 层）**通过 AdaIN 注入到 attention norm 里
- motion frames 作为额外的时序条件
- 使用 FramePacking 做长时序

这个设计的思想：**音频信号不是连续每帧都注入，而是选择性地在 12 个关键层注入**，避免过拟合音频噪声。

### 3.4 Animate-14B（人物动画与替换）

**独特的额外组件**：

```python
clip_checkpoint       = 'models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth'
lora_checkpoint       = 'relighting_lora.ckpt'          # 重光照 LoRA
use_face_encoder      = True
motion_encoder_dim    = 512
```

Animate 有两种模式：
- **animate**：输入 ref image + pose video + face video → 生成参考人物做目标动作
- **replace**：额外输入 background video + mask video → 替换原视频中的人物

Replace 模式还会加载 `relighting_lora.ckpt`——这是专门训练的 LoRA，让替换进去的人物和原视频的光照匹配。

### 3.5 能力矩阵速查

| 任务 | 输入 | 控制信号 | 推荐模型 |
|------|------|---------|---------|
| 文生视频 | prompt | 无 | T2V-A14B 或 TI2V-5B |
| 图生视频 | prompt + image | 无 | I2V-A14B 或 TI2V-5B |
| 音频驱动视频 | prompt + image + audio | 音频波形 | S2V-14B |
| 人物动画 | ref image + pose/face video | 骨骼+面部 | Animate-14B |
| 人物替换 | ref image + src video + mask | 骨骼+面部+背景+mask | Animate-14B |

---

## 四、架构与训练深度剖析

### 4.1 DiT Transformer Block（`wan/modules/model.py::WanAttentionBlock`）

Wan 的 DiT block 和标准 DiT（Peebles & Xie 2023）有几个重要差异：

```python
class WanAttentionBlock(nn.Module):
    def __init__(self, dim, ffn_dim, num_heads, ...):
        self.norm1 = WanLayerNorm(dim, eps)        # 自注意力前 norm
        self.self_attn = WanSelfAttention(...)     # self-attn + 3D RoPE
        self.norm3 = WanLayerNorm(..., elementwise_affine=True)  # cross-attn 前 norm
        self.cross_attn = WanCrossAttention(...)   # text 条件注入
        self.norm2 = WanLayerNorm(dim, eps)        # FFN 前 norm
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim)
        )
        # 每个 block 独立学 6 组 modulation（timestep 条件）
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
```

**关键设计决策**（论文 Sec 4.2.1）：
1. **共享的 MLP + block-wise bias**：`time_embedding` + `time_projection` 两层 MLP 是**所有 block 共享**的，但每个 block 有自己的 `modulation` bias。论文说"减少 25% 参数量、同参数量下性能还更好"。
2. **cross-attention 而非 in-context concat**：文本条件通过 cross-attention 注入，而不是 concat 到序列里。这对长序列视频很关键——省掉了 `text_len × video_len` 的 attention 代价。
3. **RMSNorm for QK**：`qk_norm = True` 让 Q、K 做 RMSNorm，对超长序列训练稳定性很重要（`WanSelfAttention` line 123-124）。

### 4.2 3D RoPE（`model.py::rope_apply`）

Wan 用的是**三维分块 RoPE**：

```python
# dim 被切成三段分给 T、H、W
# 比例：c - 2*(c//3)  :  c//3  :  c//3  ≈ 1/3 : 1/3 : 1/3
freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
# 每个 token 的位置编码是 (f, h, w) 三维联合
freqs_i = cat([
    freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),  # 时间
    freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),  # 高度
    freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),  # 宽度
], dim=-1)
```

**max_seq_len = 1024** 的 RoPE 频率预计算（`model.py` line 400-405）：

```python
self.freqs = torch.cat([
    rope_params(1024, d - 4 * (d // 6)),    # time
    rope_params(1024, 2 * (d // 6)),        # height
    rope_params(1024, 2 * (d // 6))         # width
], dim=1)
```

这套 RoPE 支持任意分辨率、任意时长的 token 网格（只要每维不超过 1024）。

### 4.3 Flow Matching 训练目标

Wan 用 **Rectified Flow** 做训练目标（论文 Sec 4.2.2，Eq.1-3）：

```
Interpolation:  x_t = t·x_1 + (1-t)·x_0    (x_1 = 真实数据, x_0 = 噪声)
Velocity:       v_t = dx_t/dt = x_1 - x_0
Loss:           L = E[||u(x_t, ctxt, t; θ) - v_t||²]
```

**timestep 采样**：logit-normal 分布（不是均匀）—— 给中等噪声水平更多训练机会。

**两种推理求解器**（`wan/utils/fm_solvers.py` + `fm_solvers_unipc.py`）：
- **FlowDPMSolverMultistepScheduler**（DPM++）：40KB 实现
- **FlowUniPCMultistepScheduler**（UniPC）：33KB 实现

两者都支持 `sample_shift` 参数——shift 越大，去噪更集中在高噪声区域；shift 越小，分散到全流程。Wan 的 T2V shift 默认 **12.0**，I2V 默认 **5.0**，TI2V-5B 默认 **5.0**——T2V shift 比别的大一倍，因为纯文本起步需要更多早期全局建模。

### 4.4 三阶段训练流程（论文 Sec 4.2.2）

```
Stage 1: 256px 纯图预训练
         ↓ 建立图文对齐 + 几何结构 prior
Stage 2-A: 256px 图 + 192px 5s 视频 联合训练
         ↓ 开始学视频动力学
Stage 2-B: 480px 图 + 480px 5s 视频
Stage 2-C: 720px 图 + 720px 5s 视频
         ↓ 空间分辨率逐步上调
Stage 3: Post-training
         高质量视频 480p + 720p 联合 SFT
```

**关键决策**：**先训图再训视频**。论文 Sec 4.2.2 解释："直接用 81 帧 720P 视频预训练会因为吞吐量低 + 批次小而训不好，先用 256px 图把跨模态对齐打好，再往上加。"

**优化器**：AdamW，weight_decay=1e-3，初始 lr=1e-4，FID/CLIP Score plateau 时自动降。bf16-mixed 精度。

### 4.5 后训练（Post-training）

论文对 post-training 数据构造讲得很细致（Sec 3.2）：
- 从原始视频筛出"高美学 + 清晰 + 动态合理"的子集
- 人工打分 1-10，取 7 分以上
- 按类型均衡（人物/风景/动物/特效/文本等）

训练时分辨率仍然是 480px + 720px 联合，模型架构与优化器配置和预训练完全一致。这阶段的目标是**美学对齐**，不是能力提升。

---

## 五、基础设施与性能优化

### 5.1 训练并行策略（论文 Sec 4.3）

Wan 团队对训练基础设施的描述非常硬核。核心挑战：

- **DiT 是瓶颈**：占总计算量 85%+
- **激活内存爆炸**：1M tokens + batch=1 时激活达 **8TB**
- **Attention 开销**：1M tokens 时 attention 占端到端训练时间 95%

**分布式策略组合**：

```
2D Context Parallel (CP)
├── 外层 Ring Attention       跨机通信
└── 内层 Ulysses              机内 all-to-all

+ FSDP（参数、梯度、优化器状态分片）
+ DP（最外层数据并行）
```

**128 GPU 配置示例**（论文 Fig 11）：
- Ulysses = 8（机内）
- Ring = 2（跨机）
- FSDP = 32
- DP = 4
- global batch size = 8 × 单机 batch

**效果**：256K 序列 + 16 GPU 时，CP 通信开销从纯 Ulysses 的 >10% 降到 **<1%**。

### 5.2 内存优化

**Activation Offloading 优先于 Gradient Checkpoint**。原因：
- 1 层 DiT 的 PCIe 传输时间 ≈ 1-3 层 DiT 的计算时间
- 只要 attention 计算够重，offloading 可以完全隐藏通信
- GC 会重算 → 增加 FLOPs
- Offloading 不重算 → 只是 PCIe 开销

当 CPU 内存也要爆时才启用 GC——**优先给 "GPU 内存/计算比值高"的层**上 GC。

### 5.3 推理优化（论文 Sec 4.4）

**Diffusion Cache**——Wan 自研的两层缓存：

1. **Attention Cache**：验证集选定若干 step，跑完整 attention；其余 step **复用之前的 attention 输出**。
2. **CFG Cache**：扩散晚期，条件分支和无条件分支输出非常相似；**每 N 步跑一次无条件分支**，其余直接复用。

两者合起来让 14B T2V 推理速度提升 **1.62×**。

**量化**：
- **FP8 GEMM**：per-tensor 量化权重 + per-token 量化激活，bf16 GEMM 的 **2 倍速**，整个 DiT 模块 **1.13× 加速**。
- **8-bit FlashAttention**：FlashAttention3 的 FP8 版在视频生成上有显著质量下降，SageAttention 则在 Hopper 上没优化。团队自研了一个 Hopper-friendly 的 int8+fp16 混合精度 attention。

### 5.4 Sequence Parallel 实现（`wan/distributed/sequence_parallel.py`）

源码 `sp_attn_forward` + `sp_dit_forward` 是 **USP (Unified Sequence Parallelism)** 思路的轻量实现：

```python
# sp_attn_forward: 把序列切分后跑 distributed attention
# sp_dit_forward: 把 rope_apply、patch_embedding 等都改成 SP-aware 版本
```

**rope_apply 的 SP 改造**（line 52-58）：

```python
sp_size = get_world_size()
sp_rank = get_rank()
freqs_i = pad_freqs(freqs_i, s * sp_size)
s_per_rank = s
# 每个 rank 只拿自己那段频率
freqs_i_rank = freqs_i[(sp_rank * s_per_rank):((sp_rank + 1) * s_per_rank), :, :]
```

这是把 Wan 能跑 1M tokens 的关键——**每张卡只拿一段 RoPE freq + 一段序列，通过 all-to-all 交换 QKV 计算 attention**。

---

## 六、评测与 SOTA 对比

### 6.1 Wan-Bench 2.0

Wan 团队自建了 benchmark（论文 Sec 4.6）。14 个维度：
- Large Motion Generation
- Human Artifacts
- Pixel-level Stability
- ID Consistency
- Physical Plausibility
- Smoothness
- Comprehensive Image Quality
- Scene Generation Quality
- Stylization Ability
- Single Object Accuracy
- Multiple Object Accuracy
- Spatial Position Accuracy
- Camera Control
- Action Instruction Following

每个维度用不同方法评测：
- Motion → RAFT 光流分析
- 图像质量 → LAION aesthetic + MUSIQ
- 语义对齐 → CLIP similarity
- 动作理解 → Qwen2-VL 视频问答

最后用**人类偏好加权**合成总分——从 >5000 对人类偏好对比中学出每个维度的权重。

### 6.2 Wan 14B vs SOTA（论文 Table 2）

| 模型 | Weighted Score |
|------|:-------------:|
| Mochi | 0.639 |
| CNTopB | 0.690 |
| Hunyuan | 0.673 |
| Sora | 0.700 |
| CNTopA | 0.693 |
| **Wan 14B** | **0.724** |
| Wan 1.3B | 0.689 |

**Wan 14B 超过 Sora**（这是 2025 年初的结果，Sora 是 2024 年底的版本）。

### 6.3 VBench 公开榜

Wan 14B 在 VBench 达到 **86.22%** 总分：
- 视觉质量：86.67%
- 语义一致性：84.44%

这是榜首成绩。

### 6.4 Wan2.2 vs Wan2.1

README 给出的演进数据：
- 训练数据：图像 **+65.6%**，视频 **+83.2%**
- 美学数据：全新精标（光线/构图/对比/色调等维度标签）
- 运动能力：复杂动作显著增强
- MoE 架构：参数翻倍，激活不变

### 6.5 TI2V-5B 性能

- 5 秒 720P 视频：单卡 RTX 4090 **9 分钟内**
- 对比 I2V-A14B 8 卡 H100：**A14B 质量更高，5B 速度快 10×+**
- 这是"开源 720P@24fps 最快模型之一"

---

## 七、开源完整度审计

这是这份报告最关键的一节——**Wan 到底开源到了什么程度**。

### 7.1 代码开源度：⭐⭐⭐⭐（推理全齐，训练部分可推断）

**完整开源**：
- ✅ 五款模型的全部推理代码（`generate.py` + 五个 pipeline 文件）
- ✅ DiT 完整架构代码（`wan/modules/model.py`）
- ✅ 两版 VAE 完整实现（`vae2_1.py` 663 行 + `vae2_2.py` 1051 行）
- ✅ umT5 文本编码器（`t5.py`）
- ✅ **Animate 预处理代码**（骨骼提取、面部 retargeting）
- ✅ **S2V 的音频 encoder 集成**
- ✅ FSDP + 2D Context Parallel（Ulysses + Ring）分布式推理
- ✅ Flow Matching 求解器（DPM++ + UniPC 两套 40KB+33KB）
- ✅ ComfyUI / Diffusers 官方 PR 已合并

**没开源**：
- ❌ 预训练代码（`train.py` 不存在）
- ❌ 预训练数据（billions of images/videos 的具体来源未披露）
- ❌ Post-training 数据
- ❌ 美学标注数据（Wan2.2 的卖点之一，但数据没给）
- ❌ Streamer 和一致性蒸馏的训练代码（只在论文里讲了方法）
- ❌ VACE 编辑框架的训练代码（VACE 是另一个项目）

但是——**论文 60 页把训练方法、数据筛选流程、超参、并行策略全部讲清楚了**，社区复现不缺技术细节，只缺算力。

### 7.2 权重开源度：⭐⭐⭐⭐⭐（满分）

- 5 款模型全部开源（HuggingFace + ModelScope 双发）
- 支持 BF16 / FP8 量化版（社区分发）
- 支持 LoRA 扩展（官方 LoRA + 大量社区 LoRA）
- **License Apache-2.0**，支持商用
- 生成内容归用户所有（README 明确写了）

### 7.3 论文开源度：⭐⭐⭐⭐⭐

- 60 页完整技术报告
- arXiv 公开
- v2 更新过一次
- **训练基础设施的细节全部披露**（FSDP + 2D CP 配置、量化方案、缓存策略）—— 这在视频模型论文里极其罕见

### 7.4 数据开源度：⭐（几乎为零）

- ❌ 不开源训练数据
- ❌ 不开源美学标注
- ❌ 不开源 dense caption 数据集
- ⚠️ 只提供了 Wan-Bench 2.0 的**评测协议**，不公开基准数据

这是视频生成领域的标准做法——数据是护城河，论文可以讲，数据不能放。

### 7.5 生态开源度：⭐⭐⭐⭐⭐

README 的"Community Works"列出 7 个重大二次开发：
- **Prompt Relay**：推理时的 prompt 时序控制
- **Helios**：1 分钟 19.5 FPS 长视频（PKU）
- **LightX2V**：加速框架 + 量化 + 蒸馏（ModelTC）
- **HuMo**：人物中心统一框架（文+图+音）
- **FastVideo**：稀疏 attention 蒸馏
- **Cache-dit**：全面缓存加速
- **Kijai ComfyUI WanVideoWrapper**：ComfyUI 主流 wrapper
- **DiffSynth-Studio**：低显存 + FP8 + SP + LoRA 训练 + **全量训练**

再加上本报告前面研究的 **LingBot-World、LingBot-VA、X-WAM**——这些都是基于 Wan2.2 二次开发的一线项目。

### 7.6 总体评价

Wan2.2 的开源策略可以用一句话总结：**给你武器 + 讲清楚怎么造武器 + 不给你原料**。推理、模型、论文全开；训练代码和数据不给。但论文的技术细节足够多，像 **DiffSynth-Studio 这样的第三方已经实现了全量训练代码**，说明"不开源训练代码"不是真的壁垒。

这是**最务实的开源姿势**——既保持了阿里的商业边界（你不知道我们精调数据是什么），又让社区能完整复用推理栈和部分训练栈，生态飞速发展。

---

## 八、为什么全行业都基于它搞二次开发

### 8.1 技术侧原因

**1. 基座架构质量高**
- DiT 40 层 + 5120 dim + MoE 双专家 —— 容量足够任何下游任务用
- 3D RoPE + RMSNorm + Flash Attention 3 —— 训练稳定性一流
- Wan-VAE 的 feature cache 机制 —— 天然支持流式推理

**2. umT5 原生支持中英文**
- 这是中文社区的刚需
- HunyuanVideo、CogVideoX 的中文支持都不如 Wan

**3. 分辨率覆盖完整**
- 480P（低算力友好）+ 720P（主力）两档
- I2V / T2V / TI2V / S2V / Animate 五个任务都有权重
- 5B 小模型覆盖消费级，14B MoE 覆盖工作站

**4. 代码风格干净**
- 源码 7500+ 行，模块化清晰
- 几乎没有历史包袱（Wan2.2 相对 Wan2.1 是干净重构）
- 直接能被 `from wan.modules.model import WanModel` 拿来就用

### 8.2 生态侧原因

**1. Apache-2.0 商用友好**
- 不像 Llama 那种 <7 亿用户要额外授权的条款
- 直接商用、分发、改造、再开源都没问题

**2. 阿里官方持续维护**
- 2025 年 2 月到 11 月九个月内发布了 Wan2.1 → Wan2.2 → S2V → Animate
- ComfyUI / Diffusers PR 官方合并
- Issue 响应及时

**3. 社区飞轮形成**
- Kijai Wrapper、DiffSynth、LightX2V、Wan2GP 等第三方项目形成加速生态
- HuggingFace 上 Wan 的衍生 LoRA / 量化版 / 蒸馏版每天都在增长

**4. 论文足够硬核**
- 60 页技术报告是"教科书级别"的视频模型设计参考
- 光 VAE 一节就够写硕士论文
- FSDP + 2D CP 的分布式策略部分在工业界被反复借鉴

### 8.3 衍生项目矩阵（真实数据）

| 项目 | 机构 | 基于 Wan 什么 | 做了什么 |
|------|------|--------------|---------|
| **LingBot-World** | 蚂蚁灵波 | Wan2.2 I2V-A14B | 动作控制 + 分钟级一致性 + 实时因果蒸馏 |
| **LingBot-VA** | 蚂蚁灵波 | Wan2.2 基座 | MoT 双流，视频-动作联合建模 |
| **X-WAM** | 清华+小米 | Wan2.2-TI2V-5B | 4D 世界模型，同步出 RGB+Depth+Action |
| **Helios** | 北大元组 | Wan2.1 | 单 H100 19.5 FPS，1 分钟长视频 |
| **HuMo** | Phantom Video | Wan | 人中心统一生成（文+图+音） |
| **LightX2V** | ModelTC | Wan2.1/2.2 | 加速 + 蒸馏 + 量化全家桶 |
| **DiffSynth-Studio** | ModelScope | Wan2.2 | 低显存训练 + FP8 + LoRA + **全量训练** |
| **Wan2GP** | 社区 | Wan2.1/2.2 + 其他 | 消费级 GPU 视频生成聚合 |
| **FastVideo** | Hao AI Lab | Wan | 稀疏 attention 蒸馏 |

这 9 个项目覆盖了"长视频、实时、具身、消费级、全量训练、稀疏 attention"六大方向，Wan 一个基座养活了整条赛道。

---

## 九、硬件门槛与可玩性

### 9.1 官方给出的硬件表（README computational efficiency section 背后的数据）

| 模型 | 单卡最低显存 | 推荐配置 | 720P 5 秒视频耗时（参考） |
|------|:------------:|:--------:|:-------------------------:|
| T2V-A14B | 80GB（H100/A100） | 8×H100 | ~几十秒 |
| I2V-A14B | 80GB | 8×H100 | ~几十秒 |
| **TI2V-5B** | **24GB（RTX 4090）** | 4×H100 或单 4090 | **<9 分钟**（单 4090）|
| S2V-14B | 80GB | 8×H100 | ~几十秒 |
| Animate-14B | 80GB | 8×H100 | ~几十秒 |

**降显存的关键 flag**：
- `--offload_model True`：每层计算后 offload 到 CPU
- `--convert_model_dtype`：模型参数转到 bf16
- `--t5_cpu`：T5 text encoder 放 CPU
- `--dit_fsdp --t5_fsdp`：分布式分片
- `--ulysses_size 4/8`：序列并行

**社区量化版的真实可玩性**：
- GGUF 4-bit 量化：TI2V-5B 在 12GB 卡上能跑
- FP8 量化：A14B 在单 4090 + CPU offload 能勉强跑推理（速度很慢）
- DiffSynth 的 layer-by-layer offload：8GB 卡也能推，但要跑很久

### 9.2 训练门槛

| 目标 | 显卡需求 | 代码可得 |
|------|---------|---------|
| **从零预训练 Wan** | 数百张 H100 | ❌ 代码没开源 |
| **LoRA 微调 A14B** | 8×A100 80GB | ✅ DiffSynth 支持 |
| **LoRA 微调 5B** | 4×4090 | ✅ DiffSynth 支持 |
| **全量微调 A14B** | 32×H100 | ⚠️ DiffSynth 第三方实现 |
| **全量微调 5B** | 8×H100 | ✅ DiffSynth 有 |

### 9.3 对个人学习者的可行路径

```
1. 在 HuggingFace Spaces 白嫖体验（TI2V-5B）
2. 租 3090/4090 云机（~2-5 元/小时）跑推理
3. 用 LightX2V 蒸馏版或 4-bit 量化版
4. DiffSynth-Studio 跑 LoRA 微调（单 4090 可行）
5. 想预训练？—— 放弃吧
```

---

## 十、给不同角色的建议

### 10.1 给研究者

**如果你做视频生成方向**：
- **Wan2.2 是当前开源世界必读的论文**（相当于 LLM 圈的 Llama 2）
- 重点精读：MoE 双专家（Sec 4.2 + README）、Wan-VAE（Sec 4.1）、2D CP 分布式（Sec 4.3）
- 可研究方向：
  - 用 Wan 基座做世界模型（已有 LingBot、X-WAM 做了）
  - 研究 MoE 的最优切换阈值（目前 0.875/0.900 是经验值）
  - Post-training 数据的 scaling law
  - 视频 RLHF（目前几乎没有成熟方案）

**如果你做具身智能方向**：
- **Wan2.2 是你最可能的起点基座**（因为视频先验强且中英文支持好）
- 参考 LingBot-VA 和 X-WAM 怎么在 Wan 上挂 action adapter
- 不要自己从零训 —— 训练代价你扛不起

### 10.2 给工程师/创业者

**如果你要做视频生成产品**：
- **TI2V-5B 是消费级部署的最优解**——RTX 4090 能跑 720P@24fps
- A14B 要靠云 GPU 集群跑，成本 0.5-2 元/视频（720P×5s）
- 商用没有 License 问题（Apache-2.0）
- 配合 LightX2V 蒸馏版可以降一半推理成本

**如果你要做具身智能/机器人产品**：
- **LingBot-VA 比 Wan 更适合做机器人基座**（因为它已经把视频-动作联合训了）
- 但 VAE 和 DiT 部分可以直接继承 Wan
- 训练数据是真正的瓶颈，不是模型

### 10.3 给学习者

**学习路径建议**（基于 Wan 生态学视频模型）：
```
Week 1-2: 读 Wan 论文（60 页） + 跑通 TI2V-5B 推理
Week 3-4: 精读 wan/modules/model.py + vae2_2.py
Week 5-6: 理解 MoE 切换 + 试跑 I2V-A14B（云 GPU）
Week 7-8: 用 DiffSynth-Studio 做 LoRA 微调
Week 9-10: 研究一个衍生项目（LingBot-World / X-WAM / Helios 选一）
Week 11-12: 自己做一个小的 fine-tune 应用
```

**副轨**：同步读 Flow Matching、MoE、Rectified Flow、DiT 的原始论文，不要只看 Wan 的再述。

---

## 附录 A：关键源码文件精读优先级

| 文件 | 行数 | 重要度 | 关键内容 |
|------|:----:|:------:|---------|
| `wan/modules/model.py` | 546 | ⭐⭐⭐⭐⭐ | DiT 主干 + RoPE + AdaLN modulation |
| `wan/modules/vae2_2.py` | 1051 | ⭐⭐⭐⭐⭐ | 3D Causal VAE + Feature Cache |
| `wan/modules/vae2_1.py` | 663 | ⭐⭐⭐⭐ | 兼容 Wan2.1 权重的 VAE |
| `wan/text2video.py` | 378 | ⭐⭐⭐⭐⭐ | T2V pipeline + MoE switching |
| `wan/image2video.py` | 431 | ⭐⭐⭐⭐ | I2V pipeline（多了 mask 通道） |
| `wan/textimage2video.py` | 619 | ⭐⭐⭐⭐ | TI2V-5B 统一 pipeline |
| `wan/speech2video.py` | 706 | ⭐⭐⭐ | S2V with audio adapter |
| `wan/animate.py` | 648 | ⭐⭐⭐ | Animate 两种模式调度 |
| `wan/distributed/sequence_parallel.py` | 176 | ⭐⭐⭐ | 2D CP 轻量实现 |
| `wan/utils/fm_solvers.py` | ~900 | ⭐⭐⭐ | DPM++ 求解器 |
| `wan/utils/fm_solvers_unipc.py` | ~700 | ⭐⭐⭐ | UniPC 求解器 |
| `wan/configs/*.py` | 20-60 | ⭐⭐⭐⭐ | 五套超参速查 |
| `wan/modules/t5.py` | 513 | ⭐⭐ | umT5 encoder（可复用） |
| `LingBot_World_paper.pdf` / `wan_paper` | 60 pages | ⭐⭐⭐⭐⭐ | 论文全文 |

## 附录 B：和 LingBot-World 的关系速查

本报告前面写过 LingBot-World 深度解读，那份报告里有大量"继承自 Wan2.2"的描述。这里统一列出继承关系：

| LingBot-World 组件 | 继承自 Wan2.2 的什么 |
|-------------------|---------------------|
| DiT 主干 WanModel | `Wan2.2-I2V-A14B` 的 40 层 DiT |
| MoE 双专家切换 | `boundary` + high/low noise 模型 |
| VAE | Wan2.1-VAE（4×8×8） |
| Flow Matching | UniPC + DPM++ solvers |
| 分布式策略 | FSDP + Ulysses |
| RoPE 3D 编码 | `rope_apply` 原样使用 |
| Text encoder | umT5-XXL |
| LingBot 新增的 | camera/action AdaLN 注入 + causal + KV cache |

**一句话**：LingBot-World 是"给 Wan2.2-I2V-A14B 加了 **action control adapter** 和 **causal+KV cache 蒸馏**"——所以它的论文才写"built upon Wan2.2"。

## 附录 C：术语速查

- **MoE**：Mixture-of-Experts，多专家稀疏激活
- **A14B**：Active 14B，MoE 中每步激活 14B 参数
- **DiT**：Diffusion Transformer（Peebles & Xie 2023）
- **Flow Matching / Rectified Flow**：扩散训练的现代替代方案，直接学速度场
- **VAE**：Variational Autoencoder，压缩高维视频到低维 latent
- **umT5**：多语言 Flan-T5 变体
- **Context Parallel / Ulysses / Ring Attention**：长序列分布式训练三件套
- **FSDP**：Fully Sharded Data Parallel
- **Feature Cache**：Wan-VAE 支持任意长视频编解码的缓存机制
- **SNR**：Signal-to-Noise Ratio，MoE 切换的依据
- **boundary**：MoE 切换阈值，T2V 为 0.875，I2V 为 0.900
- **shift**：Flow Matching 的 noise schedule 偏移参数

## 附录 D：一句话总结

> Wan2.2 是 2025 年视频生成赛道的 Llama 时刻——它不是最强的闭源模型，但它开源得最彻底、架构最干净、生态最繁荣。任何人做视频生成、世界模型、具身智能基座研究，绕不开 Wan。

---

**报告字数**：约 11000 字
**阅读时间**：40-50 分钟
**数据来源**：Wan arXiv 2503.20314 完整 60 页 + GitHub 仓库 Wan-Video/Wan2.2 完整源码（7500+ 行核心代码）+ HuggingFace 五款模型卡片 + README 507 行 + 社区生态项目盘点
**不覆盖内容**：
- 训练代码（官方未开源，只能读论文理论）
- 训练数据集具体构成（官方不披露）
- 闭源商业产品（Wan 2.5、2.7 等仅 API 形式）
- 具体下游 LoRA 效果（社区评测不一致）

**相关文档**：
- `./LingBot-World_蚂蚁灵波开源世界模拟器深度解读.md`（基于 Wan2.2 的世界模型应用）
- `./X-WAM_统一4D世界行动建模与异步去噪框架.md`（另一个基于 Wan2.2-TI2V-5B 的衍生）
- `../解剖一只麻雀/01-LingBot-World开源审计报告.md`（LingBot 开源审计）
- `../解剖一只麻雀/02-世界模型开源项目全景调研与选型评估.md`（选型评估）
