# LingBot-World：蚂蚁灵波开源世界模拟器深度解读

> **论文标题**：Advancing Open-source World Models
> **arXiv**：2601.20540（2026-01-29 初版）
> **机构**：蚂蚁灵波科技（Robbyant，Ant Group 旗下具身智能子公司）
> **项目主页**：https://technology.robbyant.com/lingbot-world
> **Github**：https://github.com/robbyant/lingbot-world（3664 stars / Apache-2.0）
> **HuggingFace**：https://huggingface.co/collections/robbyant/lingbot-world
> **核心定位**：**首个完全开源、支持分钟级时序、实时交互的通用域视频世界模拟器**——站在 Wan2.2 14B×2 MoE 肩膀上，把视频生成模型打磨成"打得开、玩得久"的交互式世界
>
> **与同时发表的 LingBot-VA（arXiv 2601.21998）的关系**：LingBot-VA 是机器人向的视频-动作联合模型；LingBot-World 是通用域世界模拟器。两者是 Robbyant 同一生态下的两条技术线，共享 Wan2.2 基座，共享 MoT 思想，但目标域完全不同
>
> **本报告阅读前提**：已读完仓库 README、48 MB 论文 PDF 全部 31 页、全部核心源码（`generate.py` / `image2video.py` / `wan/modules/model.py` / `wan/modules/model_fast.py` / `wan/image2video_fast.py` / `wan/utils/cam_utils.py` / `wan/utils/wasd_ijkl_to_c2ws.py` / `wan/configs/wan_i2v_A14B.py` / `wan/distributed/sequence_parallel.py`）

---

## 一、定位与核心主张

### 1.1 一句话概括

LingBot-World 是一个**从视频生成模型（Wan2.2 I2V-14B MoE）出发，通过三阶段渐进式训练（Pre-training → Middle-training → Post-training），最终演化出「动作可控 + 分钟级时序一致性 + <1 秒延迟 + 16 FPS」交互能力的开源世界模拟器**。

### 1.2 它试图解决的三个痛点

论文 Introduction 把现有视频世界模型的瓶颈归纳为三条：

| # | 痛点 | LingBot-World 的做法 |
|---|------|--------------------|
| 1 | **高质量交互数据稀缺**——网上视频都是"被动帧"，缺少 agent-environment 因果对 | 构建三源混合数据引擎（真实视频 + 游戏录屏 + UE 合成），通过游戏和 UE 拿到 RGB↔动作的严格对齐 |
| 2 | **分钟级一致性未被解决**——标准扩散模型在 10 秒后就会"灾难性遗忘" | 渐进式课程训练（5 秒 → 60 秒）+ MoE 双专家 + flow shift 渐进放大 |
| 3 | **扩散采样计算勉强，无法实时** | Stage III 通过 causal attention + KV cache + few-step distillation（DMD + 对抗训练）把 bidirectional 蒸馏成 autoregressive |

### 1.3 和竞品的定位表（论文 Table 1）

| 模型 | 领域 | 时长 | 动态度 | 分辨率 | 实时 | 开源 |
|------|:----:|:----:|:-----:|:-----:|:----:|:----:|
| Matrix-Game 2.0 | 游戏 | 短 | 低 | 480p | ✓ | ✓ |
| Yume-1.5 | 通用 | 短 | 低 | 480p | ✗ | ✓ |
| HY-World 1.5 | 通用 | 中 | 低 | 720p | ✓ | ✓ |
| Mirage 2 | 通用 | 长 | 中 | 480p | ✓ | **✗** |
| Genie 3 | 通用 | 长 | 中 | 720p | ✓ | **✗** |
| **LingBot-World** | **通用** | **长** | **高** | **720p** | **✓** | **✓** |

**一句话结论**：唯一同时做到"通用域 + 长时序 + 高动态 + 实时 + 开源"的项目。这是它最硬的差异化。

---

## 二、数据引擎（Section 2）

论文用了整整 5 页讲数据引擎，这部分是 LingBot-World 真正的护城河之一。

### 2.1 三源混合采集

```
真实视频     → 第一人称/第三人称的人/动物/车辆
游戏数据     → RGB 严格对齐 WASD + 相机参数（关键！）
UE 合成    → 带精确 ground-truth 相机内外参的程序化渲染
```

**为什么要三源而不是只爬视频？** 因为"动作-画面"的对齐性在真实视频里几乎不存在。游戏数据直接从引擎拿键盘状态，UE 则能生成任意"合法"相机轨迹。这两条人工源合在一起，解决了世界模型最卡脖子的"动作标注"问题。

### 2.2 游戏数据采集的四大类别

- **Navigation**（free / loop / transition）——覆盖随机、闭环、大跨度场景切换
- **Sightseeing**——精细观察 + 绕行 landmark 拍多视角
- **Long-tail**——静止 360°旋转、固定角度盯动态元素、后退导航
- **World interaction**——拾取/开门/战斗/破坏等因果事件

**价值点**：这份类别清单可以直接作为机器人遥操作数据采集的模板。

### 2.3 UE 合成的两种轨迹生成模式

- **Procedural**：几何图案合成（随机矩形、360°多转）+ 多点插值（含 look-back 回看，专门加强空间记忆）
- **Real-world import**：把物理设备录的路径导进 UE，保留手持抖动的自然感

### 2.4 数据 profiling 三层过滤

```
Basic Filtering：分辨率/时长阈值 + PySceneDetect/Koala-36M + TransNet v2 切片
Semantic Analysis：VLM 评分视觉质量、动作强度、场景类型
Geometric Labeling：MegaSaM 伪标注相机内外参（兜底通用视频无几何信息的问题）
```

### 2.5 层次化 caption 策略（核心创新之一）

每个视频生成**三种互补的文本条件**：

| Caption 类型 | 内容 | 作用 |
|------------|------|-----|
| **Narrative Caption** | 环境 + 相机运动 + 时间演化一体化叙事 | 全局语义 prompt |
| **Scene-Static Caption** | 只描述静态环境，不描述相机和动作 | **关键**：motion control 与 scene generation 解耦 |
| **Dense Temporal Caption** | 带时间戳的细粒度事件描述（JSON 数组） | 支持 prompt-time-alignment 训练 |

**Scene-static** 是最妙的一手——让模型在训练时分不清"这段视频是动作驱动还是场景使然"，从而把"场景生成"和"动作响应"分离。后面 action injection 才能**只学动作、不污染视觉先验**。

---

## 三、模型架构（Section 3）

### 3.1 统一目标函数（Eq. 1）

```
max  E[ log p_θ( x_{t:t+L} | x_<t, a_{t:t+L} ) ]
 θ
```

这个目标函数写得很通用——当 `t=0` 时退化为 bidirectional 世界模型（Stage II），当 `t≥0` 且条件 `x_<t` 时变成 causal autoregressive 模型（Stage III）。论文用"多阶段演化策略"统一这两种训练模式。

### 3.2 三阶段训练流水线

```
Stage I: Pre-training        → 纯视频先验（直接继承 Wan2.2 I2V-14B×2 MoE）
Stage II: Middle-training    → bidirectional 世界模型（加入动作 + 长时序一致性）
Stage III: Post-training     → causal autoregressive（causal attention + KV cache + 蒸馏）
```

### 3.3 Stage I：预训练——Wan2.2 基座

直接继承 **Wan2.2 I2V-A14B**（MoE 双专家 diffusion transformer），不做改动。论文的说法很直接："We adopt the 14B-parameter Wan2.2 image-to-video diffusion model as our pre-trained model."

**基座参数（从 `wan/configs/wan_i2v_A14B.py` 读出来的）**：

```python
dim          = 5120     # hidden dim
ffn_dim      = 13824    # FFN dim
num_heads    = 40
num_layers   = 40       # 40 层 DiT block
patch_size   = (1, 2, 2)
vae_stride   = (4, 8, 8)
sample_steps = 70       # 推理默认 70 步
boundary     = 0.947    # 高/低噪声专家切换阈值
# 两份 checkpoint：
low_noise_checkpoint  = 'low_noise_model'
high_noise_checkpoint = 'high_noise_model'
# 快速版额外一份：
fast_noise_checkpoint = 'lingbot_world_fast'
```

**MoE 设计细节**：每个专家约 14B 参数，总参数 ~28B，但每一步 denoising 只激活一个专家——推理成本≈dense 14B。这是 Wan2.2 已有的设计，LingBot-World 只是继承。

- **高噪声专家**（t ≥ 0.947×T）：建模全局结构、粗布局
- **低噪声专家**（t < 0.947×T）：打磨细节、高频纹理

### 3.4 Stage II：中期训练——注入世界知识

这是 LingBot-World 真正发生改变的阶段，做了三件事：

#### 3.4.1 Fundamental World Model（奠基）

**渐进式课程训练**：
- Round 1：5 秒片段，先把内部生成域从 Wan2.2 的窄分布拓宽
- 渐进延长到 **60 秒**
- **flow shift 随视频长度放大**——长视频更依赖高噪声 step 建模全局结构

**Multi-task 训练**：同时做 I2V 和 V2V（video continuation）。用 I2V 从单帧推未来，用 V2V 从历史片段外推未来。联合训练让模型学到一个**统一的世界转移函数**。

#### 3.4.2 Action-Conditioned World Model（可控）

这是架构层面最核心的修改。原 Wan2.2 是纯 I2V，没有动作输入。LingBot-World 的做法：

**Action 表示（hybrid 策略）**：
- 连续量：相机 6-DoF 姿态 → **Plücker embedding**（6 维射线 + 方向）
- 离散量：WASD / IJKL 键盘 → **multi-hot 向量**
- 融合：沿通道维度 concat

**Action 注入机制（AdaLN-style）**：

我把源码里的 action injection 抠出来（`wan/modules/model.py` line 256-263）：

```python
# cam injection
if dit_cond_dict is not None and "c2ws_plucker_emb" in dit_cond_dict:
    c2ws_plucker_emb   = dit_cond_dict["c2ws_plucker_emb"]
    c2ws_hidden_states = self.cam_injector_layer2(
                             torch_F.silu(
                                 self.cam_injector_layer1(c2ws_plucker_emb)))
    c2ws_hidden_states = c2ws_hidden_states + c2ws_plucker_emb   # 残差
    cam_scale          = self.cam_scale_layer(c2ws_hidden_states)
    cam_shift          = self.cam_shift_layer(c2ws_hidden_states)
    x = (1.0 + cam_scale) * x + cam_shift                        # AdaLN
```

这段代码在 **每个 DiT block 的 self-attention 之后、cross-attention 之前**插入，是典型的 AdaLN：先用一个 MLP + 残差把 Plücker embedding 投影到 hidden 维度，再拆出 scale/shift 两组参数，对 video latent 做 channel-wise 的仿射变换。

**关键实现细节**：从 `wan/modules/model.py` line 394-398 可以看到：

```python
if control_type == 'cam':
    control_dim = 6     # 相机 Plücker = 6 (rays_o + rays_d)
elif control_type == 'act':
    control_dim = 7     # 动作模式 = 3 (rays_d only) + 4 (WASD)
```

所以两套权重（`-cam` / `-act`）在第一个 patch embedding 层的输入维度是**不同的**——这是权重不能互换的根本原因。

**参数高效 fine-tuning**：
- **冻结主 DiT block**（保留 Wan2.2 学到的视觉先验）
- **只训练 action adapter**（patch_embedding_wancamctrl + c2ws_hidden_states_layer1/2 + 每个 block 的 cam_injector_layer1/2 + cam_scale/shift_layer）
- 初始化策略：所有 cam_* 层都 Xavier init，bias 置零（论文 Sec 3.3.2，代码 line 597-615）

为什么要这样？论文原话："It effectively disentangles the inherent video generation capability from the action control capability."——动作数据质量参差，全量 fine-tune 会破坏 Wan2.2 辛苦训好的视觉先验。

#### 3.4.3 并行基础设施

- **FSDP2**：模型参数 / 梯度 / 优化器状态全分片
- **Context Parallel (Ulysses)**：沿时间维切分 token，all-to-all 通信做分布式 attention

`wan/distributed/sequence_parallel.py` 里实现了完整的序列并行，包括 `rope_apply` 的跨卡 pad/拆分逻辑——这部分代码是**训练 long-token 视频 DiT 的高价值参考材料**。

### 3.5 Stage III：后期训练——Causal 化与蒸馏

这是 LingBot-World-Fast 的诞生过程。两个关键技术：

#### 3.5.1 Causal 架构适配

**Model initialization**：从 Stage II 的**高噪声专家**初始化 student（低噪声专家被丢弃了）。论文原话："Adaptation from the high-noise expert yields superior action-conditioned dynamics modeling compared to the low-noise counterpart."

**Block causal attention**：把 Wan2.2 的 full bidirectional attention 换成 block causal：
- **chunk 内部**：bidirectional（保留局部帧间一致性）
- **chunk 之间**：causal（当前 chunk 只能看之前 chunk）

这样设计的妙处：既能流式生成，又在短时尺度上保留了扩散模型原生的双向建模能力。

**Training protocol（Eq. 2）**：

```
L = E[ || G_θ(x^i_t, t, a) - x^i_0 ||² ]
```

每个 chunk 分配独立的 noise timestep（diffusion forcing），训练限定在少数几个 distillation target timestep。**关键 trick**：额外加入 timestep=0 的 clean frame 监督，弥补从高噪声专家初始化不会干净编码的问题。

#### 3.5.2 Few-step Distillation with Long-Horizon Training

**Self-rollout 训练**（self-forcing 范式）：让 student 在自己生成的序列上训，用 rolling KV cache 管理历史。解决了"train 用 GT 历史 / test 用生成历史"的分布偏移。为了控制计算量，用**stochastic gradient truncation**——只对最近 K 步反传梯度。

**DMD + 对抗训练**：

DMD 梯度（Eq. 3）：

```
∇_θ E_t[ KL(p_{θ,t} || p_{data,t}) ]
  = -E[ (s_real(x̂_t, t, a) - s_fake(x̂_t, t, a)) · ∂x̂/∂θ ]
```

对应的 tractable loss（Eq. 4）：

```
L_DMD(θ) = E[ ½ · ||x̂ - sg[x̂ - (μ_real - μ_fake)]||² ]
```

**Two-time-scale update**：每更新 student 一次，先更新 μ_fake 多次，让 fake score 跟上 student 的分布演化。

**对抗增强**（Eq. 5、6）：

```
L_G = E_{p(x̃)}[ f(1 - D(μ_fake(x̃_t, t, a))) ]
L_D = E_{p(x)}[ f(D(μ_fake(x_t, t, a))) ] - E_{p(x̃)}[ f(1 - D(μ_fake(x̃_t, t, a))) ]
```

`f(·)` 是 softplus。论文明确说**不加 R1/R2 正则**，DMD objective 本身已经足够稳定。对抗 head 只更新 D，μ_fake 只被 DMD loss 更新。

**为什么加对抗？** 因为 DMD 训练里 student 和 teacher 都不直接看真实数据，student 会继承 teacher 的 bias。对抗训练从真实数据那里"借"监督，让 student 能超越 teacher 的上限。这是这篇报告在 post-training 环节最有含金量的设计。

### 3.6 Fast 版核心代码解读

`wan/modules/model_fast.py` 是 LingBot-World-Fast 的架构实现，这部分代码**比论文更值得读**，因为论文里对 causal + KV cache 的具体 tokens 管理写得很粗。

#### 3.6.1 `CausalWanSelfAttention.forward()`——KV cache 管理

核心逻辑三步走：

**Step 1 — Causal RoPE**：根据当前 chunk 在全序列中的位置（`start_frame = current_start // frame_seqlen`）应用旋转位置编码：

```python
current_start_frame = current_start // frame_seqlen
roped_query = causal_rope_apply(q, grid_sizes, freqs, start_frame=current_start_frame)
roped_key   = causal_rope_apply(k, grid_sizes, freqs, start_frame=current_start_frame)
```

**Step 2 — Sliding-Window + Sink Token KV 管理**：

```python
sink_tokens = self.sink_size * frame_seqlen     # 保留最早 N 帧不被驱逐

if self.local_attn_size != -1 and (current_end > kv_cache["global_end_index"]) and \
   (num_new_tokens + kv_cache["local_end_index"] > kv_cache_size):
    # 当新 token 放不下，需要驱逐
    num_evicted_tokens = num_new_tokens + kv_cache["local_end_index"] - kv_cache_size
    num_rolled_tokens  = kv_cache["local_end_index"] - num_evicted_tokens - sink_tokens
    # 把 [sink + evicted, local_end] 的内容左移到 [sink, ...]，覆盖被驱逐位置
    kv_cache["k"][:, sink_tokens : sink_tokens + num_rolled_tokens] = \
        kv_cache["k"][:, sink_tokens + num_evicted_tokens : ...].clone()
    # 把新 token 放到尾部
    kv_cache["k"][:, local_start_index:local_end_index] = roped_key
```

**这四行代码就是 StreamingLLM 里"attention sink"思想在视频扩散上的实现**：最早的几帧（sink）永远保留，中间的历史帧做 sliding-window 滚动淘汰，新 token 总是插入尾部。这保证了**长视频不漂移**（sink 锚定初始帧布局）+ **显存不爆炸**（滚动淘汰旧帧）。

**Step 3 — 实际 attention 只在 max_attention_size 窗口内计算**：

```python
k_cache = kv_cache["k"][:, max(0, local_end_index - max_attention_size) : local_end_index]
v_cache = kv_cache["v"][:, max(0, local_end_index - max_attention_size) : local_end_index]
x = attention(roped_query, k_cache, v_cache)
```

#### 3.6.2 `WanI2VFast.generate()`——chunk-by-chunk 推理

Fast 版推理流程（`wan/image2video_fast.py` line 421-477）：

```
1. 初始化 self_kv_cache 和 cross_kv_cache（两类独立 cache）
2. 把 latent 按 chunk_size 切成 (latent_chunks, condition_chunks, plucker_chunks)
3. for chunk_id in range(num_chunks):
     4. 当前 chunk 跑 few-step 去噪（默认只 4 步：[0, 179, 358, 679]）
        - 每步 noise_pred = model(...)
        - 通过 _convert_flow_pred_to_x0 恢复 x0：x0 = x_t - sigma_t * flow_pred
        - 如果不是最后一步，给 x0 加下一 step 的噪声继续去噪
     5. chunk 去噪完成后，用 timestep=0 的 clean x0 再跑一次 forward 更新 KV cache
     6. 把 x0 拼入 pred_latent_chunks
7. 所有 chunk concat 后 VAE decode
```

**几个关键数字**：
- 默认 `chunk_size = 3`（一次处理 3 帧 latent，对应 9 帧视频）
- 默认 `timesteps_index = [0, 179, 358, 679]`——**4 步推理**（对比 Base 版的 70 步，提速 ~17 倍）
- `cross-attention KV` 第一次就 cache 住（文本 context 不变），后续只更新 `self-attention KV`

#### 3.6.3 `_convert_flow_pred_to_x0`——flow matching 推理的数学

```python
# pred       = noise - x0
# x_t        = (1 - sigma_t) * x0 + sigma_t * noise
# 代入 =>  x0 = x_t - sigma_t * pred
x0_pred = xt - sigma_t * flow_pred
```

这是 Flow Matching 推理的标准反解——不用 scheduler.step()，而是直接算 x0。对于 few-step distillation 训出的模型，每步直接预测 x0 比积分 ODE 更简洁。

---

## 四、WASD/IJKL → 相机矩阵的桥接（实用细节）

这块是工程上最有学习价值的小技巧。`wan/utils/wasd_ijkl_to_c2ws.py` 里实现了一个 DSL（domain-specific language）：

### 4.1 动作字符串语法

```
"w-10,a-10,d-10,iw-15,none-10,j-10,l-10,s-15"
```

每段 `<keys>-<frames>`：
- `w/a/s/d` — WASD 移动（前/左/后/右）
- `i/j/k/l` — IJKL 相机旋转（上/左/下/右）
- `none` — 不按任何键
- 多键组合：`iw` = I + W 同时按

### 4.2 DSL → WASD/IJKL multi-hot 数组

`segments_to_wasd_ijkl()` 把字符串展开成 shape=(F, 4) 的 one-hot 数组，分别给 wasd 和 ijkl。

### 4.3 multi-hot → 相机矩阵（核心）

`generate_and_save_trajectory()` 的算法：

```python
move_speed         = 0.05             # 每帧移动 5cm
rotate_speed_rad   = np.deg2rad(2.0)  # 每帧旋转 2°
pitch_limit        = np.deg2rad(85)   # pitch 限制 ±85°（防翻跟头）
current_c2w        = np.eye(4)        # 初始单位矩阵

for frame_keys in arrow_actions:
    # 1. 更新旋转（先 yaw 后 pitch，lock roll）
    if 'i' in frame_keys:  pitch_delta += rotate_speed
    if 'k' in frame_keys:  pitch_delta -= rotate_speed
    if 'j' in frame_keys:  yaw_delta   -= rotate_speed
    if 'l' in frame_keys:  yaw_delta   += rotate_speed
    R_new = R_yaw @ R_current @ R_pitch  # yaw 外侧，pitch 内侧

    # 2. 更新平移（forward/right 向量先投到水平面再归一）
    vec_forward = R_new[:, 2]            # 相机 z 轴
    vec_right   = R_new[:, 0]            # 相机 x 轴
    forward_flat = [vec_forward[0], 0, vec_forward[2]]  # 丢掉 y 分量
    forward_flat /= |forward_flat|        # 归一

    if 'w':  T += forward_flat * move_speed
    if 's':  T -= forward_flat * move_speed
    if 'd':  T += right_flat   * move_speed
    if 'a':  T -= right_flat   * move_speed
```

**这个设计的精妙点**：
- **yaw 在外层**——保证转身时 pitch 不被"带偏"
- **forward/right 投到水平面再归一**——确保"前进"永远是水平方向，不会因为抬头就往天上飞
- **pitch 限位 ±85°**——防止欧拉角奇异性（gimbal lock 前的保护）

### 4.4 从相机矩阵到 Plücker embedding

`wan/utils/cam_utils.py::get_plucker_embeddings()`：

```python
# 1. 生成像素网格，每个像素加 0.5 bias（中心采样）
grid_xy = create_meshgrid(n_frames, h, w, bias=0.5)

# 2. 反投影：(i, j) → 相机坐标系下的射线方向
xs = (i - cx) / fx * zs
ys = (j - cy) / fy * zs
directions = normalize([xs, ys, zs])   # [f, h*w, 3]

# 3. 射线方向从相机系变世界系
rays_d = directions @ R_c2w.T          # [f, h*w, 3]

# 4. Plücker = (rays_o, rays_d) 拼 6 维
rays_o = T_c2w                         # [f, 3] → expand 到 [f, h*w, 3]
plucker = cat([rays_o, rays_d], dim=-1)  # [f, h*w, 6]
```

**`only_rays_d=True` 的特殊分支**：当是 action 模式（用 WASD 驱动平移，用 c2w rotation 控姿态）时，rays_o 丢掉，只保留 rays_d（3 维），模型输入的 control_dim = 3 + 4 = 7（对应 act 配置）。

这个设计有微妙的信息论考虑：**act 模式下 WASD 已经编码了平移意图，再喂 rays_o 反而会让模型在"动作信号"和"隐式相机轨迹"之间找捷径**，绕开真正的语义学习。

---

## 五、实验评估（Section 4）

### 5.1 VBench 定量结果

论文 Table 2，在 100 条 >30 秒视频上的 VBench：

| 模型 | Imaging Quality | Aesthetic | **Dynamic Degree** | Motion Smooth | Temporal Flicker | Overall Consistency |
|-----|:---------------:|:---------:|:------------------:|:-------------:|:----------------:|:-------------------:|
| Yume-1.5 | 0.5838 | 0.5185 | 0.7612 | 0.9709 | 0.9545 | 0.1994 |
| HY-World 1.5 | 0.6512 | 0.5487 | 0.7217 | 0.9897 | **0.9773** | 0.2016 |
| **LingBot-World** | **0.6683** | **0.5660** | **0.8857** | 0.9895 | 0.9648 | **0.2178** |

**关键观察**：
- **Dynamic Degree 0.8857** vs HY-World 的 0.7217 —— 差了 **+16.4%**，这是同赛道压倒性优势。反映视频里的运动丰富度，不是"静止图"。
- **Overall Consistency 0.2178** 也是第一 —— 长时序语义连贯
- **Temporal Flickering 0.9648** 略低于 HY-World —— 但代价是**几乎持平的平滑度 + 更高的动态度**，这是合理的 trade-off

### 5.2 Emergent Memory（定性）

论文 Fig. 12 的五行展示（非常硬核）：

| 测试 | 现象 |
|----|------|
| Row 1-3：Stonehenge / 雕像 / 建筑被镜头转走 60 秒后回头 | 结构完整保留（无 explicit 3D 表示！） |
| Row 4：向前移动 → 转回看 → 远处桥变近了 | **模型推理了未观测时段的空间演化** |
| Row 5：车出画面 → 继续行驶 → 再入画面时在物理合理位置 | **模型模拟了 off-screen 的非可见状态演化** |

Row 4 和 Row 5 是真正让人"这东西不是 video player"的证据——模型在隐式建模未观测世界的动力学，而不仅仅是缓存之前看过的像素。

### 5.3 10 分钟极限测试

论文 Fig. 13，生成了长达 **10 分钟** 的连贯视频，无显著视觉质量降级或叙事断裂。这在视频扩散模型历史上是突破性的（主流模型 30 秒就崩）。

### 5.4 三个下游应用

**5.4.1 Promptable World Events**（Fig. 14）
- 全局事件：`"winter"` / `"night"` / `"pixel art"` —— 保留运动的同时切换全局视觉域
- 局部事件：`"fireworks"` / `"birds"` / `"fish"` —— 在物理一致的位置插入新物体

这展示了**语义控制和动作控制的正交性**——动作驱动相机，文字驱动内容演化。

**5.4.2 Action Agent**（Fig. 15）
- 用 Qwen3-VL-2B 作为 backbone 在 image-action 对上 fine-tune
- 输入单张图 → 预测未来 10 秒的 WASD + IJKL 动作序列
- 把预测动作丢给 LingBot-World 做 rollout
- 效果：**单图输入 → 自驱探索视频**，不需要人类操作

**5.4.3 3D Reconstruction**（Fig. 16）
- 用 VGGT / Depth Anything 3 从生成视频反推 3D 点云
- 结果证明 LingBot-World **隐式学到了几何一致性**（不用 3DGS 这种显式表示）
- 点云可进一步作为具身训练的数据源

---

## 六、与 LingBot-VA / X-WAM / Genie3 的对比

### 6.1 vs LingBot-VA（同门师兄）

| 维度 | LingBot-World | LingBot-VA |
|------|:-------------:|:----------:|
| 目标域 | 通用视频世界 | 机器人操纵 |
| 基座 | Wan2.2 I2V-A14B | Wan2.2（定制）|
| 动作空间 | WASD + 相机 6DoF | 机器人关节 + 本体感受 |
| MoE | 高/低噪声 ×1（2 专家） | Dual-stream MoT（视频流 + 动作流）|
| 开源完整度 | **只开源推理** | **完整开源训练** |
| 学术背景 | arXiv preprint | RSS 2026 |
| 硬件门槛 | 4×H100 起 | 消费级多卡 4090 可跑 |
| 上生产路径 | demo 级 | RoboTwin / LIBERO 完整闭环 |

**一句话**：LingBot-World 是"给你看成果"，LingBot-VA 是"教你做"。对学习者而言，后者价值更大。

### 6.2 vs X-WAM（清华+小米）

| 维度 | LingBot-World | X-WAM |
|------|:-------------:|:----:|
| 发布 | 2026-01 | 2026-04 |
| 基座 | Wan2.2 I2V-A14B | Wan2.2-TI2V-5B |
| 输出 | RGB 视频 | **RGB + Depth + 状态 + 动作** 四模态 |
| 3D 建模 | 隐式（从视频反推）| **显式 4D**（深度视频同步生成）|
| 去噪策略 | 双向 → 因果蒸馏 | **ANS 异步噪声采样**（各模态独立 schedule） |
| 定位 | 通用域世界模拟器 | 具身统一 4D 模型 |

**互补关系**：LingBot-World 是"从 video gen 扩到 world"，X-WAM 是"从 world gen 直接出 4D 和动作"。前者走成熟路线、做好交互，后者赌新架构、做通用性。

### 6.3 vs Genie 3 / Mirage 2（闭源对手）

LingBot-World 在 VBench 上没和 Genie 3、Mirage 2 直接比（因为它们闭源无法跑）。论文 Table 1 的定性对比上，LingBot-World 在 **Dynamic Degree** 这一维宣称"High" vs Genie 3 / Mirage 2 的 "Medium"——但这是自我评估，需要保留怀疑。

**公允地说**：Genie 3 的视觉风格和动态表现（DeepMind Blog 的 demo）仍处于 SOTA；LingBot-World 的差异化不是"超过 Genie 3"，而是"**开源可用的第一个够得上 Genie 3 量级的通用世界模型**"。

---

## 七、开源完整度复盘

这个项目的开源策略很"战略性"——把推理/权重/论文给得彻底，但训练代码/数据完全不放。

### 7.1 完整开源的部分
- 两套推理代码：`generate.py`（Base，70 步）+ `generate_fast.py`（Fast，4 步）
- 三套权重：`-base-cam`、`-base-act`、`-fast`
- 48 MB 技术报告 PDF（完整，31 页 + 6 张大图）
- 6 组示例输入（image + prompt + intrinsics + poses + WASD/IJKL）
- 完整的 Flow Matching solver（DPM++ 和 UniPC 两套）
- FSDP + Ulysses 分布式代码
- VAE 2.1 / 2.2 实现

### 7.2 完全没开源的部分
- 训练脚本（没有 `train.py`）
- 训练数据（issue #17 无回复）
- 训练目标的具体实现（issue #20 无回复）
- 数据采集引擎代码（issue #33 无回复）
- Web 交互 demo（issue #3、#46 无回复）
- 训练成本（issue #36 无回复）

### 7.3 硬件门槛（issue #1 实测数据）

| 用户 | 硬件 | 任务 | 结果 |
|------|------|------|-----|
| VictorStarkSnow | 8× A100-40GB | 10s × 480P | 47 分钟 |
| Reactantvr | 单张 RTX 6000 Pro | 5s × 480P | 24 分钟 / 55GB VRAM |
| yanis-falaki | 4× H100-80GB | 480P 推理 | 峰值 **196 GiB 显存** |

脚本默认 `--nproc_per_node=8 --ulysses_size=8`，预期**最低 8 卡 A100/H100**。消费级可行路径只有社区 nf4 4-bit 量化版。

---

## 八、技术学习价值

这份源码+论文组合对"世界模型研究者"的学习价值排序（高到低）：

### 8.1 最高价值（必读）

1. **`model_fast.py::CausalWanSelfAttention`**——StreamingLLM-style sliding window + sink tokens 在视频扩散上的完整工业实现。这是从 Transformer LM 技巧到视频 DiT 的迁移样本，教科书级。
2. **Section 3.4.2 Few-step Distillation**——DMD + self-forcing + 对抗训练的三合一 post-training 范式。对任何想把慢速扩散模型蒸馏成实时模型的研究者都是直接可参考的食谱。
3. **数据引擎层次化 caption**（Sec 2.3）——scene-static + narrative + dense-temporal 三级 caption 是"解耦动作与场景"的简洁方案，对 VLA / 机器人 imitation learning 也有直接启发。

### 8.2 中等价值

4. **`image2video_fast.py` chunk-by-chunk + few-step 推理循环**——工程化的 autoregressive 扩散推理模板，可以直接抄改到自己的项目。
5. **`cam_utils.py::get_plucker_embeddings`**——把相机 c2w + 内参转 Plücker 的参考实现，数值稳定性处理（bias=0.5）值得借鉴。
6. **`wasd_ijkl_to_c2ws.py::generate_and_save_trajectory`**——keyboard → camera matrix 的桥接算法，pitch 限位 + forward/right 水平化这两个细节很专业。
7. **MoE 双专家切换逻辑**（`image2video.py::_prepare_model_for_timestep`）——timestep 阈值驱动的模型切换 + offload 策略，对多专家 diffusion 推理框架有参考价值。

### 8.3 低价值

8. **VAE 和 T5 编码器**——Wan2.2 搬过来的，不是 LingBot-World 原创
9. **UniPC / DPM++ solver**——标准 flow matching 求解器，有很多更好的教材
10. **Animate / S2V 子模块**——Wan Animate / Speech-to-Video 的残留代码，在 LingBot-World 里**没被调用**

---

## 九、我的观点：这是一个什么档次的工作

### 9.1 客观评价

**学术层面**：arxiv preprint，未见顶会录用（截至 2026-05）。作者列表 22 人，更像工业界技术报告的规格。**单独拉出来投顶会会很吃力**——因为它的绝大多数组件（Wan2.2 基座、DMD、self-forcing、StreamingLLM 的 sink token、Plücker embedding、AdaLN 注入）都是已有技术的工程组合，没有理论层面的新发现。

**工程层面**：是**值得致敬的开源诚意**。在 2026 年初这个时点，"开源能对标 Genie 3 的通用域世界模型"本身就是一个 milestone。28B 参数的模型、10 分钟连贯生成、<1 秒延迟 + 16 FPS 这三个数字合在一起，已经超过 2025 年几乎所有开源项目。

**产业层面**：蚂蚁灵波这条产品线（lingbot-map/world/depth/vla/va）布局非常完整，LingBot-World 是其中**最吸睛的 PR 武器**。这份报告的受众不是研究者，是**具身智能行业的技术决策者**——"看，我们蚂蚁什么都能做。"

### 9.2 给学习者的建议

**如果你的目标是从 0 到 1 复现并训练一个世界模型**：
- 不要选 LingBot-World——训练代码没开源，硬件也要 8×H100
- **选 LingBot-VA** 作为主解剖对象（训练/数据/评测全齐）

**如果你的目标是读懂"视频生成 → 交互世界"的完整技术链**：
- LingBot-World 的论文 + `model_fast.py` 是**最好的开源教材之一**
- 重点吃透 Sec 3.4（causal adaptation + few-step distillation）+ Section 2（数据引擎的层次化 caption）
- 用 `cahlen/lingbot-world-base-cam-nf4` 量化版在你的多卡 4090/5090 上跑一次推理，体验"分钟级一致性"是什么感觉

**如果你的目标是做自己的世界模型创业/产品**：
- LingBot-World 的**开源许可是 Apache-2.0**，可以商用
- 但注意基座 Wan2.2 的许可链是否兼容你的商业用途
- Fast 版已具备 <1 秒延迟，**够做游戏/内容创作 demo**，不够做机器人 policy

### 9.3 一句话总结

> LingBot-World 不是一篇论文贡献很大的工作，但它是 2026 年初**开源世界模型生态里最重要的一次供给**。它把闭源的 Genie 3 级能力（分钟级 + 实时 + 通用域）打包成 Apache-2.0 的代码给到社区，即使没开训练端，已经是具身 AI 民主化方向的一个关键 step。

---

## 附录 A：关键文件清单（供后续精读）

| 文件 | 行数 | 重要度 | 关键内容 |
|------|:---:|:------:|---------|
| `wan/modules/model_fast.py` | 652 | ⭐⭐⭐⭐⭐ | Causal attention + KV cache + sink token 完整实现 |
| `wan/image2video_fast.py` | 524 | ⭐⭐⭐⭐⭐ | chunk-by-chunk few-step 推理循环 |
| `wan/modules/model.py` | 615 | ⭐⭐⭐⭐ | Base 版 DiT + AdaLN action injection |
| `wan/image2video.py` | 578 | ⭐⭐⭐⭐ | 70 步推理 pipeline + MoE 双专家切换 |
| `wan/utils/wasd_ijkl_to_c2ws.py` | 229 | ⭐⭐⭐⭐ | 动作字符串 DSL + 相机轨迹算法 |
| `wan/utils/cam_utils.py` | 150 | ⭐⭐⭐ | Plücker embedding 实现 |
| `wan/distributed/sequence_parallel.py` | 475 | ⭐⭐⭐ | FSDP2 + Ulysses 序列并行 |
| `wan/configs/wan_i2v_A14B.py` | 39 | ⭐⭐ | 超参数速查 |
| `wan/utils/fm_solvers.py` + `_unipc.py` | 40KB + 33KB | ⭐⭐ | Flow Matching 求解器 |
| `LingBot_World_paper.pdf` | 31 pages | ⭐⭐⭐⭐⭐ | 论文全文 |

## 附录 B：论文五个核心方程

1. **统一世界模型目标**（Eq. 1）：`max_θ E[log p_θ(x_{t:t+L} | x_<t, a_{t:t+L})]`
2. **Causal adaptation loss**（Eq. 2）：`L = E[||G_θ(x^i_t, t, a) - x^i_0||²]`
3. **DMD 梯度**（Eq. 3）：`∇_θ E_t[KL] = -E[(s_real - s_fake) · ∂x̂/∂θ]`
4. **DMD tractable loss**（Eq. 4）：`L_DMD = E[½·||x̂ - sg[x̂ - (μ_real - μ_fake)]||²]`
5. **对抗双目标**（Eq. 5、6）：`L_G = E[f(1-D(μ_fake))]`，`L_D = E[f(D(μ_fake_real))] - E[f(1-D(μ_fake_syn))]`

## 附录 C：VBench 指标解读

- **Imaging Quality**：单帧画面质量（细节、清晰度）
- **Aesthetic Quality**：美学吸引力
- **Dynamic Degree**：视频中的运动丰富度（越高越好）
- **Motion Smoothness**：相邻帧过渡平滑性
- **Temporal Flickering**：闪烁程度（越高越好，不闪）
- **Overall Consistency**：和 prompt 的语义对齐度

## 附录 D：x_WAM / Genie 3 / LingBot 三角对照速查

```
共同点：
├─ 都基于 Wan2.2 家族
├─ 都做 video world model 方向
└─ 都解决 action-conditional video gen 问题

各自独特：
├─ X-WAM     → 4 模态异步噪声 + 3D reconstruction in one shot
├─ Genie 3   → 闭源，DeepMind 级别视觉效果
└─ LingBot-World → 开源 + 实时 + 分钟级一致性
```

---

**报告字数**：约 9000 字
**阅读时间**：30-40 分钟
**数据来源**：GitHub 仓库源码完整阅读 + 论文 PDF 31 页全文 + README 官方文档 + 同公司 LingBot-VA 报告对照
**不覆盖内容**：训练端（因官方未开源）、具体 VBench 测试集构成（论文未披露）、商业化路径（本报告聚焦技术）
**相关文档**：
- `/解剖一只麻雀/01-LingBot-World开源审计报告.md`（开源完整度审计）
- `./X-WAM_统一4D世界行动建模与异步去噪框架.md`（架构对比）
- `./Interactive_World_Simulator_交互式世界模拟器.md`（实时世界模型对比）
