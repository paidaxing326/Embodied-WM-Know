---
title: PointWorld — 3D 世界模型规模化研究深度解读
authors: [Wenlong Huang, Yu-Wei Chao, Arsalan Mousavian, Ming-Yu Liu, Dieter Fox, Kaichun Mo, Li Fei-Fei]
institutions: [Stanford University, NVIDIA]
year: 2026
arxiv: "2601.03782"
github: "NVlabs/PointWorld"
tags: [世界模型, 3D点流, 机器人操控, PointTransformerV3, DINOv3, 异方差不确定性, NVIDIA]
---

# PointWorld: Scaling 3D World Models for In-The-Wild Robotic Manipulation — 深度解读

> **一句话定义**：PointWorld 是 NVIDIA + Stanford 联合推出的大规模预训练 3D 世界模型，从部分可观测 RGB-D 捕获和机器人动作出发，预测全场景 3D 点流（point flows）——场景和机器人都统一为 3D 点流表示，实现了跨域（真实 DROID + 仿真 BEHAVIOR）的通用 3D 动力学预测。

---

## 1. 研究背景与动机

### 1.1 世界模型的根本问题

世界模型（World Model）的核心任务：给定当前观测 + 动作指令，预测未来状态。在机器人操控领域，这意味着：
- 看到厨房桌面（RGB-D 多视角）
- 机器人执行"拿起杯子"
- 预测桌面所有物体的 3D 运动轨迹

### 1.2 已有方案的瓶颈

| 方案类型 | 代表工作 | 核心瓶颈 |
|---------|---------|---------|
| 2D 视频/图像预测 | UniSim, Genie, GameNGen | 缺乏 3D 结构，无法泛化视角；2D 像素预测噪声大 |
| 2D 动作条件视频 | VideoGPT, SVD | 无法精确预测物体 3D 运动，难以闭环控制 |
| NeRF/Gaussian 3D 重建 | NeRF, 3DGS | 需大量视角优化，无法实时预测未来 |
| 3D 点云处理 | PointNet++, PointTransformer | 缺乏预训练大规模模型，小模型泛化差 |

PointWorld 的核心洞察：**3D 点流（3D Point Flow）是最自然的 3D 世界模型表示**——
- 场景：每个 3D 点从当前位置流向未来位置
- 机器人：机械臂的 3D 点也从当前位置流向未来位置
- 两者统一表示，天然支持动作条件预测

### 1.3 "In-The-Wild" 的含义

不是实验室干净桌面，而是 DROID 数据集——**真实人类家庭厨房、凌乱桌面、多视角、遮挡、噪声深度**。这是机器人操控世界模型真正需要面对的场景。

---

## 2. 核心架构：四大模块深度解析

PointWorld 的整体架构可拆解为四大模块：

```
┌──────────────────────────────────────────────────────────────────┐
│                    PointWorld Architecture                        │
│                                                                  │
│  ┌─────────────┐    ┌─────────────┐    ┌──────────────────────┐ │
│  │ Scene       │    │ Robot       │    │ Temporal              │ │
│  │ Encoder     │───▶│ Feature     │───▶│ Embedding             │ │
│  │ (DINOv3 +   │    │ Projection  │    │ (SinCos + MLP)       │ │
│  │  Proj Head) │    │ + MLP       │    │                       │ │
│  └─────────────┘    └─────────────┘    └──────────────────────┘ │
│         │                   │                    │               │
│         │ scene_feat0       │ robot_feat_seq     │ time_emb      │
│         ▼                   ▼                    ▼               │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │              Dynamics Predictor (PTv3)                       │ │
│  │                                                              │ │
│  │  Input: scene_coord0 + robot_coord_seq → unified point cloud │ │
│  │  PTv3 backbone → per-point features                          │ │
│  │  FiLM skip (scene) + FiLM robot global summary               │ │
│  │  ├── dynamics_head: MLP → 3D flow per scene point × (T-1)   │ │
│  │  └── log_var_head: MLP → uncertainty per scene point × (T-1)│ │
│  └─────────────────────────────────────────────────────────────┘ │
│         │                                                        │
│         ▼                                                        │
│  Output: scene_flows (B,T,Ns,3) + confidence (B,T,Ns)           │
└──────────────────────────────────────────────────────────────────┘
```

### 2.1 场景编码器（SceneEncoder2D + SceneFeatureEncoder）

**设计哲学**：3D 点云缺乏纹理/语义信息，需要从 2D 图像"注入"语义。PointWorld 用 DINOv3 ViT-L16 作为冻结的 2D backbone，通过多视角投影将 2D 语义特征"贴回" 3D 点。

#### 2.1.1 DINOv3 ViT-L16 Backbone

- **模型**：`dinov3_vitl16`，特征维度 `feat_dim=1024`
- **多层特征提取**：`selected_layers = [4, 11, 17, 23]`（取 ViT 的第 4、11、17、23 层中间特征）
- **拼接**：4 层 × 1024 = 4096 维原始特征
- **冻结**：`self._freeze_encoder()` — DINOv3 全部参数 `requires_grad_(False)`，只训练投影头
- **ImageNet 标准化**：`mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]`
- **torch.compile**：对 `_maybe_no_grad` 路径做 `torch.compile` 加速推理

#### 2.1.2 多视角投影聚合

核心流程（`SceneEncoder2D.forward`，源码 `scene_featurizer.py:140-320`）：

1. **Gather camera tensors**：将所有相机（1-3 个）的 RGB、Depth、Intrinsic、Extrinsic 堆叠为 `(B, C, H, W, 3)` 等批量张量
2. **Backbone features**：DINOv3 处理所有视角的 RGB → patch tokens `(B*C, patch_h, patch_w, feat_dim)`
3. **3D → 2D 投影**：每个场景 3D 点通过 `extrinsic @ pts_h` 和 `intrinsic @ pts_cam` 投影到每个相机图像平面
4. **Visibility + Depth consistency mask**：
   - `in_img`：投影像素是否在图像范围内 + 深度是否正值
   - `depth_ok`：投影深度与实际深度的一致性检查（双向阈值：`thr_behind = args.depth_threshold`，`thr_front = 0.5 * depth_threshold`）
   - 最终 `visible = in_img & depth_ok & cam_exists`
5. **Feature sampling**：从 ViT patch token grid 用 `F.grid_sample` 采样投影位置的 2D 特征
6. **跨相机聚合**：`feat_sum_dino / cam_counts` — 多视角特征取均值后投影到 `channels` 维度

> **[源码验证]** `depth_threshold` 默认值 = `0.003`（3mm），`thr_front` = `1.5mm`。这是一个非常严格的深度一致性检查——只有投影深度和实际深度差距小于 3mm 的点才被认为"可见"。

#### 2.1.3 SceneFeatureEncoder：双通道融合

```python
# scene_featurizer.py:366-394
scene_feat0 = torch.cat([
    self.scene_encoder_norm(backbone_scene_feat0),   # DINOv3 backbone 特征
    self.scene_raw_norm(self.scene_raw_feat_proj(scene_feat0)),  # 原始数据集特征
], dim=-1)
return self.scene_proj(scene_feat0)  # 2*channels → channels
```

**双通道设计**：
- **Backbone 通道**：DINOv3 2D 投影的语义特征（经过 LayerNorm + Linear）
- **Raw 通道**：数据集原始 scene_features（scene_colors + scene_normals + gripper_open + dist2robot）
- 两者 concat 后 `Linear(2*channels, channels)` 融合

> 这是关键的"信息注入"设计——DINOv3 提供语义/纹理理解，原始特征提供几何/颜色/机器人距离信息。

### 2.2 机器人特征编码（Robot Feature Pipeline）

#### 2.2.1 RobotSampler：GPU 加速正运动学点采样

源码 `robot_sampler.py` 是一个独立的模块，核心功能：

- **URDF 加载**：通过 `urdfpy.URDF.load()` + `pytorch_kinematics` 构建运动学链
- **Mesh 预采样**：`presample(num_points)` — 按面积比例从机器人视觉 mesh 上采样 3D 点和法线（默认 `max_robot_points=500`，只采样 gripper 部分 `gripper_only=True`）
- **GPU 批量 FK**：`compute_points(joint_values)` — 给定关节角度，通过 `chain.forward_kinematics()` 批量计算所有 link 的 4×4 变换矩阵，然后将预采样点从 mesh local frame 变换到 world frame
- **Robotiq 夹爪 mimic joints**：`build_robotiq_joint_dict()` 将单个 `finger_joint` 扩展为 6 个联动关节

> **[源码验证]** 双臂处理：BEHAVIOR 域支持双臂机器人，通过 `determine_gripper_filter()` 智能选择哪只手参与（基于碰撞检测和距离判断）。

#### 2.2.2 Robot 特征构造

```python
# dataset_components/robot.py:135-199
robot_features = [robot_flows, robot_colors, robot_normals,
                  gripper_open, robot_velocity, robot_acceleration]
# 全部 concat → (T, NR, Fr)
```

六维机器人特征：
| 特征 | 维度 | 来源 |
|-----|------|------|
| `robot_flows` | 3 | RobotSampler FK 计算 |
| `robot_colors` | 3 | 固定紫红色 `(1.0, 0.0, 1.0)` |
| `robot_normals` | 3 | RobotSampler mesh 法线 |
| `gripper_open` | 1 | 夹爪开合状态 |
| `robot_velocity` | 3 | 中点法速度 `(flows[t+1]-flows[t-1])/2` |
| `robot_acceleration` | 3 | 中点法加速度 |

#### 2.2.3 Robot Projection + Temporal Embedding + Type Embedding

```python
# pointworld/base.py:467-473
robot_raw = self.robot_proj(robot_feat_seq)          # MLP: Fr → channels
time_emb = self.time_embed(self.time_steps.view(1,T)) # SinCos + MLP
robot_feat = robot_raw + time_emb + self.robot_type_emb  # 三者相加
```

三重注入：
1. **MLP 投影**：原始 robot 特征维度 → `channels`
2. **时间嵌入**：`TemporalEmbedding` — SinCos 位置编码 + 2 层 MLP（`Linear → SiLU → Linear`），时间步归一化到 `[0,1]`
3. **类型嵌入**：`robot_type_emb` — 可学习的 `nn.Parameter(1, channels)`，区分 scene vs robot 点

### 2.3 Dynamics Predictor（核心：PTv3 + FiLM + 双头输出）

#### 2.3.1 Point Transformer V3 (PTv3) Backbone

源码 `pointworld/base.py:153-205` + `ptv3/ptv3.py` + `ptv3/ptv3_arch.yaml`：

PTv3 是 Pointcept 团队的第三代点云 Transformer，PointWorld 做了适配并提供了三种规模：

| 规格 | enc_depths | enc_channels | enc_heads | dec_depths | dec_channels | dec_heads | channels_max |
|-----|-----------|-------------|----------|-----------|-------------|----------|-------------|
| **small** | [2,2,2,6,2] | [C,C,128,256,512] | [2,4,8,16,32] | [2,2,2,2] | [C,C,128,256] | [4,4,8,16] | 128 |
| **base**（默认） | [4,4,4,8,8,12,4] | [C,C,C,384,384,512,768] | [4,4,4,8,8,16,24] | [2,2,2,2,2,2] | [C,C,256,384,384,512] | [4,4,4,8,8,16] | 256 |
| **large** | [4,4,8,8,12,12,4] | [256,384,384,512,512,768,1024] | [8,12,12,16,16,24,32] | [4,4,4,4,4,4] | [256,384,384,512,512,768] | [8,12,12,16,16,24] | 256(eq) |

**PTv3 关键参数**（从 `build_ptv3` 调用）：
- `order=("z", "z-trans", "hilbert", "hilbert-trans")` — 4 种空间序列化顺序
- `stride=(2,2,2,2,2)` — 5 级下采样 stride=2
- `mlp_ratio=4, qkv_bias=True`
- `drop_path=0.3` — DropPath 随机深度
- `pre_norm=True, shuffle_orders=True`
- `enable_rpe=False` — 关闭相对位置编码
- `patch_size` 默认 `256`（`args.ptv3_patch_size`）

#### 2.3.2 统一点云构建与 PTv3 前向传播

```python
# pointworld/base.py:242-265
coord = torch.cat([scene_coord0, robot_coord_seq.reshape(B, T*Nr, 3)], dim=1)  # (B, Ns+T*Nr, 3)
feat  = torch.cat([scene_feat0,  robot_feat.reshape(B, T*Nr, D)], dim=1)
exists = torch.cat([scene_exists0, robot_exists.reshape(B, T*Nr)], dim=1)
is_robot = ... # bool mask

data_dict = {
    "coord": coord[exists],    # 只处理存在的点
    "feat" : feat[exists],
    "batch": batch[exists],
    "is_robot": is_robot[exists],
    "grid_size": self._grid_size,
}
point = self.predictor_model(data_dict)  # PTv3 forward
```

**关键设计**：
- Scene 和 Robot 点**合并为统一点云**送入 PTv3
- `is_robot` 标记区分两种点，PTv3 内部通过 `grid_size=0.015` 做 voxelization
- `exists` mask 过滤掉不存在/padding 的点，**动态点数**处理

#### 2.3.3 FiLM Skip Connection + Robot Global Summary

```python
# pointworld/base.py:267-312
# 1. Skip connection: 输入 scene 特征调制 PTv3 输出
skip_modulated = scene_feat0 * self.skip_film_gamma + self.skip_film_beta

# 2. Robot global summary: 所有 robot 点 max pooling → broadcast 到每个 scene 点
robot_global_summary = valid_robot_features.max(dim=0, keepdim=True)[0]  # (1, D)
robot_global_summary = robot_global_summary.expand(-1, Ns, -1)  # (B, Ns, D)
robot_modulated = robot_global_summary * self.robot_film_gamma + self.robot_film_beta

# 3. 三路融合
padded_scene_feat = scene_feat_output + skip_modulated + robot_modulated
```

**FiLM（Feature-wise Linear Modulation）**的设计动机：
- PTv3 输出可能丢失原始输入信息 → **skip connection 用 FiLM 调制**而非简单加法
- 机器人信息需要全局传播到每个场景点 → **max pooling 聚合**所有 robot 特征后 FiLM 调制
- `film_gamma` 和 `film_beta` 是可学习的 `nn.Parameter`，初始化为 `(1, 0)`（即恒等映射起点）

#### 2.3.4 双头输出：Dynamics + Uncertainty

```python
# pointworld/base.py:217-229
self.dynamics_head = nn.Sequential(MLP(in_channels, 128, 128), nn.Linear(128, 3 * (T-1)))
self.log_var_head = nn.Sequential(MLP(in_channels, 128, 128), nn.Linear(128, 1 * (T-1)))

# dynamics_head 初始化：Kaiming normal + scale (1.0 for DROID, 0.0 for BEHAVIOR)
# log_var_head bias 初始化：log(0.005^2) ≈ -11.5（极小初始方差 → 高初始置信度）
```

**时序设计**：
- `CONTEXT_HORIZON = 1`（t=0 是输入，已知）
- `PRED_HORIZON = 10`（预测 t=1..10）
- 第一个时间步输出零流 `(0, 0, 0)`，只预测后 9 步的相对流

### 2.4 异方差不确定性（Heteroscedastic Uncertainty）

这是 PointWorld 最独特的设计之一——**每个场景点都预测自己的不确定性**。

#### 2.4.1 不确定性建模

```python
# pointworld/base.py:59-94
# 置信度计算：
var = torch.exp(log_var)
conf = 1.0 - (var - VAR_FLOOR) / (VAR_CEILING - VAR_FLOOR)
# VAR_FLOOR = 1e-6, VAR_CEILING = 1e2
```

- **log_var_head** 输出每个点的 `log(variance)`
- **置信度 confidence** = `1 - normalized_variance`，范围 `[0, 1]`
- **VAR_FLOOR** = `1e-6`（最小方差），**VAR_CEILING** = `1e2`（最大方差）

#### 2.4.2 模域 vs 真实域的不确定性差异

```python
# pointworld/base.py:70-94
# SIM_DOMAIN_KEYWORDS = ["behavior"]
# 对于仿真域：log_var 设为常数（仿真数据无噪声，不需要不确定性建模）
# 对于真实域：log_var 正常预测
```

> **[源码验证]** 仿真域的不确定性被替换为真实域 log_var 的均值——这是合理的，因为 BEHAVIOR 仿真数据精确无噪声，不应产生"不确定"预测。

#### 2.4.3 Confidence Annotation（专家置信度标注）

评估流程中有一个独特步骤——**用专家模型（test split 上训练的模型）生成置信度标注**，过滤低置信度点：

```python
# evaluation/annotation.py:215-310
# 1. 在 test split 上训练 expert model
# 2. Expert model 推理 → confidence map (B,T,Ns)
# 3. 低置信度点 → voxel 化存储到 H5 文件
# 4. 评估时用 expert confidence 做 scene_filter_mask
```

> 这解决了 DROID 真实数据的标注噪声问题——专家模型对哪些点"看不清"，就过滤掉这些点的评估指标。

---

## 3. 数据流与训练管线

### 3.1 数据集

| 数据集 | 类型 | 机器人 | 视角 | 场景 |
|-------|------|--------|------|------|
| **DROID** | 真实世界 | Franka + Robotiq | 多相机（1-6） | 家庭厨房、凌乱桌面 |
| **BEHAVIOR** (B1K) | 仿真 | Franka 双臂 | 多相机（左右） | OmniGibson 仿真环境 |

#### 3.1.1 WebDataset 格式

数据以 WebDataset (tar shards) 格式存储，每个 shard 包含：
- 多相机 `*_initial_rgb.jpg` + `*_initial_depth.npy` + `*_intrinsic.npy` + `*_extrinsic.npy`
- 多相机 `*_scene_flows.npy` + `*_scene_colors.npy` + `*_scene_normals.npy`（int8 量化）
- `joint_positions.npy` + `gripper_positions.npy`
- BEHAVIOR 还有 `*_local_scene_points.pyd` + `*_scene_mesh_trajectories.pyd` + `clip_attributes.pyd`

#### 3.1.2 相机采样

```python
# dataset_components/cameras.py
# 训练：随机采样 min=1, max=3 个相机
# 评估：固定采样 min=2, max=2 个相机
# 选中相机重命名为 cam0, cam1... 标准前缀
```

#### 3.1.3 DROID 深度处理

- 深度图以 `uint16 mm` 存储，解码时 `/1000.0` 转为 `float32 meters`
- 相机分辨率：`180×320`（从 DROID 原始 `640×360` 缩放）
- DROID gripper_pose 从 wrist frame 转 TCP frame：`wrist2world @ inv(DROID_WRIST2TCP)`

#### 3.1.4 BEHAVIOR 仿真数据处理

- 通过 `SimDataDecoder` 解码 mesh trajectories
- 每个 mesh 有 local points/colors/normals + SE(3) trajectory `(T, 7)` (xyz + quat)
- 通过 Numba 加速的 `transform_points_kernel` 和 `transform_normals_kernel` 批量变换到世界坐标
- `base_pose` 提供机器人底座的移动变换

### 3.2 Transform Pipeline（数据增强）

训练时的完整管线（`dataset_components/pipeline.py`）：

```
原始样本
→ center_shift (均值中心化)
→ filter_within_bounds (±3m 范围裁剪)
→ assert_camera_payload_resolution (验证 180×320)
→ grid_sample_transform (voxel 下采样, grid_size=0.015)
→ sphere_crop_transform (多球体裁剪, prob=1.0, r=[0.10, 0.80])
→ enforce_max_num_points (max=12000)
→ random_rotate_around_z_axis (随机 Z 旋转)
→ random_scale_transform (scale [0.90, 1.10])
→ random_flip_transform (p=0.5, X/Y 轴翻转)
→ center_shift (二次中心化)
→ chromatic_auto_contrast (p=0.2, blend=0.2)
→ chromatic_translation (p=0.95, ratio=0.02)
→ chromatic_jitter (p=0.95, std=0.02)
→ normalize_colors (uint8 → float32 /255)
→ make_gt_copy (保存 GT 副本)
→ sample_and_apply_scene_context_mask (context_horizon=1, 其余帧用前一帧替换)
→ gather_features (robot + scene 特征聚合)
→ compute_helper_variables (moved/static mask, relative flows, weights)
→ convert_to_tensors (numpy → torch)
```

#### 3.2.1 SphereCrop（核心增强）

```python
# dataset_components/transforms.py:62-119
# 多球体裁剪：围绕离机器人最远的点做球体裁剪
# 每次选离 robot 最远的 candidate，在其位置做 radius 球裁剪
# 最多 3 个球体，裁剪到 max_scene_points 以内
```

> **[源码验证]** SphereCrop 是 PointWorld 的独创增强——不是随机裁剪，而是**裁剪离机器人最远的区域**。这迫使模型聚焦于机器人交互区域附近，而非远处的墙壁/地板。

#### 3.2.2 Context Masking

```python
# t=0: 完整观测（context）
# t=1..T: 替换为 t-1 的位置（即"不知道未来"，只能从当前位置预测）
# scene_context_mask = (T, NS, 1)，t>=context_horizon 时为 0
```

### 3.3 训练超参数

从 `arguments.py` 提取的关键默认值：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `batch_size` | 22 | |
| `num_epochs` | 200 | |
| `base_lr` | 0.0001 | AdamW 学习率 |
| `weight_decay` | 0.01 | |
| `grad_clip_max_norm` | 5.0 | |
| `predictor_dim` | 256 | PTv3 channel 维度 |
| `ptv3_size` | `base` | PTv3 规格 |
| `ptv3_patch_size` | 256 | |
| `grid_size` | 0.015 | voxel 下采样粒度 |
| `max_scene_points` | 12000 | |
| `max_robot_points` | 500 | |
| `huber_delta` | 5.0 | Huber loss delta |
| `confidence_thres` | 0.8 | 置信度过滤阈值 |
| `depth_threshold` | 0.003 | 3mm 深度一致性 |
| `dynamics_head_init_scale` | 1.0 (DROID) / 0.0 | dynamics head 初始化缩放 |
| `amp` | True | bfloat16 混合精度 |
| `T = CONTEXT_HORIZON + PRED_HORIZON` | 1 + 10 = 11 | 总时间步 |

### 3.4 训练循环

```python
# training/trainer.py:366-563
# - AdamW optimizer (lr=1e-4, wd=0.01)
# - AMP (bfloat16 or float16)
# - GradScaler + gradient clipping (max_norm=5.0)
# - NaN detection: 连续 NaN 梯度检测 → skip batch
# - 每 eval_freq=600 batches 做一次 eval
# - 每 save_freq=1800 batches 保存 checkpoint
# - DDP 支持: torchrun --nproc_per_node=N
# - wandb 日志记录
```

---

## 4. 损失函数深度解析

### 4.1 核心：不确定性加权 NLL + Huber

```python
# pointworld/losses.py:23-62
huber = HuberLoss(delta=args.huber_delta, reduction="none")
error_term = huber(output_norm, gt_target_norm)  # (B,T,NS,3)

# NLL loss:
per_dim_loss = 0.5 * (error_term / var + UNCERTAINTY_LOGVAR_WEIGHT * log_var_clamped)
per_point_loss = per_dim_loss.mean(dim=-1)  # (B,T,NS)
dynamics_loss = (per_point_loss * weights).sum()
```

**数学表达**：

$$L_{NLL} = \frac{1}{2} \sum_{i} \left( \frac{H(\hat{y}_i, y_i)}{\sigma_i^2} + \lambda \log \sigma_i^2 \right)$$

其中：
- $H(\hat{y}, y)$ = Huber loss (delta=5.0)
- $\sigma_i^2 = \exp(\log\_var_i)$ = 预测方差
- $\lambda = UNCERTAINTY\_LOGVAR\_WEIGHT = 1.0$ = 正则化系数

**直觉**：
- 高不确定性的点（大 $\sigma^2$）：误差被缩小（"我不确定，所以误差容忍度高"）
- 低不确定性的点（小 $\sigma^2$）：误差被放大（"我很确定，所以误差惩罚重"）
- $\log \sigma^2$ 正则项防止模型把所有点都标为"不确定"来逃避损失

### 4.2 输出归一化

```python
# pointworld/norm_stats.py:206-223
# normalize: (og - per_step_mean) / per_step_std
# unnormalize: normalized * per_step_std + per_step_mean
# per-domain, per-timestep 的均值和方差从 precomputed JSON 加载
```

**关键**：归一化是 **per-domain, per-timestep** 的——DROID 和 BEHAVIOR 有不同的统计量，每个时间步也有自己的均值和方差。这确保了不同尺度的运动在不同域和时间步上被合理归一化。

### 4.3 权重与 Mask

```python
# pointworld/losses.py:99-161
weights = data_dict["point_weights"]  # 可选的 per-point 权重
pred_exists_supervised = pred & exists & supervised  # 只在预测帧 + 存在 + 有监督的点计算损失
# DROID: supervised = visibility & depth_valid_mask
# BEHAVIOR: supervised = all points (仿真数据全部有监督)
```

### 4.4 Filtered Metrics（评估专用）

```python
# pointworld/losses.py:220-248
# 在评估时，用 expert confidence mask 过滤低置信度点
# 关键指标: filtered_l2_moved/mean
filt = data_dict["scene_filter_mask"].bool()
valid = filt & pred_exists_supervised
```

---

## 5. 评估体系

### 5.1 核心指标

| 指标 | 含义 | 重要性 |
|------|------|--------|
| `l2_moved/mean` | 运动点的平均 L2 误差 | 核心指标——机器人交互区域的预测精度 |
| `l2_static/mean` | 静止点的平均 L2 误差 | 不应动的物体是否被错误预测为运动 |
| `l2/mean` | 所有点的平均 L2 误差 | 全局精度 |
| **`filtered_l2_moved/mean`** | **专家置信度过滤后的运动点 L2** | **论文核心指标** |
| `confidence/mean` | 平均置信度 | 不确定性建模质量 |
| `pred_moved/max` | 最大预测运动幅度 | 预测范围 |

### 5.2 评估流程

```
1. 训练 expert model（在 test split 上）
2. Expert model 推理 → confidence annotation → H5 文件
3. 加载 target model checkpoint
4. 对每个 test sample:
   a. 前向推理 → scene_flows + confidence
   b. 注入 expert confidence mask
   c. 计算 filtered L2 metrics
5. 输出 metrics.json
```

### 5.3 可视化

基于 `viser` 的实时 3D 可视化系统（`visualization/viser_flow/`）：
- 帧序列播放
- GT vs Prediction 切换
- 上采样点渲染
- 场景流/机器人流密度和厚度控制
- 叠加透明度控制

---

## 6. 关键创新总结

### 6.1 统一 3D 点流表示

场景和机器人**统一为 3D 点流**——不需要分别建模场景变化和机器人运动，一个表示搞定。这是对 2D 视频/像素流世界模型的根本升级。

### 6.2 DINOv3 多视角 2D→3D 特征注入

不依赖纯 3D 特征（点云缺乏纹理语义），而是用冻结的 DINOv3 ViT-L16 从多视角 RGB 图像提取语义特征，通过 3D→2D 投影 + 深度一致性 + 跨相机聚合"注入" 3D 点。这是 2D 视觉预训练和 3D 结构理解的桥梁。

### 6.3 PTv3 作为 3D Dynamics Backbone

首次将 PointTransformerV3 用于世界模型的动力学预测，而非传统的 2D CNN/Transformer。PTv3 的 voxelization + serialized attention + multi-scale encoder-decoder 天然适配 3D 点流的时空预测。

### 6.4 FiLM 双重调制

- **Skip FiLM**：保留原始输入信息
- **Robot Global Summary FiLM**：将机器人全局信息传播到每个场景点
两个 FiLM 用可学习的 γ/β 参数，从恒等映射起步。

### 6.5 异方差不确定性 + Expert Confidence Filtering

不是"模型说预测完了就算完"——每个点都输出置信度，低置信度点在评估中被过滤。对于真实世界噪声数据（DROID），这是确保公平评估的关键。

### 6.6 跨域训练

同一个模型在 DROID（真实）和 BEHAVIOR（仿真）上联合训练，仿真域的不确定性用常数替代，真实域正常预测。这是迈向通用世界模型的重要一步。

---

## 7. 源码架构全景

```
PointWorld/
├── pointworld/          # 核心模型代码
│   ├── base.py          # BaseModel: Scene Encoder + Dynamics Predictor + Loss
│   ├── embeddings.py    # TemporalEmbedding: SinCos + MLP
│   ├── losses.py        # NLL + Huber loss, per-domain metrics
│   ├── metrics.py       # L2, confidence, weights 统计
│   ├── norm_stats.py    # Per-domain per-timestep 归一化
│   ├── checkpoint_contract.py  # Checkpoint 元数据契约
│   └── urdfpy_compat.py  # numpy 版本兼容
├── scene_featurizer.py  # SceneEncoder2D + SceneFeatureEncoder
├── robot_sampler.py     # RobotSampler: URDF → GPU FK → 点采样
├── ptv3/                # PointTransformerV3 (vendored + adapted)
│   ├── ptv3.py          # PTv3 实现 + MLP
│   ├── ptv3_arch.yaml   # small/base/large 架构配置
│   ├── structure.py     # 点云结构定义
│   ├── module.py        # 注意力/MLP 模块
│   └── serialization/   # z-order + hilbert 曲线序列化
├── dataset_components/  # 数据管线
│   ├── pipeline.py      # Transform pipeline (增强链)
│   ├── dataloader.py    # WebDataset + DataLoader 构建
│   ├── decoders.py      # WDS 解码 + SimDataDecoder
│   ├── cameras.py       # 多相机采样 + 标准化
│   ├── robot.py         # 机器人特征聚合 + 双臂处理
│   ├── transforms.py    # 全部数据增强 (SphereCrop, GridSample, 等)
│   ├── constants.py     # 所有 release 固定常量
│   ├── collate.py       # Batch collation
│   └── utils.py         # Hash, rotate 等工具
├── training/
│   └── trainer.py       # Trainer: DDP + AMP + wandb + NaN 检测
│   └── checkpointing.py # Checkpoint 保存/加载
├── evaluation/
│   ├── tester.py        # Tester (extends Trainer): 全量评估 + 可视化
│   ├── metrics.py       # _WeightedStat: 统计聚合
│   ├── annotation.py    # ConfidenceHelper: expert confidence H5 管理
│   └── meta.py          # 评估元数据累加器
├── visualization/
│   ├── prediction_viz/  # PredictionVisualizer: 3D 预测可视化
│   └── viser_flow/      # viser-based 实时 3D viewer
├── deploy/              # 部署工具 (transform_utils_torch.py, robots.py)
├── assets/              # URDF + mesh 文件 (franka, r1pro)
├── stats/               # norm_stats.json (droid, droid_behavior)
├── transform_utils.py   # SE(3) 变换工具 (从 OmniGibson adapted)
├── utils.py             # NaN 检测, soft selector, robot URDF resolve
├── arguments.py         # 全部 CLI 参数定义
├── train.py             # 训练入口
├── eval.py              # 评估入口
└── environments/        # conda env yml
```

---

## 8. 与同类工作的对比定位

| 维度 | PointWorld | UniSim | 3D-Diffuser-Actor | DINO-WM |
|------|-----------|--------|-------------------|---------|
| **表示** | 3D 点流 | 2D 视频 | 3D 点云 + diffusion | 2D 特征 |
| **场景编码** | DINOv3 ViT-L16 多视角投影 | 视频 encoder | 点云 encoder | DINOv2 单视角 |
| **动力学 backbone** | PTv3 (点云 Transformer) | 2D Transformer | Diffusion | 2D Transformer |
| **机器人表示** | 3D 点流 (FK 点采样) | 2D 动作向量 | 3D 关键点 | 2D 动作 |
| **不确定性** | 异方差 per-point | 无 | 无 | 无 |
| **跨域训练** | DROID + BEHAVIOR | 单域 | 单域 | 单域 |
| **多视角** | 1-6 相机 | 单视角 | 单视角 | 单视角 |
| **预测输出** | 每点 3D 位移 × 10 步 | 未来视频帧 | 未来关键点位置 | 未来特征 |

PointWorld 的核心差异化：
1. **纯 3D 表示**——不回到 2D，场景和机器人都在 3D 空间预测
2. **多视角 2D→3D 注入**——利用 2D 预训练但不被 2D 限制
3. **per-point 不确定性**——不是全局"我预测得好不好"，而是"这个点我预测得好不好"
4. **可扩展**——PTv3 支持 small/base/large 三档，点云天然支持变长输入

---

## 9. 局限性与未来方向

### 9.1 当前局限（从源码和 README 提取）

1. **Precomputed datasets 和 pretrained checkpoints 尚未发布**（仍在 NVIDIA 内部审核）
2. **Eval 输出在 GPU 上非确定性**——小范围 run-to-run 变差
3. **Partial-batch 比较敏感**——`eval_num_batches` 和 `num_workers` 需匹配
4. **只支持 Franka + Robotiq / R1Pro**——其他机器人需要新 URDF
5. **分辨率固定 180×320**——高分辨率需修改 pipeline
6. **BEHAVIOR 域的不确定性是常数**——仿真数据无法真正评估不确定性建模质量

### 9.2 潜在研究方向

1. **闭环控制**：当前是开环预测，需要结合 MPPI/采样规划做闭环部署
2. **更长时序预测**：当前 PRED_HORIZON=10，可扩展到更长时序（需要 hierarchical/multi-scale）
3. **更多机器人**：扩展 URDF 支持到更多机器人形态
4. **Diffusion + Point Flow**：将确定性预测替换为扩散生成，支持多模态未来
5. **更大规模预训练**：更多域、更多数据、更大模型
6. **语言条件**：加入语言指令作为额外条件信号

---

## 10. 核心数据参数速查表

| 项 | 值 |
|----|-----|
| 总时间步 T | 11 (context=1 + pred=10) |
| 场景点数 Ns | ≤ 12000 |
| 机器人点数 Nr | 500 (gripper only) |
| PTv3 规格 | base (7 enc stages, 6 dec stages) |
| predictor_dim (channels) | 256 |
| DINOv3 | ViT-L16, 4 层 [4,11,17,23], feat_dim=1024 |
| 多层 concat | 4096 → Linear → 256 |
| 深度一致性阈值 | 3mm (behind) / 1.5mm (front) |
| Voxel grid_size | 0.015 |
| Huber delta | 5.0 |
| VAR_FLOOR / VAR_CEILING | 1e-6 / 1e2 |
| 训练相机数 | 1-3 (随机) |
| 评估相机数 | 2 (固定) |
| 图像分辨率 | 180 × 320 |
| Batch size | 22 |
| 学习率 | 1e-4 (AdamW) |
| Weight decay | 0.01 |
| Drop path | 0.3 |
| AMP dtype | bfloat16 |
| DDP | 支持 |
| 数据格式 | WebDataset (tar shards) |
| 仿真域不确定性 | 常数（真实域 log_var 均值） |
| dynamics_head init scale | 1.0 (DROID) / 0.0 (BEHAVIOR) |
| log_var_head init bias | log(0.005²) ≈ -11.5 |

---

## 11. 参考资源

- 论文：[arXiv 2601.03782](https://arxiv.org/abs/2601.03782)
- 项目主页：[point-world.github.io](https://point-world.github.io/)
- 源码仓库：[github.com/NVlabs/PointWorld](https://github.com/NVlabs/PointWorld)
- YouTube 视频：[XPOsCwrYdk0](https://youtu.be/XPOsCwrYdk0)
- DINOv3：[github.com/facebookresearch/dinov3](https://github.com/facebookresearch/dinov3)
- PTv3：[github.com/Pointcept/PointTransformerV3](https://github.com/Pointcept/PointTransformerV3)
- DROID 数据集：[github.com/stanford-tross-group/droid_dataset](https://github.com/stanford-tross-group/droid_dataset)
- BEHAVIOR / OmniGibson：[github.com/StanfordVL/OmniGibson](https://github.com/StanfordVL/OmniGibson)

---

> **声明**：本报告基于 PointWorld v1 源码（`main` 分支，commit 截至 2026-01）的逐文件深度阅读撰写。论文 HTML/PDF 因网络限制未能直接获取，所有技术细节均从源码交叉验证推导，与 README 元数据对齐。标注 `[源码验证]` 的内容直接对应代码实现。