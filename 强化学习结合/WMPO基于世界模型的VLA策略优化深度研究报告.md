# WMPO：基于世界模型的 VLA 策略优化深度研究报告

研究对象：

- 论文：WMPO: World Model-based Policy Optimization for Vision-Language-Action Models，arXiv HTML v1：<https://arxiv.org/html/2511.09515v1>
- 项目主页：<https://wm-po.github.io/>
- 官方源码：<https://github.com/WM-PO/WMPO>，本次拉取到的源码提交为 `c836d74 clean repo`
- 研究时间：2026-06-11

## 1. 一句话结论

WMPO 的核心价值在于：它把 VLA 的在线强化学习从昂贵的真实机器人交互中“搬到”一个像素级视频生成世界模型里完成，并用轨迹级成功/失败奖励驱动 GRPO 更新策略。论文的关键判断是，面向 VLA 的世界模型不能只停留在抽象 latent dynamics 中；由于 VLA 的视觉表征来自大规模真实图像预训练，世界模型最终必须产生足够接近真实观测的像素轨迹，才能让策略在“想象环境”里学到可迁移的动作修正能力。

我的判断：WMPO 是“生成式世界模型 + on-policy VLA RL”方向里相当清晰的一次系统集成。它的强项不只是提出算法，而是把 OpenVLA-OFT、OpenSora、VideoMAE、verl、Mimicgen 这些部件打通成一个可运行训练管线。它的主要风险也来自这里：系统非常重、任务和路径较硬编码、依赖高质量世界模型和奖励模型，真实机器人上的复杂接触误差仍然可能被世界模型漏掉。

## 2. 论文要解决的问题

VLA 模型通常通过 imitation learning 从专家演示学习。这个范式的问题是：模型只见过成功轨迹，遇到训练分布外状态时容易错误累积；一旦动作把物体推偏、夹爪卡住、末端执行器进入不良位姿，模仿学习策略通常不知道如何恢复。

强化学习理论上可以通过失败和自我试错解决这个问题，但真实机器人 rollout 昂贵、慢、不安全，而且 on-policy RL 尤其吃交互量。已有实用做法常退到 off-policy 或 DPO 式偏好学习，但它们很难持续从当前策略分布中学习，容易受静态数据限制。

WMPO 的问题设定就是：能不能用一个 action-conditioned video world model 替代真实环境，让 VLA 在世界模型中进行 on-policy GRPO，从而保留 on-policy 学习的优势，同时显著降低真实机器人交互成本？

## 3. 方法总览

WMPO 的训练循环由三部分组成：

1. 想象轨迹生成：从真实初始状态 `s0` 出发，VLA 策略产生一个 action chunk，世界模型根据最近观测帧和动作 chunk 生成下一段视频帧；重复这个过程直到完整轨迹结束。
2. 轨迹采样与打分：对同一个初始状态采样多条轨迹，用 VideoMAE 奖励模型判断每条轨迹是否成功。
3. 策略更新：对同组轨迹的成功/失败奖励做组内标准化，使用 GRPO 风格的 clipped policy objective 更新 VLA。

形式化地，论文把任务写成 MDP `M=(S,A,P,R)`：

- 状态 `S = I x G`，其中 `I` 是图像观测序列，`G` 是语言指令。论文明确假设机器人状态可以仅由图像观测定义，POMDP 和更复杂状态留作未来工作。
- 动作 `A` 是 action chunk。每个 action chunk 长度为 `K`，每个动作是 `D` 维控制向量；策略优化时每个维度离散成 256 个 bins。
- 转移函数由世界模型实现：`s_{t+1} ~ p_phi(s_{t+1} | s_t, a_t)`。
- 奖励函数由学习得到的 `R_psi` 实现，输入完整轨迹，输出二值成功标签。

目标是最大化世界模型生成轨迹上的期望成功奖励：

```text
max_theta E_{tau ~ pi_theta, p_phi}[R_psi(tau)]
```

这个目标的关键不是公式本身，而是把真实环境 `P` 替换成世界模型 `p_phi` 后，仍然让策略看到“像真实相机观测”的图像序列。

## 4. 世界模型设计

论文使用 OpenSora 的视频扩散 backbone，但为机器人操控做了几处关键改造。

第一，像素空间对齐。WMPO 不让 VLA 直接在 RSSM 一类抽象 latent state 中训练，而是让世界模型在 VAE latent 中扩散生成后，再 decode 回像素空间给 VLA 使用。论文认为这是因为 VLA 的视觉理解来自 web-scale 图像/视频预训练，像素级轨迹能更好复用这些表征。这个判断很重要：WMPO 的世界模型虽然内部仍在 latent 中扩散，但训练策略时暴露给 VLA 的接口是图像，而不是世界模型自己的隐变量。

第二，VAE 替换。论文称将 OpenSora 原有 3D VAE 换为 SDXL 的 2D VAE，以减少过强时序压缩带来的细粒度运动失真。机器人操作中的微小位移、接触、姿态偏差比普通视频更敏感，3D 时序压缩可能让物体和夹爪关系被抹平。

第三，长时序自回归与 noisy-frame conditioning。世界模型一次预测 `K=8` 帧，随后把已生成帧作为下一段生成的条件。长时间滚动会积累误差，所以论文在训练时给条件帧加入扩散噪声，强度为 50/1000 diffusion steps，使模型习惯“不完美的历史帧”。这是为了让它在用自己生成的帧继续生成时不迅速崩坏。

第四，frame-level action control。论文把动作信号注入 AdaLN 调制，使每一帧的动作与对应视觉变化对齐。源码中对应 `dependencies/opensora/opensora/models/stdit/stdit3.py`：`ActionEncoder` 将 action chunk 中每一帧的动作向量经过 MLP 映射到 hidden size，并加到 timestep embedding 上；`STDiT3Block` 再通过 scale/shift/gate 调制 spatial 与 temporal transformer blocks。这个实现与论文“动作和扩散 timestep 一起做帧级 AdaLN 调制”的描述吻合。

第五，policy behavior alignment。世界模型先在 Open X-Embodiment 轨迹上预训练，再用当前策略自己采集的真实 rollout 对下游任务微调。这样做的理由是：大规模专家数据主要是成功轨迹，缺少失败状态；如果世界模型只学成功演示，它想象不出当前策略会制造的失败、卡住和碰撞状态，强化学习就会在错误环境中更新。

## 5. 奖励模型

WMPO 使用轻量 VideoMAE 成功/失败分类器作为轨迹级奖励模型。

论文的数据构造方式是：

- 成功轨迹的 terminal clip 是正样本。
- 成功轨迹的中间 clip 是负样本。
- 失败轨迹中的任意 clip 或 terminal clip 是负样本。
- 推理时用长度 `L=8` 的滑动窗口、stride `s=1` 扫描完整轨迹；只要某个 clip 的成功概率超过阈值 `tau_thr`，轨迹就被判定为成功。

源码中有两条奖励模型路径：

- `reward_model/videomae.py` 是独立 DDP 训练脚本，使用 `VideoMAEForVideoClassification`、`num_frames=8`、`num_labels=2`，训练时用 `CrossEntropyLoss`，并在验证时扫阈值找 F1 最优。
- `verl/workers/fsdp_workers.py` 的 `_build_world_model` 会在 WMPO 训练过程中加载 VideoMAE reward model，并把它传给 `RobWMHFRollout`。

一个细节：论文说奖励模型使用 binary cross-entropy loss，但源码的主训练实现实际是二分类 `CrossEntropyLoss`；独立推理脚本 `reward_model/inference_videomae.py` 又用 `sigmoid(logits)` 的第二维做阈值判断，而训练评估处用 `softmax(logits)[:, 1]` 扫阈值。二分类下这通常不改变排序，但阈值数值不可直接跨脚本解释。

各任务开源脚本中的 reward threshold 是任务相关的：

| 任务 | 阈值 |
| --- | ---: |
| Coffee | 0.82 |
| StackThree | 0.66 |
| ThreePieceAssembly | 0.62 |
| Square | 0.96 |

这说明奖励模型不是一个全任务统一标定的 verifier，而是每个任务单独训练/调阈值的成功分类器。

## 6. GRPO 策略优化

WMPO 采用 Group Relative Policy Optimization。对每个真实初始状态，采样 `G=8` 条世界模型轨迹，得到奖励 `R_i in {0,1}`。组内优势为：

```text
A_i = (R_i - mean(R_1...R_G)) / std(R_1...R_G)
```

如果同组 8 条轨迹全成功或全失败，组内标准差接近 0，没有相对学习信号。论文采用 Dynamic Sampling：直接丢弃这种 group，继续采样直到 batch 填满。

源码中这个逻辑落在 `verl/trainer/ppo/ray_trainer.py`：

- 训练 batch 从 `StateDataset` 中取初始状态。
- 每个初始状态复制 `n_samples=8`。
- 生成后用 `RobRewardManager.verify` 读 `complete` 字段作为成功标签。
- `filter` 函数将每个 group 的平均成功率限制在 `accuracy_lower_bound=0.1` 到 `accuracy_upper_bound=0.9`，等价于过滤掉全 0 或全 1 的 group。
- `compute_grpo_outcome_advantage` 按 `uid` 聚合同一初始状态的 8 条轨迹，计算组内标准化优势。

策略目标是 PPO/GRPO 式 clipped objective：

```text
min(r_i,t(theta) * A_i,
    clip(r_i,t(theta), 1 - eps_low, 1 + eps_high) * A_i)
```

其中 `eps_low=0.20`，`eps_high=0.28`。论文还强调遵循 DAPO，去掉 KL regularization，不需要 reference model，从而降低显存并鼓励探索。开源脚本中也设置 `algorithm.kl_ctrl.kl_coef=0.00`。

值得注意的是，源码里的 `RobRewardManager` 会把完成奖励放在 `finish_step * action_token_len - 1` 的 token 位置，其他 token reward 为 0；`compute_advantage` 再用 `finish_step` 构造 mask，只对完成前的动作 token 计算优势。默认配置里 `verifier.reward_coef=5`，但在 GRPO 组内标准化后，单纯比例缩放不会改变优势方向。

## 7. 实验设置与结果

### 7.1 仿真任务

论文使用 Mimicgen 中四个精细操作任务：

- Coffee_D0
- StackThree_D0
- ThreePieceAssembly_D0
- Square_D0

基础策略是 OpenVLA-OFT，在每个任务 300 条专家轨迹上 SFT。为简化实验，作者去掉了机器人 proprioceptive state 和 wrist camera，只使用主视角图像输入。action chunk 长度为 `K=8`。

实验使用两种真实 rollout budget：`P=128` 和 `P=1280`。这些真实轨迹由 base policy 采集，用于 fine-tune world model 和训练 reward model。世界模型用 `c=4` 条件帧和一个 action chunk 预测后续 `K=8` 帧。每个任务评测 128 个不同初始状态，报告平均成功率。

### 7.2 主结果

| Rollout budget P | 方法 | Coffee | StackThree | ThreePieceAssembly | Square | Mean |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| - | Base policy | 43.8 | 46.9 | 19.5 | 24.2 | 33.6 |
| 128 | GRPO | 38.3 | 52.3 | 17.2 | 25.0 | 33.2 |
| 128 | DPO | 43.8 | 53.9 | 23.4 | 28.1 | 37.3 |
| 128 | WMPO | 61.7 | 56.3 | 37.5 | 32.8 | 47.1 |
| 1280 | GRPO | 47.7 | 54.7 | 20.3 | 25.8 | 37.1 |
| 1280 | DPO | 52.3 | 57.0 | 26.7 | 33.6 | 42.4 |
| 1280 | WMPO | 75.0 | 64.1 | 46.1 | 45.3 | 57.6 |

论文给出的关键结论：

- `P=128` 时，WMPO 平均 47.1%，比最强 baseline DPO 的 37.3% 高 9.8 个百分点。
- `P=1280` 时，WMPO 平均 57.6%，比 DPO 的 42.4% 高 15.2 个百分点。
- GRPO 在真实环境中直接做 on-policy 更新时受 rollout 数量限制，更新次数少，尤其小预算下不稳定。
- DPO 可重复利用离线数据，但不能随当前策略分布继续在线改进，性能容易平台化。
- 奖励模型在所有任务上 F1 超过 0.95，作者认为它能可靠地区分成功/失败，并缓解 reward hacking。

### 7.3 涌现行为

论文重点展示了 Square 任务的自纠正行为：base policy 遇到碰撞后继续把方块顶在杆上，直到超时失败；WMPO 策略会学会抬起、重新对齐、再插入。这个行为通常不会出现在纯专家演示中，因为专家数据很少包含失败和恢复过程。

作者还比较了成功轨迹的平均长度，WMPO 成功轨迹更短。解释是：WMPO 通过失败惩罚减少卡住状态，从而让策略更快、更流畅地完成任务。

### 7.4 泛化扰动

论文测试三类分布外扰动：

- Square：stick 位置从固定变为矩形范围内随机。
- StackThree：桌面背景换成灰色。
- ThreePieceAssembly：红色底座换成深木色底座。

| 方法 | Position Disruption | Background Disruption | Texture Disruption | Mean |
| --- | ---: | ---: | ---: | ---: |
| Base policy | 14.1 | 46.1 | 10.9 | 23.7 |
| GRPO | 15.6 | 47.7 | 10.9 | 24.7 |
| DPO | 16.4 | 34.4 | 7.8 | 19.5 |
| WMPO | 22.3 | 50.0 | 16.4 | 29.6 |

WMPO 全部扰动下最好，但绝对成功率仍不高。我的解读是：WMPO 确实提升了策略鲁棒性，但不是“泛化已解决”。尤其 Position/Texture 扰动下成功率仍在 20% 左右，说明策略对视觉变化和几何变化仍然敏感。

### 7.5 Lifelong learning

论文在 StackThree 上做迭代学习：每轮收集 `P=128` 条当前策略真实轨迹，执行 WMPO 后再用新策略继续收集下一轮。DPO baseline 使用同样数据；另以 300、428、556 条专家演示训练的 imitation policy 作参考。结果显示 WMPO 能稳定持续提升，而 DPO 迭代不稳定。

这里的意义是：WMPO 的真实数据不要求人工专家继续标注，而是由当前策略自己采集。这更接近部署后的持续学习场景。

### 7.6 真实机器人

真实实验任务是 Mobile ALOHA 上的 “Insert the square into the stick”，方块与杆之间 clearance 只有 5mm。设置：

- 用 200 条高质量专家演示 fine-tune OpenVLA-OFT base policy。
- 部署 base policy 采集 128 条轨迹。
- 用这些轨迹进一步 fine-tune world model，并在 world model 中优化策略。
- 与同数据集训练的 offline DPO 对比。
- 每个模型真实评测 30 次。

成功率：

| 方法 | 成功率 |
| --- | ---: |
| Base policy | 53% |
| DPO | 60% |
| WMPO | 70% |

论文附录也给出一个失败案例：世界模型直到最后一帧都基本准确，但没能捕捉方块因为细微扰动卡在杆上的瞬间。这是世界模型用于机器人接触任务的核心风险：大部分视觉轨迹正确，不代表关键接触事件一定正确。

## 8. 源码总体结构

官方仓库不是一个轻量 demo，而是多个大系统拼起来的训练栈。核心目录：

| 路径 | 作用 |
| --- | --- |
| `README.md` | 项目说明、数据/权重组织、训练和评测入口 |
| `download_hf.py` | 从 Hugging Face `fangqi/WMPO` 下载 `checkpoint_files/**` 与 `data_files/**` |
| `install.sh` | 安装 OpenSora、OpenVLA-OFT、robosuite、robomimic、mimicgen 等依赖 |
| `examples/mimicgen/*/train_wmpo_*.sh` | 四个任务、两种 rollout budget 的 WMPO 训练入口 |
| `examples/mimicgen/*/evaluate.sh` | 评测入口 |
| `examples/opensora/*.sh` | world model 训练入口，实际配置在下载的 checkpoint 目录中 |
| `reward_model/` | 独立 VideoMAE 奖励模型训练、推理、阈值搜索脚本 |
| `verl/` | 修改过的 verl RL 训练框架，包含 trainer、worker、rollout、GRPO/PPO 算法 |
| `dependencies/opensora/` | 修改过的 OpenSora，加入机器人动作条件与 VLA 数据集 |
| `dependencies/openvla-oft/` | OpenVLA-OFT 依赖代码 |

README 说明完整 checkpoint 和数据约 `364GiB + 530GiB`。这意味着源码虽开源，但复现实验实际依赖大规模下载与多机 GPU 环境。

## 9. 源码中的端到端训练流程

以 `examples/mimicgen/coffee/train_wmpo_128.sh` 为例，训练入口执行：

```text
python -m verl.trainer.main_ppo
```

脚本通过 Hydra 覆盖大量配置，包括：

- `data.n_samples=8`
- `data.filter_accuracy=True`
- `data.accuracy_lower_bound=0.1`
- `data.accuracy_upper_bound=0.9`
- `data.train_batch_size=64`
- `actor_rollout_ref.actor.optim.lr=5e-6`
- `actor_rollout_ref.actor.clip_ratio_low=0.2`
- `actor_rollout_ref.actor.clip_ratio_high=0.28`
- `actor_rollout_ref.rollout.temperature=1.6`
- `actor_rollout_ref.wm.enable=True`
- `algorithm.adv_estimator=grpo`
- `algorithm.kl_ctrl.kl_coef=0.00`

这些与论文 Table 4 基本一致。

端到端路径如下：

1. `verl/trainer/main_ppo.py` 读取配置并初始化 Ray。
2. 如果 `actor_rollout_ref.wm.enable=True`，worker mapping 使用 `RobWMActorRolloutRefWorker`，否则使用真实环境 rollout worker。
3. `RayTrainer._create_dataloader` 从 `verl/utils/dataset/{task}_d0_states.pkl` 读取初始环境状态。
4. `RayTrainer.fit` 每轮从 dataloader 取 batch，并将每个初始状态复制 `n_samples=8` 次。
5. `actor_rollout_wg.generate_sequences` 根据 meta 中的 `use_wm=True` 路由到 world model rollout。
6. `RobWMHFRollout._generate_wm_minibatch` 根据 `state_id` 从 `data_files/first_images/{task}/{state_id}.png` 取初始图像。
7. `run_wm_inference` 循环执行：当前图像进入 VLA，VLA 采样动作 chunk，动作 chunk 进入世界模型，世界模型生成后续 8 帧，取最后一帧作为下一轮 VLA 输入。
8. 完整视频生成后，`predict_success` 用 VideoMAE 滑窗判断是否成功，并返回 `complete` 与 `finish_step`。
9. `RobRewardManager` 将 `complete` 转成稀疏 token-level reward。
10. `compute_advantage` 根据同一 `uid` 的 8 条轨迹计算 GRPO 组内优势。
11. actor worker 计算旧策略 log prob、新策略 log prob 和 clipped policy loss，更新 VLA。

这个代码路径说明，WMPO 不是简单“离线用世界模型生成数据再 SFT”。它在每轮训练中用当前策略和世界模型闭环生成新轨迹，再进行 on-policy 风格更新。

## 10. 真实环境 rollout 与评测路径

仓库中也保留了真实模拟环境 rollout 类 `verl/workers/rollout/rob_rollout.py`。它通过 robomimic/Mimicgen 环境 reset 到指定 state，然后让 VLA 在环境中执行动作 chunk，直到成功或达到任务最大步数。

任务最大步数在 `robwm_rollout.py` 与 `rob_rollout.py` 中硬编码：

| 任务 | max_steps |
| --- | ---: |
| Coffee | 256 |
| StackThree | 320 |
| ThreePieceAssembly | 384 |
| Square | 184 |
| ALOHA | 224 |

评测脚本 `examples/mimicgen/*/evaluate.sh` 设置 `trainer.val_only=True`。`RayTrainer._validate` 中的 meta 没有传 `use_wm=True`，因此默认调用真实环境 rollout，而不是世界模型 rollout。这一点很重要：训练时在世界模型中优化，论文表格成功率则是回到真实仿真环境评测。

## 11. 世界模型训练实现

世界模型训练入口是 `examples/opensora/{task}_{budget}.sh`，底层调用：

```text
dependencies/opensora/scripts/train_web.py
```

不过具体 world model 配置文件位于下载后的：

```text
checkpoint_files/world_models/{task}/P_128/train_mimicgen_cfg.py
checkpoint_files/world_models/{task}/P_1280/train_mimicgen_cfg.py
```

这些配置不在 Git 仓库静态源码里，需要执行 `download_hf.py` 获取。

OpenSora 训练脚本会：

- 构建 `SimpleVLAWebDataset`。
- 加载 VAE 并计算 latent size。
- 构建 STDiT3 diffusion model。
- 构建 scheduler。
- 用 ColossalAI Booster、HybridAdam、EMA 做分布式训练。

`SimpleVLAWebDataset` 的关键逻辑：

- 从 WebDataset tar 中读取 `video.npy`、`action.npy`、`meta.json`。
- 默认 `To=4` 条件帧，`Ta=8` 待预测动作/视频帧。
- 按 `finish_step` 截取有效窗口。
- 视频归一化到 `[-1,1]`。
- 动作用统计量 `q01/q99` 归一化到 `[-1,1]`。

这与论文“给定 4 帧条件 + 1 个 action chunk，预测 8 帧”的设置一致。

## 12. 在线更新世界模型的代码能力

论文主流程强调 policy behavior alignment，即用 policy rollout 对世界模型微调。开源脚本的 `train_wmpo_*.sh` 中设置：

```text
+actor_rollout_ref.wm.update_wm=False
```

也就是说，默认 WMPO 训练脚本加载已经训练好的 world model，不在每个策略 epoch 中继续更新 world model。

但 `verl/workers/fsdp_workers.py` 中确实保留了 `_update_world_model` 和 `_update_terminal_model` 相关代码：

- `_build_world_model_dataloader` 从 rollout shard 构建 world model 训练数据。
- `_update_world_model` 先评估 latent L2，再对 diffusion model 做若干步训练，并更新 EMA。
- `_build_terminal_model_dataloader` 混合 rollout shards 和 expert shards，用于更新 VideoMAE terminal reward model。

这说明项目实现上支持交替更新，只是默认复现实验使用下载好的 world model checkpoint。报告解读时需要区分“论文方法中的 alignment 数据阶段”和“开源脚本默认的策略优化阶段”。

## 13. 复现门槛与工程风险

这个项目的复现门槛高，主要体现在以下方面。

硬件规模：论文附录写明 OpenVLA-OFT SFT 用 8 张 H100，world model 训练和 policy optimization 用 32 张 H100。开源脚本默认 `NUM_NODES=4`、`NUM_GPUS_PER_NODE=8`。单机小卡环境基本只能做代码阅读或小规模改造，不能期望复现主结果。

数据和模型体积：README 明确提示 checkpoint 与数据约 `364GiB + 530GiB`。`download_hf.py` 会下载 `checkpoint_files/**` 和 `data_files/**`，如果只关心某个任务，应该手动改 allow patterns。

系统依赖复杂：`install.sh` 包含 `sudo apt install`、OpenSora editable install、OpenVLA-OFT、Mujoco、robosuite、robomimic、mimicgen、robosuite-task-zoo 等。Python 依赖还包括 TensorFlow 2.19、xformers、flash-attn、diffusers、Ray、ColossalAI 相关栈。

路径硬编码：部分脚本有 `/mnt/hdfs/zhufangqi/...`、`/opt/tiger/...` 等作者内部路径，尤其 `reward_model` 独立脚本默认值明显需要用户修改。训练脚本还会把 `checkpoint_files/modeling_prismatic.py` 等文件复制进模型目录。

配置依赖下载文件：世界模型的 `inference_cfg.py` 和 `train_mimicgen_cfg.py` 不在 Git 源码中，而在 Hugging Face checkpoint 目录里。只 clone GitHub 不足以运行完整实验。

任务扩展性有限：当前 Mimicgen 脚本只覆盖四个任务，`RobWMHFRollout` 中任务名、语言描述、最大步数、first image 路径都有硬编码。迁移到新任务需要补数据统计、初始状态、初始图像、reward model、world model 配置和任务描述。

## 14. 方法优势

WMPO 的第一大优势是把 on-policy 更新重新带回 VLA 场景。真实环境 GRPO 在 rollout budget 小时几乎没法稳定更新，DPO 又受静态数据限制；WMPO 用世界模型生成当前策略分布下的大量轨迹，能持续制造组内成功/失败对比。

第二大优势是 reward 设计朴素。它不做复杂 reward shaping，只判断轨迹是否成功。这样避免了手工设计 dense reward，也降低了 reward hacking 空间。VideoMAE 滑动窗口让成功检测不一定依赖固定终止时刻。

第三大优势是像素级世界模型与 VLA 表征兼容。很多 model-based RL 的 latent world model 很高效，但对于已经在图像上预训练的 VLA foundation model，latent 接口会造成表示错位。WMPO 的设计让 VLA 继续吃图像。

第四大优势是系统验证较完整。论文不只报告仿真主表，还给出自纠正行为、轨迹长度、扰动泛化、lifelong learning 和真实机器人结果。源码也开出了训练脚本、数据下载、奖励模型、世界模型训练入口，不只是伪代码。

## 15. 局限与批判性分析

第一，世界模型误差会直接变成策略优化偏差。论文附录真实机器人失败案例已经说明，世界模型可能在最后微小接触事件上判断错误。机器人精细装配常常成败就在毫米级接触，视觉上“差不多”不等于动力学上正确。

第二，reward model 仍可能被策略利用。作者报告 reward model F1 超过 0.95，但它是在已收集轨迹分布上验证的。策略在世界模型中优化后可能进入 reward model 未覆盖的视觉状态，出现“看起来像成功 clip”的伪成功。当前方法主要依赖高质量 policy behavior alignment 和滑窗阈值缓解。

第三，真实预算公平性需要细读。论文保证不同方法使用相同真实 rollout budget `P`，但 WMPO 额外依赖大规模 OXE 预训练世界模型、OpenSora 生成模型和大量计算。若比较“真实机器人交互”成本，WMPO 明显占优；若比较“总训练算力和系统复杂度”，成本并不低。

第四，基线 GRPO 被真实 rollout 数量严重限制。附录说明 batch size 64、group size 8 意味着一次更新至少 512 条真实轨迹，动态采样还会过滤更多。因此 `P=128` 时大 batch GRPO 不可行，`P=1280` 也只能更新一两次。这个问题本身正是 WMPO 要解决的瓶颈，但也意味着主表中的 GRPO baseline 不是“算力充分的 GRPO”，而是“同真实交互预算下的直接 GRPO”。

第五，当前动作空间是离散 action tokens。论文附录也承认目前聚焦离散动作表示，未来才扩展到 flow-based policies 和 FlowGRPO。对于高频连续控制或需要多模态连续动作分布的任务，当前实现未必最自然。

第六，观测简化限制了结论外推。论文为了简化去掉 proprioceptive state 和 wrist camera，并假设图像足以定义状态。这对 Mimicgen 桌面任务可行，但对遮挡、多接触、力控、双臂协作、非刚体操作未必成立。

第七，泛化仍然有限。扰动表中 WMPO 平均 29.6%，高于 baseline，但绝对成功率不高。它证明了方向有效，不证明世界模型训练自动带来强泛化。

第八，缺少足够细的消融表。论文文字强调 noisy-frame conditioning、frame-level action control、policy behavior alignment 都关键，但 v1 正文没有看到系统的 quantitative ablation 表来分别证明每个组件贡献。

## 16. 对“强化学习结合世界模型”方向的启示

WMPO 对具身世界模型研究有几个重要启示。

首先，世界模型不一定要替代策略表征。很多路线会把策略也迁入世界模型 latent；WMPO 反而强调保持 VLA 原有图像输入接口，只让世界模型负责生成图像。这是一种“世界模型作为环境模拟器，而不是策略状态空间”的路线。

其次，成功/失败的失败数据比专家成功数据更关键。Policy behavior alignment 的真正价值是让世界模型学会当前策略会怎样失败。对于强化学习来说，会失败的世界模型比只会生成漂亮成功演示的世界模型更有用。

再次，trajectory-level outcome reward 和 group-relative advantage 很适合稀疏机器人任务。机器人操作往往难以定义每步 reward，但最终成功可以判断。GRPO 的组内标准化把同一初始条件下的相对好坏变成学习信号，避免训练 value function。

最后，生成式世界模型的实用性取决于闭环稳定性，而不是单段视频质量。WMPO 特别处理自回归长时序、条件帧噪声和动作帧对齐，说明 robotics world model 不能只看短视频生成指标，要看闭环 rollout 能否维持动力学一致性。

## 17. 如果要在本地继续研究或复现

建议按从轻到重的顺序做：

1. 只读代码：clone 官方仓库，重点读 `README.md`、`examples/mimicgen/*/train_wmpo_*.sh`、`verl/trainer/main_ppo.py`、`verl/trainer/ppo/ray_trainer.py`、`verl/workers/rollout/robwm_rollout.py`、`dependencies/opensora/opensora/models/stdit/stdit3.py`。
2. 下载单任务 checkpoint：修改 `download_hf.py` 的 `allow_patterns`，只下载一个任务的 SFT/WMPO/world_model/reward_model 和必要 data_files。
3. 先跑 evaluate：比训练便宜，能验证环境、Mimicgen、OpenVLA-OFT checkpoint、Ray worker 是否能通。
4. 再跑 world model inference：用 `trainer.rollout_before_train=True` 或 `_save_rollouts` 路径生成可视化轨迹，检查世界模型是否稳定。
5. 最后跑小规模训练：把 `NUM_NODES`、`NUM_GPUS_PER_NODE`、batch、micro batch 调小，但这只能验证代码通路，不应期待复现论文性能。

## 18. 值得继续追问的研究问题

- 如果加入 proprioception、wrist camera、force/torque，世界模型如何融合这些非图像状态？
- 世界模型是否能输出不确定性，用于过滤高风险 imagined trajectories？
- Reward model 能否从 task-specific VideoMAE 变成多任务通用 verifier？
- 能否用真实环境少量回放校正 reward hacking，比如每轮将高奖励想象轨迹抽样回真实环境验证？
- Flow-based VLA action head 与 FlowGRPO 结合后，是否比 256-bin 离散动作更适合精细操作？
- Policy behavior alignment 的最小真实 rollout 数是多少？失败样本比例如何影响世界模型？
- 当前方法对 deformable object、液体、接触丰富装配是否仍成立？

## 19. 总结

WMPO 是一条很有代表性的路线：用像素级视频生成世界模型承接 VLA 的 on-policy 强化学习，把真实机器人交互预算从“每次策略更新都要大量 rollout”转变为“先采少量 policy behavior 轨迹对齐世界模型，再在世界模型里密集优化策略”。论文结果显示，在相同真实 rollout budget 下，它明显优于直接 GRPO 和 offline DPO，并在 Square 任务中出现自纠正行为。

但它不是廉价方案。它把真实交互成本转移到了大规模世界模型预训练、任务级 reward model、复杂分布式训练系统和严格的数据组织上。对于研究来说，WMPO 最值得借鉴的是范式：让世界模型生成 VLA 可消费的像素观测，让当前策略在想象环境里产生成功/失败对比，再用 group-relative on-policy 目标更新策略。对于工程落地来说，真正的难点仍是世界模型的接触精度、reward verifier 的可靠性、以及新任务迁移时的数据和配置成本。

## 20. 本次核验过的主要源码文件

- `README.md`
- `download_hf.py`
- `install.sh`
- `requirements.txt`
- `align.json`
- `examples/mimicgen/coffee/train_wmpo_128.sh`
- `examples/mimicgen/coffee/train_wmpo_1280.sh`
- `examples/mimicgen/square/train_wmpo_128.sh`
- `examples/mimicgen/square/evaluate.sh`
- `examples/opensora/coffee_128.sh`
- `reward_model/videomae.py`
- `reward_model/inference_videomae.py`
- `reward_model/find_thre.py`
- `verl/trainer/main_ppo.py`
- `verl/trainer/ppo/ray_trainer.py`
- `verl/trainer/ppo/core_algos.py`
- `verl/workers/fsdp_workers.py`
- `verl/workers/rollout/rob_rollout.py`
- `verl/workers/rollout/robwm_rollout.py`
- `verl/utils/dataset/rob_dataset.py`
- `dependencies/opensora/scripts/train_web.py`
- `dependencies/opensora/opensora/models/stdit/stdit3.py`
- `dependencies/opensora/opensora/datasets/simplevla_webdataset.py`
- `dependencies/opensora/opensora/datasets/success_classifier_vit_v1_3.py`
- `dependencies/opensora/opensora/datasets/action_tokenizer.py`

## 21. 参考链接

- 论文 HTML v1：<https://arxiv.org/html/2511.09515v1>
- 项目主页：<https://wm-po.github.io/>
- 官方 GitHub：<https://github.com/WM-PO/WMPO>
- Hugging Face 权重与数据入口：<https://huggingface.co/fangqi/WMPO>
