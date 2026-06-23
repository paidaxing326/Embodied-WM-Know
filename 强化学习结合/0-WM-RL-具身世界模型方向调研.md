# WM + RL：具身世界模型作为强化学习训练场的方向调研

> **主题**：Embodied World Model + Reinforcement Learning / VLA-RL / World Action Model  
> **时间**：2026-06  
> **目标**：解释这个方向到底在做什么、要回答什么核心问题、截至 2026-06 各家推进到哪里、目前效果与瓶颈是什么，并给出高质量论文和开源项目学习清单。

> **时间口径**：本文的“年份”优先按论文首次公开时间（arXiv v1 / 官网首次发布）或项目首次开源时间计算；会议年份、代码后续更新和榜单收录会单独说明。尤其要避免把“2025 年预印本 + 2026 年会议发表/仓库更新”的工作误写成单纯 2026 年工作。

---

## 目录

1. [一句话结论](#1-一句话结论)
2. [这个方向到底是什么](#2-这个方向到底是什么)
3. [为什么现在会火](#3-为什么现在会火)
4. [标准技术闭环](#4-标准技术闭环)
5. [核心科学问题与工程问题](#5-核心科学问题与工程问题)
6. [截至 2026-06 的行业与学术版图](#6-截至-2026-06-的行业与学术版图)
7. [几条主流技术路线](#7-几条主流技术路线)
8. [代表论文与项目详解](#8-代表论文与项目详解)
9. [当前明确发现的问题](#9-当前明确发现的问题)
10. [接下来最值得做的工作方向](#10-接下来最值得做的工作方向)
11. [高质量论文与开源项目清单](#11-高质量论文与开源项目清单)
12. [建议学习路线](#12-建议学习路线)

---

## 1. 一句话结论

**WM + RL 的核心不是“做一个更漂亮的视频生成模型”，也不是简单“用神经网络替代 Isaac / MuJoCo”。它真正想回答的问题是：能不能用从真实世界交互数据中学出来的世界模型，给 VLA / VA 机器人策略提供一个便宜、可控、足够可信的闭环试错环境，让策略通过 RL 从“模仿专家”进化到“能自我改进”。**

你的直觉“用 WM 替代仿真器作为 VLA / VA 的 RL 训练场，缩小 Sim2Real Gap”是对的，但要补充三点：

1. **短期不是完全替代物理仿真器，而是替代一部分训练、评估、数据生成和失败恢复。**
2. **Sim2Real Gap 不会消失，只会转化成 WM2Real Gap。** 世界模型如果从真实机器人数据学习，确实可能更贴近真实视觉、物体、场景分布；但它也会 hallucinate，RL 又特别擅长钻模型漏洞。
3. **该方向的胜负手不是视频预测质量，而是闭环可控性、奖励可信度、长时序稳定性和防模型利用。**

---

## 2. 这个方向到底是什么

### 2.1 最小定义

给定机器人交互数据：

```text
D = {(o_t, a_t, r_t, o_{t+1}, instruction, success)}
```

训练一个 action-conditioned world model：

```text
M_phi(history, action, goal/instruction)
    -> next observation / latent state / reward / termination / uncertainty
```

再把策略放进这个模型里 rollout：

```text
policy pi_theta(o_t, instruction) -> action a_t
world model M_phi(o_t, a_t) -> imagined next observation o_{t+1}
reward/verifier R_psi(o_t, a_t, o_{t+1}, goal) -> score
RL algorithm -> update pi_theta
```

这就是“在世界模型里做 RL”。

### 2.2 和传统 model-based RL 的关系

它本质上是 model-based RL 的具身大模型版本。

经典 model-based RL，例如 Dreamer、TD-MPC、MBPO、MuZero，解决的是：

> 学一个环境模型，用模型生成 imagined rollout，再训练策略或做规划。

具身 WM + RL 继承这个思想，但对象变了：

| 传统 MBRL | 具身 WM + RL |
|---|---|
| 状态常是低维 proprioception 或游戏像素 | 状态是多视角 RGB、语言、机械臂状态、触觉、深度、3D 表征 |
| 动作空间较规整 | 动作可能是关节、末端位姿、action chunk、离散 action token |
| 奖励通常来自环境 | 奖励常常需要 VLM、成功判别器、goal matching 或人工规则 |
| 模型规模中小 | 世界模型可能是 diffusion / transformer / video foundation model |
| 任务多为 benchmark | 任务是开放词汇、开放物体、长时序机器人操作 |

### 2.3 和 VLA / VA 的关系

- **VA**：Vision-Action，输入视觉，输出动作，不一定有语言。
- **VLA**：Vision-Language-Action，输入视觉和语言指令，输出机器人动作。

当前 VLA / VA 大多来自 imitation learning / behavior cloning：

```text
大量示教数据 -> 监督学习 -> 能模仿数据分布内行为
```

问题是：

1. demo 覆盖不到所有失败状态；
2. BC 不会主动试错；
3. 长任务中一旦偏离示教轨迹，就容易崩；
4. 很难从稀疏成功信号中继续提升。

WM + RL 想补的正是这一块：

```text
VLA/VA 先靠 imitation 得到基础能力
再在 world model 里做 trial-and-error / self-correction / policy improvement
最后迁移到真实机器人
```

---

## 3. 为什么现在会火

### 3.1 真实机器人 RL 太贵

在真实机械臂上做 RL 有几个硬约束：

1. 采样慢：一次操作可能几十秒。
2. 成本高：机器人、场地、维护、标注都贵。
3. 容易损坏：随机探索会撞、摔、夹坏物体。
4. 稀疏奖励难：很多任务只有成功/失败。
5. 长尾失败多：真实环境的物体、光照、遮挡、接触状态极其复杂。

所以“直接 real-world RL”很难规模化。

### 3.2 物理仿真器有 Sim2Real Gap

Isaac、MuJoCo、ManiSkill、RoboSuite、RoboCasa、RoboTwin 等仿真器很重要，但它们的差距也明显：

1. 接触、摩擦、软物体、液体、透明反光物体难建模；
2. 真实视觉分布和仿真视觉分布差距大；
3. 资产建模成本高；
4. 任务越开放，手工构造环境越难；
5. 仿真中学到的 reward shortcut 到真实世界可能失效。

### 3.3 机器人数据和视频生成模型同时成熟

2023-2026 之间几个条件凑齐了：

1. Open X-Embodiment、DROID、BridgeData、LIBERO、RoboMimic、RoboTwin 等机器人数据集和 benchmark 变多；
2. diffusion / transformer 视频模型可以做动作条件预测；
3. VLA 基座模型出现，例如 RT 系列、OpenVLA、pi0、RDT、GR00T；
4. RL 后训练在 LLM 领域被验证，机器人社区开始迁移 PPO、GRPO、RFT、RLVR 等思路；
5. GPU 算力和生成模型基础设施更成熟。

于是一个自然问题出现：

> 能不能把真实机器人数据学成一个 world simulator，让机器人策略在里面练？

这就是 2026 年 WM + RL 的主线。

---

## 4. 标准技术闭环

一个典型系统可以抽象成六个模块：

```text
真实/仿真/视频数据
        │
        ▼
动作条件世界模型 M_phi
        │
        ├── 预测未来观测/隐状态
        ├── 预测 reward / done / success
        └── 输出 uncertainty / confidence
        │
        ▼
策略 pi_theta 在 WM 中 rollout
        │
        ▼
奖励模型 / 验证器 / VLM judge 打分
        │
        ▼
RL / RFT / policy optimization 更新策略
        │
        ▼
真实机器人验证与收集失败样本
        │
        └────────── 回灌世界模型与策略
```

### 4.1 世界模型负责什么

一个高质量具身世界模型至少要做四件事：

1. **状态转移**：动作之后世界怎么变。
2. **视觉结果**：物体位置、姿态、遮挡、手爪接触、场景变化。
3. **任务进展**：是否接近目标、是否成功、是否失败。
4. **不确定性估计**：哪些 rollout 可信，哪些不该用于 RL。

### 4.2 RL 负责什么

RL 的任务不是从零开始乱试，而是在已有 VLA / VA 能力上做策略改进：

1. 从失败状态恢复；
2. 选择更稳健的动作；
3. 优化长时序任务；
4. 提升稀疏成功率；
5. 探索 demo 中没有覆盖的变体。

### 4.3 为什么这件事难

因为 RL 会主动寻找模型里的漏洞。

如果世界模型在某类动作上预测错误，策略会学会利用这个错误。例如：

```text
真实世界：夹爪没碰到杯子，杯子不会移动。
世界模型：只要夹爪靠近杯子，杯子就被预测为移动。
RL 结果：策略学会“假装靠近”，在 WM 里刷成功，真实机器人失败。
```

这就是模型利用，也叫 model exploitation / reward hacking / simulator hacking。

---

## 5. 核心科学问题与工程问题

### 5.1 问题一：世界模型到底要预测什么

有三类选择：

| 类型 | 预测对象 | 优点 | 问题 |
|---|---|---|---|
| Latent WM | 隐状态、reward、value | 快、适合 RL、误差可控 | 可解释性差，和 VLA 视觉输入不一定对齐 |
| Video WM | 未来 RGB / 多视角视频 | 直观，可给 VLA 继续读图 | 慢，长时序漂移，像素好看不代表物理正确 |
| 3D / object-centric WM | 物体、几何、接触、场景图 | 更物理、更可规划 | 数据和标注难，开放世界泛化难 |

2026 年的趋势是融合：

```text
视频生成负责观测
latent dynamics 负责 rollout 效率
3D / object / contact 表征负责物理约束
VLM / verifier 负责语义奖励
```

### 5.2 问题二：世界模型能否被动作真正控制

很多视频模型可以预测“看起来合理的未来”，但机器人 RL 需要的是：

```text
不同 action -> 不同 future
同一 task 下 action 的微小差别 -> 接触结果的正确变化
```

如果模型只是根据语言和场景生成“任务大概会成功”的视频，它不能当训练场。

所以 action controllability 是最关键指标之一。

### 5.3 问题三：长时序 rollout 会不会漂移

机器人任务常常需要几十到几百步：

```text
靠近物体 -> 对准 -> 接触 -> 抓取 -> 抬起 -> 移动 -> 放置 -> 松手
```

世界模型一步预测还行，多步自回归后容易：

1. 物体融化、漂移、穿模；
2. 手爪和物体接触关系不稳定；
3. 任务进度被过度乐观预测；
4. 细节被平均化；
5. 错误累积导致策略学到假动作。

因此很多系统只做短 rollout、分段 rollout、keyframe rollout，或者把 WM 用作 evaluator 而不是完整环境。

### 5.4 问题四：奖励可信度从哪里来

真实仿真器里 reward 可以手写；世界模型里 reward 往往也要学。

常见方案：

1. 成功分类器：判断最终状态是否成功。
2. VLM reward：用 VLM 读图判断任务进展。
3. goal image matching：和目标图像/视频对齐。
4. learned value / reward head：随世界模型一起训练。
5. verified reward：只对可验证的状态给奖励，降低 hallucination。
6. 人工规则 + 几何检测：在结构化 benchmark 中使用。

关键问题是：**reward model 本身也会被策略利用。**

### 5.5 问题五：如何避免 policy exploit world model

常见防护手段：

1. 短 horizon rollout，减少误差累积；
2. uncertainty penalty，对 OOD 状态惩罚；
3. ensemble world model，用模型分歧估计不确定性；
4. conservative policy update，限制策略偏离数据分布；
5. rollout filtering，用 VLM / rule / physics checker 过滤不可信轨迹；
6. real-world relabeling，把真实失败样本回灌；
7. co-training policy and WM，让两者共同进化；
8. world model adversarial test，主动找模型漏洞。

### 5.6 问题六：评估指标不能只看 FVD

视频世界模型常用 FVD、LPIPS、PSNR、SSIM，但它们不够。

机器人 WM 更该看：

1. 动作条件敏感性；
2. 接触预测准确率；
3. success ranking correlation；
4. policy evaluation correlation；
5. rollout 后真实迁移提升；
6. OOD 检测能力；
7. 多步物理一致性；
8. RL 训练后真实 success rate 是否上升。

一句话：**能不能让策略变强，比视频漂不漂亮更重要。**

---

## 6. 截至 2026-06 的行业与学术版图

这一节按“各家在做什么、效果如何、问题在哪里”来整理。这里的“各家”包括公司、研究院和高质量开源学术团队。

### 6.1 Google DeepMind：VLA 基座 + 世界模拟器评估，偏闭源高算力路线

公开脉络：

1. RT-1 / RT-2 / RT-X / Open X-Embodiment 奠定了 VLA 和跨机器人数据混训路线。
2. Gemini Robotics 把 Gemini 系列能力接到机器人控制上，强调语义理解、泛化和交互。
3. Veo / video world model 被用于机器人策略评估和模拟未来结果的公开讨论中。

做的工作：

- 更偏 foundation model + policy evaluation / safety simulation；
- 强调多模态大模型对真实机器人任务的理解与泛化；
- 世界模型更像“评估器、想象器、安全检查器”，公开资料里还不是完整开源的 WM+RL 训练场。

效果：

- 高质量闭源 demo 强，数据和算力优势明显；
- 对行业判断很重要：世界模型不仅用于训练，也用于评估和安全。

问题：

- 开源程度有限；
- 很难复现实验；
- 是否能大规模稳定做 RL policy improvement，公开证据还不足。

参考：

- Gemini Robotics: <https://deepmind.google/discover/blog/gemini-robotics-brings-ai-into-the-physical-world/>
- Veo World Simulator for Gemini Robotics: <https://veo-robotics.github.io/>
- Open X-Embodiment / RT-X: <https://robotics-transformer-x.github.io/>

### 6.2 NVIDIA：Cosmos + GR00T，最明确的工业级 world foundation model 路线

公开脉络：

1. Cosmos World Foundation Models 面向物理 AI，提供生成式世界模型基础设施。
2. GR00T 系列面向通用人形机器人策略。
3. Cosmos Policy / World Action Model 把世界模型和动作模型进一步融合。

做的工作：

- 用 world foundation model 生成/预测物理世界；
- 用合成数据、机器人轨迹、未来预测辅助机器人策略训练；
- 发布较多开源模型、推理框架和训练工具。

效果：

- 工业生态位很清楚：给机器人公司提供“数据生成 + 仿真增强 + 策略训练”的底座；
- 开源程度比多数大厂更高；
- 对 humanoid 和通用机器人方向影响大。

问题：

- 真实复杂接触任务的闭环 RL 证据仍需更多公开 benchmark；
- WFM 生成质量和策略提升之间的因果关系仍要量化；
- 大模型成本高，普通团队复现压力大。

参考：

- Cosmos: <https://github.com/NVIDIA/Cosmos>
- Cosmos Policy: <https://github.com/NVlabs/cosmos-policy>
- GR00T: <https://developer.nvidia.com/isaac/gr00t>

### 6.3 Meta FAIR：JEPA 式 latent world model，偏表征和规划，不是直接 VLA-RL

公开脉络：

1. JEPA / V-JEPA 主张在 latent space 中预测未来，不追求像素级重建。
2. V-JEPA 2 把视频表征、物理理解和机器人控制联系起来。

做的工作：

- 学习物理世界的抽象 latent 表征；
- 用 latent prediction 支撑规划和控制；
- 更关注“理解世界”的表征，而不是用视频生成器作为训练场。

效果：

- 对“世界模型一定要生成像素吗”提出了重要反例；
- latent 方案更高效，更接近 RL 所需的状态表示。

问题：

- 和 VLA 后训练、RL rollout 的直接结合还不如 Dreamer / TD-MPC 这条线清晰；
- 对复杂语言任务、长时序开放操作的公开验证仍有限。

参考：

- V-JEPA: <https://ai.meta.com/blog/v-jepa-yann-lecun-ai-model-video-joint-embedding-predictive-architecture/>
- V-JEPA 2: <https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks/>

### 6.4 Physical Intelligence：强 VLA / generalist policy，公开上不是 WM+RL 主线

公开脉络：

1. pi0 / pi0.5 / openpi 是当前通用机器人策略的重要参考。
2. 其核心公开叙事更偏大规模行为克隆、flow matching action expert、跨任务泛化。

做的工作：

- 构建强大的 generalist robot policy；
- 用大量真实机器人数据训练；
- 开源 openpi，方便社区复现 VLA / diffusion policy 风格策略。

效果：

- 是 WM+RL 很重要的“policy base”参考；
- 代表了“先把 imitation base 做强，再考虑 RL/post-training”的路线。

问题：

- 公开材料中，world model 作为 RL 训练场不是其主叙事；
- 如果要做 WM+RL，pi0/openpi 更像要被 post-train 的底座。

参考：

- openpi: <https://github.com/Physical-Intelligence/openpi>
- pi0: <https://www.physicalintelligence.company/blog/pi0>

### 6.5 Stanford / Berkeley / UW 等学术团队：从机器人视频 WM 到 VLA-RL 的核心试验场

公开脉络：

1. DROID、BridgeData、Open X-Embodiment 等真实机器人数据支撑了大量 world model 工作。
2. 2025-2026 期间，Ctrl-World、UWM、VLA-RFT、World-Gymnast 等项目把 action-conditioned video model、policy rollout、RL 后训练逐渐接起来。其中 Ctrl-World、UWM、VLA-RFT 是 2025 年首次公开，World-Gymnast 是 2026 年工作。

做的工作：

- 训练多视角动作条件机器人世界模型；
- 用 WM 评估策略结果；
- 用 WM 生成失败恢复、反事实 rollout 和 RL 训练数据；
- 探索 policy-world model co-improvement。

效果：

- 这是当前最接近“开源 WM+RL 实验闭环”的区域；
- 论文和代码多，可学习性强；
- 很多工作已经从“预测视频”推进到“让策略变强”。

问题：

- 多数还在 LIBERO、DROID、RoboMimic、RoboTwin 等受控 benchmark；
- 真实开放家庭/工厂环境仍难；
- 长 horizon 和接触物理仍是硬瓶颈。

参考：

- DROID: <https://droid-dataset.github.io/>
- Ctrl-World: <https://ctrl-world.github.io/>
- UWM: <https://weirdlabuw.github.io/uwm/>
- World-Gymnast: <https://world-gymnast.github.io/>

### 6.6 ETH / Legged Robotics：从神经动力学模型到真实机器人控制，偏经典 MBRL 工程化

公开脉络：

1. legged robotics 更关注 locomotion / whole-body control。
2. Robotic World Model 项目用 learned world model 做 policy optimization。

做的工作：

- 学神经世界模型替代部分仿真；
- 用 model-based policy optimization 提高真实机器人样本效率；
- 更接近传统 MBRL + 真实机器人控制。

效果：

- 对腿式机器人、连续控制、动力学建模很有参考价值；
- 工程问题更扎实，不只是视频生成。

问题：

- 与 VLA / 语言指令 / 开放物体操作的连接较弱；
- 更多是控制侧世界模型，而不是生成式视觉世界模型。

参考：

- Robotic World Model: <https://github.com/leggedrobotics/robotic_world_model>

### 6.7 国内高校与开源团队：2025 下半年到 2026 年加速涌现 VLA-RL / WMPO / World-Env 类工作

公开脉络：

1. RLinf、SimpleVLA-RL、VLA-RFT、WMPO 等多在 2025 年公开/开源，World-Gymnast、WoVR、RehearseVLA / World-Env、DreamZero、Fast-WAM 等在 2026 年继续把 WM/VLA/RL 主线推高。
2. 很多工作承接 LLM RL 后训练思想，把 GRPO、PPO、RFT、verified reward 迁移到 VLA。

做的工作：

- 在 LIBERO / RoboTwin / ManiSkill / RoboCasa 中做 VLA RL；
- 用 world model / video simulator 进行 post-training；
- 用 VLM reward、成功验证器、trajectory verifier 控制奖励质量；
- 研究 world model 和 policy 的共同迭代。

效果：

- 论文速度快，开源多；
- 对复现 VLA-RL 管线很有帮助；
- 已经能看到部分 benchmark 上明显提升。

问题：

- 很多结果仍是仿真 benchmark；
- 真实机器人验证有限；
- reward 和 WM 的可靠性仍是核心短板；
- 论文之间 benchmark 不统一，横向比较困难。

参考：

- RLinf: <https://github.com/RLinf/RLinf>
- SimpleVLA-RL: <https://github.com/PRIME-RL/SimpleVLA-RL>
- VLA-RFT: <https://github.com/OpenHelix-Team/VLA-RFT>
- RehearseVLA: <https://github.com/iSEE-Laboratory/RehearseVLA>
- WMPO: <https://wm-po.github.io/>

### 6.8 ByteDance / 视频生成系团队：机器人视频世界模型，偏生成与仿真

公开脉络：

1. IRASim 等工作强调 fine-grained robot manipulation world model。
2. 这条线通常从视频生成、动作条件生成、交互细节建模切入。

做的工作：

- 用 diffusion / transformer 生成机器人交互视频；
- 让模型理解夹爪、物体、接触、遮挡、微小动作差异；
- 服务于策略评估、数据增强和未来训练场。

效果：

- 视频生成质量和动作可控性在提高；
- 对视觉世界模型本身很有参考价值。

问题：

- 从“生成合理视频”到“支撑 RL 训练并真实迁移”仍有距离；
- reward、uncertainty 和闭环稳定性需要额外系统。

参考：

- IRASim: <https://gen-irasim.github.io/>

### 6.9 其他机器人公司：公开证据有限，不能把 demo 直接等同于 WM+RL

Figure、1X、Tesla、Unitree、Boston Dynamics 等机器人公司都在强化真实机器人数据、仿真、VLA / VA 策略和闭环部署能力，但截至 2026-06，公开资料里能清楚验证的“世界模型作为 VLA/VA 的 RL 训练场”证据仍有限。

更稳妥的判断是：

1. 它们大概率都会用仿真、合成数据、离线日志回放、策略评估器和安全检查器；
2. 是否已经把 learned world model 大规模用于 RL post-training，公开材料不足；
3. 对行业研究来说，不应把高质量机器人 demo 自动解读成 WM+RL 成熟落地。

这部分值得持续跟踪，但当前不宜作为可复现学习主线。

---

## 7. 几条主流技术路线

### 7.1 路线 A：Latent World Model + Actor-Critic

代表：DreamerV3、DayDreamer、TD-MPC2、BOOM、Robotic World Model。

基本做法：

```text
encoder: observation -> latent state
dynamics: latent + action -> next latent
reward/value: latent + action -> reward/value
policy: 在 latent imagination 中训练
```

优点：

1. 训练和 rollout 快；
2. 适合 actor-critic；
3. 对连续控制有效；
4. 工程链路成熟。

缺点：

1. 与 VLA 的视觉/语言输入结合不自然；
2. latent 错了不容易发现；
3. 对开放场景和语义任务支持弱。

适合：

- locomotion；
- dexterous / continuous control；
- 低成本 real robot RL；
- 需要高样本效率的控制问题。

### 7.2 路线 B：Action-conditioned Video World Model

代表：IRASim、Ctrl-World、UWM、WorldEval、World-Env。

基本做法：

```text
past frames + robot action + instruction
    -> future frames / future video
```

优点：

1. 和 VLA 输入天然对齐；
2. 可视化强，容易检查；
3. 能利用大规模视频生成模型能力；
4. 可以做 policy evaluation、data augmentation、failure imagination。

缺点：

1. 慢；
2. 长 rollout 漂移；
3. 好看的视频不等于物理正确；
4. 接触、遮挡、反光、透明、软体仍难。

适合：

- manipulation；
- VLA post-training；
- policy evaluation；
- 失败样本生成。

### 7.3 路线 C：World Model as Evaluator / Reward Model

代表：WorldEval、Veo-style policy evaluation、VLM reward、GenReward 类工作。

基本做法：

```text
不一定用 WM 训练策略
先用 WM / VLM 判断某条策略 rollout 是否会成功
再用于 policy selection、reranking、offline RL 或 RL reward
```

优点：

1. 比完整神经训练场更稳；
2. 更容易落地；
3. 可用于安全过滤和部署前评估。

缺点：

1. 不能直接产生大量可靠训练经验；
2. evaluator 偏差会误导策略选择；
3. 对 OOD 策略仍不可靠。

适合：

- 多策略筛选；
- policy reranking；
- 安全评估；
- data curation。

### 7.4 路线 D：VLA 在世界模型中做 RL / RFT

代表：World-Gymnast、WMPO、VLA-RFT、RehearseVLA、WoVR。

基本做法：

```text
VLA policy -> action
world model -> imagined future
reward/verifier -> success score
GRPO/PPO/RFT -> update VLA
```

优点：

1. 最贴近“WM 替代训练场”的目标；
2. 可以直接提升 VLA success rate；
3. 能利用 LLM RL 后训练经验。

缺点：

1. 最容易被 WM / reward hallucination 毒害；
2. 算力成本高；
3. debug 难；
4. 真实迁移证据还不够多。

适合：

- 受控 manipulation benchmark；
- VLA 后训练；
- policy self-improvement 研究。

### 7.5 路线 E：World Action Model

代表：UWM、DreamZero、Cosmos Policy、WAM survey。

World Action Model 的想法是：

> 世界模型和动作模型不再分开。模型既理解未来会怎样，也能知道为了达到未来应该做什么动作。

它同时支持：

1. forward dynamics：动作导致什么未来；
2. inverse dynamics：想达到目标需要什么动作；
3. policy generation：直接生成动作；
4. planning：在未来空间里搜索。

这条线可能是 VLA 和 WM 融合后的下一阶段。

---

## 8. 代表论文与项目详解

### 8.0 代表工作的时间核验表

下表按“首次公开/论文时间”为主口径，会议年份或代码时间单独列出：

| 工作 | 首次公开/论文时间 | 会议/代码补充 | 时间判断 |
|---|---:|---|---|
| DreamerV3 | 2023 | - | 经典 latent WM+RL 基线 |
| DayDreamer | 2022 | - | 真实机器人 MBRL 早期标杆 |
| TD-MPC2 | 2024 | - | TD-MPC 系列连续控制基线 |
| IRASim | 2024-06 arXiv | ICCV 2025 | 不能归为 2026 新工作 |
| UWM | 2025-04 arXiv | RSS/ICML 2025，GitHub 2025-04 | 2025 工作 |
| WorldEval | 2025 | GitHub 2025-05 | 世界模型评估器方向 |
| Ctrl-World | 2025-10 arXiv | ICLR 2026，GitHub 2025-10 | 首发是 2025，会议是 2026 |
| VLA-RFT | 2025-10 arXiv | GitHub 2025-09/10 | 2025 VLA 强化微调工作 |
| WMPO | 2025-11 arXiv | GitHub 2025-11 | 2025 工作，不是 2026 |
| Veo World Simulator | 2025-12 arXiv | - | 2025 年底工作 |
| Cosmos Policy | 2025-12 GitHub | 2026 持续更新 | 代码/生态时间，论文时间需另核 |
| World-Gymnast | 2026-02 arXiv | GitHub 2026-01 | 2026 工作 |
| WoVR | 2026-02 arXiv | - | 2026 工作 |
| DreamZero | 2026-02 arXiv | GitHub 2026-01 | 2026 WAM 工作 |
| RehearseVLA / World-Env | 2026 | CVPR 2026，GitHub 2026-03 | 2026 工作 |
| Fast-WAM | 2026-03 arXiv | - | 2026 WAM 工作 |
| GenReward | 2025-11-30 arXiv / 编号 2512 | - | 2025 年底公开，不能归为 2026 工作 |

### 8.1 DreamerV3：理解 latent imagination RL 的必读基线

链接：

- 项目：<https://danijar.com/project/dreamerv3/>
- 代码：<https://github.com/danijar/dreamerv3>

核心思想：

- 学一个 recurrent state-space world model；
- 在 latent imagination 中 rollout；
- 用 actor-critic 训练策略。

为什么重要：

- 它告诉我们“世界模型不一定要生成像素，能支撑价值学习就够”；
- 是后续机器人 WM+RL 的重要思想源头。

局限：

- 和大规模 VLA / language-conditioned manipulation 不是一套工程范式；
- 面向开放语义任务时需要额外模块。

### 8.2 DayDreamer：真实机器人上用世界模型做 RL 的早期标杆

链接：

- 代码：<https://github.com/danijar/daydreamer>
- 项目：<https://danijar.com/project/daydreamer/>

核心思想：

- 把 Dreamer 风格方法搬到真实机器人；
- 通过少量真实交互学习多个机器人控制任务。

为什么重要：

- 证明 real-world model-based RL 可以显著提高样本效率；
- 是“真实机器人 + 世界模型 + RL”的基础参考。

局限：

- 任务规模和语义复杂度远低于当前 VLA 目标；
- 不是开放词汇机器人策略。

### 8.3 TD-MPC2：高维连续控制里的强世界模型规划基线

链接：

- 项目：<https://www.tdmpc2.com/>
- 代码：<https://github.com/nicklashansen/tdmpc2>

核心思想：

- 学 latent world model；
- 用 MPPI 做 online planning；
- 同时训练 policy、value、dynamics。

为什么重要：

- 对高维连续控制非常强；
- 是很多后续 policy + planner + world model 工作的工程基础；
- BOOM 这类工作就在其谱系上解决 planner-policy divergence。

局限：

- 更偏控制，不是 VLA；
- 对多模态语言和开放视觉任务不是原生支持。

### 8.4 BOOM：把 planner 和 policy divergence 问题挑明

本仓库已有深度解读：`强化学习结合/BOOM_用世界模型自举off-policy强化学习深度解读.md`

链接：

- 代码：<https://github.com/molumitu/BOOM_MBRL>

核心思想：

- 在 TD-MPC2 式世界模型 + MPPI 框架中，解决 planner 采数据、policy 被训练之间的分布错配；
- 用 likelihood-free forward KL 和 Q-weighted imitation 把 policy 拉回 planner 的高质量动作。

对 WM+RL 的启发：

- 在世界模型中做 RL 时，行为分布错配非常危险；
- 训练场、planner、policy 必须形成闭环对齐。

### 8.5 Robotic World Model：真实机器人控制侧的 learned simulator

链接：

- 代码：<https://github.com/leggedrobotics/robotic_world_model>

核心思想：

- 用 learned world model 支撑机器人 policy optimization；
- 关注不确定性、离线数据、真实机器人迁移。

为什么重要：

- 它更像“神经仿真器替代一部分动力学仿真”的工程路线；
- 对 locomotion / whole-body control 很有参考价值。

局限：

- 和 VLA / 语言条件 manipulation 的连接较弱。

### 8.6 IRASim：细粒度机器人交互视频世界模型

链接：

- 项目：<https://gen-irasim.github.io/>
- 论文：<https://arxiv.org/abs/2406.14540>

核心思想：

- 训练可控的机器人 manipulation 视频世界模型；
- 强调 fine-grained interaction，例如夹爪与物体的接触、移动、遮挡。

为什么重要：

- 代表“视频生成式 WM 进入机器人操作”的路线；
- 对后续 WM 作为策略评估器、数据生成器很重要。
- 时间上是 2024-06 arXiv、ICCV 2025 论文，不应写成 2026 新工作。

局限：

- 视频预测本身不等于能支持 RL；
- 仍需要 reward、uncertainty、closed-loop rollout 系统。

### 8.7 Ctrl-World：DROID 上的可控机器人世界模型

链接：

- 项目：<https://ctrl-world.github.io/>
- 论文：<https://arxiv.org/abs/2510.10125>
- 代码：<https://github.com/Robert-gyj/Ctrl-World>

核心思想：

- 在真实机器人数据上训练多视角动作条件世界模型；
- 支持 future prediction、policy evaluation、trajectory imagination。

为什么重要：

- 是 2025 年首发、ICLR 2026 发表的机器人 video WM 开源项目；
- 更接近 VLA 输入形式；
- 能用于评估和增强策略。

局限：

- 长时序 rollout 和复杂接触仍难；
- policy improvement 需要额外 RL / reward 设计。

### 8.8 UWM：把视频生成和动作模型统一起来

链接：

- 项目：<https://weirdlabuw.github.io/uwm/>
- 论文：<https://arxiv.org/abs/2504.02792>
- 代码：<https://github.com/WEIRDLabUW/unified-world-model>

核心思想：

- Coupling video and action diffusion；
- 同一模型可以做 video prediction、action generation、inverse dynamics。

为什么重要：

- 这是 World Action Model 思路的早期强代表；
- 它不只问“动作之后会怎样”，也问“要达到这个未来该怎么动”。
- 时间上是 2025-04 arXiv，并非 2026 才出现。

局限：

- 是否能稳定支撑长程 RL，还需要更多实证。

### 8.9 WorldEval：世界模型作为机器人策略评估器

链接：

- 代码：<https://github.com/liyaxuanliyaxuan/Worldeval>

核心思想：

- 用 world model 预测策略执行结果；
- 衡量它对真实策略成功率排序的相关性。

为什么重要：

- 评估器可能比完整训练场更先落地；
- 对“世界模型是否真的懂策略后果”提出了更合适的指标。

局限：

- 评估相关性不等于能做 RL；
- evaluator 自身也有 OOD 风险。

### 8.10 World-Gymnast：直接把世界模型当机器人 RL 训练环境

链接：

- 项目：<https://world-gymnast.github.io/>
- 论文：<https://arxiv.org/abs/2602.02454>
- 代码：<https://github.com/world-gymnast/world-gymnast>

核心思想：

- 构建一个 learned world model environment；
- 在里面用 RL 训练机器人策略；
- 目标是让机器人通过“想象中的试错”提升。

为什么重要：

- 这是最贴近“WM 替代训练场”的公开工作之一；
- 名字里的 Gymnast 也很直接：世界模型变成 gym。

局限：

- 关键仍在世界模型可靠性和 reward 可信度；
- 要警惕策略利用模型幻觉。

### 8.11 WMPO：World Model-based Policy Optimization for VLA

链接：

- 项目：<https://wm-po.github.io/>
- 论文：<https://arxiv.org/abs/2511.09515>
- 代码：<https://github.com/WM-PO/WMPO>

核心思想：

- 用世界模型支持 VLA policy optimization；
- 面向 VLA post-training，而不是单纯训练视频预测。

为什么重要：

- 是 2025 年底公开的“VLA + WM + RL”代表工作，后续在 2026 年继续被讨论和复现；
- 关注的就是 VLA 在世界模型中改进策略。

局限：

- 真实部署和跨 benchmark 泛化仍要持续观察；
- reward / verifier 质量决定上限。

### 8.12 VLA-RFT：把 verified reward 引入 VLA 强化微调

链接：

- 项目：<https://vla-rft.github.io/>
- 论文：<https://arxiv.org/abs/2510.00406>
- 代码：<https://github.com/OpenHelix-Team/VLA-RFT>

核心思想：

- Reinforcement Fine-Tuning VLA；
- 重视 verified reward，避免 VLA 在不可靠奖励上乱学。

为什么重要：

- LLM 领域 RLVR / RFT 的思想正在迁移到机器人；
- 对“reward 必须可信”这个问题很有针对性。
- 时间上是 2025-10 arXiv、2025-09/10 开源，不应归为 2026 新工作。

局限：

- 可验证任务范围有限；
- 开放真实场景中的自动验证仍难。

### 8.13 RehearseVLA / World-Env：用世界模型做虚拟排练

链接：

- 代码：<https://github.com/iSEE-Laboratory/RehearseVLA>
- 项目：<https://world-env.github.io/>

核心思想：

- 把世界模型当虚拟环境，让 VLA 在执行前或训练中 rehearsal；
- 强调 physically-consistent world model 与 VLM reflector。

为什么重要：

- “排练”是很贴切的定位：不一定要求世界模型完美，只要能暴露失败、提供改进信号；
- 对实际工程更现实。

局限：

- 仍需证明在更多真实长任务中能稳定增益；
- physically consistent 的定义和评估仍在形成。

### 8.14 WoVR：把“可靠世界模型”问题正面拎出来

链接：

- 论文：<https://arxiv.org/abs/2602.13977>

核心思想：

- World Models as Reliable Simulators for Post-Training VLA Policies；
- 关注世界模型作为 post-training simulator 时的可靠性。

为什么重要：

- 它直指本方向最大问题：WM 不是生成视频就能当 simulator；
- 强调可靠性、可验证性和策略后训练。

局限：

- 需要看后续代码、真实机器人和多任务复现情况。

### 8.15 RLinf / SimpleVLA-RL：VLA-RL 基础设施

链接：

- RLinf: <https://github.com/RLinf/RLinf>
- SimpleVLA-RL: <https://github.com/PRIME-RL/SimpleVLA-RL>

时间备注：

- RLinf 仓库创建于 2025-08，SimpleVLA-RL 仓库创建于 2025-05；二者在 2026 仍有更新，不能把更新日期误读为首次公开年份。

核心思想：

- 提供 VLA 强化学习训练框架；
- 通常先在 LIBERO / RoboTwin 等仿真环境中训练；
- 支持 PPO / GRPO 等后训练机制。

为什么重要：

- 即使不用 WM，也必须先理解 VLA-RL 怎么跑通；
- 它们是未来接入 learned world model 的基础设施。

局限：

- 当前更多依赖传统仿真器；
- 接入 WM 后会引入更多稳定性问题。

### 8.16 OpenVLA / openpi：VLA policy base

链接：

- OpenVLA: <https://github.com/openvla/openvla>
- OpenVLA-OFT: <https://github.com/moojink/openvla-oft>
- openpi: <https://github.com/Physical-Intelligence/openpi>

为什么重要：

- WM+RL 通常不是从零训练策略，而是在已有 VLA / VA 上 post-train；
- OpenVLA 和 openpi 是理解 policy side 的关键开源项目。

局限：

- 它们本身不是 WM+RL；
- 需要外接 world model、reward、RL trainer。

---

## 9. 当前明确发现的问题

### 9.1 世界模型越强，RL 越可能找到更隐蔽的漏洞

弱世界模型的问题容易看出来；强世界模型的问题更危险，因为它生成的视频看起来合理，但关键物理细节错了。

例如：

1. 夹爪和物体没有真实接触，却预测物体被带走；
2. 物体姿态在遮挡后被错误补全；
3. 透明/反光/软物体行为被平均化；
4. 任务快成功时模型过度乐观；
5. VLM reward 看到“像成功”就打高分。

这会让 policy 学到“在模型里成功、在真实中失败”的动作。

### 9.2 多步闭环比单步预测难一个量级

很多论文展示 one-step / short-horizon prediction 很好，但 RL 需要闭环：

```text
模型预测下一步 -> 策略基于预测图再出动作 -> 模型继续预测 -> ...
```

每一步都会放大前一步错误。

因此未来应该少看单纯视频指标，多看：

1. closed-loop rollout stability；
2. policy ranking correlation；
3. RL 后真实成功率；
4. 模型不确定性是否能识别失败预测。

### 9.3 Reward 比 dynamics 更容易被低估

很多讨论集中在“世界模型能不能预测未来”，但 RL 训练还需要 reward。

一个世界模型即使未来预测不错，如果 reward 错了，策略也会跑偏。

特别是 VLA 任务中，reward 往往是语义性的：

```text
"把红色杯子放到盘子左边"
"打开抽屉后拿出毛巾"
"把掉在桌边的物体扶正"
```

这些成功条件不是简单几何距离可以判断的。

### 9.4 数据覆盖决定 WM 上限

如果训练数据里几乎没有失败、碰撞、滑落、误抓、遮挡恢复，那么世界模型也不会知道这些事情怎么发生。

而 RL 恰好会探索到这些边界状态。

所以高质量 WM+RL 数据集应该包含：

1. 成功轨迹；
2. 失败轨迹；
3. 恢复轨迹；
4. 人类纠正轨迹；
5. 多视角同步；
6. robot proprioception；
7. action chunk；
8. 任务语义标注；
9. success / failure labels；
10. 真实部署日志。

### 9.5 跨本体 action space 仍然很难

VLA 想跨机器人，但 WM 往往强依赖 action representation：

```text
关节角
末端位姿
夹爪开合
双臂协同
移动底盘
灵巧手
humanoid whole-body action
```

如果 action space 不统一，世界模型很难同时服务多种机器人。

可能路线：

1. 用 end-effector / task-space action 作为中间层；
2. 用 action token 做离散化；
3. 用 embodiment adapter；
4. 学 inverse dynamics，把目标变化映射到具体机器人动作；
5. 用 World Action Model 统一 forward 和 inverse。

### 9.6 真实验证仍是稀缺证据

2026 年很多论文 benchmark 很漂亮，但行业真正关心的是：

```text
在未见真实物体、真实桌面、真实光照、真实接触条件下，
WM+RL 后的策略是否比 imitation base 稳定提升？
```

这个证据目前还不充分。

### 9.7 计算成本会成为工程瓶颈

VLA 本来就大，再套一个视频世界模型做 rollout，成本会很高：

1. rollout 慢；
2. 多策略并行难；
3. PPO / GRPO 需要大量样本；
4. diffusion video model 推理成本高；
5. 长时序任务需要 memory 和 cache。

所以未来很可能出现分层架构：

```text
高层：VLM / VLA 负责语义和子目标
中层：world model 负责短程未来和失败预测
低层：小 policy / MPC / controller 负责快速控制
```

---

## 10. 接下来最值得做的工作方向

### 10.1 做“可靠性优先”的世界模型，而不是“好看优先”的世界模型

建议指标：

1. action sensitivity；
2. contact correctness；
3. physical consistency；
4. success ranking correlation；
5. OOD uncertainty calibration；
6. closed-loop rollout degradation；
7. policy improvement after RL。

### 10.2 构建 WM+RL 专用 benchmark

现在很多工作用 LIBERO、RoboTwin、RoboCasa、DROID，但还缺一个专门评估 WM+RL 的 benchmark。

它应该包含：

1. policy base；
2. world model training data；
3. held-out real / sim tasks；
4. reward verifier；
5. evaluation robot / simulator；
6. metrics for WM, reward, policy improvement；
7. ablation protocol；
8. failure replay set。

### 10.3 研究“模型可信区域内”的保守 RL

关键不是让策略在 WM 里无限探索，而是：

```text
只在 WM 有把握的区域做策略改进
遇到不确定状态就回到真实采样或传统仿真器
```

可做方向：

1. ensemble uncertainty；
2. epistemic / aleatoric decomposition；
3. confidence-aware reward；
4. conservative rollout truncation；
5. offline RL regularization；
6. model disagreement penalty。

### 10.4 把失败数据变成核心资产

当前很多机器人数据集偏成功 demo。WM+RL 更需要失败。

值得做：

1. 自动收集失败；
2. 分类失败类型；
3. 学 recovery policy；
4. 用 WM 生成近失败状态；
5. 让 VLA 学会“发现不对劲并修正”。

### 10.5 做 VLA + WM + Verifier 的三模型协同

未来系统很可能是：

```text
VLA policy: 负责提出动作
World Model: 负责预测后果
Verifier / Critic: 负责判断后果是否可信和有价值
```

三者共同训练，而不是只训练其中一个。

### 10.6 探索 World Action Model

WAM 可能把三个问题统一：

1. forward：这个动作会发生什么；
2. inverse：想发生这个结果该做什么；
3. policy：当前应该怎么做。

如果 WAM 成熟，它可能比“VLA + 外置 WM + 外置 RL”更简洁。

---

## 11. 高质量论文与开源项目清单

### 11.1 综述与导航

| 名称 | 类型 | 价值 |
|---|---|---|
| [World Model for Robot Learning: A Comprehensive Survey](https://arxiv.org/abs/2605.00080) | Survey | 当前最贴近机器人 WM 的系统综述 |
| [World Action Models: The Next Frontier in Embodied AI](https://arxiv.org/abs/2605.12090) | Survey | 理解 WAM 概念和 VLA-WM 融合趋势 |
| [Awesome World Models](https://github.com/knightnemo/Awesome-World-Models) | List | 查世界模型论文入口 |
| [Awesome World Model for Robotics Policy](https://github.com/NTUMARS/Awesome-World-Model-for-Robotics-Policy) | List | 查机器人策略相关 WM 工作 |
| [Awesome-WAM](https://github.com/OpenMOSS/Awesome-WAM) | List | 查 WAM / VLA-WM 新工作 |

### 11.2 经典 world model RL

| 名称 | 链接 | 推荐理由 |
|---|---|---|
| DreamerV3 | <https://github.com/danijar/dreamerv3> | latent imagination RL 标准参考 |
| DayDreamer | <https://github.com/danijar/daydreamer> | 真实机器人 world model RL 早期标杆 |
| TD-MPC2 | <https://github.com/nicklashansen/tdmpc2> | 高维连续控制强基线 |
| BOOM | <https://github.com/molumitu/BOOM_MBRL> | planner-policy divergence 问题讲得清楚 |
| Robotic World Model | <https://github.com/leggedrobotics/robotic_world_model> | learned simulator + 真实机器人控制 |

### 11.3 机器人视频世界模型 / 神经仿真器

| 名称 | 时间口径 | 链接 | 推荐理由 |
|---|---:|---|---|
| IRASim | 2024-06 arXiv / ICCV 2025 | <https://gen-irasim.github.io/> | 细粒度机器人交互视频 WM |
| Ctrl-World | 2025-10 arXiv / ICLR 2026 | <https://ctrl-world.github.io/> | DROID 上的可控多视角机器人 WM |
| UWM | 2025-04 arXiv / RSS、ICML 2025 | <https://weirdlabuw.github.io/uwm/> | video/action diffusion 统一 |
| WorldEval | 2025 | <https://github.com/liyaxuanliyaxuan/Worldeval> | 用 WM 评估机器人策略 |
| Veo World Simulator | 2025-12 arXiv | <https://veo-robotics.github.io/> | 用视频世界模型评估 Gemini Robotics 策略 |
| RehearseVLA / World-Env | 2026 / CVPR 2026 | <https://github.com/iSEE-Laboratory/RehearseVLA> | VLA 在世界模型中虚拟排练 |

### 11.4 WM + VLA-RL / post-training

| 名称 | 时间口径 | 链接 | 推荐理由 |
|---|---:|---|---|
| World-Gymnast | 2026-02 arXiv / GitHub 2026-01 | <https://world-gymnast.github.io/> | 世界模型作为 RL gym |
| WMPO | 2025-11 arXiv / GitHub 2025-11 | <https://wm-po.github.io/> | 面向 VLA 的 world model policy optimization |
| VLA-RFT | 2025-10 arXiv / GitHub 2025-09/10 | <https://github.com/OpenHelix-Team/VLA-RFT> | verified reward + VLA 强化微调 |
| WoVR | 2026-02 arXiv | <https://arxiv.org/abs/2602.13977> | 可靠 world simulator for VLA post-training |
| GenReward | 2025-11-30 arXiv / 编号 2512 | <https://arxiv.org/abs/2512.00961> | 用视频扩散模型生成 goal-driven reward |

### 11.5 VLA-RL 基础设施与 policy base

| 名称 | 链接 | 推荐理由 |
|---|---|---|
| OpenVLA | <https://github.com/openvla/openvla> | 开源 VLA 基座 |
| OpenVLA-OFT | <https://github.com/moojink/openvla-oft> | OpenVLA 高效微调 |
| openpi | <https://github.com/Physical-Intelligence/openpi> | pi0/openpi 系策略实现 |
| RLinf | <https://github.com/RLinf/RLinf> | VLA-RL 训练框架 |
| SimpleVLA-RL | <https://github.com/PRIME-RL/SimpleVLA-RL> | 简洁 VLA-RL baseline |

时间备注：RLinf 仓库创建于 2025-08，SimpleVLA-RL 仓库创建于 2025-05；2026 年的 push / release 只能说明持续维护，不能替代首次公开年份。

### 11.6 World Action Model

| 名称 | 时间口径 | 链接 | 推荐理由 |
|---|---:|---|---|
| UWM | 2025-04 arXiv | <https://weirdlabuw.github.io/uwm/> | 同时学习视频预测、动作生成和 inverse dynamics |
| DreamZero | 2026-02 arXiv / GitHub 2026-01 | <https://dreamzero0.github.io/> | 明确提出 WAM 作为 zero-shot policy |
| DreamZero 代码 | 2026-01 GitHub | <https://github.com/dreamzero0/dreamzero> | 可跑的 WAM 推理发布包 |
| Cosmos Policy | 2025-12 GitHub / 2026 持续更新 | <https://github.com/NVlabs/cosmos-policy> | NVIDIA 面向视频模型到控制规划的路线 |
| Fast-WAM | 2026-03 arXiv | <https://yuantianyuan01.github.io/FastWAM/> | 讨论 WAM 是否需要测试时显式想象未来 |

### 11.7 数据集与 benchmark

| 名称 | 链接 | 推荐理由 |
|---|---|---|
| Open X-Embodiment | <https://robotics-transformer-x.github.io/> | 跨机器人大规模数据 |
| DROID | <https://droid-dataset.github.io/> | 真实机器人交互数据，适合训练 WM |
| LIBERO | <https://libero-project.github.io/> | VLA / lifelong manipulation 常用 benchmark |
| RoboTwin | <https://robotwin-benchmark.github.io/> | 双臂操作和 VLA-RL 常用 benchmark |
| RoboCasa | <https://robocasa.ai/> | 家庭操作仿真环境 |
| ManiSkill | <https://maniskill.ai/> | 机器人操作仿真 benchmark |

---

## 12. 建议学习路线

### 第一阶段：补 model-based RL 基础

先读：

1. DreamerV3；
2. TD-MPC2；
3. DayDreamer；
4. BOOM。

要掌握：

1. latent dynamics；
2. imagined rollout；
3. actor-critic in world model；
4. planner-policy divergence；
5. model exploitation。

### 第二阶段：理解 VLA / VA 策略

建议看：

1. OpenVLA；
2. OpenVLA-OFT；
3. openpi；
4. RDT / RT-X / pi0 相关论文。

要掌握：

1. action token / action chunk；
2. language-conditioned manipulation；
3. behavior cloning 的局限；
4. VLA 后训练为什么需要 RL。

### 第三阶段：看机器人世界模型

建议看：

1. IRASim；
2. Ctrl-World；
3. UWM；
4. WorldEval。

要掌握：

1. action-conditioned video prediction；
2. multi-view prediction；
3. policy evaluation correlation；
4. video metric 和 policy metric 的差异。

### 第四阶段：进入 WM + VLA-RL 主线

建议看：

1. World-Gymnast；
2. WMPO；
3. VLA-RFT；
4. RehearseVLA；
5. WoVR。

要掌握：

1. VLA 在世界模型里 rollout；
2. reward / verifier 设计；
3. PPO / GRPO / RFT 怎么接机器人；
4. 如何防止 model exploitation；
5. 如何评估真实迁移。

### 第五阶段：关注 WAM

建议看：

1. UWM；
2. DreamZero；
3. Cosmos Policy；
4. World Action Models survey。

要掌握：

1. forward dynamics；
2. inverse dynamics；
3. action generation；
4. world model 和 policy 的统一。

---

## 最后总结

WM + RL 在具身智能里的核心价值，是给机器人策略提供一种新的自我改进机制：

```text
真实数据学世界模型
世界模型生成可控试错
奖励/验证器筛掉幻觉
RL 提升策略
真实机器人验证并回灌失败
```

但这个方向还没到“神经仿真器完全替代物理仿真器”的阶段。2026 年更现实的判断是：

1. **评估器和数据增强会最先落地。**
2. **短 horizon、受控任务的 VLA post-training 会继续快速进展。**
3. **真实长时序开放任务仍卡在物理一致性、奖励可靠性、OOD 不确定性和跨本体动作表示。**
4. **行业下一步最需要的不是再多一个漂亮视频 WM，而是能证明 policy improvement 的可靠闭环。**

所以如果要在这个方向继续做，最有价值的问题不是泛泛地“训练一个世界模型”，而是：

> 在什么任务范围、什么数据覆盖、什么可信度约束下，世界模型生成的 imagined experience 能稳定提升真实机器人策略？

这个问题回答清楚，WM + RL 才真正从概念变成机器人训练基础设施。
