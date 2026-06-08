# BOOM：用世界模型自举 Off-policy 强化学习深度解读

> **论文**: Bootstrap Off-policy with World Model (BOOM)
> **作者**: Guojian Zhan¹², Likun Wang¹, Xiangteng Zhang¹, Jiaxin Gao¹, Masayoshi Tomizuka², Shengbo Eben Li¹（通讯作者 lishbo@tsinghua.edu.cn）
> **机构**: ¹清华大学 人工智能学院 & 车辆与运载学院　²加州大学伯克利 BAIR
> **发表**: NeurIPS 2025
> **论文链接**: arxiv.org/abs/2511.00423 (openreview id=zNqDCSokDR) | **代码**: github.com/molumitu/BOOM_MBRL
> **解读时间**: 2026-06-02 ｜ 解读基于论文全文（含全部定理证明、附录）与官方源码逐行交叉验证

---

## 目录

1. [一句话概览与核心贡献](#1-一句话概览与核心贡献)
2. [背景与动机：Actor Divergence 这个"原罪"](#2-背景与动机actor-divergence-这个原罪)
3. [预备知识：Off-policy RL 与 MPPI 在线规划](#3-预备知识off-policy-rl-与-mppi-在线规划)
4. [核心方法：BOOM 的自举闭环](#4-核心方法boom-的自举闭环)
5. [理论分析：两条不等式撑起的安全网](#5-理论分析两条不等式撑起的安全网)
6. [源码架构全景（逐行验证）](#6-源码架构全景逐行验证)
7. [实验设置与结果](#7-实验设置与结果)
8. [消融研究：三个组件各值多少](#8-消融研究三个组件各值多少)
9. [与相关工作的对比](#9-与相关工作的对比)
10. [关键创新与贡献总结](#10-关键创新与贡献总结)
11. [局限性、未来方向与反思](#11-局限性未来方向与反思)
12. [对"冲世界模型赛道"的学习价值](#12-对冲世界模型赛道的学习价值)
13. [参数参考表](#13-参数参考表)

---

## 1. 一句话概览与核心贡献

> **BOOM 解决的核心问题**：当"在线规划（planning）"遇上"off-policy 强化学习"时，**采数据的行为体（planner）和被训练的策略网络（policy）不是同一个东西**，这个错配（actor divergence）会同时毒害价值学习和策略改进。BOOM 用一个**自举闭环**把二者重新拽到一起：策略给规划器一个初始解，规划器精修出更优动作再反过来"对齐拉拽"策略，世界模型在中间同时支撑两者。

BOOM 是 **TD-MPC2 谱系**的直系改进（共用世界模型、共用 MPPI 规划器、共用代码库），它的全部增量浓缩在**策略损失的一个对齐项**上。三个关键贡献：

| 贡献 | 一句话本质 | 解决什么 |
|---|---|---|
| **① Likelihood-free 前向 KL 对齐** | 用 $-\log\pi(a\mid s)$ 直接模仿 buffer 里 planner 存下的动作，**不需要知道 planner 的动作概率密度** | MPPI 是采样式非参数分布，似然不可解析，传统 reverse KL 用不了 |
| **② Soft Q-weighted 机制** | 用 $w_i=\mathrm{softmax}(Q_i/\tau)$ 给每条历史动作加权，高价值的多学、低价值的少学 | buffer 里早期 planner 动作质量参差不齐，均匀模仿会被"垃圾经验"带偏 |
| **③ 即插即用 + SOTA** | 整套改动只加在 policy loss 上，世界模型训练完全沿用 TD-MPC2 | 在 DMC Suite 与 Humanoid-Bench 共 14 个高维任务上全面登顶 |

**最直观的成绩**：Humanoid-Bench 平均回报 **820.6**，比第二名 DreamerV3(10M) 的 555.6 高 **+47.7%**，比同门 BMPC 高 **+60.5%**；在 H1hand-hurdle（跨栏）这种最难任务上比次优高 **+121%**。

---

## 2. 背景与动机：Actor Divergence 这个"原罪"

### 2.1 为什么要把"规划"塞进强化学习

纯 model-free 的 RL（如 SAC）靠试错学习，在高维连续控制（人形 67/21 维、狗 223/38 维）上经常**直接学不动、趋近于 0 分**。在线规划（online planning）提供了一个强力武器：**用学到的动力学模型做前瞻 rollout**，在每一步真正动手前先在"脑内"模拟若干条未来轨迹，挑出回报最高的那条来执行。TD-MPC、TD-MPC2 把"学世界模型 + MPPI 规划"这套打法在高维任务上做到了 SOTA。

### 2.2 但这里藏着一个结构性矛盾

问题出在：**真正下场采数据的，是"策略 + MPPI 规划器"的组合体**，论文记作行为策略 $\beta = \pi + \text{MPPI}$；而你要训练、最终要部署的，是那个**纯策略网络 $\pi$**。二者不是一回事——这就是 **actor divergence（行为体偏离）**。

在 off-policy 框架下，这个偏离会引爆两个连锁问题：

```
                  planner β 采的数据  d^β(s,a)
                          │
            ┌─────────────┴─────────────┐
            ▼                           ▼
  ① 价值学习的分布偏移            ② 策略改进不可靠
  Q 只在 β 访问过的区域准         策略靠 Q 的梯度更新
  在 π 偏好但 β 没覆盖的          但 Q 在那些区域是错的
  区域 → 系统性高估(OOD)    →    错误的 Q 把策略推向坑里
            │                           │
            └────────── 整个训练失稳 ◀───┘
```

- **① 价值学习的分布偏移**：价值函数在行为分布 $d^\beta$ 上最小化 Bellman 误差，于是它只在 planner 走过的窄区域学得准，在策略 $\pi$ 偏好但 planner 很少涉足的 **OOD 区域系统性高估**。
- **② 策略改进不可靠**：策略靠最大化 $Q$ 来更新，而 $Q$ 在那些 OOD 区域本身就是错的偏高估计，于是策略被"骗"着往坏动作走，训练崩坏。

### 2.3 最棘手的一点：planner 的似然算不出来

最致命的难点在于：MPPI 这类**采样式规划器产生的是非参数动作分布**。它先采一堆高斯样本，再用回报做 softmax 加权平均 + 重采样——经过这套加权重采样后，**最终动作早已不服从任何高斯形式，其精确似然 $\beta(a\mid s)$ 在实践中不可解析**。这就堵死了"用 reverse KL 做模仿学习"的常规路：reverse KL 需要知道 $\beta(a\mid s)$。

> 这是理解 BOOM 全部设计动机的钥匙：**既要对齐 planner，又拿不到 planner 的概率密度，怎么办？**

---

## 3. 预备知识：Off-policy RL 与 MPPI 在线规划

### 3.1 Off-policy RL 的核心与软肋

MDP 五元组 $(\mathcal{S},\mathcal{A},P,r,\gamma)$，动作价值函数：

$$Q^{\pi}(s,a)=\mathbb{E}_{\pi}\Big[\textstyle\sum_{t=0}^{\infty}\gamma^{t}r(s_t,a_t)\,\big|\,s_0{=}s,a_0{=}a\Big]$$

Off-policy 的精髓是**把"采数据"和"改策略"解耦**：用行为策略 $\beta$ 采的历史数据（存在 replay buffer 里反复复用）来优化目标策略 $\pi$，从而获得高样本效率。但它有一个前提假设——**当 $\beta$ 偏离 $\pi$ 太远，分布偏移就会摧毁价值估计**。这正是第 2 章问题的理论根源。

### 3.2 MPPI：采样式在线规划器

MPPI（Model Predictive Path Integral）是连续控制里最常用的采样规划器。每个规划步：

1. **采样**：借动力学模型，从因子化高斯采 $N_p$ 条动作轨迹 $a_t^i\sim\mathcal{N}(\mu_t,\sigma_t^2)$，$t=0,\dots,H{-}1$；
2. **评估**：每条轨迹用 reward + 终值估回报 $G^i=\sum_{t=0}^{H-1}r_t^i+\hat{v}_H^i$；
3. **重加权**：对回报做 softmax $w^i=\frac{\exp(G^i-\max_j G^j)}{\sum_k\exp(G^k-\max_j G^j)}$；
4. **更新分布**：$\mu_t\leftarrow\sum_i w^i a_t^i$，迭代数轮收敛。

测试时执行首动作 $\mu_0$，训练时加高斯噪声探索。**关键限制**：最终动作是对采样候选的加权平均，**不来自任何参数化概率策略**，经过 reweighting/resampling 后已非真高斯，**精确似然不可解**。

---

## 4. 核心方法：BOOM 的自举闭环

### 4.1 三个组件 + 一个闭环

BOOM 由**策略 π、规划器 P、世界模型**三个紧耦合组件构成，核心是一个 **bootstrap loop（自举闭环）**：

```
         ┌────────────────────────────────────────────┐
         │                                            │
         │   策略 π ──提供初始解──▶ 规划器 P (MPPI)      │
         │     ▲                       │              │
         │     │                       │ 模型预测优化   │
         │  行为对齐                    ▼ 精修出更优动作  │
         │  (Q-加权前向KL)  ◀──bootstrap── a_β          │
         │                                            │
         └──────────────┬─────────────────────────────┘
                        │ 二者都由世界模型支撑
              ┌─────────▼──────────┐
              │   世界模型 (TD-MPC2 式)  │
              │  encoder h / dynamics f │
              │  reward R / value Q     │
              │  双重角色：① 给规划器做rollout │
              │           ② 给策略提供Q目标  │
              └────────────────────────┘
```

- **策略 → 规划器**：策略给 MPPI 提供 `num_pi_trajs=24` 条初始轨迹，加速搜索（源码 `boom_alg.py:157-168`）；
- **规划器 → 策略**：planner 通常产出更高质量的动作，把策略向这些动作对齐，给了策略强力的改进指引；
- **世界模型**：一边让规划器做 receding-horizon 模拟采高质量轨迹，一边给策略提供 Q 值做性能改进。

### 4.2 世界模型：完全沿用 TD-MPC2

世界模型训练 4 个组件：编码器 $z=h(s)$、潜空间动力学 $z'=f(z,a)$、奖励预测 $R(z,a)$、价值函数 $Q(z,a)$。注意所有状态 $s$ 先编码进潜空间 $z$。联合用 TD 损失训练：

$$\mathcal{L}_{\text{model}}=\mathbb{E}_{(s,a,r,s')_{0:H}}\Big[\sum_{t=0}^{H}\gamma^{t}\big(\underbrace{\|f(z_t,a_t)-\text{sg}(h(s_t'))\|_2^2}_{\text{一致性/动力学}}+\underbrace{\text{CE}(R_t,r_t)}_{\text{奖励}}+\underbrace{\text{CE}(Q_t,q_t)}_{\text{价值}}\big)\Big] \tag{1}$$

其中 $q_t$ 是用策略 $\pi$ 算的目标值，sg 是 stop-gradient，CE 是交叉熵（价值和奖励都用 **two-hot 离散回归** + symlog 变换，源码 `math.py:66-95`）。

> **这一段一字未改**——BOOM 把全部创新留给了下面的策略损失。这也是它"易实现"卖点的来源。

### 4.3 创新①：Likelihood-free 前向 KL 对齐

既然 buffer 里所有动作都是 planner $\beta$ 产生的，那**直接模仿它们**就能对齐——还不损失训练效率。问题是 $\beta$ 似然不可解，reverse KL 用不了。BOOM 的破解是**改用前向 KL**：

$$\mathrm{KL}(\beta\,\|\,\pi)=\mathbb{E}_{a\sim\beta}\big[\log\beta(a\mid s)\big]-\mathbb{E}_{a\sim\beta}\big[\log\pi(a\mid s)\big] \tag{2}$$

**精妙之处**：第一项只依赖 $\beta$，对 $\pi$ 的参数是常数，**优化时直接丢掉**。于是只剩：

$$\boxed{\mathcal{L}_{\text{align}}=\mathbb{E}_{(s,a,\dots)_{0:H}}\big[-\log\pi(a\mid s)\big]} \tag{3}$$

这就是一个**最大似然 / 行为克隆式**的项——鼓励 $\pi$ 给 planner 选过的动作更高概率，**全程不需要 $\beta(a\mid s)$**。这是把"非参数 planner 的本事"蒸馏进"参数化策略"的一个简洁而有原则的机制。

> **前向 vs 反向 KL 的本质区别（附录 C.1 的玩具实验）**：当 planner 是多峰分布、而策略是单峰高斯时——
> - **前向 KL（BOOM 用的）= mode-covering（覆盖众数）**：策略会调整均值方差去**覆盖 planner 的多个高价值峰**，不会过早塌缩；
> - **反向 KL = mode-seeking（追逐众数）**：策略只贴上**某一个峰**，可能恰好贴到次优峰，丢掉全局结构。
>
> 这解释了为什么消融里前向 KL 持续更优。

### 4.4 创新②：Soft Q-weighted 机制

buffer 里的历史 planner 动作**质量参差不齐**（早期模型差，planner 也菜）。均匀模仿会被垃圾经验带偏。BOOM 借鉴 planner 自身的"价值加权选择"原则，定义一个**软目标分布**：

$$p\propto\exp(Q/\tau),\qquad w_i=\frac{\exp(Q_i/\tau)}{\sum_{j=1}^{N}\exp(Q_j/\tau)} \tag{4a}$$

$\tau>0$ 控制锐度（默认 1）。带权对齐损失：

$$\mathcal{L}_{\text{align}}=\mathbb{E}_{(s,a,\dots)_{0:H}}\sum_{t=0}^{H-1}\sum_{i=1}^{N}w_i\big[-\log\pi(a_i\mid s_i)\big] \tag{4b}$$

效果：**策略优先学高价值动作**，把对齐资源集中到"有前途的经验"上，既加速学习又消化了 planner 历史动作的质量波动。论文指出这里的 $Q$ 也可换成 $V$ 或优势 $A$，选 $Q$ 只因最易获取。这套思路与 offline RL 里的 **AWAC（Advantage-Weighted）** 一脉相承。

### 4.5 创新③：自举策略目标

把对齐项并入标准策略损失（最大化 Q），得到最终目标：

$$\boxed{\mathcal{L}_{\text{policy}}=-\mathbb{E}_{(s,a,\dots)_{0:H}}\sum_{t=0}^{H-1}\Big[\underbrace{Q(s,\pi(s))}_{\text{自身价值改进}}+\lambda_{\text{align}}\cdot\underbrace{\mathcal{L}_{\text{align}}}_{\text{向planner对齐}}\Big]} \tag{5}$$

策略既要"按自己的 Q 估计变好（max Q）"，又要"贴住 buffer 里高质量的 planner 动作（min 对齐损失）"。$\lambda_{\text{align}}$ 是可调系数。

### 4.6 为什么世界模型也跟着变好？

这是一个**正反馈飞轮**，论文给了两条因果链：

```
策略向planner对齐 → 采集数据与策略行为更一致 → 分布偏移↓
       │
       ├─▶ ① Q 学得更准（训练在policy-relevant轨迹上）
       │        │
       │        └─▶ 规划用的Q更可靠 → planner产出更优动作 → 又给策略更好的对齐目标 ↺
       │
       └─▶ ② TD式联合训练下，Q梯度更准
                → 流入encoder/dynamics/reward的梯度更有信息量
                → 整个世界模型质量↑
```

policy 与 planner 的一致性 ↑ → Q 估计更准 → planner 动作更优 → 又反哺策略，**自举着一起冲向最优解**。

### 4.7 算法伪代码（Algorithm 1）

```
输入: 策略 π_θ, 编码器 h_ξ, 动力学 f_ψ, 奖励 R_ω, 价值 Q_φ, 规划器 P
1.  // Warmup: 世界模型预训练
2.  用随机动作与环境交互, 存 (s, a_rand, r, s') 入 buffer D
3.  最小化 L_model (式1) 更新 h,f,R,Q
4.  for each iteration:
5.    // 数据采集 (用规划器)
6.    编码 z = h_ξ(s)
7.    规划 a_β ~ β = P(π_θ, f_ψ, R_ω, Q_φ, z)
8.    与环境交互, 存 (s, a_β, r, s') 入 D
9.    // 世界模型 + 策略学习
10.   从 D 采 rollout batch
11.   最小化 L_model (式1) 更新 h,f,R,Q
12.   最小化 L_policy (式5) 更新 π_θ   ← BOOM 的全部增量在这一行
```

---

## 5. 理论分析：两条不等式撑起的安全网

BOOM 证明了：只要把 $\mathrm{KL}(\beta\|\pi)$ 压小，actor divergence 的两大危害都被数学上界住。

### 5.1 定理 1：自举对齐控制"回报差"

> 设 $|r(s,a)|\le R_{\max}$，$\gamma\in[0,1)$。若 $\mathrm{KL}(\beta\|\pi)\le\varepsilon$，则
> $$\big|J(\beta)-J(\pi)\big|\le\frac{R_{\max}}{1-\gamma}\sqrt{2\varepsilon} \tag{6}$$

**含义**：策略与 planner 的**实际回报差被 KL 上界控制**。贴住 planner，就不会因分布错配掉性能。

**证明骨架**（附录 A.2）：回报差 → 逐步奖励期望差（三角不等式）→ 用 **Lemma 3**（全变差界期望差）得 $\le 2R_{\max}\cdot\mathrm{TV}$ → 用 **Pinsker 不等式**（Lemma 2）$\mathrm{TV}\le\sqrt{\tfrac12\mathrm{KL}}$ → 等比级数求和 $\sum\gamma^t=\frac{1}{1-\gamma}$，得证。

### 5.2 定理 2：自举对齐控制"Q 值差"

> 设 $Q(s,a)$ 对 $a$ 是 $L_Q$-Lipschitz，$\mathrm{KL}(\beta\|\pi)\le\varepsilon$，则
> $$|Q(s,a_\beta)-Q(s,a_\pi)|\le L_Q\cdot\|a_\beta-a_\pi\|_2\le L_Q\cdot D(\varepsilon) \tag{7}$$
> 进一步把 MPPI 动作分布近似为高斯混合 $\beta(s)=\sum_{i=1}^K w_i\mathcal{N}(\mu_i,\Sigma_i)$，策略 $\pi(s)=\mathcal{N}(\mu_\pi,\Sigma_\pi)$，则以至少 $1-\delta$ 概率：
> $$D(\varepsilon)\le\min\!\Big(2\sqrt{d},\ \max_i\big(\sqrt{2\Lambda_i\log\tfrac{K}{\delta}}+\sqrt{2\varepsilon\Lambda(\Sigma_\pi)/w_i}\big)+\sqrt{2\Lambda(\Sigma_\pi)\log\tfrac1\delta}\Big) \tag{8}$$

**含义**：策略与 planner 的 **Q 值偏差被对齐程度控制**——这正是第 2 章"OOD 高估"的克星。对齐越紧（$\varepsilon$ 越小），价值偏差越小，策略更新越稳。

**证明骨架**（附录 A.3）：Lipschitz → 三角分解 $\|a_\beta-a_\pi\|_2$ 为三段（GMM 采样偏差 + 分量均值偏移 + 策略采样偏差）→ 分别用 **Lemma 4**（高斯样本集中不等式，基于 Laurent–Massart）和 **Lemma 5**（高斯间 KL 下界均值距离）→ 再用归一化动作空间直径 $2\sqrt d$ 取 min 收紧。

> 两条定理合起来给了 BOOM 一个干净的故事：**"对齐"不是经验技巧，而是有界保证的——它同时夹住了回报差和价值差，从根上掐灭 actor divergence。**

---

## 6. 源码架构全景（逐行验证）

代码基于官方 TD-MPC2 codebase（`github.com/nicklashansen/tdmpc`），结构精简。

### 6.1 目录结构

```
boom/
├── boom_alg.py          # BOOM 主算法：act/plan/update/update_pi ★核心
├── train.py             # 入口
├── config.yaml          # 全部超参
├── common/
│   ├── world_model.py   # WorldModel: encoder/dynamics/reward/pi/Q ★核心
│   ├── math.py          # two_hot离散回归, gaussian_logprob, squash
│   ├── scale.py         # RunningScale 5%-95%分位的running尺度归一
│   ├── layers.py        # mlp/Ensemble/SimNorm/NormedLinear
│   └── buffer.py        # replay buffer
├── trainer/
│   └── online_trainer.py # 在线训练循环 ★
└── envs/                 # DMC/Humanoid/Metaworld/Maniskill 等封装
humanoid_bench/           # Humanoid-Bench 环境(H1hand机器人)
```

### 6.2 自举闭环在代码里长什么样

**[源码验证]** `boom_alg.py:235-282` 的 `update_pi` 是论文式(5)的落地，把"max Q"和"Q-加权前向 KL 对齐"两块拼在一起：

```python
# ① max-Q 项：标准策略改进
_, pis, log_pis, log_std = self.model.pi(zs, task)
qs = self.model.Q(zs, pis, task, return_type="min")
qs = self.scale(qs)                                   # RunningScale 归一
rho = torch.pow(self.cfg.rho, torch.arange(len(qs)))  # 时间步衰减 rho^t
q_loss = ((self.cfg.entropy_coef * log_pis - qs).mean(dim=(1,2)) * rho).mean()

# ② Q-加权前向 KL 对齐项 (式4b)
std = log_std.exp().detach()
std = torch.max(std, self.cfg.min_std * torch.ones_like(std))
eps = (pis - mu) / std                                # mu=buffer里planner的动作均值
forward_kl = math.gaussian_logprob(eps, std.log(), size=action_dims).mean(dim=-1)
forward_kl = self.scale(forward_kl) if self.scale.value > 2.0 else torch.zeros_like(forward_kl)
forward_kl = torch.softmax(qs.detach().squeeze(), dim=-1) * forward_kl   # ← soft Q-weight
fkl_loss = -(forward_kl.sum(dim=-1) * rho).mean()

# ③ 合并 (式5)
pi_loss = q_loss + (self.cfg.action_dim * self.lamda) * fkl_loss
```

**几个值得注意的实现细节**（论文未明说、读代码才看得到）：

1. **对齐系数按动作维度区间硬切换**（`boom_alg.py:270-274`）：代码逻辑是 `λ = dim(A)/1000  if 24 ≤ action_dim ≤ 48  else  dim(A)/50`。即**狗(38 维)落在区间内 → dim/1000；而 humanoid-DMC(21 维) 和 H1hand(61 维) 都在区间外 → dim/50**。论文把它笼统写成"DMC 用 dim/1000、Humanoid-Bench 用 dim/50"，其实是**近似说法**——真正的 key 是动作维区间而非数据集。这是按任务手调的工程经验值。
2. **前向 KL 的"延迟启动"**：只有当 `RunningScale.value > 2.0` 才启用对齐项，否则置零——**训练早期 Q 尺度还没起来时不强行对齐**，避免被噪声 Q 误导。
3. **soft Q-weight 用的是 `softmax(Q)` 跨 batch 归一**（`τ` 默认隐含为 1），与式(4a)一致。
4. **planner 初始解注入**（`boom_alg.py:157-168`）：MPPI 的 512 个样本里前 24 条直接用策略 rollout 填充（`num_pi_trajs=24`），即"策略给规划器初始解"。

### 6.3 世界模型组件

**[源码验证]** `world_model.py:21-48`：

| 组件 | 实现 | 备注 |
|---|---|---|
| `_encoder` | MLP，末层 `SimNorm` 单纯形归一 | state obs 直接 MLP 编码 |
| `_dynamics` | `latent+action+task → latent`，末层 SimNorm | 潜空间预测下一状态 |
| `_reward` | MLP → `num_bins=101` | two-hot 离散回归 |
| `_pi` | MLP → `2×action_dim`（μ 和 log_std） | **对角高斯策略** |
| `_Qs` | `Ensemble` of `num_q=5` 个 Q | 用 `functorch` 向量化集成 |
| `_target_Qs` | deepcopy + Polyak 软更新 | `tau` 软更新 |

**[源码验证]** `world_model.py:212-243` 的 `Q` 函数：从 5 个 Q 里**随机抽 2 个取 min**（`return_type="min"`），这是 TD-MPC2 式的保守价值估计（缓解高估）。

**两个反常识的工程开关**（`layers.py:72-92`）：`SimNorm` 在 `24 ≤ action_dim ≤ 48` 时**直接跳过**（对狗这种高动作维任务禁用单纯形归一）——又一处按本体维度做的特判，说明这套方法对不同形态机器人需要少量针对性调参。

### 6.4 训练循环

**[源码验证]** `online_trainer.py:119-189`：标准 online RL 循环。值得注意——它**每 100 步批量预采 100 个 replay batch 缓存**（`replay_sample_list`），再逐个喂给 `agent.update`，减少 buffer 采样开销；`seed_steps` 步随机探索预热，预热结束时一次性做 `seed_steps` 次更新预训练世界模型。评估时**同时评 planner（MPPI）和纯策略 π 两套**（`eval_pi`），正好对应"训练用 β、部署用 π"的设定。

---

## 7. 实验设置与结果

### 7.1 基线与基准

**4 个代表性在线 RL 基线**：

| 基线 | 类别 | 一句话 |
|---|---|---|
| **SAC** | model-free off-policy | 最大熵 SOTA model-free |
| **DreamerV3** | imagination-driven MBRL | 想象式 SOTA，评 2M & 10M 两种预算 |
| **TD-MPC2** | planning-driven MBRL | BOOM 的直系前身，MPPI 在线规划 |
| **BMPC** | planning-driven MBRL | TD-MPC2 变体，**只模仿 planner、丢掉 Q**，带 relabel 机制 |

**14 个高维任务**：DMC Suite 选 7 个最难的（humanoid 67/21、dog 223/38）+ Humanoid-Bench 7 个（Unitree H1hand 机器人 151/61，含走滑坡、穿杆林、连续跨栏等长程目标任务）。3 个随机种子，单 RTX 3090Ti，Dog-run 训 2M 步约 50 小时。

### 7.2 总成绩表（Total Average Return, 3 seeds 均值±标准差）

| 任务 | SAC | DreamerV3 (2M / 10M) | TD-MPC2 | BMPC | **BOOM(ours)** |
|---|---|---|---|---|---|
| Humanoid-stand | 9.0 | 264.5 / 717.0 | 913.3 | <u>947.9</u> | **962.1** |
| Humanoid-walk | 173.8 | 251.5 / 755.6 | 884.8 | <u>935.1</u> | **936.1** |
| Humanoid-run | 1.6 | 62.5 / 353.5 | 316.2 | <u>531.2</u> | **582.8** |
| Dog-stand | 197.6 | 35.4 / 35.4 | 936.4 | <u>971.3</u> | **986.8** |
| Dog-walk | 24.7 | 9.1 / 9.1 | 885.0 | <u>942.9</u> | **965.4** |
| Dog-trot | 67.1 | 7.9 / 8.4 | 884.4 | <u>911.3</u> | **947.9** |
| Dog-run | 16.5 | 4.3 / 4.3 | 427.0 | <u>673.7</u> | **820.7** |
| **AVG. DMC** | 58.8 | 87.9 / 269.0 | 745.6 | <u>835.8</u> | **877.7** |
| H1hand-stand | 74.1 | 220.3 / 845.4 | 728.7 | 780.0 | **926.1** |
| H1hand-walk | 27.0 | 161.3 / 744.0 | 644.2 | 672.6 | **935.4** |
| H1hand-run | 14.1 | 55.8 / 622.4 | 66.1 | 236.0 | **682.2** |
| H1hand-sit | 268.4 | 687.3 / 699.1 | 693.7 | 688.2 | **918.1** |
| H1hand-slide | 19.0 | 162.6 / 367.6 | 141.3 | 440.1 | **926.1** |
| H1hand-pole | 122.5 | 334.3 / 577.4 | 207.5 | 739.9 | **930.5** |
| H1hand-hurdle | 12.9 | 26.6 / 135.7 | 59.0 | 197.1 | **435.6** |
| **AVG. H-Bench** | 68.5 | 233.0 / 555.6 | 338.8 | 511.7 | **820.6** |

> 注：DreamerV3 列原文给出 2M 与 10M 两个预算的成绩。BOOM **在全部 14 个任务上都拿了第一**。

### 7.3 关键读数

- **DMC Suite**：BOOM 877.7，比次优 BMPC(835.8) **+5.0%**，比 TD-MPC2 **+17.7%**；Dog-run 单项比次优 **+21.8%**。
- **Humanoid-Bench**：BOOM 820.6，比 DreamerV3(10M) **+47.7%**，比 BMPC **+60.5%**；最难的 H1hand-slide **+110.5%**、H1hand-hurdle **+121%**。
- **横向观察**：SAC/DreamerV3 在高维任务常**整体趴窝（接近 0 分，如 Dog 系列）**；TD-MPC2/BMPC 能学但**不稳、训练曲线震荡**（BMPC 因丢掉 Q、只模仿 planner，对历史动作质量波动尤其敏感）。BOOM 学得更快、更稳、终值更高。

---

## 8. 消融研究：三个组件各值多少

所有消融在 **Dog-run（223/38，最高维）** 上做。

| 消融项 | 对比设置 | 结论 |
|---|---|---|
| **① 对齐度量** | 前向 KL vs 反向 KL | 前向 KL 持续更高回报。反向 KL 需用高斯代理近似 planner 似然，**近似不准反而有害**，且 mode-seeking 易贴到次优峰 |
| **② Soft Q-weight** | Q 加权 vs 均匀加权 | Q 加权持续**加速训练 + 提升终值**，因为它能消化 buffer 里动作质量的参差 |
| **③ 对齐系数 λ** | 0.1× / 1× / 10× 默认值 | 跨该范围**表现稳定**，说明 BOOM 对这个超参不敏感、不需精调 |

> 消融讲清了一件事：**前向 KL（似然无关）和 Q 加权（处理质量波动）都是必要的，且整套方法对超参鲁棒。**

---

## 9. 与相关工作的对比

### 9.1 在 MBRL 谱系里的位置

```
                    Model-Based RL
        ┌───────────────────┴───────────────────┐
   planning-driven                        imagination-driven
   (用规划器采数据)                          (用想象rollout更新策略)
        │                                       │
  PETS, PlaNet, LOOP                      SimPLe, IRIS, Dreamer系列
        │                                       │
  TD-MPC / TD-MPC2 ──┐                    DreamerV3 (代表作)
        │            │                          │
  BMPC(丢Q只模仿)   TDM(PC)²(并行工作)            └─ 直接从策略采样,
        │            │                              收敛慢,长rollout
   ★ BOOM ◀──────────┘                              误差累积
   (max-Q + Q加权前向KL对齐)
```

### 9.2 与几个关键对手的精确区别

| 对手 | 它怎么做 | BOOM 的不同 |
|---|---|---|
| **TD-MPC2** | 联合学世界模型 + MPPI 规划，但**无视 actor divergence** | BOOM 加自举对齐，专治这个偏离 |
| **BMPC** | **彻底放弃显式策略优化，纯模仿 planner** + relabel 历史动作 | BOOM **保留 max-Q**，且用 Q 加权处理动作质量波动 → 更稳、效率更高（BMPC 曲线震荡） |
| **TDM(PC)²**（并行同期工作） | 也发现"对齐 planner 有益"，但用 **TD3+BC 式 + 反向 KL** | BOOM 更像 **AWAC：critic 加权 + 前向 KL** |
| **DreamerV3** | 想象式，直接从学到的策略采样交互 | 收敛需更多迭代；BOOM 全任务超越 |

### 9.3 与 Offline RL 的本质区别（论文专门澄清）

虽然"对齐 + 价值最大化"看着像 offline RL，但二者**范式根本不同**：

- **学习范式**：Offline RL 围绕**固定数据集**，靠 BC 把策略约束在数据支撑内，怕外推误差；BOOM 围绕**策略本身**为优化目标，在线交互中 policy 与 planner **协同进化奔向最优**。
- **学习目标**：Offline RL 里 max-Q 与 BC **天然冲突**（max-Q 想往数据外推，BC 把它拉回来），被迫保守折中；BOOM 里二者**互补**——planner 通常比策略好，对齐 planner 既提性能又增强一致性 → Q 更准 → 策略更好 → planner 更优，**正反馈而非拉锯**。

---

## 10. 关键创新与贡献总结

| # | 创新 | 为什么重要 | 落点 |
|---|---|---|---|
| 1 | **Likelihood-free 前向 KL 对齐** | 第一次让"对齐采样式非参数 planner"在似然不可解下变得可行；mode-covering 避免塌缩 | 式(2)(3)、`update_pi` |
| 2 | **Soft Q-weighted 对齐** | 用 critic 给历史动作排优先级，消化 buffer 质量波动，AWAC 式加权 | 式(4)、`softmax(Q)` |
| 3 | **自举闭环 + 飞轮** | policy↔planner↔world-model 三者互相抬升，把 off-policy 的高效与 planning 的高质量统一 | Algorithm 1 |
| 4 | **两条理论保证** | 回报差(定理1)与 Q 值差(定理2)都被 KL 上界控制，actor divergence 数学可控 | 附录 A |
| 5 | **极简 + SOTA** | 仅改 policy loss，世界模型沿用 TD-MPC2；14 任务全胜 | 全文 |

**论文自己点出的两条更大启示**：① **价值函数仍是 planning-driven MBRL 的根本瓶颈**——价值估准了，规划和终值都跟着好；② **最大熵温度系数 α 的自动调节**是未来关键，因为 planning 主导采样时，planner 探索熵与策略熵的关系极其复杂。

---

## 11. 局限性、未来方向与反思

### 11.1 论文承认的局限

- **规划开销大**：数据采集时跑 MPPI（512 样本 × 6~8 轮迭代 × H=3 rollout），比 model-free 直接采样慢得多，限制训练吞吐。
- **依赖较准的动力学模型**：模型预测误差大或长 rollout 分布偏移时，规划质量和价值估计都会退化。

### 11.2 我的批判性补充

1. **"即插即用"是双刃剑**：增量只在 policy loss，确实优雅；但代价是**继承了 TD-MPC2 的全部包袱**——SimNorm 开关、λ 系数都要按动作维度特判（`24≤dim≤48` 的硬编码出现了三次），说明跨本体迁移并非零成本。
2. **前向 KL 的实现 ≠ 公式字面**：式(3)写的是 $-\log\pi(a_{\text{planner}}\mid s)$，而代码 `eps=(pis-mu)/std` 实际是"**策略采样动作 pis** 在以 **planner 均值 mu** 为心、**策略 std** 为宽的高斯下的对数概率"。这是个合理但与公式略有出入的工程近似，读者若只看论文容易误判。
3. **窗口意义上的价值**：BOOM 的全部成绩在**仿真**（DMC/Humanoid-Bench），未触碰真机。论文也把"sparse-reward / 真实机器人 / 带噪观测"列为未来工作——这恰恰是从"刷 benchmark"到"落地"之间最深的一道沟。
4. **理论与实践的缝隙**：定理把 MPPI 近似成 GMM 才推得动 Q 值界，但真实 MPPI 经过重采样后未必是干净 GMM；理论是"安慰性保证"而非紧界。

### 11.3 未来方向

- 加速在线规划计算；
- 引入**不确定性感知**机制处理多峰/不可靠 planner 输出；
- 用**更具表达力的策略类**（如扩散策略 diffusion）替换对角高斯，进一步提升上限；
- α 温度的自动调节机制。

---

## 12. 对"冲世界模型赛道"的学习价值

把 BOOM 放进 [[快与慢-双非本科冲世界模型赛道的局势与节奏判断]] 里那张"硬通货清单"对照看，它的学习性价比很高——一篇论文几乎踩满了清单上的高频考点：

| 硬通货考点（24 份 JD 反推） | BOOM 里的对应 | 掌握后能讲什么 |
|---|---|---|
| **世界模型 / 序列建模** | TD-MPC2 式潜空间 world model（encoder/dynamics/reward/value 联合训练） | 能讲清"latent world model 怎么训、为什么用 two-hot 离散回归" |
| **强化学习（PPO/SAC/Decision Transformer）** | off-policy actor-critic、最大熵、target Q 软更新、Q 集成取 min | 能讲 actor divergence 这个真问题，而非只会背 SAC 公式 |
| **MPC / 规划** | MPPI（采样 + 回报 softmax 加权 + 迭代） | 能讲 sampling-based MPC 与 policy 的耦合 |
| **PyTorch 工程** | `functorch` 向量化 Q 集成、`torch.compile`、RunningScale 归一 | 能讲实现层的加速与稳定技巧 |
| **理论功底** | Pinsker / 全变差 / 高斯集中不等式 | 面试里能把"为什么对齐有用"上升到不等式层面 |

**一句话**：BOOM 是 TD-MPC2 体系一个"小切口、强论证、满成绩"的范例。它示范了一种**做研究/做项目的高性价比打法**——不追求推倒重来，而是**精准找到前作的结构性裂缝（actor divergence），用一个有理论支撑的小改动（一个对齐项）撬动全面提升**。这套"找裂缝 + 小切口 + 强论证"的思路，对要在窗口期内快速做出"能独立负责的模块"的人，比盲目追新架构更值得学。

> ⚠️ 提醒：这是**仿真 benchmark 的 SOTA**，不是真机方案。简历上写"复现/理解 BOOM"是漂亮的对口证据（命中世界模型 + RL + MPC 三个考点），但别把它当成"我能做具身落地"——那道沟（sim-to-real、真机噪声、稀疏奖励）BOOM 自己都还没过。

---

## 13. 参数参考表

**[源码验证]** 来自 `config.yaml`，与论文 Table 2 交叉核对（少量字段论文表与默认配置不一致，以代码默认为准并标注）。

| 类别 | 超参 | 值 |
|---|---|---|
| **训练** | 学习率 lr | 3e-4 |
| | 编码器 lr 缩放 enc_lr_scale | 0.3（→ 编码器 9e-5） |
| | replay batch size | 256 |
| | buffer size | 1,000,000 |
| | 总步数 steps | 2,001,000 |
| | 梯度裁剪 grad_clip_norm | 20 |
| | target 软更新 tau | 0.01（论文 Table2 写 0.5，以代码为准） |
| | 折扣 discount | denom=5, min=0.95, max=0.995（按 episode 长度自适应） |
| **世界模型** | reward_coef / value_coef | 0.1 / 0.1 |
| | consistency_coef（动力学） | 20 |
| | rho（时间步衰减） | 0.5 |
| | Q 集成数 num_q | 5 |
| | 价值 bin 数 num_bins | 101（vmin=-10, vmax=+10） |
| | latent_dim / mlp_dim | 512 / 512 |
| | SimNorm 维度 simnorm_dim | 8 |
| | dropout | 0.01 |
| **规划器(MPPI)** | iterations | 6（动作维>20 时 8） |
| | num_samples（种群） | 512 |
| | num_elites（精英） | 64 |
| | num_pi_trajs（策略初始解） | 24 |
| | horizon H | 3 |
| | min_std / max_std | 0.05 / 2 |
| | temperature | 0.5 |
| **Actor** | log_std 范围 | [-10, 2] |
| | 熵系数 entropy_coef (α) | 1e-4 |
| | 对齐系数 λ_align | `24≤dim(A)≤48` 时 dim(A)/1000，否则 dim(A)/50（按动作维区间特判） |

---

> **解读方法说明**：本报告基于 arXiv:2511.00423 论文全文（含第 3 章方法、第 5 章理论、附录 A 全部 5 条引理与 2 个定理证明、附录 B 环境与超参、附录 C 前向/反向 KL 玩具实验）与 `github.com/molumitu/BOOM_MBRL` 官方源码（`boom_alg.py` / `world_model.py` / `online_trainer.py` / `math.py` / `layers.py` / `scale.py` / `config.yaml`）逐行交叉验证而成。标注 **[源码验证]** 处为代码与论文公式的对照确认点，其中第 11.2 节指出了一处"公式字面与实现的细微差异"。
