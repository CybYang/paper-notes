---
title: OpenHLM：面向全身人形 Loco-Manipulation 的 VLA 系统设计
date: 2026-07-28
description: 以 Whole-Body Controller、VLA Design 与 Heterogeneous Co-Training 三个阶段为主线，梳理 OpenHLM 的数据采集、模型训练、动作接口与闭环推理流程。
tags:
  - Loco-Manipulation
  - Whole-Body-Control
  - VLA
  - Flow-Matching
  - Imitation-Learning
  - Human-Data
  - Robot-Teleoperation
  - Retargeting
  - Pretraining
  - Co-training
  - Gripper
  - Humanoid
---

# OpenHLM：面向全身人形 Loco-Manipulation 的 VLA 系统设计

OpenHLM 关注全身人形机器人在移动与操作之间的统一协调。与将上半身 manipulation 和下半身 locomotion 分开的传统方案不同，OpenHLM 使用一个高层 VLA 同时生成手臂、夹爪、腿、腰和 root 的全身参考动作，再由低层 SONIC controller 负责动作跟踪与动态平衡。

论文的三个 Phase 是一条 **system design roadmap**，用于依次确定数据采集接口、VLA 设计和训练数据组成，并不是推理时串联的三个网络。

<div class="paper-overview" markdown>

![OpenHLM 的全身控制、VLA 设计与异构共训练路线](../assets/papers/openhlm-teaser.png){ loading=lazy }

<span class="paper-overview__caption">图：OpenHLM 通过全身控制器与遥操作、VLA 设计、异构共训练三个 Phase 逐步确定最终系统。图片来自论文官方项目主页。</span>

</div>

<!-- more -->

## 论文信息

- **论文：** [OpenHLM: An Empirical Recipe for Whole-Body Humanoid Loco-Manipulation](https://arxiv.org/abs/2606.22174)
- **作者：** Yingdong Hu, Haodong Zhu, Boyuan Zheng, Yihang Hu, Tong Zhang, Zunhao Chen, Junming Zhao, Ruiqian Nai, Yang Gao
- **机构：** Tsinghua University, Shanghai Qi Zhi Institute, Spirit AI
- **版本：** arXiv:2606.22174v1, 2026-06-20
- **项目主页：** [OpenHLM](https://openhlm-project.github.io/)
- **代码：** [OpenHLM-project/OpenHLM](https://github.com/OpenHLM-project/OpenHLM)
- **数据与模型：** [Dataset](https://huggingface.co/datasets/OpenHLM/OpenHLM-data) · [Checkpoints](https://huggingface.co/OpenHLM/OpenHLM-ckpts)
- **机器人平台：** Unitree G1
- **末端执行器：** 双 ChangingTek CTAG2F90-D parallel gripper
- **高层策略：** π0.5-initialized VLA
- **低层控制器：** SONIC whole-body motion tracker

## 1. 整体框架

OpenHLM 使用两层控制架构：

```text
High-Level Policy
数据采集时：Human Operator
部署时：OpenHLM VLA
        ↓
Whole-Body Reference Action
        ↓
SONIC Low-Level Controller
        ↓
Target Joint Positions
        ↓
PD Controller
        ↓
Unitree G1
```

三个 Phase 分别解决不同的问题：

```text
Phase I：Whole-Body Controller & Teleoperation
确定“怎样采到适合训练 whole-body VLA 的数据”
        ↓
Joint-based Whole-Body Teleoperation
GMR Online Retargeting
SONIC + 0.2 s Future Preview

Phase II：Whole-Body VLA Policy Design
确定“π0.5 怎样适配 G1 的全身动作空间”
        ↓
34-D State / Action
Absolute Joint Targets
Proprioception
Multi-step Flow Matching

Phase III：Heterogeneous Co-Training
确定“怎样减少昂贵的全身遥操作数据”
        ↓
Full Loco-Manip Teleop
+
Stationary G1 Teleop / HuMI
        ↓
Single Multi-task OpenHLM Policy
```

最终得到的不是多个任务分别对应多个 policy，而是 **一个 language-conditioned multi-task VLA**。语言指令、视觉和当前机器人状态共同决定模型在当前时刻应该执行哪一种全身动作。

---

## 2. Phase I：Whole-Body Controller & Teleoperation

### 2.1 Phase I 流程

```text
Human Operator
        ↓
PICO4U
├── Head-Mounted Display
├── Left / Right Controller
└── Two Leg Trackers
        ↓
Human Whole-Body Motion
        ↓
GMR Online Retargeting
        ↓
G1 Joint-Space Reference
│
├── 14-D Arm Joints
├── 12-D Leg Joints
├── 3-D Waist Joints
└── Root Roll / Pitch / Yaw Rate
        ↓
32-D Body Reference
        │
        │ + Left / Right Gripper Width
        ↓
34-D Whole-Body Action
        ↓
SONIC
Future Reference Preview：0.2 s
        ↓
High-Frequency Joint Targets
        ↓
PD Controller
        ↓
Real G1 Execution
        │
        ├── Head RGB
        ├── Left Wrist RGB
        ├── Right Wrist RGB
        ├── 34-D Proprioception
        ├── Language Instruction
        └── 34-D Demonstration Action
                 ↓
          VLA Training Data
```

### 2.2 Joint-based Whole-Body Teleoperation

数据采集时，人类操作者相当于高层 policy。PICO4U 获取头、双手和双腿的运动信息，GMR 将人体动作实时 retarget 到 Unitree G1 的 joint space。

最终采用的 body action 为 32 维：

```text
Dual Arms        14-D
Dual Legs        12-D
Waist             3-D
Root              3-D
─────────────────────
Body Action      32-D
```

其中 root 的 3 维为：

```text
root roll
root pitch
yaw angular velocity
```

再加入两个平行夹爪各 1 维开合量，VLA 最终使用：

$$
34D = 32D\ \text{body} + 2D\ \text{gripper}
$$

因此 OpenHLM 使用的是 **双平行夹爪，不是灵巧手**。

### 2.3 SONIC Low-Level Controller

SONIC 不负责理解语言和选择目标物体，它负责将高层给出的 whole-body reference 转化为真实机器人能够稳定跟踪的动作。

```text
Reference Whole-Body Motion
+
Current Robot State
+
Future Reference Preview
        ↓
SONIC Motion Tracker
        ↓
Target Joint Positions
        ↓
PD Tracking
```

高层决定“想怎样动”，SONIC 负责“尽量稳定地把动作做出来”。

论文测试 future-frame preview latency：

```text
0.0 s → 操作响应快，但 locomotion 容易抖动和跺脚
0.2 s → 平滑性与操作延迟之间最佳折中
0.4 s → 性能下降
0.6 s → 操作延迟明显，任务性能大幅下降
```

最终固定为：

$$
\Delta t = 0.2\ \text{s}
$$

### 2.4 Teleoperation Interface 对比

论文依次比较：

```text
Decoupled Control
        ↓
上半身 IK + 下半身 locomotion controller
        ↓
腿只能间接由 navigation command 控制

VR 3-Point
        ↓
Head + Two Wrist Poses + Navigation Command
        ↓
下半身由 motion planner 推断

Joint-based Whole-Body
        ↓
Human Motion → GMR
        ↓
直接得到所有 G1 body joints

SMPL-based Whole-Body
        ↓
保留 81-D SMPL-style action
        ↓
VLA 直接学习高维人体表示
```

Joint-based 方法最终胜出，原因主要有两点：

1. **Expressivity 更强。** 腿和脚可以直接参与操作，例如下蹲扩大工作空间、用脚踩垃圾桶踏板。
2. **Action representation 更紧凑。** 32-D robot joint-space 比 81-D SMPL action 更容易学习，减少冗余自由度带来的协调误差。

### 2.5 Phase I 的训练与 Loss

Phase I 本身不是一个新的 OpenHLM 神经网络训练阶段。SONIC 和 GMR 都来自已有工作，论文在这一阶段主要通过真实机器人实验确定数据采集接口。

因此不存在独立的 $\mathcal{L}_{\mathrm{phase1}}$。

OpenHLM 论文也没有重新给出 SONIC 的 RL / motion-tracking training objective。Phase I 的产物是 **统一的数据格式和可执行的全身动作空间**，供 Phase II 的 VLA 学习。

---

## 3. Phase II：Whole-Body VLA Policy Design

### 3.1 Phase II 流程

```text
Joint-based Teleoperation Dataset
│
├── Head RGB
├── Left Wrist RGB
├── Right Wrist RGB
├── Language Instruction
├── Current 34-D Proprioception s_t
└── Future 34-D Whole-Body Actions
        ↓

从 trajectory 选择时刻 t
        ↓

Current Images I_t
+
Language l
+
Current State s_t
        │
        ▼
────────────────────────────────
π0.5-based OpenHLM VLA
────────────────────────────────

Images + Language
        ↓
PaliGemma VLM
        ↓
Vision-Language Context
        │
        │
Current Proprioception s_t
        │
        │
GT Future Action Chunk A*
        ↓
Flow Matching 构造 Noisy Action Chunk A_τ
        ↓
Action Input Projection
        ↓
Action Expert
        ↑
        │ Vision-Language Context
        │ Proprioception
        │ Flow Time τ
        ↓
Predicted Vector Field
        ↓
Action Output Projection
        ↓
Flow-Matching Loss
        ↓
Backpropagation
        ↓
Humanoid-adapted OpenHLM VLA
```

### 3.2 Training Sample

从一条 demonstration trajectory 中取当前时刻 $t$。

模型条件输入为：

$$
(I_t,\ l,\ s_t)
$$

其中：

- $I_t$：当前头部和双腕 RGB；
- $l$：当前任务的 language instruction；
- $s_t$：34-D robot proprioception。

监督目标不是单独的下一帧动作，而是长度为 50 的 future action chunk：

$$
A_t^*
=
[a_t,a_{t+1},\ldots,a_{t+49}]
\in\mathbb{R}^{50\times34}
$$

34-D state / action 的默认排列为：

```text
Left Arm        7
Left Gripper    1
Right Arm       7
Right Gripper   1
Left Leg        6
Right Leg       6
Waist           3
Root            3
─────────────────
Total          34
```

OpenHLM 保留 π0.5 原有的 bimanual ordering，将 humanoid-specific 的腿、腰和 root 维度接在后面。

### 3.3 Action Projection 与 Weight Surgery

π0.5 原有 action projection 最多支持 32-D action，而 OpenHLM 需要 34-D，因此 Action Expert 的输入和输出线性层都要扩展。

论文采用 **weight surgery**：

```text
Pretrained 32-D Projection
        ↓
复制原有 pretrained weights
        ↓
扩展到 34-D
        ↓
只对新增参数使用 Xavier Initialization
```

相比将整个 action projection 随机初始化，这种方式尽量保留 π0.5 已经学习到的 robot action representation。

### 3.4 Absolute Action 与 Proprioception

OpenHLM 最终选择 **absolute joint targets**：

```text
Policy Output：
目标关节绝对位置
```

而不是：

```text
Policy Output：
相对当前状态的 joint delta
```

同时给 VLA 输入完整的 34-D proprioception。原因是头部与腕部相机无法可靠观察下半身姿态，因此仅靠视觉很难判断腿、腰和 root 当前处于什么状态。

论文发现，单独去掉 proprioception 或切换为 relative action 只产生一定下降，但 **同时去掉 proprioception + 使用 relative action 会出现严重 drift**。

### 3.5 π0.5 Pretraining

OpenHLM 的 VLA 主体直接由 **π0.5** 初始化：

```text
π0.5
│
├── PaliGemma VLM
└── Robotics Action Expert
        ↓
G1 Whole-Body Fine-tuning
```

π0.5 的 robot pretraining 数据主要来自 static / wheeled dual-arm robots，并不包含 humanoid 数据，但仍然能够迁移到 G1。

消融结果显示：

```text
π0.5 Initialization       → 最好
PaliGemma Initialization  → 明显下降
Random Initialization     → 大幅下降
```

这说明仅有视觉语言表示还不够，π0.5 在机器人数据中学习到的 manipulation prior 和 closed-loop correction behavior 对 OpenHLM 很重要。

### 3.6 Flow Matching 与 Loss

OpenHLM 沿用 π0.5 的 **Flow Matching Action Expert**。

训练时有 GT future action chunk $A^*$，先构造一个 noisy/intermediate action $A_\tau$：

```text
Noise
  │
  │  interpolation at flow time τ
  ▼
A_τ
  │
  │ + Vision / Language
  │ + Robot State
  ▼
Action Expert
  ↓
Predicted Vector Field
  ↕
Target Flow Direction
  ↓
Flow-Matching Loss
```

OpenHLM 论文没有重新展开 π0.5 的具体 flow-matching 方程。其核心目标可以概括为：

$$
L_{\text{FM}}
=
\mathbb{E}
\left[
\left\|
v_\theta(A_\tau,\tau\mid I_t,l,s_t)
-
u_\tau
\right\|_2^2
\right]
$$

其中：

- $A_\tau$：由 noise 和 GT action 构造的中间 action；
- $v_\theta$：Action Expert 预测的 vector field；
- $u_\tau$：指向真实 action distribution 的目标 flow direction。

因此高层 VLA 的训练本质属于 **supervised imitation learning**：GT supervision 来自 teleoperation demonstration，不使用 task reward，也不是 PPO。

### 3.7 Multi-step Action Generation

π0.5 推理时通过多步积分将 noisy action 逐渐变成可执行 action。

论文比较：

```text
10-step Flow Matching
        vs.
1-step Flow Matching
        vs.
1-step Drifting Model
```

单步方法推理更快，validation action MSE 甚至更低，但真实机器人 task progress 下降约 20 个百分点。作者推测单步生成的动作虽然 L2 误差较小，但更抖、更缺乏 temporal smoothness。

最终保留：

```text
Multi-step Flow Matching
Typical Integration Steps：10
```

### 3.8 训练设置

默认 VLA training：

```text
Optimizer             AdamW
β1 / β2               0.9 / 0.95
Peak Learning Rate    1e-4
LR Schedule           Cosine Decay
Warmup                1K Steps
Training Steps        30K
Batch Size            128
Action Horizon        50
Image Resolution      224 × 224
Training Hardware     4 × A800
Training Time         ~24 h
```

图像增强包括 head image 的 random crop / rotation，以及所有图像的 brightness、contrast、saturation augmentation。

在 8 个训练任务、每个任务 40 demonstrations 的设置下，最终 humanoid-adapted VLA 达到约 89% average task progress。

---

## 4. Phase III：Heterogeneous Co-Training

### 4.1 Phase III 流程

```text
────────────────────────────────────
Data Stream A
Full Loco-Manipulation Teleoperation
Tasks 1–8
────────────────────────────────────

Real G1
Head / Wrist RGB
+
Language
+
34-D Proprioception
+
34-D Whole-Body Action
        │
        │
        ├──────────────────────┐
        │                      │
        │                      │
        ▼                      ▼
────────────────────    ────────────────────
Data Stream B           Data Stream C
Stationary G1 Teleop    Stationary HuMI
Tasks 9–12              Tasks 9–12
────────────────────    ────────────────────

Same G1                  Human + UMI Grippers
Feet stay in place       + Wrist GoPro
Manipulation only        + Body Trackers
        │                      ↓
        │                  Offline IK
        │                      ↓
        │                G1 Joint Targets
        │                      │
        └──────────┬───────────┘
                   ↓
          Unified Training Format
                   ↓
       Mixed Dataset / Co-training
                   ↓
          Same OpenHLM VLA
                   ↓
       Same Flow-Matching Loss
                   ↓
       Single Multi-task Policy
```

### 4.2 Stationary G1 Teleoperation

Stationary data 使用同一台 G1、同一套 joint-based teleoperation，只去掉长距离 locomotion：

```text
Full Loco-Manip Demo
走到桌前
→ 操作
→ 走回来

Stationary Demo
机器人已经位于工作区
→ 操作
→ 结束
```

它仍然可以包含：

- arm / gripper manipulation；
- torso movement；
- height adjustment；
- squatting；
- in-place turning。

因此它不是“只有夹爪数据”，而是 **没有长距离 walking 的 same-embodiment whole-body manipulation data**。

这类数据与真实 G1 的 sensing 和 action distribution 一致，所以不仅能够提供新物体和新语言监督，也能够教新的 manipulation motion。

### 4.3 HuMI Data

HuMI 是一种 robot-free demonstration interface。人类使用两个 UMI-style hand-held grippers 和 body trackers 完成操作。

```text
Human Demonstration
│
├── Two Gripper Trajectories
├── Gripper Widths
├── Pelvis / Foot Trackers
└── Wrist GoPro RGB
        ↓
Offline IK
        ↓
Whole-Body G1 Joint Targets
        ↓
Shift Action by 0.2 s
        ↓
与 Teleop Data 对齐的 State / Action Format
        ↓
Co-training
```

动作向前平移 0.2 s，用于近似真实 teleoperation pipeline 中 SONIC 的 future-frame preview latency。

HuMI 与真实 G1 数据仍存在明显 domain gap：

```text
Vision Gap：
GoPro vs. Robot RealSense / Head Camera

Action Gap：
Human motion after IK
vs.
Real robot-in-the-loop teleoperation
```

因此在当前数据规模下，HuMI 更适合提供 **semantic supervision**，而不是新的 robot motion supervision。

### 4.4 Co-training 不是独立的新 Policy

Phase III 没有引入新的网络结构，也没有增加额外 head。

论文描述的是：

```text
Tasks 1–8 Full Loco-Manip Data
+
Tasks 9–12 Cheap Data
        ↓
共同训练同一个 OpenHLM VLA
```

而不是：

```text
先训练 locomotion policy
→ 再训练 manipulation policy
→ 再训练一个 fusion policy
```

论文也没有将 Phase III 描述成只使用新任务数据对 8-task checkpoint 做一次 new-data-only sequential fine-tuning。核心是 **heterogeneous co-training**，即保留原全身数据的同时混入 cheaper data。

### 4.5 Phase III Loss

三种数据最终都服务于相同的 VLA action prediction，因此继续使用同一个：

$$
L_{\text{FM}}
$$

没有新增：

- semantic alignment loss；
- contrastive loss；
- RL reward；
- motion-specific auxiliary loss。

不同数据流的作用来自 **数据分布互补**，而不是额外损失函数。

### 4.6 Stationary 与 HuMI 的作用差异

论文把 held-out tasks 分为两类。

#### Motion-Reuse Tasks：Tasks 9–11

新任务的动作模式在 Tasks 1–8 中已经出现，主要变化是：

```text
New Object
+
New Language Prompt
```

Stationary 和 HuMI 都可以显著提升性能，说明两者都能提供新的 semantic grounding。

#### New-Motion Task：Task 12 Pouring

Pouring 需要训练集此前不存在的 vessel-tilt motion。

```text
Stationary G1 Data
→ 能学会新的 pouring motion

HuMI Data
→ 当前数据规模下未能可靠迁移新的 motion
```

因此论文的核心结论是：

```text
Stationary G1 Teleop：Semantic + New Motion

HuMI：Mainly New Semantics
```

---

## 5. 核心模型与算法

### 5.1 π0.5 VLA

OpenHLM 没有重新设计一个新的 VLA backbone，而是保留 π0.5 的内部结构，重点修改 humanoid action / state interface。

```text
Images + Language
        ↓
PaliGemma VLM
        ↓
Vision-Language Context
        │
        │
Noisy Future Action + Proprioception
        ↓
Action Expert
        ↓
Whole-Body Action Chunk
```

PaliGemma 主要提供 scene / object / language condition，Action Expert 负责连续机器人动作生成。

### 5.2 Flow Matching

Flow Matching 是连续生成模型。训练时学习一个 vector field，使随机 noise 能沿连续轨迹流向真实动作分布。

```text
Training:
GT Action + Noise
        ↓
Intermediate Action
        ↓
Learn Correct Flow Direction

Inference:
Random Noise
        ↓
Repeated Vector-Field Integration
        ↓
Robot Action
```

与直接回归一个平均动作相比，这种生成式 action head 可以表达多模态连续动作分布。

### 5.3 GMR Retargeting

GMR 负责 Human-to-Robot motion retargeting：

```text
Human Whole-Body Motion
        ↓
Robot Kinematic Constraints
        ↓
G1 Joint-Space Motion
```

OpenHLM 的关键选择是 **在线完成 retargeting，然后直接把 robot joint-space motion 作为 VLA supervision**，而不是让 VLA 自己学习从高维 SMPL motion 到机器人动作的映射。

### 5.4 SONIC

SONIC 是 learned whole-body motion tracker。它与高层 VLA 的职责分工为：

```text
OpenHLM VLA
负责 task-level whole-body reference generation
        ↓
SONIC
负责 dynamics-aware tracking and balance
        ↓
PD Controller
负责关节目标跟踪
```

因此 OpenHLM 的“一个 policy 控制全身”指的是 **高层任务策略统一输出全身参考动作**，并不意味着从视觉直接输出电机 torque。

### 5.5 Heterogeneous Co-Training

Co-training 的核心不是额外模型，而是让同一个 policy 同时看到不同来源的 demonstrations：

```text
Expensive / High-Fidelity
Full G1 Loco-Manip Data
        +
Cheaper / Limited
Stationary G1 or HuMI Data
        ↓
Shared VLA
```

完整 G1 数据保留 locomotion 和真实 robot motion prior，便宜数据用于扩展新的对象、语言和部分 manipulation skill。

---

## 6. 推理流程

训练完成后，PICO、GMR、HuMI、offline IK 都不再参与部署。

```text
Current Observation
│
├── Head RGB
├── Left Wrist RGB
├── Right Wrist RGB
├── Language Instruction
└── 34-D Proprioception
        ↓
π0.5-based OpenHLM VLA
        ↓
PaliGemma
        ↓
Vision-Language Context
        │
        │ + Current State
        │ + Random / Noisy Action Chunk
        ▼
Action Expert
        ↓
10-step Flow-Matching Integration
        ↓
50 × 34 Future Action Chunk
        ↓
只执行最多前 25 Steps
        ↓
30 Hz Whole-Body Reference Commands
        ↓
SONIC Low-Level Controller
        ↓
50 Hz Joint Targets
        ↓
PD Controller
        ↓
Real G1
        ↓
New Images + New Proprioception
        ↓
Next VLA Inference
```

Action horizon 为 50，action step 以 30 Hz 表示，因此一次 VLA inference 预测约：

$$
50 / 30 \approx 1.67\ \text{s}
$$

但最多只执行前 25 步：

$$
25 / 30 \approx 0.83\ \text{s}
$$

随后重新观察并再次生成 action chunk。论文将这一周期写为每 $5/6$ 秒重新进行一次 inference。

因此 OpenHLM 使用的是 **chunked closed-loop control**：

```text
Observe
→ Predict a Future Chunk
→ Execute Part of the Chunk
→ Observe Again
→ Replan
```

VLA 不需要显式切换 walking / grasping / placing skill。任务阶段由语言、视觉和 robot state 隐式决定，同一个 policy 连续生成 walking、squatting、grasping、foot manipulation 和 return motion。

---

## 7. 消融实验

论文的消融主要服务于三个 Phase 的设计选择。

| 设计问题 | 最终结论 |
|---|---|
| Teleoperation Interface | Joint-based whole-body teleoperation 最稳定且表达能力最完整。 |
| Joint Space vs. SMPL | 32-D robot joint-space 优于 81-D SMPL representation。 |
| Future Preview | 0.2 s 在平滑 locomotion 与 teleoperation latency 之间最合适。 |
| Action Projection | Weight surgery 略优于完全随机初始化。 |
| Action Ordering | 保留 π0.5 pretrained bimanual ordering 更合适。 |
| Absolute vs. Relative Action | Absolute joint target 更稳定。 |
| Proprioception | 保留全身 proprioception，尤其可以避免与 relative action 组合产生 drift。 |
| Pretraining | π0.5 robot pretraining 显著优于仅 PaliGemma 或随机初始化。 |
| Action Generation | 10-step flow matching 明显优于两种 one-step 方案。 |
| Whole-Body Data Scaling | 20 demos 后已有明显提升，40 demos / task 后接近饱和。 |
| Heterogeneous Data | Stationary data 可传递 semantics + motion；HuMI 当前主要传递 semantics。 |

一个重要现象是：**validation action MSE 与真实机器人性能并不一致**。PaliGemma 初始化和 π0.5 初始化的离线 action MSE 接近，但 π0.5 在抓取失败后的闭环重试能力明显更强；one-step generation 的 action MSE 更低，真实机器人表现反而更差。

---

## 8. HLM-12 与系统表现

HLM-12 包含 12 个 language-conditioned tasks，覆盖四类能力：

```text
1. Pick-and-Place with Locomotion
2. Whole-Body Workspace Extension
3. Using Body Parts as Manipulators
4. Loco-Manipulation under Environmental Constraints
```

代表任务包括：

- Cola Placement；
- Shelf Cup / Cube Transfer；
- Bottle Disposal；
- Jar Opening；
- Toy Stowing；
- Sword Extraction；
- Cart Pushing；
- Shuttlecock Setup；
- Pouring。

8 个 full-teleop training tasks 上，OpenHLM 达到约 **89% average task progress**。

在 4 个 whole-body teleop 从未覆盖的 held-out tasks 上：

```text
8-task Baseline          33%
Stationary Co-training   87%
HuMI Co-training         67%
12-task Teleop Oracle    94%
```

其中 HuMI 的主要损失来自 Pouring 这一 new-motion task。

在额外的 long-horizon fruit arrangement task 上：

```text
Ψ0                         48.8%
GR00T N1.6                 57.5%
OpenHLM + HuMI             87.5%
OpenHLM Teleop Oracle      97.5%
```

OpenHLM + HuMI 的 demonstration time 为 1.14 h，而两个 baseline 都约为 2.70 h。

---

## 9. Conclusion

OpenHLM 的核心结论不是提出一个新的 Transformer 结构，而是给出一套经过系统实验验证的 whole-body humanoid VLA recipe：

```text
Joint-based Whole-Body Teleoperation
        +
GMR Online Retargeting
        +
SONIC Whole-Body Tracking
        +
π0.5 Robot Pretraining
        +
34-D Whole-Body State / Action Interface
        +
Absolute Joint Targets
        +
Proprioception
        +
Multi-step Flow Matching
        +
Heterogeneous Co-training
        ↓
Single Multi-task Whole-Body VLA
```

这套系统说明，humanoid loco-manipulation 的高层策略不需要将 locomotion 和 manipulation 显式拆成两个 policy。一个 language-conditioned VLA 可以同时输出腿、腰、手臂和夹爪动作，下蹲、转身、用脚踩踏板、抓取和携物行走都可以出现在同一个 action chunk 中；动态平衡和高频跟踪则交给 SONIC。

论文同时说明，**robot pretraining 的价值不能只通过离线 action MSE 衡量**。π0.5 虽然没有使用 humanoid 数据预训练，但其在大量机器人操作数据中形成的闭环 manipulation prior 仍能够跨 embodiment 迁移到 G1。

Heterogeneous co-training 进一步降低了全身遥操作成本。真实 G1 stationary data 可以同时补充新的 semantic grounding 与新 motion，而 HuMI 在当前规模下主要适合扩展新物体和新语言，尚不能稳定提供此前未出现的机器人动作模式。

从后续 loco-manipulation 研究角度看，OpenHLM 已经提供了完整的 **视觉 + 语言 + 全身状态 → 全身动作** baseline，但仍使用两维 parallel gripper command，没有 dexterous finger-level action，也没有 tactile feedback。这为进一步研究灵巧手、触觉条件下的 whole-body coordination、接触恢复以及 manipulation precision 与 balance 之间的协调留下了明显空间。
