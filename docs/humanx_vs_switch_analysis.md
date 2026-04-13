# Humanx Policy 单独使用 vs Switch Policy 中使用 —— 差异分析

## 1. 概述

| 维度 | 单独 Humanx (`g1_humanx`) | Switch 中的 Humanx (`g1_switch`) |
|------|--------------------------|----------------------------------|
| Pipeline 类型 | `RlPipeline` | `RlMultiPolicyPipeline` |
| 配置类 | `RlPipelineCfg` | `RlMultiPolicyPipelineCfg` |
| 策略字段 | `policy: G1HumanxPolicyCfg` | `policies[1]: G1HumanxPolicyCfg` |
| 初始活跃策略 | Humanx 自身 | AMO (`policies[0]`) |
| 环境配置 | 自定义 `G1MujocoEnvCfg(sim_decimation=10, ...)` | 默认 `G1MujocoEnvCfg()` |

---

## 2. 配置差异详细对比

### 2.1 ONNX 模型文件不同

```
# g1_humanx（单独使用）
policy_file_override = "assets/models/g1/humanx/model_689000.onnx"

# g1_switch 中的 Humanx
policy_file_override = "assets/models/g1/humanx/jumpshot.onnx"
```

**影响**：两个 ONNX 模型可能是不同训练阶段或不同训练配置的产物，网络权重不同会直接影响输出行为。

### 2.2 motion_adjustments 不同

```
# g1_humanx（单独使用）—— 有 motion_adjustments
motion_adjustments = {
    -6: 0.4,     # right_wrist_pitch_joint
    -13: -0.2,   # left_shoulder_roll_joint
    -8: -0.3,    # right_shoulder_yaw_joint
    4: 0.1,      # left_ankle_pitch_joint
    10: 0.1,     # right_ankle_pitch_joint
}

# g1_switch 中的 Humanx —— 无 motion_adjustments（使用默认空 dict）
motion_adjustments = {}
```

**影响**：
- `motion_adjustments` 会修改 `init_angles`（motion 首帧关节角 + 偏移）
- 同时在 `PolicyWrapper._apply_motion_adjustments` 中，对**非策略控制的关节**产生额外的 `_pd_adjustments` 偏移
- 单独使用时，这些偏移帮助机器人在运动过程中保持更好的姿态；Switch 中缺失这些偏移

### 2.3 policy_name 不同

```
# g1_humanx
policy_name = "fake_action"   # 影响 motion_data_path 的默认推导

# g1_switch 中的 Humanx
policy_name = "jumpshot"      # 默认值
```

但两者都显式指定了 `policy_file_override` 和/或 `motion_data_path`，所以 `policy_name` 的差异主要影响日志显示。

### 2.4 ⚠️ 环境 sim_decimation 不同（关键差异）

```
# g1_humanx（单独使用）
env: G1MujocoEnvCfg = G1MujocoEnvCfg(
    forward_kinematic=None,
    update_with_fk=False,
    born_place_align=True,
    sim_decimation=10,        # ← 每个控制步仿真 10 次
)

# g1_switch
env: G1MujocoEnvCfg = G1MujocoEnvCfg()  # 使用默认值
# 默认 sim_decimation = 20              # ← 每个控制步仿真 20 次
```

**影响**：
- `sim_decimation` 决定了每个控制步中 MuJoCo 物理仿真的子步数
- `control_dt = sim_dt × sim_decimation`
  - 单独使用：`control_dt = 0.001 × 10 = 0.01s`（100Hz 控制匹配 100Hz 策略）
  - Switch 中：`control_dt = 0.001 × 20 = 0.02s`（50Hz 控制，但策略频率仍为 100Hz）
- **这意味着 Switch 中每个控制步的物理仿真时间更长，PD 控制器的响应特性完全不同**

### 2.5 环境其他配置差异

```
# g1_humanx（单独使用）
forward_kinematic = None
update_with_fk = False
born_place_align = True

# g1_switch（默认值）
update_with_fk = True         # 默认开启正运动学更新
born_place_align = False      # 默认值（未显式设置）
```

---

## 3. 策略输入（Observation）对比

### 3.1 观测空间结构（两者一致）

两者使用相同的 `G1HumanxPolicyCfg`，观测键和维度完全一致：

| 观测分量 | 维度 | 说明 |
|---------|------|------|
| `actions` | 29 | 上一步网络输出（clip 后、乘 scale 前） |
| `base_ang_vel` | 3 | 基座角速度 × 0.25 |
| `dof_pos` | 29 | (当前关节角 - default_pos) × 1.0 |
| `dof_vel` | 29 | 关节角速度 × 0.05 |
| `history_obs_buf` | 372 | 4 步历史 × (29+3+29+29+3) = 372 |
| `pd_error` | 29 | (last_action × action_scales + default_dof_pos) - dof_pos |
| `projected_gravity` | 3 | 重力在基座坐标系的投影 |
| **总计** | **494** | 按 `sorted(actor_obs_keys)` 拼接 |

### 3.2 ⚠️ 观测数值差异（关键问题）

虽然结构一致，但**实际数值会有差异**：

#### (a) `dof_pos` 的基准不同
- `dof_pos_minus_default = dof_pos - self.default_dof_pos`
- `default_dof_pos` 来自 `G1_29AsapDoF.default_pos`，两者一致
- **但切换瞬间的实际 `dof_pos` 可能与策略训练时的初始状态差距很大**

#### (b) `last_action` 在切换时被重置为零
```python
# HumanxPolicy.reset()
def reset(self):
    self.timestep = 0
    self.flag_motion_done = False
    self.last_action = np.zeros(self.num_actions, dtype=np.float32)
```
- 切换时 `switch_policy` 会调用两次 `reset()`（第82行和第88行）
- `last_action = 0` → `actions` 观测为全零 → `pd_error = default_dof_pos - dof_pos`
- **如果切换瞬间机器人不在 default_pos 附近，pd_error 会非常大，导致网络输出异常**

#### (c) 历史缓冲被清空
```python
# reset() 中
self.history_buf.clear()
for _ in range(self.history_buf.maxlen):
    obs_a = [np.zeros(...) for k in sorted(self.history_obs_dims.keys())]
    self.history_buf.appendleft(obs_a)
```
- 切换后 `history_obs_buf` 全为零
- 策略训练时的历史缓冲是连续的，突然变零会导致网络行为异常

#### (d) warmup 期间的观测
- 切换前 10 步（`DELAY_STEPS_SWITCH = 10`），目标策略会被 warmup：
```python
# PolicyManager.step()
for idx in self.warmup_policy_indices:
    if idx != self.current_policy_id:
        self.policy_by_id(idx).get_observation(env_data, ctrl_data)
```
- 但 warmup 只调用 `get_observation`，**不调用 `get_action`**
- 所以 `last_action` 在 warmup 期间始终为零，历史缓冲中的 `actions` 项也全为零
- **warmup 并不能真正预热策略的内部状态**

---

## 4. 策略输出（Action）对比

### 4.1 动作空间（两者一致）

| 参数 | 值 |
|------|-----|
| 动作维度 | 29（与 `action_dof.num_dofs` 一致） |
| `actions_scale` | 0.25 |
| `action_clip` | 100.0 |
| action_scales[j] | `0.25 × dof_effort_limits[j] / stiffness[j]` |

### 4.2 PD 目标计算

```python
# PolicyWrapper.get_pd_target()
action = self.policy.get_action(obs)           # ONNX 推理 → clip → × action_scales
pd_target = action + self.policy.default_pos   # 加上默认关节角
pd_target = self.actions_adapter.fit(pd_target, template=env_dof_cfg.default_pos)
if self._has_pd_adjustments:
    pd_target += self._pd_adjustments          # 加上 motion_adjustments 偏移
```

### 4.3 ⚠️ motion_adjustments 对 pd_target 的影响

单独使用时有 `motion_adjustments`，会产生 `_pd_adjustments`：
- 对于**不在策略动作空间中的关节**（如果有的话），直接加偏移到 pd_target
- 对于**在策略动作空间中的关节**，偏移只影响 `init_angles`，不影响运行时 pd_target

Switch 中的 Humanx 没有 `motion_adjustments`，所以：
- `init_angles` = motion 首帧原始值（无偏移）
- `_pd_adjustments` = 全零

---

## 5. 初始角度（init_angles）对比

### 5.1 来源

两者都从 `jumpshot.pkl` 的 motion 数据第一帧读取：
```python
self.init_angles = self.motion_data[self.motion_name]['dof'][0, :].copy()
```

### 5.2 偏移差异

```
# g1_humanx（单独使用）
init_angles[idx] += offset  # 对 5 个关节施加偏移
# 索引 -6, -13, -8, 4, 10

# g1_switch 中的 Humanx
# 无偏移，init_angles = motion 首帧原始值
```

### 5.3 init_angles 的使用场景

| 使用场景 | 单独 Humanx | Switch 中 Humanx |
|---------|------------|-----------------|
| `get_init_dof_pos()` | 返回带偏移的 init_angles | 返回原始 init_angles |
| Pipeline prepare 阶段 | 用于初始姿态插值 | 用于切换时的目标姿态（如果 `switch_prepare_duration_s > 0`） |
| 观测中的 `dof_pos` | 不直接使用 init_angles | 不直接使用 init_angles |

---

## 6. ⚠️ 策略切换时的状态变化（核心问题分析）

### 6.1 切换流程时序

```
触发 [POLICY_SWITCH],1
    │
    ├── 1. target_policy.reset()          ← last_action=0, history=0, timestep=0
    ├── 2. 加入 warmup_policy_indices
    │
    ├── 3. 等待 10 步（DELAY_STEPS_SWITCH）
    │   └── 每步: target_policy.get_observation()  ← 只读观测，不推理动作
    │
    ├── 4. _prepare_for_switch(policy_id)
    │   └── if switch_prepare_duration_s <= 0: return  ← g1_switch 默认为 0，直接跳过！
    │
    ├── 5. target_policy.reset()          ← 再次重置！之前 warmup 积累的历史全部清空
    │
    └── 6. set_policy(policy_id)
        ├── env.update_dof_cfg(override_cfg=humanx_action_dof)  ← 更新环境刚度/阻尼
        ├── freq = 100Hz
        └── 开始用 Humanx 策略控制
```

### 6.2 关键问题

#### 问题 1：无过渡插值（`switch_prepare_duration_s = 0`）
- `g1_switch` 没有设置 `switch_prepare_duration_s`，默认为 0
- `_prepare_for_switch` 直接 return，**没有从当前姿态到目标姿态的平滑过渡**
- 切换瞬间，机器人可能处于 AMO 策略的某个姿态，与 Humanx 期望的初始姿态差距很大

#### 问题 2：双重 reset 清空了 warmup 效果
```python
def switch_policy(self, policy_id, on_before_set=None):
    self.policy_by_id(policy_id).reset()       # 第一次 reset
    self.warmup_policy_indices.add(policy_id)
    
    def _do_switch():
        if on_before_set is not None:
            on_before_set(policy_id)
        self.policy_by_id(policy_id).reset()   # 第二次 reset！
        self.set_policy(policy_id)
```
- 10 步 warmup 期间积累的历史缓冲在第二次 `reset()` 中被完全清空
- **warmup 完全无效**

#### 问题 3：环境刚度/阻尼突变
- `set_policy` 调用 `env.update_dof_cfg(override_cfg=self.policy.cfg_action_dof)`
- Humanx 使用 `G1_29HumanxDoF` 的刚度/阻尼，与 AMO 使用的可能不同
- 刚度/阻尼的突然变化会导致 PD 控制器行为突变

#### 问题 4：sim_decimation 不匹配
- Switch 环境使用默认 `sim_decimation=20`，而单独 Humanx 使用 `sim_decimation=10`
- Humanx 策略以 100Hz 运行，但环境的 `control_dt = 0.02s`（50Hz 物理更新）
- **控制频率与物理仿真频率不匹配，可能导致控制不稳定**

#### 问题 5：机器人物理状态不连续
- 切换前机器人在 AMO 策略控制下运动，关节角度、速度、姿态都是 AMO 的状态
- 切换后 Humanx 策略的 `last_action = 0`，`pd_error = default_pos - current_pos`
- 如果当前姿态偏离 `default_pos` 较大，第一步的 pd_target 会产生很大的跳变

---

## 7. 摔倒原因分析

综合以上分析，策略切换后机器人摔倒的可能原因按优先级排列：

### 高优先级（最可能的原因）

| # | 原因 | 详细说明 |
|---|------|---------|
| 1 | **sim_decimation 不匹配** | Switch 环境 `sim_decimation=20`（control_dt=0.02s），而 Humanx 训练时可能使用 `sim_decimation=10`（control_dt=0.01s）。物理仿真步长翻倍会导致 PD 控制器响应完全不同，力矩计算 `torque = (pd_target - dof_pos) * stiffness - dof_vel * damping` 在更长的仿真间隔下会产生更大的位置偏差累积 |
| 2 | **无过渡插值** | `switch_prepare_duration_s=0` 导致切换瞬间 pd_target 跳变。从 AMO 的输出直接切到 Humanx 的输出，中间没有任何平滑过渡 |
| 3 | **双重 reset 清空历史** | warmup 积累的历史在切换时被第二次 `reset()` 清空，策略收到全零历史，行为异常 |

### 中优先级

| # | 原因 | 详细说明 |
|---|------|---------|
| 4 | **ONNX 模型不同** | `model_689000.onnx` vs `jumpshot.onnx`，可能是不同训练阶段的模型 |
| 5 | **缺少 motion_adjustments** | 单独使用时有 5 个关节的偏移调整，Switch 中没有 |
| 6 | **初始姿态差距大** | AMO 运动中的姿态可能与 Humanx 期望的起始姿态差距很大 |

### 低优先级

| # | 原因 | 详细说明 |
|---|------|---------|
| 7 | **环境 FK 配置不同** | 单独使用 `update_with_fk=False`，Switch 使用默认 `True` |
| 8 | **born_place_align 不同** | 可能影响坐标系对齐 |

---

## 8. 建议修复方案

### 方案 1：统一环境配置（优先级最高）

```python
# g1_cfg.py 中 g1_switch 的 env 配置应与 g1_humanx 一致
class g1_switch(RlMultiPolicyPipelineCfg):
    env: G1MujocoEnvCfg = G1MujocoEnvCfg(
        forward_kinematic=None,
        update_with_fk=False,
        born_place_align=True,
        sim_decimation=10,          # ← 与单独 Humanx 一致
    )
```

### 方案 2：添加切换过渡时间

```python
class g1_switch(RlMultiPolicyPipelineCfg):
    switch_prepare_duration_s: float = 1.0  # 1 秒过渡插值
```

### 方案 3：修复双重 reset 问题

在 `rl_multi_policy_pipeline.py` 中，去掉切换时的第二次 reset，保留 warmup 积累的历史：

```python
def switch_policy(self, policy_id, on_before_set=None):
    self.policy_by_id(policy_id).reset()
    self.warmup_policy_indices.add(policy_id)

    def _do_switch():
        if on_before_set is not None:
            on_before_set(policy_id)
        # self.policy_by_id(policy_id).reset()  # ← 去掉第二次 reset
        self.set_policy(policy_id)
```

### 方案 4：统一模型和 motion_adjustments

```python
# g1_switch 中的 Humanx 配置应与 g1_humanx 一致
G1HumanxPolicyCfg(
    policy_file_override="assets/models/g1/humanx/model_689000.onnx",
    motion_adjustments={
        -6: 0.4,
        -13: -0.2,
        -8: -0.3,
        4: 0.1,
        10: 0.1,
    },
    motion_data_path="assets/motions/g1/humanx/jumpshot.pkl",
)
```

### 方案 5：增强 warmup 机制

当前 warmup 只调用 `get_observation`，不调用 `get_action`，导致 `last_action` 始终为零。
可以在 warmup 中同时调用 `get_action` 来更新内部状态：

```python
def step(self, env_data, ctrl_data):
    for idx in self.warmup_policy_indices:
        if idx != self.current_policy_id:
            obs, _ = self.policy_by_id(idx).get_observation(env_data, ctrl_data)
            self.policy_by_id(idx).policy.get_action(obs)  # ← 也执行推理，更新 last_action
    self.timer.tick()
```

---

## 9. 推荐修复优先级

1. **首先**：统一 `sim_decimation=10`（方案 1）—— 这是最可能的根本原因
2. **其次**：添加 `switch_prepare_duration_s=1.0`（方案 2）—— 提供平滑过渡
3. **然后**：修复双重 reset（方案 3）—— 保留 warmup 效果
4. **同时**：统一模型和 motion_adjustments（方案 4）—— 确保行为一致
5. **最后**：增强 warmup 机制（方案 5）—— 让策略内部状态更合理
