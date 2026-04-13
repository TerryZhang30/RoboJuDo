# G1 关节索引与名称对照表

## 1. Env / Motion 29-DoF（G1_29DoF / G1_29AsapDoF / G1_29HumanxDoF）

此关节顺序同时用于：环境空间、motion pkl 数据、Humanx 策略的 obs_dof / action_dof。

| 索引 | 负索引 | 关节名称 | 部位 |
|------|--------|---------|------|
| 0 | -29 | left_hip_pitch_joint | 左腿 |
| 1 | -28 | left_hip_roll_joint | 左腿 |
| 2 | -27 | left_hip_yaw_joint | 左腿 |
| 3 | -26 | left_knee_joint | 左腿 |
| 4 | -25 | left_ankle_pitch_joint | 左腿 |
| 5 | -24 | left_ankle_roll_joint | 左腿 |
| 6 | -23 | right_hip_pitch_joint | 右腿 |
| 7 | -22 | right_hip_roll_joint | 右腿 |
| 8 | -21 | right_hip_yaw_joint | 右腿 |
| 9 | -20 | right_knee_joint | 右腿 |
| 10 | -19 | right_ankle_pitch_joint | 右腿 |
| 11 | -18 | right_ankle_roll_joint | 右腿 |
| 12 | -17 | waist_yaw_joint | 腰 |
| 13 | -16 | waist_roll_joint | 腰 |
| 14 | -15 | waist_pitch_joint | 腰 |
| 15 | -14 | left_shoulder_pitch_joint | 左臂 |
| 16 | -13 | left_shoulder_roll_joint | 左臂 |
| 17 | -12 | left_shoulder_yaw_joint | 左臂 |
| 18 | -11 | left_elbow_joint | 左臂 |
| 19 | -10 | left_wrist_roll_joint | 左手腕 |
| 20 | -9 | left_wrist_pitch_joint | 左手腕 |
| 21 | -8 | left_wrist_yaw_joint | 左手腕 |
| 22 | -7 | right_shoulder_pitch_joint | 右臂 |
| 23 | -6 | right_shoulder_roll_joint | 右臂 |
| 24 | -5 | right_shoulder_yaw_joint | 右臂 |
| 25 | -4 | right_elbow_joint | 右臂 |
| 26 | -3 | right_wrist_roll_joint | 右手腕 |
| 27 | -2 | right_wrist_pitch_joint | 右手腕 |
| 28 | -1 | right_wrist_yaw_joint | 右手腕 |

---

## 2. AMO obs_dof（G1AmoDoF，23 关节）

AMO 的观测空间。相比 29-DoF 去掉了 6 个 wrist 关节。

| AMO 索引 | 关节名称 | 对应 29-DoF 索引 | 部位 |
|----------|---------|-----------------|------|
| 0 | left_hip_pitch_joint | 0 | 左腿 |
| 1 | left_hip_roll_joint | 1 | 左腿 |
| 2 | left_hip_yaw_joint | 2 | 左腿 |
| 3 | left_knee_joint | 3 | 左腿 |
| 4 | left_ankle_pitch_joint | 4 | 左腿 |
| 5 | left_ankle_roll_joint | 5 | 左腿 |
| 6 | right_hip_pitch_joint | 6 | 右腿 |
| 7 | right_hip_roll_joint | 7 | 右腿 |
| 8 | right_hip_yaw_joint | 8 | 右腿 |
| 9 | right_knee_joint | 9 | 右腿 |
| 10 | right_ankle_pitch_joint | 10 | 右腿 |
| 11 | right_ankle_roll_joint | 11 | 右腿 |
| 12 | waist_yaw_joint | 12 | 腰 |
| 13 | waist_roll_joint | 13 | 腰 |
| 14 | waist_pitch_joint | 14 | 腰 |
| 15 | left_shoulder_pitch_joint | 15 | 左臂 |
| 16 | left_shoulder_roll_joint | 16 | 左臂 |
| 17 | left_shoulder_yaw_joint | 17 | 左臂 |
| 18 | left_elbow_joint | 18 | 左臂 |
| 19 | right_shoulder_pitch_joint | **22** | 右臂 |
| 20 | right_shoulder_roll_joint | **23** | 右臂 |
| 21 | right_shoulder_yaw_joint | **24** | 右臂 |
| 22 | right_elbow_joint | **25** | 右臂 |

> 注意：索引 0-18 与 29-DoF 一一对应，但从索引 19 开始发生偏移——AMO 跳过了 29-DoF 中的 left_wrist（19-21），所以 AMO[19] 对应 29-DoF[22]。

---

## 3. AMO action_dof（G1AmoLowerDoF，15 关节）

AMO 的动作空间。只包含下肢和腰部，不控制上肢。

| action 索引 | 关节名称 | 对应 29-DoF 索引 | 对应 AMO obs 索引 |
|------------|---------|-----------------|-----------------|
| 0 | left_hip_pitch_joint | 0 | 0 |
| 1 | left_hip_roll_joint | 1 | 1 |
| 2 | left_hip_yaw_joint | 2 | 2 |
| 3 | left_knee_joint | 3 | 3 |
| 4 | left_ankle_pitch_joint | 4 | 4 |
| 5 | left_ankle_roll_joint | 5 | 5 |
| 6 | right_hip_pitch_joint | 6 | 6 |
| 7 | right_hip_roll_joint | 7 | 7 |
| 8 | right_hip_yaw_joint | 8 | 8 |
| 9 | right_knee_joint | 9 | 9 |
| 10 | right_ankle_pitch_joint | 10 | 10 |
| 11 | right_ankle_roll_joint | 11 | 11 |
| 12 | waist_yaw_joint | 12 | 12 |
| 13 | waist_roll_joint | 13 | 13 |
| 14 | waist_pitch_joint | 14 | 14 |

---

## 4. Humanx obs_dof / action_dof（G1_29HumanxDoF，29 关节）

Humanx 的观测和动作空间完全相同，均为 29 关节，与 Env 29-DoF 一致。

| 索引 | 关节名称 | 对应 29-DoF 索引 |
|------|---------|-----------------|
| 0-5 | left_hip_* (6 joints) | 0-5 |
| 6-11 | right_hip_* (6 joints) | 6-11 |
| 12-14 | waist_* (3 joints) | 12-14 |
| 15-21 | left_shoulder/elbow/wrist (7 joints) | 15-21 |
| 22-28 | right_shoulder/elbow/wrist (7 joints) | 22-28 |

完全一一对应，无索引偏移。

---

## 5. 索引偏移对照（AMO obs ↔ 29-DoF 中不一致的部分）

| 关节名称 | AMO obs 索引 | 29-DoF 索引 | 偏移量 |
|---------|-------------|------------|--------|
| left_hip ~ left_elbow | 0-18 | 0-18 | 0 |
| left_wrist_roll | **不存在** | 19 | — |
| left_wrist_pitch | **不存在** | 20 | — |
| left_wrist_yaw | **不存在** | 21 | — |
| right_shoulder_pitch | **19** | **22** | **+3** |
| right_shoulder_roll | **20** | **23** | **+3** |
| right_shoulder_yaw | **21** | **24** | **+3** |
| right_elbow | **22** | **25** | **+3** |
| right_wrist_roll | **不存在** | 26 | — |
| right_wrist_pitch | **不存在** | 27 | — |
| right_wrist_yaw | **不存在** | 28 | — |

> AMO 去掉了 6 个 wrist 关节（左 3 + 右 3），导致右臂 4 个关节的索引相比 29-DoF 偏移了 +3。
> 这就是为什么 AMO 和 Humanx/Motion 之间不能直接按索引映射，必须按关节名匹配。

---

## 6. g1_switch 中 motion_adjustments 索引含义

### AMO 的 motion_adjustments（29-DoF 空间索引）

```python
motion_adjustments = {
    -5: 0.3,    # 29-DoF[24] = right_shoulder_yaw_joint  (AMO obs[21])
    -12: -0.2,  # 29-DoF[17] = left_shoulder_yaw_joint   (AMO obs[17])
    -1: -0.3,   # 29-DoF[28] = right_wrist_yaw_joint     (AMO 中不存在)
    4: 0.05,    # 29-DoF[4]  = left_ankle_pitch_joint    (AMO obs[4])
    10: 0.05,   # 29-DoF[10] = right_ankle_pitch_joint   (AMO obs[10])
}
```

### g1_humanx 的 motion_adjustments（29-DoF 空间索引）

```python
motion_adjustments = {
    -6: 0.4,    # 29-DoF[23] = right_shoulder_roll_joint  (Humanx[23])
    -13: -0.2,  # 29-DoF[16] = left_shoulder_roll_joint   (Humanx[16])
    -8: -0.3,   # 29-DoF[21] = left_wrist_yaw_joint       (Humanx[21])
    4: 0.1,     # 29-DoF[4]  = left_ankle_pitch_joint     (Humanx[4])
    10: 0.1,    # 29-DoF[10] = right_ankle_pitch_joint    (Humanx[10])
}
```
