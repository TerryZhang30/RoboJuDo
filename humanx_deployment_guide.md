# Humanx / Humanx Loop 部署指南

本文参考仓库根目录的 [README.md](../README.md)，整理 RoboJuDo 的环境配置、sim2sim 运行方式和 Unitree G1 真机运行方式。重点覆盖当前仓库已经注册好的 Humanx 与 Humanx Loop 策略：

| 场景 | 配置名 | 环境 | 策略 |
| --- | --- | --- | --- |
| Humanx sim2sim | `g1_humanx` | `MujocoEnv` | `G1HumanxPolicyCfg` |
| Humanx 真机 | `g1_humanx_real` | `UnitreeCppEnv` | `G1HumanxPolicyCfg` |
| Humanx Loop sim2sim | `g1_humanx_loop` | `MujocoEnv` | `G1HumanxLoopPolicyCfg` |
| Humanx Loop 真机 | `g1_humanx_loop_real` | `UnitreeCppEnv` | `G1HumanxLoopPolicyCfg` |

## 1. 基础环境配置

### 1.1 创建 Python 环境

建议使用 Python 3.11，与 [pyproject.toml](../pyproject.toml) 中的要求保持一致。

```bash
conda create -n robojudo python=3.11 -y
conda activate robojudo
```

### 1.2 安装 RoboJuDo

在仓库根目录执行：

```bash
cd /home/zzx/Documents/RoboJuDo

# 可选：如果只需要 CPU 推理，可以先安装 CPU 版 torch
pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install -e .
```

Humanx / Humanx Loop 使用 `onnxruntime` 加载 `.onnx` 模型，并用 `joblib` 读取 motion `.pkl` 文件；这些依赖已经写在 [requirements.txt](../requirements.txt) 中，会随 `pip install -e .` 安装。

### 1.3 安装 sim2sim 可视化模块

sim2sim 依赖 MuJoCo 与 viewer。`submodule_cfg.yaml` 默认启用了 `mujoco_viewer`：

```bash
python submodule_install.py
```

如果只想显式安装 viewer：

```bash
python submodule_install.py mujoco_viewer
```

### 1.4 真机 SDK 配置

Humanx 真机配置默认使用 `UnitreeCppEnv`，需要先安装 Unitree 官方 C++ SDK，再安装本仓库内置的 `unitree_cpp` 绑定。更详细步骤见 [docs/unitree_setup.md](unitree_setup.md)。

```bash
# 安装 unitree_cpp 之前，请先安装 Unitree 官方 unitree_sdk2
python submodule_install.py unitree_cpp

# 验证导入
python -c "from robojudo.environment import UnitreeCppEnv"
```

如果导入失败，优先检查 `unitree_sdk2` 是否已正确安装，以及 `unitree_cpp` 是否重新编译安装。

## 2. Humanx / Humanx Loop 资源检查

当前 Humanx 相关资源放在：

```text
assets/models/g1/humanx/
assets/motions/g1/humanx/
```

默认配置使用的核心文件如下：

| 配置 | 模型文件 | Motion 文件 |
| --- | --- | --- |
| `g1_humanx` / `g1_humanx_real` | `assets/models/g1/humanx/model_689000.onnx` | `assets/motions/g1/humanx/BMaster_fake_action_and_shot_hoi_wsf.pkl` |
| `g1_humanx_loop` / `g1_humanx_loop_real` | `assets/models/g1/humanx/loop.onnx` | `assets/motions/g1/humanx/BMaster_fake_action_and_shot_hoi_loop.pkl` |

注意：当前 [g1_cfg.py](../robojudo/config/g1/g1_cfg.py) 中部分 Humanx 路径使用了绝对路径 `/home/zzx/Documents/RoboJuDo/...`。如果仓库移动到其他目录，需要同步修改 `policy_file_override` 和 `motion_data_path`。

## 3. 运行 Humanx Sim2Sim

### 3.1 Humanx

Humanx 是一次性 Human-Object Interaction 动作策略。当前 `g1_humanx` 配置中启用了球体，并设置 `sim_decimation=10`、`sim_slowmo_factor=3.0`，方便慢速观察动作。

```bash
python scripts/run_pipeline.py -c g1_humanx
```

键盘控制：

| 按键 | 命令 | 作用 |
| --- | --- | --- |
| `i` | `[SIM_REBORN]` | 重生仿真机器人 |
| `o` | `[SHUTDOWN]` | 关闭 pipeline |
| `r` | `[MOTION_RESET]` | 重置动作与仿真状态 |

### 3.2 Humanx Loop

Humanx Loop 是带 motion phase 的循环/重放版本，观测中额外包含 `ref_motion_phase`。它适合调试可重复执行的 HOI 动作。

```bash
python scripts/run_pipeline.py -c g1_humanx_loop
```

键盘控制：

| 按键 | 命令 | 作用 |
| --- | --- | --- |
| `i` | `[SIM_REBORN]` | 重生仿真机器人 |
| `o` | `[SHUTDOWN]` | 关闭 pipeline |
| `r` | `[MOTION_RESET]` | 完整重置动作与仿真状态 |
| `t` | `[MOTION_REPLAY]` | 不重置物理状态，只重放 motion phase；如果球未释放，会重新把球 hold 到手部附近 |

## 4. 真机运行前检查

真机部署风险很高。运行任何策略前，至少完成以下检查：

1. 确认机器人急停手段可用，操作者可以立刻断电或进入阻尼/停止状态。
2. 先在 sim2sim 中验证同一个配置可以正常运行。
3. 检查机器人周围留有足够空间，Humanx 动作可能包含大幅度上肢和身体运动。
4. 确认 `net_if` 对应的是与机器人通信的网卡。
5. 确认真机配置中的 `control_dt=0.01` 与策略频率 `freq=100` 对齐。

### 4.1 配置网卡

Humanx 真机配置在 [g1_cfg.py](../robojudo/config/g1/g1_cfg.py) 中：

```python
class g1_humanx_real(g1_humanx):
    env: G1RealEnvCfg = G1RealEnvCfg(
        env_type="UnitreeCppEnv",
        unitree=G1UnitreeCfg(net_if="eth0", control_dt=0.01),
    )
```

Humanx Loop 真机配置类似：

```python
class g1_humanx_loop_real(g1_humanx_loop):
    env: G1RealEnvCfg = G1RealEnvCfg(
        env_type="UnitreeCppEnv",
        unitree=G1UnitreeCfg(net_if="eth0", control_dt=0.01),
    )
```

如果策略直接运行在 G1 板载电脑上，通常使用 `eth0`。如果从外部电脑通过网线连接机器人，需要按 [docs/unitree_setup.md](unitree_setup.md) 找到实际网卡名，并替换 `net_if`。

## 5. 运行 Humanx 真机策略

### 5.1 启动流程

执行：

```bash
python scripts/run_pipeline.py -c g1_humanx_real
```

`run_pipeline.py` 检测到真实环境后，会按以下流程启动：

1. 等待零力矩启动：按 `Start` 继续，按 `L2` 退出。
2. `prepare()`：在约 2 秒内把机器人插值到 Humanx 初始姿态。
3. 等待正式启动确认：按 `L1` 开始策略，按 `L2` 退出。
4. 策略循环运行，pipeline 以 `freq=100` 发送目标关节位置。

Humanx 真机控制：

| 按键 | 命令 | 作用 |
| --- | --- | --- |
| `Start` | 启动零力矩阶段 | 结束零力矩等待，进入准备姿态 |
| `L1` | 启动确认 | 从准备姿态进入策略运行 |
| `L2` | `[SHUTDOWN]` | 关闭环境输出 |
| `Y` | `[MOTION_RESET]` | 重置动作与 pipeline |
| `X` | `[MOTION_FADE_IN]` | 当前 Humanx 主 pipeline 中未显式消费，属于保留映射 |
| `B` | `[MOTION_FADE_OUT]` | 当前 Humanx 主 pipeline 中未显式消费，属于保留映射 |

## 6. 运行 Humanx Loop 真机策略

执行：

```bash
python scripts/run_pipeline.py -c g1_humanx_loop_real
```

启动流程与 `g1_humanx_real` 相同：

1. 按 `Start` 结束零力矩等待。
2. 等待 2 秒准备姿态插值。
3. 按 `L1` 正式启动策略。
4. 运行中按 `L2` 退出。

Humanx Loop 真机控制：

| 按键 | 命令 | 作用 |
| --- | --- | --- |
| `Start` | 启动零力矩阶段 | 结束零力矩等待，进入准备姿态 |
| `L1` | 启动确认 | 从准备姿态进入策略运行 |
| `L2` | `[SHUTDOWN]` | 关闭环境输出 |
| `Y` | `[MOTION_RESET]` | 完整重置动作与 pipeline |
| `A` | `[MOTION_REPLAY]` | soft reset motion phase，适合 Loop 策略重放动作 |

## 7. Humanx 与 Humanx Loop 的主要差异

| 维度 | Humanx | Humanx Loop |
| --- | --- | --- |
| Policy class | `HumanxPolicy` | `HumanxLoopPolicy` |
| Config class | `G1HumanxPolicyCfg` | `G1HumanxLoopPolicyCfg` |
| 默认模型 | `model_689000.onnx` | `loop.onnx` |
| 默认 motion | `BMaster_fake_action_and_shot_hoi_wsf.pkl` | `BMaster_fake_action_and_shot_hoi_loop.pkl` |
| 观测 | 基础状态、历史、PD error | Humanx 观测 + `ref_motion_phase` |
| 重放方式 | 主要使用 `[MOTION_RESET]` | 支持 `[MOTION_REPLAY]` soft reset |
| sim2sim 按键 | `i` / `o` / `r` | `i` / `o` / `r` / `t` |
| 真机额外按键 | `Y` reset | `Y` reset，`A` replay |

两者都使用 29-DoF G1 配置，策略频率为 100 Hz，并通过 motion 第一帧生成准备阶段的初始姿态。

## 8. 常见问题

### 8.1 找不到模型或 motion 文件

报错类似：

```text
FileNotFoundError: Model file not found at ...
```

检查：

```bash
ls assets/models/g1/humanx
ls assets/motions/g1/humanx
```

如果仓库路径不是 `/home/zzx/Documents/RoboJuDo`，修改 [g1_cfg.py](../robojudo/config/g1/g1_cfg.py) 中 Humanx 配置的绝对路径。

### 8.2 `UnitreeCppEnv` 导入失败

重新确认：

```bash
python submodule_install.py unitree_cpp
python -c "from robojudo.environment import UnitreeCppEnv"
```

若仍失败，检查 Unitree 官方 `unitree_sdk2` 是否按默认路径或可被编译系统找到。

### 8.3 真机无响应

优先检查：

1. `net_if` 是否是机器人通信网卡。
2. 程序是否仍停在 `Start` 或 `L1` 等待阶段。
3. Unitree 低层状态和遥控器数据是否正常到达。
4. 是否按下了 `L2` 或其他 shutdown 触发。

### 8.4 运行中 frame drop

Humanx / Humanx Loop 真机频率为 100 Hz，G1 板载电脑建议使用 `UnitreeCppEnv`。如果持续掉帧：

1. 确认没有在真机上开启不必要的可视化或重日志。
2. 确认使用的是 `UnitreeCppEnv` 而不是 `UnitreeEnv`。
3. 降低其他进程负载，必要时只保留部署进程。

