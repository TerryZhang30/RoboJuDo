# Pickball BC 真机部署代码逻辑分析

本文分析当前 `pickball_bc` 策略从仿真视觉反馈迁移到 Unitree G1 真机部署时，哪些代码可以直接复用，哪些只服务于仿真，以及真机还需要补齐的接口和安全逻辑。

## 结论概览

当前代码已经具备真机部署的核心骨架：

- `RlPipeline` 可以同时驱动仿真环境和真机环境。
- `UnitreeCppEnv` / `UnitreeEnv` 已经能读取真机关节、IMU、里程计，并发送关节位置目标。
- `PickballBCPolicy` 已经是独立策略，直接继承 `Policy`，可以加载 ONNX、构造视觉+本体观测、输出 29 DoF 动作。
- `PickballBCPolicy` 内已有动作跳变保护，支持 `raise`、`hold`、`zero`、`freeze` 模式。
- `G1PickballBCPolicyCfg` 已经把 G1 29DoF、动作缩放、图像输入 shape、ONNX 路径、motion 初始姿态配置集中起来。

但还不能直接真机闭环运行视觉策略，主要缺口是：

- 真机环境当前不会提供 `depth_image` / `image` / `camera_image`。
- 仿真中的 `DepthCameraCfg` 和 `MujocoEnv._render_depth_image()` 只生成 MuJoCo 合成深度图，不会访问真实相机。
- 当前图像缺失时 policy 会使用常量 fallback image，这在真机上不安全，应该改为“图像缺失或过期则拒绝动作/停机/保持安全姿态”。
- 真机发送关节目标前的关节限位裁剪目前在 `UnitreeEnv` / `UnitreeCppEnv` 中是注释状态，需要真机部署前启用。
- 当前已经加入 `g1_pickball_bc_real` / `pickball_bc_real` 配置，但仍需要在真机上确认 RealSense 驱动、相机权限和相机安装位姿。

## 当前仿真数据流

仿真入口通常是：

```bash
python scripts/run_pipeline.py -c pickball_bc
```

配置链路：

- `robojudo/config/g1/g1_cfg.py`
  - `g1_pickball_bc`
  - `pickball_bc`
- `robojudo/config/g1/policy/g1_pickball_bc_policy_cfg.py`
  - `G1PickballBCPolicyCfg`
  - `G1PickballBCDoF`
- `robojudo/policy/pickball_bc_policy.py`
  - `PickballBCPolicy`
- `robojudo/environment/mujoco_env.py`
  - `MujocoEnv`

每一帧执行顺序在 `robojudo/pipeline/rl_pipeline.py` 的 `RlPipeline.step()`：

1. `env.update()`
2. `env_data = env.get_data()`
3. `ctrl_data = ctrl_manager.get_ctrl_data(env_data)`
4. `obs, extras = policy.get_observation(env_data, ctrl_data)`
5. `pd_target = policy.get_pd_target(obs)`
6. 仿真中处理球：`_apply_ball_info(extras)`
7. `env.step(pd_target, extras.get("hand_pose", None))`
8. `post_step_callback(...)`

仿真深度图来源在 `MujocoEnv.get_data()`：

```python
env_data = super().get_data()
if self._depth_camera_cfg.enabled:
    depth_image = self._render_depth_image()
    env_data["depth_image"] = depth_image
    env_data["image"] = depth_image
return env_data
```

这意味着 `PickballBCPolicy` 不关心图像来自仿真还是真机，只要 `env_data` 中有 `depth_image`、`image` 或 `camera_image` 即可。

## PickballBCPolicy 中可复用的部分

文件：`robojudo/policy/pickball_bc_policy.py`

### ONNX 推理

可直接用于真机：

- `_resolve_onnx_providers()`
- `_preload_onnx_gpu_dependencies()`
- `_check_onnx_provider_fallback()`
- `get_action()`

这些逻辑支持：

- CPU / CUDA / TensorRT provider 选择。
- `onnx_device="auto"` 自动优先 CUDA。
- 输入名检查：当前只支持 `actor_obs` 和 `image`。
- 输出动作 clip。

真机注意点：

- 如果真机计算平台是 Jetson，建议确认 `onnxruntime-gpu`、CUDA、TensorRT provider 是否实际激活。
- 如果 provider fallback 到 CPU，100Hz 策略频率可能掉帧。
- `scripts/run_pipeline.py` 会在真机掉帧超过一定程度时触发退出逻辑。

### 本体观测构造

`get_observation()` 中以下部分可复用：

- `dof_pos`
- `dof_vel`
- `base_ang_vel`
- `projected_gravity`
- `phase`
- `pd_error`
- `actions`
- `history_actor`

这些输入都可以由真机环境提供：

- `UnitreeCppEnv.update()` 从 `unitree_cpp` 读取 motor q/dq、IMU quaternion、gyroscope。
- `UnitreeEnv.update()` 从 Unitree DDS low state 读取 motor q/dq、IMU quaternion、gyroscope。
- `Environment.get_data()` 统一打包成 `env_data`。

真机注意点：

- `base_quat` 在 G1 中来自 IMU，并受 `born_place_align` 影响。
- `projected_gravity` 对 IMU 坐标和四元数顺序很敏感，当前代码统一使用 xyzw。
- `phase` 当前使用 `phase_dt = 1/60`，不是 policy 控制频率 `dt`。这很可能是为了匹配训练导出时的时间尺度，真机上不要轻易改。

### 图像预处理接口

可复用：

- `_image_from_env_data()`
- `_coerce_image()`
- `_resize_nearest()`

当前 policy 接收以下 key：

```python
image_obs_keys = ["depth_image", "image", "camera_image"]
```

要求图像最终变成：

```python
image_shape = [224, 224, 1]
dtype = np.float32
```

支持输入格式：

- `H x W`
- `H x W x C`
- `1 x H x W x C`
- 部分 `C x H x W`

真机部署时，真实相机只需要把深度图塞进 `env_data["depth_image"]`，policy 就能继续工作。

### 动作后处理和安全门

可复用：

- `_compute_action_scales()`
- `_process_action_output()`
- `_enforce_joint_delta_limit()`

当前动作语义：

1. ONNX 输出 `raw_actions`
2. `raw_actions` 被 clip 到 `[-action_clip, action_clip]`
3. 若 `apply_action_scales=True`，返回 `raw_actions * action_scales`
4. `PolicyWrapper.get_pd_target()` 再加 `policy.default_pos`
5. 经过 DoF adapter 映射到真机环境关节顺序
6. `UnitreeCppEnv.step()` 或 `UnitreeEnv.step()` 发送关节位置目标

所以 `PickballBCPolicy.get_action()` 返回的是“相对默认姿态的动作偏移”，不是最终电机角度。最终 PD target 是：

```python
pd_target = policy_action + default_pos
```

跳变保护比较的是缩放后的 action output，相当于比较相邻帧目标关节偏移变化。当前模式含义：

- `raise`：超过阈值直接抛异常，程序停止。
- `hold`：当前帧保持上一帧安全输出；下一帧如果恢复正常，会继续使用模型输出。
- `zero`：当前帧输出 0，相当于回到默认姿态附近。
- `freeze`：触发后一直保持上一帧安全输出，直到 reset。

真机建议：

- 初次实机闭环建议使用 `freeze` 或 `raise`，不建议一开始使用 `hold`。
- `max_joint_delta_rad` 应该按每帧最大目标变化量设定。100Hz 下 0.6 rad/frame 非常大，真机建议从更小值开始。
- 需要额外增加速度限幅/斜坡滤波，单纯跳变检测不能替代动作平滑。

## 仿真专用但真机不可直接使用的部分

### MujocoEnv 深度相机

文件：

- `robojudo/environment/env_cfgs.py`
  - `DepthCameraCfg`
- `robojudo/environment/mujoco_env.py`
  - `_add_depth_camera_to_spec()`
  - `_init_depth_renderer()`
  - `_render_depth_image()`
  - `get_data()`

这些逻辑只对 MuJoCo 有效：

- 在 MJCF 里动态插入一个 camera。
- 用 `mujoco.Renderer` 渲染 synthetic depth。
- 可选录制 WebM。

真机上没有 `mujoco.Renderer`，也没有 MJCF camera，所以这部分只能作为真实相机的参数参考：

- 相机名：`d435_depth`
- 分辨率：`224 x 224`
- 深度范围：`depth_min=0.01`，`depth_max=3.0`
- 模拟安装位姿：挂在 `torso_link` 下，`pos=[0.0576235, 0.01753, 0.42987]`
- 模拟相机 `quat`

如果真机相机安装位姿与仿真不一致，策略看到的深度分布会偏移，效果可能明显变差。

### 仿真球逻辑

文件：

- `MujocoEnv._add_ball_to_spec()`
- `MujocoEnv.set_ball_pos()`
- `MujocoEnv.release_ball()`
- `RlPipeline._apply_ball_info()`

这些只对仿真球有效。真机里真实球不会被 `env.set_ball_pos()` 控制。

真机中仍可保留 `track_motion_ball=False`，让 policy 只依赖视觉和本体状态。

## 真机部署可复用的部分

### RlPipeline

文件：`robojudo/pipeline/rl_pipeline.py`

可复用：

- policy/env/controller 统一装配。
- DoF adapter：策略关节顺序和环境关节顺序不一致时自动映射。
- `prepare()`：从当前姿态插值到策略初始姿态。
- `wait_for_zero_torque_start()`：真机启动前零力矩等待。
- `wait_for_start_confirmation()`：真机开始前确认。
- `safety_check()`：检测机器人倾倒。

真机重要行为在 `scripts/run_pipeline.py`：

```python
if not cfg.env.is_sim:
    pipeline.wait_for_zero_torque_start()
    pipeline.prepare()
    pipeline.wait_for_start_confirmation()
```

只要使用 `G1RealEnvCfg`，`is_sim=False`，就会走真机启动流程。

### UnitreeCppEnv / UnitreeEnv

文件：

- `robojudo/environment/unitree_cpp_env.py`
- `robojudo/environment/unitree_env.py`

可复用：

- 读取真机 motor state。
- 读取 IMU quaternion、angular velocity。
- 读取 Unitree odometry 或 ZED odometry。
- 发送关节位置命令。
- 设置 PD gains。
- 关机 / damping mode。

区别：

- `UnitreeCppEnv` 通过 `unitree_cpp.UnitreeController`。
- `UnitreeEnv` 通过 `unitree_sdk2py` DDS。

当前 G1 真机默认配置：

```python
G1RealEnvCfg.env_type = "UnitreeCppEnv"
G1RealEnvCfg.odometry_type = "UNITREE"
```

### UnitreeCtrl

文件：

- `robojudo/controller/unitree_ctrl.py`
- `robojudo/controller/ctrl_cfgs.py`

可复用：

- 从 Unitree 遥控器读取按钮事件。
- 触发 `[SHUTDOWN]`、`[MOTION_RESET]` 等命令。

真机建议配置至少包含：

- 紧急 shutdown。
- motion reset。
- start confirm。

## 当前真机视觉缺口

最关键缺口：`UnitreeCppEnv.get_data()` 没有覆盖 `Environment.get_data()`，因此不会往 `env_data` 添加图像。

当前真机 `env_data` 只有：

```python
dof_pos
dof_vel
base_quat
base_ang_vel
base_lin_acc
base_pos
base_lin_vel
torso_pos
torso_quat
torso_ang_vel
fk_info
```

而 `PickballBCPolicy` 需要额外至少一个：

```python
depth_image
image
camera_image
```

如果没有图像，当前 policy 会走 `_fallback_image()`，即全常量图像。这个行为对仿真调试方便，但真机危险，因为策略会在没有真实视觉反馈时继续输出动作。

真机部署前建议改成：

- 增加配置 `require_image: bool = True`。
- 图像缺失时抛异常或进入安全动作。
- 图像过期时抛异常或进入安全动作。
- 打印/记录图像时间戳、shape、min/max、NaN 数量。

## 已接入的真机视觉方案

现在已经新增一个真实深度相机模块，没有把 RealSense 代码直接塞进 policy。policy 仍然只消费 `env_data["depth_image"]`。

当前结构：

```text
robojudo/tools/real_depth_camera.py
robojudo/environment/env_cfgs.py
robojudo/environment/unitree_cpp_env.py
robojudo/environment/unitree_env.py
robojudo/config/g1/g1_cfg.py
scripts/test_realsense_depth.py
```

### 1. RealDepthCameraCfg

位于 `env_cfgs.py`：

```python
class RealDepthCameraCfg(Config):
    enabled: bool = False
    camera_type: Literal["realsense"] = "realsense"
    width: int = 224
    height: int = 224
    source_width: int = 640
    source_height: int = 480
    fps: int = 30
    depth_min: float = 0.01
    depth_max: float = 3.0
    timeout_ms: int = 1000
    max_frame_age_s: float = 0.1
    serial_no: str | None = None
    use_background_thread: bool = True
```

### 2. RealDepthCamera wrapper

位于 `robojudo/tools/real_depth_camera.py`。

职责：

- 初始化真实相机。
- 获取最新 depth frame。
- 转成 `np.float32`。
- 输出 shape 为 `(224, 224, 1)`。
- 对 NaN/Inf/0 做处理。
- clip 到训练时一致的深度范围。
- 记录 timestamp。

接口：

```python
class RealDepthCamera:
    def get_depth_image(self) -> tuple[np.ndarray, float]:
        ...
```

### 3. 在 UnitreeCppEnv / UnitreeEnv 中挂相机

当前 `UnitreeCppEnv` 和 `UnitreeEnv` 都会在 `real_depth_camera.enabled=True` 时创建相机：

```python
if cfg_env.real_depth_camera.enabled:
    self.depth_camera = RealDepthCamera(cfg_env.real_depth_camera)
else:
    self.depth_camera = None
```

覆盖或扩展 `get_data()`：

```python
def get_data(self):
    env_data = super().get_data()
    if self.depth_camera is not None:
        depth_image, ts = self.depth_camera.get_depth_image()
        env_data["depth_image"] = depth_image
        env_data["image"] = depth_image
        env_data["camera_timestamp"] = ts
    return env_data
```

这样 `PickballBCPolicy` 不需要知道真实相机存在，只要继续消费 `depth_image`。

### 4. pickball_bc_real 配置

当前已经新增：

```python
@cfg_registry.register
class g1_pickball_bc_real(g1_pickball_bc):
    env: G1RealEnvCfg = G1RealEnvCfg(
        env_type="UnitreeCppEnv",
        unitree=G1UnitreeCfg(net_if="eth0", control_dt=0.01),
        real_depth_camera=RealDepthCameraCfg(enabled=True, ...)
    )

    ctrl: list[UnitreeCtrlCfg] = [
        UnitreeCtrlCfg(
            triggers={
                "L2": "[SHUTDOWN]",
                "Y": "[MOTION_RESET]",
            }
        )
    ]

    do_safety_check: bool = True
    wait_for_zero_torque_start: bool = True
    prepare_duration_s: float = 2.0
    prepare_reset_before_done: bool = False
    wait_for_start_confirmation: bool = True
    start_confirm_button: str = "L1"
    shutdown_button: str = "L2"
    start_hold_stiffness_scale: float = 3.0
    start_hold_damping_scale: float = 3.0
```

运行方式：

```bash
python scripts/run_pipeline.py -c g1_pickball_bc_real
```

相机单独测试：

```bash
python scripts/test_realsense_depth.py
```

## 真机部署前必须补强的安全点

### 1. 图像缺失不能 fallback

当前 `_image_from_env_data()` 缺图时会返回常量图像。这在真机上建议禁用。

建议：

- `require_image=True`
- 缺图直接 `RuntimeError`
- 或返回 freeze/zero 安全动作

### 2. 图像时延检查

真机相机通常 30Hz，policy 100Hz。允许重复使用上一帧图像，但必须限制最大 age。

建议：

- `env_data["camera_timestamp"]`
- `max_frame_age_s <= 0.1`
- 超时触发安全动作

### 3. 启用关节限位裁剪

`UnitreeCppEnv.step()` 和 `UnitreeEnv.step()` 中都有 position limit 裁剪代码，但当前被注释。

真机部署前建议启用：

```python
limits = self.position_limits
pd_target_clipped = np.clip(pd_target, limits[:, 0], limits[:, 1])
positions = pd_target_clipped
```

同时如果发生裁剪，应该 warning 或触发安全策略。

### 4. 动作速度/加速度限制

当前 `PickballBCPolicy` 有“相邻帧跳变检测”，但没有真正做速率限制。

建议在 policy 输出或 pipeline 发送前增加：

```python
pd_target = last_pd_target + np.clip(
    pd_target - last_pd_target,
    -max_delta_per_step,
    max_delta_per_step,
)
```

这样即使模型输出突变，也不会突然给电机大目标。

### 5. 真机初始姿态确认

`prepare()` 会插值到 `policy.get_init_dof_pos()`，该值来自 motion pkl 第一帧。

真机需要确认：

- 第一帧姿态是否物理可达。
- 双脚是否稳定。
- 球是否在真实环境中处于相机可见范围。
- `prepare_duration_s` 是否足够长。

### 6. PD gains 复核

`G1PickballBCDoF` 中 stiffness/damping 来自策略配置，不一定适合真机。

真机建议：

- 先低增益 dry-run 或吊挂测试。
- 确认腰、肩、肘、腕关节不会抖动。
- 对腕部小电机设置更严格目标变化限制。

## 推荐落地顺序

1. 保持 `pickball_bc` 仿真跑通，确认 ONNX 输入输出、深度图 shape、动作安全模式正常。
2. 新增真实深度相机 wrapper，只做采集和预处理，不接机器人。
3. 单独运行相机测试脚本，打印：
   - shape
   - dtype
   - min/max
   - NaN/Inf 数量
   - FPS
   - frame age
4. 在 `UnitreeCppEnv.get_data()` 中接入 `depth_image`，先 `act=False` 或零力矩模式跑 observation。
5. 使用 `PickballBCPolicy.get_observation()` 做干跑，确认 ONNX 不推动作也能稳定构造 obs。
6. 开启 ONNX 推理但不发送电机，记录 `raw_actions`、`pd_target`、最大关节跳变。
7. 开启真机 `prepare()`，确认初始姿态。
8. 极低风险条件下闭环：
   - `do_safety_check=True`
   - `joint_delta_safety_mode="freeze"` 或 `"raise"`
   - 开启关节限位裁剪
   - 开启图像 freshness 检查
   - 遥控器 shutdown 可用

## 当前代码可直接复用清单

| 模块 | 文件 | 真机可用性 | 说明 |
| --- | --- | --- | --- |
| Pickball BC policy | `robojudo/policy/pickball_bc_policy.py` | 高 | ONNX、obs 构造、图像预处理、动作安全门可复用 |
| Pickball BC cfg | `robojudo/config/g1/policy/g1_pickball_bc_policy_cfg.py` | 中高 | DoF、缩放、ONNX 路径可复用；真机 gains/阈值需调 |
| Pipeline | `robojudo/pipeline/rl_pipeline.py` | 高 | 真机启动、prepare、policy/env 连接可复用 |
| UnitreeCppEnv | `robojudo/environment/unitree_cpp_env.py` | 高 | 当前 G1Real 默认真机环境 |
| UnitreeEnv | `robojudo/environment/unitree_env.py` | 中高 | 另一套 SDK2Py 真机环境 |
| UnitreeCtrl | `robojudo/controller/unitree_ctrl.py` | 高 | 遥控器命令输入 |
| DepthCameraCfg/Mujoco depth | `robojudo/environment/mujoco_env.py` | 仿真专用 | 只能作为真实相机参数参考 |
| Ball helpers | `MujocoEnv` + `RlPipeline._apply_ball_info()` | 仿真专用 | 真机不会控制真实球 |

## 需要新增或修改清单

真机视觉部署最小改动：

- 在真机上安装并验证 `pyrealsense2`。
- 在 G1 上确认相机没有被 Unitree 自带视觉进程占用。
- 用 `scripts/test_realsense_depth.py` 单独确认深度图 shape、单位和 FPS。
- 在真机 env 的 `step()` 中启用关节限位裁剪。
- 增加发送前 PD target 速率限制。

完成这些后，当前 `PickballBCPolicy` 的大部分逻辑都可以继续复用，真机部署的核心工作会集中在“真实深度图接入”和“安全策略强化”两块。
