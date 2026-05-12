# HumanX ONNX 导出机制分析

## 1. 总览结论

HumanX 当前的 ONNX 导出不是独立脚本，而是挂在评估入口 `humanoidverse/eval_agent.py` 里。运行评估脚本并传入 checkpoint 后，代码会先加载训练配置和权重，再根据 `export.onnx` 开关把 policy actor 导出为 `.onnx`，最后进入仿真评估循环。

当前默认导出的 ONNX 只包含 **actor 推理网络**：

- 输入名：`actor_obs`
- 输出名：`action`
- opset：13
- 设备：导出前把 actor 深拷贝到 CPU
- 不包含 critic、动作噪声分布、仿真环境、观测拼接、动作裁剪、动作缩放、PD 控制或 torque clip

因此 ONNX 的使用方必须自己构造与训练时完全一致的 `actor_obs`，并按部署端的控制逻辑处理输出 action。

## 2. 入口和调用链

### 2.1 默认开关

`humanoidverse/config/base_eval.yaml` 中默认打开：

```yaml
export:
  policy: true
  onnx: true
```

也就是说，只要通过 `eval_agent.py` 评估 checkpoint，默认会同时尝试导出 JIT 和 ONNX。依赖列表在 `setup.py` 中包含 `onnx` 和 `onnxruntime`，但当前运行环境中 `python -c "import onnx"` 失败，实际导出前需要确认环境已安装这些包。

### 2.2 典型命令

README 中的评估入口是：

```bash
python humanoidverse/eval_agent.py \
  +checkpoint=logs/HumanX/<run_dir>/model_<iter>.pt
```

只导出 ONNX、跳过 JIT 时可以用：

```bash
python humanoidverse/eval_agent.py \
  +checkpoint=logs/HumanX/<run_dir>/model_<iter>.pt \
  export.policy=false \
  export.onnx=true \
  headless=true \
  num_envs=1
```

注意：`eval_agent.py` 目前没有 `export_only` 模式。ONNX 导出完成后会继续执行 `algo.evaluate_policy()`；对 PPO/DAgger 这类评估实现来说，评估循环通常不会自动结束。

### 2.3 `eval_agent.py` 中的流程

核心流程如下：

1. 使用 Hydra 加载 `base_eval.yaml`。
2. 如果传入 `checkpoint`，先找 checkpoint 旁边或上一级目录的 `config.yaml`。
3. 合并训练配置、`eval_overrides` 和命令行覆盖项。
4. `pre_process_config(config)` 计算观测维度。
5. 实例化环境和算法。
6. `algo.setup()` 构建网络。
7. `algo.load(config.checkpoint)` 加载 checkpoint。
8. 构造导出目录：`<checkpoint_dir>/exported/`。
9. 如果 `export.onnx=true` 且算法满足 `hasattr(algo, "actor")` 和 `hasattr(algo, "get_example_obs")`，调用：

```python
example_obs_dict = algo.get_example_obs()
export_policy_as_onnx(
    algo.inference_model,
    exported_policy_path,
    exported_onnx_name,
    example_obs_dict,
)
```

导出的文件名由 checkpoint 名称替换后缀得到，例如：

```text
logs/HumanX/<run_dir>/model_32000.pt
logs/HumanX/<run_dir>/exported/model_32000.onnx
```

仓库中已有大量历史导出产物，路径形如 `logs/HumanX/.../exported/model_*.onnx`。

## 3. 实际导出实现

ONNX 的实际逻辑在 `humanoidverse/utils/inference_helpers.py`：

```python
actor = copy.deepcopy(inference_model["actor"]).to("cpu")

class PPOWrapper(nn.Module):
    def __init__(self, actor):
        super(PPOWrapper, self).__init__()
        self.actor = actor

    def forward(self, actor_obs):
        return self.actor.act_inference(actor_obs)

wrapper = PPOWrapper(actor)
example_input_list = example_obs_dict["actor_obs"]

torch.onnx.export(
    wrapper,
    example_input_list,
    path,
    verbose=True,
    input_names=["actor_obs"],
    output_names=["action"],
    opset_version=13,
)
```

几个关键点：

- `example_obs_dict` 来自 `algo.get_example_obs()`，它会调用 `env.reset_all()`，然后把所有 obs tensor 移到 CPU。
- `torch.onnx.export` 没有配置 `dynamic_axes`，所以 batch 维默认是静态的。由于评估覆盖通常把 `num_envs` 设为 1，导出的模型大概率固定为 batch=1。
- wrapper 调的是 `actor.act_inference(actor_obs)`，所以导出的是确定性的 action mean，而不是训练时的采样动作。
- `verbose=True` 会打印较长的 ONNX 图日志。

仓库里还有一个未启用的 `export_policy_and_estimator_as_onnx()`，用于同时导出 actor 和左右手 force estimator。它需要 `inference_model` 中存在 `left_hand_force_estimator`、`right_hand_force_estimator`，并且 example obs 里存在 `long_history_for_estimator`。`eval_agent.py` 中这条调用被注释掉了。

## 4. 输入输出契约

### 4.1 输入 `actor_obs`

ONNX 输入不是原始机器人状态，而是已经由环境拼好的 policy observation。形状通常是：

```text
[num_envs, actor_obs_dim]
```

`actor_obs_dim` 的来源是 `pre_process_config()`：

- 读取 `env.config.obs.obs_dict.actor_obs`
- 读取 `env.config.obs.obs_dims`
- 展开 `obs_auxiliary` 中的 history 类观测
- 汇总到 `config.robot.algo_obs_dim_dict["actor_obs"]`

需要特别注意观测顺序：环境最终拼接时使用的是 `sorted(obs_config)`，不是 YAML 中肉眼看到的原始顺序。history 观测内部也使用 `sorted(history_config.keys())`。所以部署端如果自己拼 `actor_obs`，必须按源码排序规则复现，否则维度正确也会语义错位。

观测还会经过以下处理：

- `_get_obs_xxx()` 采集原始量
- 按 `obs_scales` 缩放
- 训练/评估配置中如果开启噪声，会加 observation noise
- 最终按 `normalization.clip_observations` 裁剪
- history 观测依赖 `HistoryHandler`，episode reset 时要清空历史

### 4.2 输出 `action`

ONNX 输出是 actor 的 action mean，形状通常是：

```text
[num_envs, robot.actions_dim]
```

以 G1 29DoF 配置为例，`robot.actions_dim=29`，所以输出是 29 维。

这个输出仍然是 **环境期望的 policy action**，不是关节目标角、不是 torque。环境侧后处理包括：

- `_pre_physics_step()` 先按 `robot.control.action_clip_value` 裁剪 action。
- 如果启用控制延迟，会进入 action queue。
- `_compute_torques()` 中计算 `actions_scaled = actions * self.action_scales`。
- HOI 环境的 `action_scales` 不是简单的 scalar，而是逐关节：

```python
control.action_scale * dof_effort_limit / p_gain
```

- P 控制下 torque 由 `actions_scaled + default_dof_pos - dof_pos` 进入 PD 控制。
- 最后可能按 torque limit 裁剪。

因此部署时不要无脑再乘一次 `action_scale`。如果部署端控制器需要的是关节目标角，应明确转换关系；如果部署端直接复刻 HumanX 环境，则 ONNX 输出应作为 policy action 输入后处理链。

## 5. 算法兼容性

### 5.1 当前通用路径基本支持

以下算法类都暴露了 `inference_model`，并且有 `get_example_obs()`，适合当前通用 ONNX 导出路径：

- `humanoidverse.agents.ppo.ppo.PPO`
- `humanoidverse.agents.dagger.DAgger`
- `humanoidverse.agents.dagger.DAggerBC`
- `humanoidverse.agents.dagger.DAggerKL`
- `humanoidverse.agents.dagger.Specialist2Generalist`
- `humanoidverse.agents.itp_ppo.itp_ppo.ITP_PPO`

其中 ITP 的 `inference_model["actor"]` 是 `policy_net`，`PolicyNetwork` 提供了 `act_inference()`，所以能适配当前 wrapper。

### 5.2 需要小心或不支持

- `policy_switch_ppo.py` 的 `inference_model` 返回 `frozen_actor`、`standing_actor`、`standing_critic`，没有通用导出 wrapper 需要的 `"actor"` 键，也没有被 `eval_agent.py` 的 `can_export` 逻辑覆盖。若要导出 policy switch，需要自定义 wrapper，把 frozen actor、standing actor、状态机/混合逻辑一起定义清楚，或分别导出子网络。
- `ppo_locomanip.py` 的 `inference_model` 直接返回 actor list，而 `export_policy_as_onnx()` 期待 `inference_model["actor"]`。它会通过 `can_export` 检查，但真正导出时容易失败。
- `bc/` 目前没有 ONNX 导出入口。它有自己的 checkpoint 加载和 `act()` 推理逻辑，但没有接到 `eval_agent.py` 的 ONNX 导出链路。
- `bc_vision/vision_daggerbc` 已新增根目录脚本 `onnx_exporter.py`，用于导出这次指定的 `bc_runs/vision_daggerbc/20260507_134418/last.pt`。它不是 `eval_agent.py` 的一部分，而是复用 HumanX 的 ONNX 导出风格单独实现，见下文“本次新增的 vision_daggerbc 导出脚本”。

## 6. 导出时最需要注意的点

### 6.1 模型 eval 模式

当前 `export_policy_as_onnx()` 只做了 `copy.deepcopy(...).to("cpu")`，没有显式 `actor.eval()`。`eval_agent.py` 里真正调用 `_get_inference_policy()` 设置 eval mode 是在导出之后的评估阶段。

现有 MLP actor 主要由 Linear/ELU 组成，影响不大；如果以后加入 Dropout、BatchNorm、RNN 等模块，导出前应显式：

```python
actor.eval()
wrapper.eval()
```

### 6.2 静态 batch 维

当前没有 `dynamic_axes`。如果导出时 `num_envs=1`，部署时传 `[N, obs_dim]` 可能不被 runtime 接受。需要动态 batch 时建议改成：

```python
dynamic_axes={
    "actor_obs": {0: "batch"},
    "action": {0: "batch"},
}
```

### 6.3 `add_ref_action=true` 的模型

`PPOActor.act_inference()` 支持 `actor_obs` 之外再传 `ref_obs/ref_dof_pos`。评估代码已经兼容这种情况，但 ONNX wrapper 只接收一个 `actor_obs` 输入。

如果 checkpoint 对应 `algo.config.add_ref_action=true`，当前导出会因为缺少 `ref_obs` 而失败，或者导出的行为不完整。此时应改 wrapper：

```python
def forward(self, actor_obs, ref_obs):
    return self.actor.act_inference(actor_obs, ref_obs)
```

并把 `input_names` 改成 `["actor_obs", "ref_obs"]`。

### 6.4 观测顺序必须完全一致

这是最容易出现“ONNX 能跑但机器人行为很差”的问题。HumanX 的拼接顺序由代码排序决定：

- 普通 obs：`sorted(obs_config)`
- history obs：`sorted(history_config.keys())`

不要根据 YAML 中写出来的顺序手工拼接。导出或部署前建议把 `algo.get_example_obs()` 打印出来的 obs key、shape 和最终部署端的拼接结果逐项比对。

### 6.5 history 和 last action

很多 HOI/motion tracking 配置的 `actor_obs` 包含 `actions`、`pd_error`、`history_actor` 等时序量。ONNX 本身是无状态 MLP，不会替你维护 history buffer。部署端必须维护：

- 上一帧 action
- history buffer
- episode reset 时的 history 清零
- 与仿真/真机控制频率一致的更新节奏

### 6.6 导出时会实例化仿真环境

因为 example input 来自 `env.reset_all()`，导出并不是纯粹读取 checkpoint。它会创建 IsaacGym/IsaacSim 环境并 reset 一步。机器上缺少 simulator、资产路径、motion 文件或 GPU 环境时，导出也会失败。

### 6.7 输出 action 的尺度

ONNX 输出是 policy action。若用于真机或别的仿真器，需要明确部署端期望：

- 如果部署端复刻 HumanX 控制链：输入 ONNX 输出，再做 action clip、delay、action_scales、PD。
- 如果部署端直接期望目标关节角：需要转换为 `default_dof_pos + action * action_scales`。
- 如果部署端直接期望 torque：还需要当前 dof_pos/dof_vel 和 PD 参数，不能只靠 ONNX 输出。

### 6.8 导出后应做数值对齐验证

建议导出后至少做三类检查：

1. `onnx.checker.check_model()` 验证模型格式。
2. 用同一份 `actor_obs` 对比 PyTorch actor 和 ONNXRuntime 输出，误差应在浮点容差内。
3. 用一段真实 rollout 的观测序列回放，确认 history/action 更新和部署端一致。

示例验证代码骨架：

```python
import numpy as np
import onnx
import onnxruntime as ort

onnx_path = "logs/HumanX/<run_dir>/exported/model_<iter>.onnx"
onnx.checker.check_model(onnx.load(onnx_path))

session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
obs = example_obs_dict["actor_obs"].cpu().numpy().astype(np.float32)
onnx_action = session.run(["action"], {"actor_obs": obs})[0]
```

再将 `onnx_action` 与 `actor.act_inference(example_obs_dict["actor_obs"])` 的 PyTorch 结果比较。

## 7. 建议改进

如果后续要把 ONNX 导出作为稳定工具使用，建议补几个小改动：

- 在 `eval_agent.py` 中加 `export_only`，导出后直接退出。
- 在 `export_policy_as_onnx()` 中显式 `actor.eval()` 和 `wrapper.eval()`。
- 增加 `dynamic_axes`，避免 batch=1 固化。
- 为 `add_ref_action`、force estimator、policy switch 分别写专用 wrapper。
- 导出后自动跑一次 ONNXRuntime/PyTorch 数值对齐。
- 把输入 shape、obs 拼接顺序、action_dim、checkpoint、git commit 写入一个 sidecar JSON，方便部署端校验。

## 8. 快速排查表

| 现象 | 常见原因 | 建议检查 |
| --- | --- | --- |
| 导出时报缺少 `onnx` | 当前 Python 环境没装 onnx | `python -c "import onnx"` |
| ONNX 输入 shape 不接受 batch N | 没有 dynamic batch | 增加 `dynamic_axes` 或按导出 batch 推理 |
| 导出时报 `act_inference` 缺少 ref input | `add_ref_action=true` | wrapper 增加 `ref_obs` 输入 |
| ONNX 能跑但动作异常 | obs 拼接顺序或尺度不一致 | 按 `sorted(obs_config)` 复现，检查 `obs_scales`、clip、history |
| action 过大或真机目标角异常 | 重复乘了 action scale | 明确 ONNX 输出是 policy action，不是目标角/torque |
| policy switch checkpoint 导不出 | 通用 wrapper 只支持 `"actor"` | 写 policy switch 专用导出逻辑 |
| 只想导出却进入仿真循环 | `eval_agent.py` 没有 `export_only` | 等导出日志出现后停止，或新增 `export_only` 开关 |

## 9. 本次新增的 vision_daggerbc 导出脚本

本次新增了项目根目录脚本：

```text
onnx_exporter.py
```

默认输入 checkpoint：

```text
bc_runs/vision_daggerbc/20260507_134418/last.pt
```

默认输出 ONNX：

```text
bc_runs/vision_daggerbc/20260507_134418/exported/last.onnx
```

运行命令：

```bash
/home/ps/miniforge3/envs/hxmimic/bin/python onnx_exporter.py --quiet
```

默认 `python` 是 `/home/ps/miniforge3/bin/python`，当前环境没有 `torch`；实际导出使用的是已有的 `hxmimic` 环境。该环境包含 `torch`、`onnx`、`onnxruntime`。

### 9.1 与 HumanX 原 ONNX 导出逻辑保持一致的地方

脚本刻意沿用 `humanoidverse/utils/inference_helpers.py` 的导出风格：

- 使用 `copy.deepcopy(policy)`，不直接改原始模型实例。
- 导出前把模型放到 CPU。
- 使用一个 `nn.Module` wrapper 包住真实 policy。
- 通过 `torch.onnx.export(...)` 导出。
- 默认 `opset_version=13`。
- 默认不设置 `dynamic_axes`，所以 batch 维固定。
- 输出名仍是 `action`。

### 9.2 vision_daggerbc 与普通 PPO 的差异

普通 PPO ONNX 只有一个输入：

```text
actor_obs -> action
```

vision_daggerbc 的 policy 需要向量观测和第一视角图像，因此新增 wrapper 的输入是：

```text
actor_obs, image -> action
```

wrapper 内部做了三件事：

1. 把 `actor_obs [B, 495]` 扩成模型训练格式 `actor_obs [B, 1, 495]`。
2. 把 `image [B, 224, 224, 1]` 扩成 `image [B, 1, 224, 224, 1]`。
3. 调用 `VisionActionMLP.forward()` 得到归一化 action 后，再用模型里的 `denormalize_actions()` 输出真实 action。

这里做 action 反归一化很重要：`VisionActionMLP.forward()` 的训练目标是 normalized action，而 `policy.act()` 在闭环评估中返回的是 denormalized action。ONNX 导出的 `action` 应该与 `policy.act()` 的语义一致。

### 9.3 已生成 ONNX 的签名和检查结果

本次已实际导出：

```text
bc_runs/vision_daggerbc/20260507_134418/exported/last.onnx
```

文件大小约 48 MB。使用 `onnx.checker.check_model()` 通过，ONNXRuntime CPU 推理也通过。

签名：

```text
opset: 13
input actor_obs: [1, 495]
input image: [1, 224, 224, 1]
output action: [1, 29]
```

用全零 dummy 输入跑 ONNXRuntime，输出 shape 为 `[1, 29]`，结果均为有限数值。

同一份 dummy 输入下，PyTorch wrapper 与 ONNXRuntime 输出最大绝对误差约为：

```text
1.94e-6
```

### 9.4 这个脚本的注意事项

- 默认导出固定 batch=1、固定 `actor_obs_dim=495`、固定 image shape `[224,224,1]`，这是为了贴近 HumanX 当前没有 `dynamic_axes` 的导出方式。
- 如果部署端要 batch N，可用 `--dynamic-batch` 重新导出；但这会和 HumanX 当前默认导出略有不同。
- trace 过程中会出现若干 `TracerWarning`，来源是 `bc_vision/model.py` 中的形状检查分支。这些分支在导出时被固定到 dummy 输入形状上；对当前固定输入签名是符合预期的。
- ONNX 只包含 vision policy 网络，不包含相机采集、vector obs 拼接、history 维护、action clip、动作缩放或 PD 控制。
