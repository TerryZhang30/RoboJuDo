import logging
import os
import time
from collections import deque

import joblib
import numpy as np
import onnxruntime as ort

from robojudo.policy import Policy, policy_registry
from robojudo.policy.policy_cfgs import PickballBCPolicyCfg
from robojudo.utils.util_func import quat_rotate_inverse_np

logger = logging.getLogger(__name__)


@policy_registry.register
class PickballBCPolicy(Policy):
    """Pickball behavior-cloning policy exported as ONNX."""

    cfg_policy: PickballBCPolicyCfg

    def __init__(self, cfg_policy: PickballBCPolicyCfg, device):
        if not os.path.isfile(cfg_policy.policy_file):
            raise FileNotFoundError(f"Model file not found at {cfg_policy.policy_file}")

        logger.debug(f"Loading pickball BC policy from {cfg_policy.policy_file}")
        providers = self._resolve_onnx_providers(cfg_policy, device)
        self._preload_onnx_gpu_dependencies(providers)
        self.session = ort.InferenceSession(cfg_policy.policy_file, ort.SessionOptions(), providers=providers)
        active_providers = self.session.get_providers()
        self._check_onnx_provider_fallback(cfg_policy, providers, active_providers)
        logger.info(f"ONNX Runtime providers active: {active_providers}")
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.output_names = [o.name for o in self.session.get_outputs()]

        super().__init__(cfg_policy=cfg_policy, device=device)

        self.obs_scales = cfg_policy.obs_scales
        self.default_dof_pos = np.asarray(self.default_dof_pos, dtype=np.float32)
        self.default_pos = np.asarray(self.default_pos, dtype=np.float32)
        self.action_scales = self._compute_action_scales(cfg_policy)

        self._load_motion(cfg_policy)
        self._last_image = self._fallback_image()
        self._warned_image_fallback = False

        history_length = max(cfg_policy.obs_hist_length.values(), default=0)
        self.history_buf = deque(maxlen=history_length)
        self.history_obs_dims = cfg_policy.obs_hist_dims

        self.reset()

    @staticmethod
    def _resolve_onnx_providers(cfg_policy: PickballBCPolicyCfg, device: str) -> list[str]:
        available = list(ort.get_available_providers())
        requested = cfg_policy.onnx_providers
        if requested is None:
            mode = str(cfg_policy.onnx_device).lower()
            if mode == "pipeline":
                mode = str(device).lower()
            if mode in ("auto", "cuda", "gpu"):
                requested = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            elif mode == "tensorrt":
                requested = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
            elif mode == "cpu":
                requested = ["CPUExecutionProvider"]
            else:
                raise ValueError(f"Unknown onnx_device: {cfg_policy.onnx_device}")

        providers = [provider for provider in requested if provider in available]
        missing = [provider for provider in requested if provider not in available]
        if missing:
            logger.warning(f"Requested ONNX providers unavailable: {missing}; available providers: {available}")
        if not providers:
            providers = ["CPUExecutionProvider"] if "CPUExecutionProvider" in available else available
        return providers

    @staticmethod
    def _preload_onnx_gpu_dependencies(providers: list[str]) -> None:
        gpu_providers = {"CUDAExecutionProvider", "TensorrtExecutionProvider"}
        if not any(provider in gpu_providers for provider in providers):
            return
        preload_dlls = getattr(ort, "preload_dlls", None)
        if preload_dlls is None:
            return
        try:
            preload_dlls()
        except Exception as exc:
            logger.warning(f"ONNX Runtime CUDA dependency preload failed: {exc}")

    @staticmethod
    def _check_onnx_provider_fallback(
        cfg_policy: PickballBCPolicyCfg,
        requested_providers: list[str],
        active_providers: list[str],
    ) -> None:
        gpu_providers = {"CUDAExecutionProvider", "TensorrtExecutionProvider"}
        requested_gpu = [provider for provider in requested_providers if provider in gpu_providers]
        active_gpu = [provider for provider in active_providers if provider in gpu_providers]
        if not requested_gpu or active_gpu:
            return

        message = (
            f"Requested ONNX GPU providers {requested_gpu}, but active providers are {active_providers}. "
            "The session is running on CPU."
        )
        if cfg_policy.onnx_require_gpu:
            raise RuntimeError(message)
        logger.warning(message)

    def _compute_action_scales(self, cfg_policy: PickballBCPolicyCfg) -> np.ndarray:
        stiffness = cfg_policy.obs_dof.stiffness
        if stiffness is None:
            raise ValueError("PickballBCPolicy requires obs_dof.stiffness to compute action scales")
        if len(cfg_policy.dof_effort_limits) != self.num_actions:
            raise ValueError("dof_effort_limits length must match action_dof.num_dofs")
        if len(stiffness) != self.num_actions:
            raise ValueError("obs_dof.stiffness length must match action_dof.num_dofs")

        effort_limits = np.asarray(cfg_policy.dof_effort_limits, dtype=np.float32)
        stiffness = np.asarray(stiffness, dtype=np.float32)
        return (cfg_policy.actions_scale * effort_limits / stiffness).astype(np.float32)

    def _load_motion(self, cfg_policy: PickballBCPolicyCfg) -> None:
        self.motion_data = joblib.load(cfg_policy.motion_data_path)
        self.motion_name = list(self.motion_data.keys())[0]
        motion = self.motion_data[self.motion_name]
        self.init_angles = np.asarray(motion["dof"][0, :], dtype=np.float32).copy()
        for idx, offset in cfg_policy.motion_adjustments.items():
            self.init_angles[idx] += offset

        self.motion_fps = int(motion["fps"])
        self.motion_length_s = (motion["dof"].shape[0] - 1) / self.motion_fps
        self._obj_pos = motion.get("obj_pos", None)
        self._obj_rot_quat = motion.get("obj_rot_quat", None)
        self._ball_release_frame = int(motion.get("stand_frame", motion["dof"].shape[0]))
        if self._obj_pos is not None:
            logger.info(
                f"Loaded pickball object trajectory: {self._obj_pos.shape[0]} frames, "
                f"release_frame={self._ball_release_frame}"
            )

    def reset(self):
        self.timestep = 0
        self.flag_motion_done = False
        self.last_action = np.zeros(self.num_actions, dtype=np.float32)
        self._last_safe_action_output = self._process_action_output(self.last_action)
        self._joint_delta_safety_frozen = False

        self.history_buf.clear()
        for _ in range(self.history_buf.maxlen or 0):
            obs_a = [
                np.zeros(self.history_obs_dims[key], dtype=np.float32)
                for key in sorted(self.history_obs_dims.keys())
            ]
            self.history_buf.appendleft(obs_a)

    def soft_reset(self):
        self.reset()

    def post_step_callback(self, commands: list[str] | None = None):
        self.timestep += 1
        if (self.timestep * self.dt) >= self.motion_length_s:
            self.flag_motion_done = True

        for command in commands or []:
            match command:
                case "[MOTION_RESET]":
                    self.reset()
                case "[MOTION_REPLAY]":
                    self.soft_reset()

    def get_init_dof_pos(self) -> np.ndarray:
        return self.init_angles.copy()

    def get_initial_ball_info(self) -> dict | None:
        if not self.cfg_policy.track_motion_ball:
            return None
        if self._obj_pos is None or len(self._obj_pos) == 0:
            return None
        return {
            "target_pos": np.asarray(self._obj_pos[0], dtype=np.float64),
            "released": bool(self.cfg_policy.release_ball_after_init),
        }

    def get_observation(self, env_data, ctrl_data):
        del ctrl_data
        self._check_image_freshness(env_data)
        self._last_image = self._image_from_env_data(env_data)

        dof_pos = np.asarray(env_data.dof_pos, dtype=np.float32)
        dof_vel = np.asarray(env_data.dof_vel, dtype=np.float32)
        projected_gravity = quat_rotate_inverse_np(env_data.base_quat, np.array([0, 0, -1]))
        phase = np.clip(np.array([(self.timestep * self.cfg_policy.phase_dt) / self.motion_length_s]), 0, 1.0)
        actions_scaled = self.last_action * self.action_scales + self.default_dof_pos

        obs_components = {
            "dof_pos": (dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
            "dof_vel": dof_vel * self.obs_scales.dof_vel,
            "base_ang_vel": np.asarray(env_data.base_ang_vel, dtype=np.float32) * self.obs_scales.base_ang_vel,
            "projected_gravity": projected_gravity,
            "phase": phase,
            "pd_error": actions_scaled - dof_pos,
            "actions": self.last_action,
            "history_actor": self._get_obs_history(),
        }

        obs = np.concatenate([obs_components[key] for key in self.cfg_policy.vector_obs_keys], axis=0).astype(
            np.float32
        )
        obs_a = [obs_components[key] for key in sorted(self.history_obs_dims.keys())]
        self.history_buf.appendleft(obs_a)

        extras = {"CALLBACK": ["[MOTION_DONE]"] if self.flag_motion_done else []}
        ball_info = self._get_ball_info()
        if ball_info is not None:
            extras["ball_info"] = ball_info
        return obs, extras

    def _get_obs_history(self) -> np.ndarray:
        if not self.history_buf:
            return np.empty(0, dtype=np.float32)
        history_list = [np.concatenate(items, axis=0) for items in zip(*self.history_buf, strict=True)]
        return np.concatenate(history_list, axis=0).astype(np.float32)

    def _fallback_image(self) -> np.ndarray:
        return np.full(self.cfg_policy.image_shape, self.cfg_policy.image_fill_value, dtype=np.float32)

    def _check_image_freshness(self, env_data) -> None:
        max_image_age_s = self.cfg_policy.max_image_age_s
        if max_image_age_s is None:
            return
        timestamp = env_data.get("camera_timestamp", None) if hasattr(env_data, "get") else None
        if timestamp is None:
            if self.cfg_policy.require_image:
                raise RuntimeError("PickballBCPolicy requires camera_timestamp but env_data did not provide it")
            return

        age = time.time() - float(timestamp)
        if age > max_image_age_s:
            raise RuntimeError(
                f"PickballBCPolicy camera image is stale: age={age:.3f}s, limit={max_image_age_s:.3f}s"
            )

    def _image_from_env_data(self, env_data) -> np.ndarray:
        for key in self.cfg_policy.image_obs_keys:
            image = env_data.get(key, None) if hasattr(env_data, "get") else getattr(env_data, key, None)
            if image is not None:
                return self._coerce_image(image)

        if self.cfg_policy.require_image:
            raise RuntimeError(
                f"PickballBCPolicy requires one of image_obs_keys={self.cfg_policy.image_obs_keys}, "
                "but none was present in env_data"
            )
        if self.cfg_policy.warn_on_image_fallback and not self._warned_image_fallback:
            logger.warning(
                "PickballBCPolicy did not receive an image/depth_image from env_data; "
                "using the configured constant image fallback."
            )
            self._warned_image_fallback = True
        return self._fallback_image()

    def _coerce_image(self, image) -> np.ndarray:
        target_h, target_w, target_c = self.cfg_policy.image_shape
        image_np = np.asarray(image, dtype=np.float32)

        if image_np.ndim == 4 and image_np.shape[0] == 1:
            image_np = image_np[0]
        if image_np.ndim == 2:
            image_np = image_np[..., None]
        if image_np.ndim != 3:
            raise ValueError(f"Expected image with 2 or 3 dims, got shape {image_np.shape}")

        if image_np.shape[0] in (1, 3) and image_np.shape[-1] not in (1, 3):
            image_np = np.transpose(image_np, (1, 2, 0))
        if target_c == 1 and image_np.shape[-1] != 1:
            image_np = image_np.mean(axis=-1, keepdims=True)
        if target_c != 1 and image_np.shape[-1] == 1:
            image_np = np.repeat(image_np, target_c, axis=-1)
        if image_np.shape[-1] != target_c:
            raise ValueError(f"Expected image channel count {target_c}, got shape {image_np.shape}")

        if image_np.shape[:2] != (target_h, target_w):
            image_np = self._resize_nearest(image_np, target_h, target_w)
        return image_np.astype(np.float32, copy=False)

    @staticmethod
    def _resize_nearest(image: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        src_h, src_w = image.shape[:2]
        y_idx = np.rint(np.linspace(0, src_h - 1, target_h)).astype(np.int64)
        x_idx = np.rint(np.linspace(0, src_w - 1, target_w)).astype(np.int64)
        return image[y_idx][:, x_idx]

    def _get_ball_info(self) -> dict | None:
        if not self.cfg_policy.track_motion_ball:
            return None
        if self._obj_pos is None:
            return {"use_hands_fallback": True, "released": True}

        frame = int(round(self.timestep * self.dt * self.motion_fps))
        frame = min(max(frame, 0), self._obj_pos.shape[0] - 1)
        ball_info = {
            "target_pos": np.asarray(self._obj_pos[frame], dtype=np.float64),
            "released": self.cfg_policy.release_ball_at_stand_frame and frame >= self._ball_release_frame,
        }
        if self._obj_rot_quat is not None:
            ball_info["target_quat"] = np.asarray(self._obj_rot_quat[frame], dtype=np.float64)
        return ball_info

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        ort_inputs = {}
        for input_name in self.input_names:
            if input_name == "actor_obs":
                ort_inputs[input_name] = np.expand_dims(obs, axis=0).astype(np.float32)
            elif input_name == "image":
                ort_inputs[input_name] = np.expand_dims(self._last_image, axis=0).astype(np.float32)
            else:
                raise ValueError(f"Unsupported ONNX input '{input_name}' for PickballBCPolicy")

        ort_outputs = self.session.run(self.output_names, ort_inputs)
        raw_actions = np.asarray(ort_outputs[0], dtype=np.float32).squeeze()
        raw_actions = np.clip(raw_actions, -self.cfg_policy.action_clip, self.cfg_policy.action_clip)
        action_output = self._process_action_output(raw_actions)
        return self._enforce_joint_delta_limit(raw_actions, action_output)

    def _process_action_output(self, raw_actions: np.ndarray) -> np.ndarray:
        raw_actions = np.asarray(raw_actions, dtype=np.float32)
        if self.cfg_policy.apply_action_scales:
            return (raw_actions * self.action_scales).astype(np.float32, copy=False)
        return raw_actions.astype(np.float32, copy=True)

    def _enforce_joint_delta_limit(self, raw_actions: np.ndarray, action_output: np.ndarray) -> np.ndarray:
        max_delta_limit = self.cfg_policy.max_joint_delta_rad
        if max_delta_limit is None:
            self.last_action = raw_actions.copy()
            self._last_safe_action_output = action_output.copy()
            return action_output

        last_output = self._last_safe_action_output
        if self.cfg_policy.joint_delta_safety_mode == "freeze" and self._joint_delta_safety_frozen:
            return last_output.copy()

        abs_delta = np.abs(action_output - last_output)
        max_idx = int(np.argmax(abs_delta))
        max_delta = float(abs_delta[max_idx])
        if max_delta <= max_delta_limit:
            self.last_action = raw_actions.copy()
            self._last_safe_action_output = action_output.copy()
            return action_output

        if max_idx < len(self.cfg_policy.dof_names):
            joint_name = self.cfg_policy.dof_names[max_idx]
        else:
            joint_name = f"joint_{max_idx}"
        message = (
            "PickballBCPolicy joint delta safety limit exceeded: "
            f"{joint_name} changed by {max_delta:.4f} rad, limit={max_delta_limit:.4f} rad"
        )
        if self.cfg_policy.joint_delta_safety_mode == "hold":
            logger.error(f"{message}; holding previous safe action output.")
            return last_output.copy()
        if self.cfg_policy.joint_delta_safety_mode == "zero":
            logger.error(f"{message}; outputting zero action.")
            zero_raw_actions = np.zeros_like(raw_actions, dtype=np.float32)
            zero_action_output = np.zeros_like(action_output, dtype=np.float32)
            self.last_action = zero_raw_actions
            self._last_safe_action_output = zero_action_output.copy()
            return zero_action_output
        if self.cfg_policy.joint_delta_safety_mode == "freeze":
            logger.error(f"{message}; freezing previous safe action output until reset.")
            self._joint_delta_safety_frozen = True
            return last_output.copy()

        raise RuntimeError(message)
