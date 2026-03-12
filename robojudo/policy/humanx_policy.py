import logging
import os
from collections import deque

import joblib
import numpy as np
import onnxruntime as ort

from robojudo.policy import Policy, policy_registry
from robojudo.policy.policy_cfgs import HumanxPolicyCfg
from robojudo.utils.util_func import quat_rotate_inverse_np

logger = logging.getLogger(__name__)


@policy_registry.register
class HumanxPolicy(Policy):
    """Human-Object Interaction Policy"""

    cfg_policy: HumanxPolicyCfg

    def __init__(self, cfg_policy: HumanxPolicyCfg, device):
        if not os.path.isfile(cfg_policy.policy_file):
            raise FileNotFoundError(f"Model file not found at {cfg_policy.policy_file}")

        logger.debug(f"Loading humanx policy from {cfg_policy.policy_file}")
        self.session = ort.InferenceSession(cfg_policy.policy_file)
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.output_names = [o.name for o in self.session.get_outputs()]

        super().__init__(cfg_policy=cfg_policy, device=device)

        self.obs_scales = cfg_policy.obs_scales

        # Load motion data
        self.motion_data = joblib.load(cfg_policy.motion_data_path)
        self.motion_name = list(self.motion_data.keys())[0]
        self.init_angles = self.motion_data[self.motion_name]['dof'][0, :].copy()

        # Apply motion adjustments
        for idx, offset in cfg_policy.motion_adjustments.items():
            self.init_angles[idx] += offset

        self.motion_length_s = (self.motion_data[self.motion_name]['dof'].shape[0] - 1) / \
                               self.motion_data[self.motion_name]['fps']

        # Compute action scales
        action_scales = []
        for j_id in range(len(cfg_policy.dof_names)):
            e = cfg_policy.dof_effort_limits[j_id]
            s = cfg_policy.kps[j_id]
            action_scales.append(cfg_policy.actions_scale * e / s)
        self.action_scales = np.array(action_scales, dtype=np.float32)

        # History management
        history_length = max(cfg_policy.obs_hist_length.values())
        self.history_buf = deque(maxlen=history_length)
        self.history_obs_dims = cfg_policy.obs_hist_dims

        self.reset()

    def reset(self):
        self.timestep = 0
        self.flag_motion_done = False
        self.last_action = np.zeros(self.num_actions, dtype=np.float32)

        # Pre-fill history with zeros
        self.history_buf.clear()
        for _ in range(self.history_buf.maxlen):
            obs_a = [np.zeros(self.history_obs_dims[k], dtype=np.float32)
                     for k in sorted(self.history_obs_dims.keys())]
            self.history_buf.appendleft(obs_a)

    def post_step_callback(self, commands: list[str] | None = None):
        self.timestep += 1
        if (self.timestep * self.dt) >= self.motion_length_s:
            self.flag_motion_done = True

    def _get_obs_history(self):
        history_list = [np.concatenate(items, axis=0) for items in zip(*self.history_buf, strict=True)]
        return np.concatenate(history_list, axis=0)

    def get_observation(self, env_data, ctrl_data):
        base_quat = env_data.base_quat
        base_ang_vel = env_data.base_ang_vel
        dof_pos = env_data.dof_pos
        dof_vel = env_data.dof_vel

        dof_pos_minus_default = dof_pos - self.default_dof_pos
        projected_gravity = quat_rotate_inverse_np(base_quat, np.array([0, 0, -1]))

        # Compute PD error
        actions_scaled = self.last_action * self.action_scales + self.default_dof_pos
        pd_error = actions_scaled - dof_pos

        # Get history
        history = self._get_obs_history()

        # HOI-specific observations (placeholders)
        local_curr_ball_position = np.zeros(3, dtype=np.float32)
        local_ref_origin_pos_xy = np.zeros(2, dtype=np.float32)
        dif_ref_heading = np.zeros(1, dtype=np.float32)

        # Build observation components
        obs_components = {
            "base_ang_vel": base_ang_vel * self.obs_scales.base_ang_vel,
            "projected_gravity": projected_gravity,
            "dof_pos": dof_pos_minus_default * self.obs_scales.dof_pos,
            "dof_vel": dof_vel * self.obs_scales.dof_vel,
            "actions": self.last_action,
            "pd_error": pd_error,
            "local_curr_ball_position": local_curr_ball_position,
            "history_obs_buf": history,
            "local_ref_origin_pos_xy": local_ref_origin_pos_xy,
            "dif_ref_heading": dif_ref_heading,
        }

        # Build observation (config-driven)
        obs = np.concatenate([obs_components[k] for k in sorted(self.cfg_policy.actor_obs_keys)], axis=0)

        # Update history
        obs_a = [obs_components[k] for k in sorted(self.history_obs_dims.keys())]
        self.history_buf.appendleft(obs_a)

        extras = {"CALLBACK": ["[MOTION_DONE]"] if self.flag_motion_done else []}
        return obs, extras

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        ort_inputs = {self.input_names[0]: np.expand_dims(obs, axis=0).astype(np.float32)}
        ort_outputs = self.session.run(self.output_names, ort_inputs)
        actions = np.asarray(ort_outputs[0]).squeeze()

        actions = np.clip(actions, -self.cfg_policy.action_clip, self.cfg_policy.action_clip)
        self.last_action = actions.copy()

        return actions * self.action_scales

    def get_init_dof_pos(self) -> np.ndarray:
        return self.init_angles.copy()
