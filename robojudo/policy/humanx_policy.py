import logging
import os

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

        self.reset()

    def reset(self):
        self.timestep = 0
        self.flag_motion_done = False
        self.actions = np.zeros(self.num_actions, dtype=np.float32)

        # Initialize history buffers
        for k, hlen in self.cfg_policy.obs_hist_length.items():
            setattr(self, f"{k}_buf", np.zeros(hlen * self.cfg_policy.obs_hist_dims[k], dtype=np.float32))

        self.obs_key_sorted = sorted(self.cfg_policy.actor_obs_keys)
        self.hist_obs_key_sorted = sorted(self.cfg_policy.obs_hist_dims.keys())

        # Initialize observation attributes
        self.local_curr_ball_position = np.zeros(3, dtype=np.float32)
        self.local_ref_origin_pos_xy = np.zeros(2, dtype=np.float32)
        self.dif_ref_heading = np.zeros(1, dtype=np.float32)

    def post_step_callback(self, commands: list[str] | None = None):
        self.timestep += 1
        if (self.timestep * self.dt) >= self.motion_length_s:
            self.flag_motion_done = True

    def _update_history(self):
        """Update history buffers"""
        for obs_name in self.hist_obs_key_sorted:
            obs_buf = getattr(self, f"{obs_name}_buf")
            obs_dims = self.cfg_policy.obs_hist_dims[obs_name]
            obs_val = getattr(self, obs_name)
            updated_hist = np.concatenate([obs_val, obs_buf[:-obs_dims]], axis=0, dtype=np.float32)
            setattr(self, f"{obs_name}_buf", updated_hist)

    def get_observation(self, env_data, ctrl_data):
        base_quat = env_data.base_quat
        base_ang_vel = env_data.base_ang_vel
        dof_pos = env_data.dof_pos
        dof_vel = env_data.dof_vel

        # Compute observations
        self.projected_gravity = quat_rotate_inverse_np(base_quat, np.array([0, 0, -1]))
        self.dof_pos = (dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos
        self.dof_vel = dof_vel * self.obs_scales.dof_vel
        self.base_ang_vel = base_ang_vel * self.obs_scales.base_ang_vel

        # Compute history buffer (use current history, not updated yet)
        hist_obs_buf = []
        for obs_name in self.hist_obs_key_sorted:
            hist_obs_buf.append(getattr(self, f"{obs_name}_buf"))
        self.history_obs_buf = np.concatenate(hist_obs_buf, axis=0, dtype=np.float32)

        # Compute PD error
        actions_scaled = self.actions * self.action_scales + self.default_dof_pos
        self.pd_error = actions_scaled - dof_pos

        # Build observation
        obs_buf = []
        for obs_name in self.obs_key_sorted:
            obs_buf.append(getattr(self, obs_name))
        obs = np.concatenate(obs_buf, axis=0, dtype=np.float32)

        # Update history AFTER building observation
        self._update_history()

        extras = {"CALLBACK": ["[MOTION_DONE]"] if self.flag_motion_done else []}
        return obs, extras

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        obs_input = obs.astype(np.float32).reshape(1, -1)
        self.actions = self.session.run(self.output_names, {self.input_names[0]: obs_input})[0].squeeze()
        self.actions = np.clip(self.actions, -self.cfg_policy.action_clip, self.cfg_policy.action_clip)

        actions_scaled = self.actions * self.action_scales
        return actions_scaled

    def get_init_dof_pos(self) -> np.ndarray:
        return self.init_angles.copy()
