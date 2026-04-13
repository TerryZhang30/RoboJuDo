import logging
import os

import numpy as np
import onnxruntime as ort

from robojudo.policy import Policy, policy_registry
from robojudo.policy.policy_cfgs import PolicyCfg
from robojudo.utils.util_func import quat_rotate_inverse_np

logger = logging.getLogger(__name__)


@policy_registry.register
class StandPolicy(Policy):
    """Anti-fall stand policy with ONNX inference (96-dim obs, 29-dof action)."""

    def __init__(self, cfg_policy: PolicyCfg, device):
        if not os.path.isfile(cfg_policy.policy_file):
            raise FileNotFoundError(f"Model file not found at {cfg_policy.policy_file}")

        logger.debug(f"Loading stand policy from {cfg_policy.policy_file}")
        self.session = ort.InferenceSession(cfg_policy.policy_file)
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.output_names = [o.name for o in self.session.get_outputs()]

        super().__init__(cfg_policy=cfg_policy, device=device)

        self.obs_scales = cfg_policy.obs_scales

        # Per-joint action scale: actions_scale * effort_limit / stiffness
        # Matches HumanxPolicy training convention
        if hasattr(cfg_policy, 'dof_effort_limits') and hasattr(cfg_policy, 'actions_scale'):
            action_scales = []
            for j_id in range(len(cfg_policy.dof_names)):
                e = cfg_policy.dof_effort_limits[j_id]
                s = cfg_policy.obs_dof.stiffness[j_id]
                action_scales.append(cfg_policy.actions_scale * e / s)
            self.action_scales = np.array(action_scales, dtype=np.float32)
        else:
            self.action_scales = np.full(self.num_actions, self.action_scale, dtype=np.float32)

        self.reset()

    def reset(self):
        self.last_action = np.zeros(self.num_actions, dtype=np.float32)

    def post_step_callback(self, commands: list[str] | None = None):
        pass

    def get_observation(self, env_data, ctrl_data):
        base_lin_vel = env_data.base_lin_vel
        if base_lin_vel is None:
            base_lin_vel = np.zeros(3, dtype=np.float32)

        projected_gravity = quat_rotate_inverse_np(env_data.base_quat, np.array([0, 0, -1]))

        obs = np.concatenate([
            self.last_action,
            env_data.base_ang_vel * self.obs_scales.base_ang_vel,
            base_lin_vel * self.obs_scales.base_lin_vel,
            (env_data.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
            env_data.dof_vel * self.obs_scales.dof_vel,
            projected_gravity,
        ])

        return obs, {}

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        ort_inputs = {self.input_names[0]: np.expand_dims(obs, axis=0).astype(np.float32)}
        ort_outputs = self.session.run(self.output_names, ort_inputs)
        actions = np.asarray(ort_outputs[0]).squeeze()

        if self.action_clip is not None:
            actions = np.clip(actions, -self.action_clip, self.action_clip)
        self.last_action = actions.copy()

        return actions * self.action_scales
