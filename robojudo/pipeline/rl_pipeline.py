import logging
import time

import numpy as np
from box import Box

import robojudo.environment
import robojudo.policy
from robojudo.controller import CtrlManager
from robojudo.environment import Environment
from robojudo.pipeline import Pipeline, pipeline_registry
from robojudo.pipeline.pipeline_cfgs import RlPipelineCfg
from robojudo.policy import Policy, PolicyCfg
from robojudo.tools.dof import DoFAdapter
from robojudo.tools.tool_cfgs import DoFConfig
from robojudo.utils.progress import ProgressBar
from robojudo.utils.util_func import get_gravity_orientation

logger = logging.getLogger(__name__)


class PolicyWrapper:
    """A wrapper for Policy to handle observation and action adaptation."""

    def __init__(self, cfg_policy: PolicyCfg, env_dof_cfg: DoFConfig, device: str):
        self.env_dof_cfg = env_dof_cfg

        policy_type = cfg_policy.policy_type
        policy_name = policy_type
        if hasattr(cfg_policy, "policy_name"):
            policy_name += "@" + cfg_policy.policy_name  # type: ignore
        # while policy_name in self.policies.keys():
        #     policy_name += "_new"
        self.name = policy_name

        policy_class: type[Policy] = getattr(robojudo.policy, policy_type)
        self.policy: Policy = policy_class(cfg_policy=cfg_policy, device=device)
        self.obs_adapter = DoFAdapter(env_dof_cfg.joint_names, self.policy.cfg_obs_dof.joint_names)
        self.actions_adapter = DoFAdapter(self.policy.cfg_action_dof.joint_names, env_dof_cfg.joint_names)

        self._orig_default_dof_pos = self.policy.default_dof_pos.copy()
        self._orig_default_pos = self.policy.default_pos.copy()
        self._pose_adjusted = False

        self._apply_motion_adjustments(cfg_policy, env_dof_cfg)
        if self._has_pd_adjustments or not np.array_equal(self._orig_default_dof_pos, self.policy.default_dof_pos):
            self._pose_adjusted = True

    def _apply_motion_adjustments(self, cfg_policy: PolicyCfg, env_dof_cfg: DoFConfig):
        """Process motion_adjustments: shift default_pos for policy-controlled joints,
        store remaining offsets for direct pd_target application (shoulders/wrists)."""
        self._pd_adjustments = np.zeros(len(env_dof_cfg.joint_names), dtype=np.float32)
        self._has_pd_adjustments = False

        motion_adjustments = getattr(cfg_policy, "motion_adjustments", None)
        if not motion_adjustments:
            return

        env_joints = env_dof_cfg.joint_names
        obs_joints = self.policy.cfg_obs_dof.joint_names
        action_joints = self.policy.cfg_action_dof.joint_names

        for raw_idx, offset in motion_adjustments.items():
            idx = raw_idx if raw_idx >= 0 else len(env_joints) + raw_idx
            joint_name = env_joints[idx]

            if joint_name in obs_joints:
                obs_idx = obs_joints.index(joint_name)
                self.policy.default_dof_pos[obs_idx] += offset

            if joint_name in action_joints:
                action_idx = action_joints.index(joint_name)
                self.policy.default_pos[action_idx] += offset
            else:
                self._pd_adjustments[idx] = offset
                self._has_pd_adjustments = True

            logger.info(f"Motion adjustment: {joint_name} (env[{raw_idx}]) += {offset}"
                        f" | action={'yes' if joint_name in action_joints else 'direct'}")

    def toggle_motion_adjustments(self):
        """Toggle between initial pose and the configured motion_adjustments."""
        if self._pose_adjusted:
            self.policy.default_dof_pos[:] = self._orig_default_dof_pos
            self.policy.default_pos[:] = self._orig_default_pos
            self._pd_adjustments[:] = 0.0
            self._has_pd_adjustments = False
            self._pose_adjusted = False
            logger.warning("Pose toggled → initial")
        else:
            self.policy.default_dof_pos[:] = self._orig_default_dof_pos
            self.policy.default_pos[:] = self._orig_default_pos
            self._pd_adjustments[:] = 0.0
            self._has_pd_adjustments = False
            self._apply_motion_adjustments(self.policy.cfg_policy, self.env_dof_cfg)
            self._pose_adjusted = True
            logger.warning("Pose toggled → adjusted")

    def get_observation(self, env_data: Box, ctrl_data: Box):
        env_data_adapted = env_data.copy()
        env_data_adapted.dof_pos = self.obs_adapter.fit(env_data_adapted.dof_pos)
        env_data_adapted.dof_vel = self.obs_adapter.fit(env_data_adapted.dof_vel)
        return self.policy.get_observation(env_data_adapted, ctrl_data)

    def get_action(self, obs):
        action = self.policy.get_action(obs)
        return self.actions_adapter.fit(action)

    def get_pd_target(self, obs):
        action = self.policy.get_action(obs)
        pd_target = action + self.policy.default_pos
        pd_target = self.actions_adapter.fit(pd_target, template=self.env_dof_cfg.default_pos)
        if self._has_pd_adjustments:
            pd_target += self._pd_adjustments
        return pd_target

    def get_init_dof_pos(self):
        return self.actions_adapter.fit(self.policy.get_init_dof_pos(), template=self.env_dof_cfg.default_pos)

    def __getattr__(self, name):
        """Fallback: delegate other func to the wrapped policy."""
        return getattr(self.policy, name)


@pipeline_registry.register
class RlPipeline(Pipeline):
    cfg: RlPipelineCfg

    def __init__(self, cfg: RlPipelineCfg):
        super().__init__(cfg=cfg)

        env_class: type[Environment] = getattr(robojudo.environment, self.cfg.env.env_type)
        self.env: Environment = env_class(cfg_env=self.cfg.env, device=self.device)

        self.ctrl_manager = CtrlManager(cfg_ctrls=self.cfg.ctrl, env=self.env, device=self.device)

        self.policy = PolicyWrapper(
            cfg_policy=self.cfg.policy,
            env_dof_cfg=self.env.dof_cfg,
            device=self.device,
        )

        self.env.update_dof_cfg(override_cfg=self.policy.cfg_action_dof)
        self.visualizer = self.env.visualizer
        self.default_stiffness = np.asarray(self.env.stiffness, dtype=np.float32).copy()
        self.default_damping = np.asarray(self.env.damping, dtype=np.float32).copy()

        self.freq = self.cfg.policy.freq
        self.dt = 1.0 / self.freq

        self.self_check()
        self.reset()

    def self_check(self):
        self.env.self_check()
        for _ in range(10):
            self.step(dry_run=True)

    def reset(self):
        logger.info("Pipeline reset")
        self.timestep = 0

        if hasattr(self.env, "reborn"):
            self.env.reborn()  # pyright: ignore[reportAttributeAccessIssue]
        self.env.reset()
        # self.env.reborn(init_qpos=[0.2, 0.2, 0.8] + [ 0.707, 0, 0, 0.707]) # FOR SIM DEBUG
        self.policy.reset()
        self.ctrl_manager.reset()

    def safety_check(self):
        if not self.do_safety_check:
            return
        gravity_ori = get_gravity_orientation(self.env.base_quat)
        angle = np.arccos(np.clip(-gravity_ori[2], -1.0, 1.0))
        if abs(angle) > 1.0:  # more than ~57 degrees
            logger.error("Robot fallen! Shutdown for safety.")
            if hasattr(self.env, "reborn"):
                self.env.reborn()  # pyright: ignore[reportAttributeAccessIssue]
            else:
                self.env.shutdown()

    def post_step_callback(self, env_data, ctrl_data, extras, pd_target):
        self.timestep += 1
        commands = ctrl_data.get("COMMANDS", [])
        for command in commands:
            match command:
                case "[SHUTDOWN]":
                    logger.warning("Emergency shutdown!")
                    self.env.shutdown()
                case "[MOTION_RESET]":
                    logger.warning("Motion reset!")
                    self.reset()
                case "[SIM_REBORN]":
                    if hasattr(self.env, "reborn"):
                        logger.warning("Simulation Env reborn!")
                        self.env.reborn()  # pyright: ignore[reportAttributeAccessIssue]
                case "[POSE_TOGGLE]":
                    self.policy.toggle_motion_adjustments()

        self.ctrl_manager.post_step_callback(ctrl_data)

        self.policy.post_step_callback(commands)
        if self.visualizer is not None:
            self.policy.debug_viz(self.visualizer, env_data, ctrl_data, extras)

        self.safety_check()
        if self.cfg.debug.log_obs:
            self.debug_logger.log(
                env_data=env_data,
                ctrl_data=ctrl_data,
                extras=extras,
                pd_target=pd_target,
                timestep=self.timestep,
            )

    def _button_pressed(self, ctrl_data, button_name: str) -> bool:
        for ctrl_name, ctrl_payload in ctrl_data.items():
            if ctrl_name == "COMMANDS" or not hasattr(ctrl_payload, "get"):
                continue
            button_events = ctrl_payload.get("button_event", [])
            for event in button_events:
                if event.get("type") == "button" and event.get("name") == button_name and event.get("pressed", False):
                    return True
        return False

    def _restore_default_gains(self):
        if hasattr(self.env, "set_gains"):
            self.env.set_gains(self.default_stiffness.tolist(), self.default_damping.tolist())

    def _set_scaled_gains(self, stiffness_scale: float, damping_scale: float):
        if hasattr(self.env, "set_gains"):
            self.env.set_gains(
                (self.default_stiffness * stiffness_scale).tolist(),
                (self.default_damping * damping_scale).tolist(),
            )

    def _poll_ctrl_data(self):
        self.env.update()
        env_data = self.env.get_data()
        ctrl_data = self.ctrl_manager.get_ctrl_data(env_data)
        return env_data, ctrl_data

    def wait_for_zero_torque_start(self):
        if not self.cfg.wait_for_zero_torque_start:
            return

        start_button = self.cfg.zero_torque_start_button
        shutdown_button = self.cfg.shutdown_button
        logger.warning(f"Waiting for {start_button} in zero torque. Press {shutdown_button} to shutdown.")
        self.ctrl_manager.reset()
        self._set_scaled_gains(0.0, 0.0)
        zero_target = np.zeros(self.env.num_dofs, dtype=np.float32)

        try:
            while True:
                _env_data, ctrl_data = self._poll_ctrl_data()

                if "[SHUTDOWN]" in ctrl_data.get("COMMANDS", []) or self._button_pressed(ctrl_data, shutdown_button):
                    logger.warning(f"Shutdown requested before prepare by {shutdown_button}.")
                    self.env.shutdown()
                    raise SystemExit(0)

                if self._button_pressed(ctrl_data, start_button):
                    logger.warning(f"Zero torque released by {start_button}.")
                    return

                self.env.step(zero_target)
                time.sleep(self.dt)
        finally:
            self._restore_default_gains()

    def wait_for_start_confirmation(self):
        if not self.cfg.wait_for_start_confirmation:
            return

        start_button = self.cfg.start_confirm_button
        shutdown_button = self.cfg.shutdown_button
        logger.warning(f"Waiting for {start_button} to start. Press {shutdown_button} to shutdown.")
        self.ctrl_manager.reset()
        desired_motor_angle = self.policy.get_init_dof_pos()
        self._set_scaled_gains(self.cfg.start_hold_stiffness_scale, self.cfg.start_hold_damping_scale)

        try:
            while True:
                _env_data, ctrl_data = self._poll_ctrl_data()

                if "[SHUTDOWN]" in ctrl_data.get("COMMANDS", []) or self._button_pressed(ctrl_data, shutdown_button):
                    logger.warning(f"Shutdown requested before start by {shutdown_button}.")
                    self.env.shutdown()
                    raise SystemExit(0)

                if self._button_pressed(ctrl_data, start_button):
                    logger.warning(f"Start confirmed by {start_button}.")
                    self.policy.reset()
                    self.timestep = 0
                    self.ctrl_manager.reset()
                    return

                self.env.step(desired_motor_angle)
                time.sleep(self.dt)
        finally:
            self._restore_default_gains()

    def step(self, dry_run=False):
        self.env.update()
        env_data = self.env.get_data()

        ctrl_data = self.ctrl_manager.get_ctrl_data(env_data)

        commands = ctrl_data.get("COMMANDS", [])
        if len(commands) > 0:
            logger.info(f"{'=' * 10} COMMANDS {'=' * 10}\n{commands}")

        obs, extras = self.policy.get_observation(env_data, ctrl_data)
        pd_target = self.policy.get_pd_target(obs)

        if not dry_run:
            self.env.step(pd_target, extras.get("hand_pose", None))

        self.post_step_callback(env_data, ctrl_data, extras, pd_target)

    def prepare(self, init_motor_angle=None):
        if init_motor_angle is not None:
            desired_motor_angle = init_motor_angle
        else:
            desired_motor_angle = self.policy.get_init_dof_pos()

        # logger.info(f"{desired_motor_angle=}")
        current_motor_angle = np.array(self.env.dof_pos)
        # logger.info(f"{current_motor_angle=}")

        if self.cfg.prepare_duration_s is None:
            traj_len = 1000
        else:
            traj_len = max(1, int(round(self.cfg.prepare_duration_s * self.freq)))
        last_step_time = time.time()
        logger.warning("prepare_init")
        pbar = ProgressBar("Prepare", traj_len)

        for t in range(traj_len):
            current_motor_angle = np.array(self.env.dof_pos)

            if self.cfg.prepare_duration_s is None:
                blend_ratio = np.minimum(t / 300, 1)
            else:
                blend_ratio = t / max(traj_len - 1, 1)
            action = (1 - blend_ratio) * current_motor_angle + blend_ratio * desired_motor_angle

            # warm up network
            self.step(dry_run=True)

            self.env.step(action)

            time_diff = last_step_time + self.dt - time.time()
            if time_diff > 0:
                time.sleep(time_diff)
            else:
                logger.error("Warning: frame drop")
            last_step_time = time.time()
            pbar.update()

            if self.cfg.prepare_reset_before_done and t == 0.9 * traj_len:
                logger.info(f"{'=' * 10} RESET ZERO POSITION {'=' * 10}")
                self.reset()

        time.sleep(0.01)
        pbar.close()
        logger.warning("prepare_done")


if __name__ == "__main__":
    pass
