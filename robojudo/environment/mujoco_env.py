import logging
import time

import mujoco
import mujoco_viewer
import numpy as np

from robojudo.environment import Environment, env_registry
from robojudo.environment.env_cfgs import MujocoEnvCfg
from robojudo.environment.utils.mujoco_viz import MujocoVisualizer
from robojudo.utils.util_func import quat_rotate_inverse_np, quatToEuler

logger = logging.getLogger(__name__)


@env_registry.register
class MujocoEnv(Environment):
    cfg_env: MujocoEnvCfg

    def __init__(self, cfg_env: MujocoEnvCfg, device="cpu"):
        super().__init__(cfg_env=cfg_env, device=device)

        self.sim_duration = cfg_env.sim_duration
        self.sim_dt = cfg_env.sim_dt
        self.sim_decimation = cfg_env.sim_decimation
        self.control_dt = self.sim_dt * self.sim_decimation

        self._ball_cfg = cfg_env.ball
        self._ball_body_id: int = -1
        self._ball_jnt_qpos_addr: int = -1
        self._ball_released: bool = False

        if cfg_env.ball.enabled:
            self.model, self.data = self._build_model_with_ball(cfg_env)
        else:
            self.model = mujoco.MjModel.from_xml_path(cfg_env.xml)  # pyright: ignore[reportAttributeAccessIssue]
            self.data = mujoco.MjData(self.model)  # pyright: ignore[reportAttributeAccessIssue]

        self.model.opt.timestep = self.sim_dt
        self._refresh_dof_indices()
        if self.model.nkey > 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)  # pyright: ignore[reportAttributeAccessIssue]
            mujoco.mj_forward(self.model, self.data)  # pyright: ignore[reportAttributeAccessIssue]
        else:
            mujoco.mj_step(self.model, self.data)  # pyright: ignore[reportAttributeAccessIssue]

        self.viewer = mujoco_viewer.MujocoViewer(
            self.model,
            self.data,
            width=1200,
            height=900,
            hide_menus=True,
            diable_key_callbacks=True,
        )
        self.viewer.cam.distance = 3.0
        self.viewer.cam.elevation = -10.0
        self.viewer.cam.azimuth = 180.0
        # self.viewer._paused = True

        if cfg_env.visualize_extras:
            self.visualizer = MujocoVisualizer(self.viewer)
        else:
            self.visualizer = None

        self.last_time = time.time()

        self.update()  # get initial state

    # ------------------------------------------------------------------
    # Ball helpers
    # ------------------------------------------------------------------

    def _build_model_with_ball(self, cfg_env: MujocoEnvCfg):
        """Use MjSpec to inject a free-body ball into the scene."""
        ball = cfg_env.ball
        spec = mujoco.MjSpec.from_file(cfg_env.xml)  # pyright: ignore[reportAttributeAccessIssue]

        body = spec.worldbody.add_body()
        body.name = "ball"
        body.pos = ball.init_pos

        fj = body.add_freejoint()
        fj.name = "ball_joint"

        geom = body.add_geom()
        geom.name = "ball_geom"
        geom.type = mujoco.mjtGeom.mjGEOM_SPHERE  # pyright: ignore[reportAttributeAccessIssue]
        geom.size = [ball.radius, 0, 0]
        geom.mass = ball.mass
        geom.rgba = ball.rgba
        geom.condim = ball.condim
        geom.friction = ball.friction
        # geom.solref = [-1000, -100]
        geom.solimp = [1.0, 1.0, 0.001, 0.5, 2.0]

        model = spec.compile()
        data = mujoco.MjData(model)  # pyright: ignore[reportAttributeAccessIssue]

        self._ball_body_id = mujoco.mj_name2id(  # pyright: ignore[reportAttributeAccessIssue]
            model, mujoco.mjtObj.mjOBJ_BODY, "ball"
        )
        ball_jnt_id = mujoco.mj_name2id(  # pyright: ignore[reportAttributeAccessIssue]
            model, mujoco.mjtObj.mjOBJ_JOINT, "ball_joint"
        )
        self._ball_jnt_qpos_addr = int(model.jnt_qposadr[ball_jnt_id])
        logger.info(
            f"Ball added: body_id={self._ball_body_id}, "
            f"qpos_addr={self._ball_jnt_qpos_addr}, nq={model.nq}"
        )
        return model, data

    @property
    def ball_enabled(self) -> bool:
        return self._ball_body_id >= 0

    def set_ball_pos(self, pos: np.ndarray, quat: np.ndarray | None = None):
        """Teleport the ball to *pos* (xyz) and zero its velocity."""
        if not self.ball_enabled:
            return
        addr = self._ball_jnt_qpos_addr
        self.data.qpos[addr: addr + 3] = pos
        if quat is not None:
            self.data.qpos[addr + 3: addr + 7] = quat
        else:
            self.data.qpos[addr + 3: addr + 7] = [1, 0, 0, 0]

        vel_addr = self.model.jnt_dofadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "ball_joint")  # pyright: ignore[reportAttributeAccessIssue]
        ]
        self.data.qvel[vel_addr: vel_addr + 6] = 0.0

    def get_ball_pos(self) -> np.ndarray:
        if not self.ball_enabled:
            return np.zeros(3)
        return self.data.xpos[self._ball_body_id].copy()

    def release_ball(self):
        """Stop kinematic tracking — ball enters free dynamics."""
        self._ball_released = True
        logger.info("Ball released into free dynamics")

    def hold_ball(self):
        """Re-enable kinematic tracking."""
        self._ball_released = False

    def get_hands_midpoint(self) -> np.ndarray:
        """Return the midpoint between left and right hand collision geoms."""
        left_id = mujoco.mj_name2id(  # pyright: ignore[reportAttributeAccessIssue]
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "left_hand_collision"
        )
        right_id = mujoco.mj_name2id(  # pyright: ignore[reportAttributeAccessIssue]
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "right_hand_collision"
        )
        if left_id < 0 or right_id < 0:
            return np.array(self._ball_cfg.init_pos, dtype=np.float64)
        return (self.data.geom_xpos[left_id] + self.data.geom_xpos[right_id]) / 2.0

    def update_dof_cfg(self, override_cfg=None):
        super().update_dof_cfg(override_cfg=override_cfg)
        if hasattr(self, "model"):
            self._refresh_dof_indices()

    def _refresh_dof_indices(self):
        qpos_indices = []
        qvel_indices = []
        for joint_name in self.joint_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)  # pyright: ignore[reportAttributeAccessIssue]
            if joint_id < 0:
                raise ValueError(f"Joint {joint_name} not found in MuJoCo model")

            qpos_addr = int(self.model.jnt_qposadr[joint_id])
            qvel_addr = int(self.model.jnt_dofadr[joint_id])
            qpos_indices.append(qpos_addr)
            qvel_indices.append(qvel_addr)

        self._dof_qpos_indices = np.asarray(qpos_indices, dtype=np.int32)
        self._dof_qvel_indices = np.asarray(qvel_indices, dtype=np.int32)

    def reborn(self, init_qpos=None):
        if init_qpos is not None:
            self.data.qpos[0:7] = init_qpos
            self.data.qvel[:] = 0.0
            self.data.ctrl[:] = 0.0
            mujoco.mj_forward(self.model, self.data)  # pyright: ignore[reportAttributeAccessIssue]
        elif self.model.nkey > 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)  # pyright: ignore[reportAttributeAccessIssue]
        else:
            mujoco.mj_resetData(self.model, self.data)  # pyright: ignore[reportAttributeAccessIssue]

        if self.ball_enabled:
            self._ball_released = False
            self.set_ball_pos(np.array([0.0, 0.0, -1.0]))
        mujoco.mj_forward(self.model, self.data)  # pyright: ignore[reportAttributeAccessIssue]

    def reset(self):
        if self.born_place_align:  # TODO: merge
            self.born_place_align = False  # disable during reset
            self.update()
            self.born_place_align = True  # enable after reset
            self.set_born_place()
            self.update()

    def set_gains(self, stiffness, damping):
        assert len(stiffness) == self.num_dofs and len(damping) == self.num_dofs
        self.stiffness = np.asarray(stiffness)
        self.damping = np.asarray(damping)

    def self_check(self):
        pass

    def set_born_place(self, quat: np.ndarray | None = None, pos: np.ndarray | None = None):
        quat_ = self.base_quat if quat is None else quat
        pos_ = self.base_pos if pos is None else pos
        super().set_born_place(quat_, pos_)

    def update(self, simple=False):  # TODO: clean sensors in xml
        """simple: only update dof pos & vel"""
        dof_pos = self.data.qpos[self._dof_qpos_indices].astype(np.float32)
        dof_vel = self.data.qvel[self._dof_qvel_indices].astype(np.float32)

        self._dof_pos = dof_pos.copy()
        self._dof_vel = dof_vel.copy()

        if simple:
            return

        quat = self.data.qpos.astype(np.float32)[3:7][[1, 2, 3, 0]]
        ang_vel = self.data.qvel.astype(np.float32)[3:6]
        base_pos = self.data.qpos.astype(np.float32)[:3]
        lin_vel = self.data.qvel.astype(np.float32)[0:3]

        if self.born_place_align:
            quat, base_pos = self.base_align.align_transform(quat, base_pos)

        lin_vel = quat_rotate_inverse_np(quat, lin_vel)
        rpy = quatToEuler(quat)

        self._base_rpy = rpy.copy()
        self._base_quat = quat.copy()
        self._base_ang_vel = ang_vel.copy()

        self._base_pos = base_pos.copy()
        self._base_lin_vel = lin_vel.copy()

        if self.update_with_fk:
            fk_info = self.fk()
            self._fk_info = fk_info.copy()
            self._torso_ang_vel = fk_info[self._torso_name]["ang_vel"]
            self._torso_quat = fk_info[self._torso_name]["quat"]
            self._torso_pos = fk_info[self._torso_name]["pos"]

    def step(self, pd_target, hand_pose=None):
        assert len(pd_target) == self.num_dofs, "pd_target len should be num_dofs of env"

        if hand_pose is not None:
            logger.info("Hand pose-->", hand_pose)

        self.viewer.cam.lookat = self.data.qpos.astype(np.float32)[:3]
        if self.viewer.is_alive:
            self.viewer.render()

        for _ in range(self.sim_decimation):
            torque = (pd_target - self.dof_pos) * self.stiffness - self.dof_vel * self.damping
            torque = np.clip(torque, -self.torque_limits, self.torque_limits)

            self.data.ctrl = torque

            mujoco.mj_step(self.model, self.data)  # pyright: ignore[reportAttributeAccessIssue]
            self.update(simple=True)
        self.update(simple=False)

    def shutdown(self):
        self.viewer.close()


if __name__ == "__main__":
    from robojudo.config.g1.env.g1_mujuco_env_cfg import G1MujocoEnvCfg

    mujoco_env = MujocoEnv(cfg_env=G1MujocoEnvCfg())
    mujoco_env.viewer._paused = False

    while True:
        # mujoco_env.update()
        mujoco_env.step(np.zeros(mujoco_env.num_dofs))
        time.sleep(0.02)
