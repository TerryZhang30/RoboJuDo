import logging
import time
from pathlib import Path

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
        self._depth_camera_cfg = cfg_env.depth_camera
        self._ball_body_id: int = -1
        self._ball_jnt_qpos_addr: int = -1
        self._ball_released: bool = False
        self._depth_camera_id: int = -1
        self._depth_renderer: mujoco.Renderer | None = None
        self._depth_image: np.ndarray | None = None
        self._depth_video_writer = None
        self._depth_video_frame_count: int = 0
        self._depth_video_source_frame_count: int = 0
        self._depth_video_atexit_registered: bool = False
        self._depth_image_sim_time: float = -1.0
        self._depth_image_wall_timestamp: float = 0.0
        self._depth_image_frame_id: int = 0
        self._depth_video_last_source_frame_id: int = -1

        if cfg_env.ball.enabled or cfg_env.depth_camera.enabled:
            self.model, self.data = self._build_model_with_extras(cfg_env)
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

        if cfg_env.depth_camera.enabled:
            self._init_depth_renderer()

        if cfg_env.visualize_extras:
            self.visualizer = MujocoVisualizer(self.viewer)
        else:
            self.visualizer = None

        self.last_time = time.time()

        self.update()  # get initial state

    # ------------------------------------------------------------------
    # Ball helpers
    # ------------------------------------------------------------------

    def _build_model_with_extras(self, cfg_env: MujocoEnvCfg):
        """Use MjSpec to inject optional task objects and sensors."""
        spec = mujoco.MjSpec.from_file(cfg_env.xml)  # pyright: ignore[reportAttributeAccessIssue]

        if cfg_env.depth_camera.enabled:
            self._add_depth_camera_to_spec(spec, cfg_env)

        if cfg_env.ball.enabled:
            self._add_ball_to_spec(spec, cfg_env)

        model = spec.compile()
        data = mujoco.MjData(model)  # pyright: ignore[reportAttributeAccessIssue]

        if cfg_env.depth_camera.enabled:
            self._depth_camera_id = mujoco.mj_name2id(  # pyright: ignore[reportAttributeAccessIssue]
                model, mujoco.mjtObj.mjOBJ_CAMERA, cfg_env.depth_camera.camera_name
            )
            if self._depth_camera_id < 0:
                raise ValueError(f"Depth camera {cfg_env.depth_camera.camera_name} was not added to MuJoCo model")
            logger.info(
                f"Depth camera added: camera_id={self._depth_camera_id}, "
                f"name={cfg_env.depth_camera.camera_name}"
            )

        if cfg_env.ball.enabled:
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

    def _add_ball_to_spec(self, spec, cfg_env: MujocoEnvCfg):
        ball = cfg_env.ball
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
        if ball.solref is not None:
            geom.solref = ball.solref
        geom.solimp = ball.solimp
        if hasattr(geom, "margin"):
            geom.margin = ball.margin

    def _add_depth_camera_to_spec(self, spec, cfg_env: MujocoEnvCfg):
        camera = cfg_env.depth_camera
        link_body = spec.worldbody.find_child(camera.link_body_name)

        if link_body is None:
            if not camera.create_link_body:
                raise ValueError(f"Depth camera link body {camera.link_body_name} not found in MuJoCo model")

            parent_body = spec.worldbody.find_child(camera.parent_body_name)
            if parent_body is None:
                raise ValueError(f"Depth camera parent body {camera.parent_body_name} not found in MuJoCo model")

            link_body = parent_body.add_body()
            link_body.name = camera.link_body_name
            link_body.pos = camera.pos
            logger.info(
                f"Created synthetic depth camera link body {camera.link_body_name} "
                f"under {camera.parent_body_name} at {camera.pos}"
            )

        cam = link_body.add_camera()
        cam.name = camera.camera_name
        cam.pos = [0.0, 0.0, 0.0]
        cam.quat = camera.quat
        cam.fovy = camera.fovy
        cam.resolution = [camera.width, camera.height]

    def _init_depth_renderer(self):
        camera = self._depth_camera_cfg
        self._depth_renderer = mujoco.Renderer(
            self.model,
            height=camera.height,
            width=camera.width,
        )
        self._depth_renderer.enable_depth_rendering()
        self._depth_image = np.full((camera.height, camera.width, 1), camera.depth_max, dtype=np.float32)

    def _depth_frame_period(self) -> float:
        fps = float(self._depth_camera_cfg.fps)
        if fps <= 0.0:
            return 0.0
        return 1.0 / fps

    def _render_depth_image(self) -> np.ndarray:
        camera = self._depth_camera_cfg
        if not camera.enabled:
            return np.empty((0, 0, 1), dtype=np.float32)
        if self._depth_renderer is None:
            self._init_depth_renderer()

        sim_time = float(self.data.time)
        frame_period = self._depth_frame_period()
        elapsed = sim_time - self._depth_image_sim_time
        if (
            self._depth_image is not None
            and self._depth_image_sim_time >= 0.0
            and elapsed >= 0.0
            and frame_period > 0.0
            and elapsed < frame_period - 1e-9
        ):
            return self._depth_image.copy()

        assert self._depth_renderer is not None
        self._depth_renderer.update_scene(self.data, camera=self._depth_camera_id)
        depth = self._depth_renderer.render().astype(np.float32)
        depth = np.nan_to_num(depth, nan=camera.depth_max, posinf=camera.depth_max, neginf=-camera.depth_max)
        depth = np.abs(depth)
        depth = np.clip(depth, camera.depth_min, camera.depth_max)
        self._depth_image = depth[..., None]
        self._depth_image_sim_time = sim_time
        self._depth_image_wall_timestamp = time.time()
        self._depth_image_frame_id += 1
        return self._depth_image.copy()

    def _depth_to_rgb_uint8(self, depth_image: np.ndarray) -> np.ndarray:
        camera = self._depth_camera_cfg
        depth = np.asarray(depth_image, dtype=np.float32)
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        depth = np.nan_to_num(depth, nan=camera.depth_max, posinf=camera.depth_max, neginf=-camera.depth_max)
        depth = np.abs(depth)
        depth = np.clip(depth, camera.depth_min, camera.depth_max)
        denom = max(camera.depth_max - camera.depth_min, 1e-6)
        gray = ((depth - camera.depth_min) / denom * 255.0).clip(0, 255).astype(np.uint8)
        return np.repeat(gray[..., None], 3, axis=-1)

    def _init_depth_video_writer(self):
        if self._depth_video_writer is not None:
            return

        import atexit
        import cv2

        camera = self._depth_camera_cfg
        path = Path(camera.record_video_path)
        if path.suffix.lower() != ".webm":
            path = path.with_suffix(".webm")
        path.parent.mkdir(parents=True, exist_ok=True)

        codecs = [camera.record_video_codec, "VP90", "VP80"]
        for codec in dict.fromkeys(codecs):
            if len(codec) != 4:
                logger.warning(f"Skipping invalid WebM codec {codec!r}; codec must be 4 characters")
                continue
            writer = cv2.VideoWriter(
                str(path),
                cv2.VideoWriter_fourcc(*codec),
                float(camera.record_video_fps),
                (int(camera.width), int(camera.height)),
            )
            if writer.isOpened():
                self._depth_video_writer = writer
                logger.info(f"Recording first-person depth video to {path} using codec={codec}")
                break
            writer.release()

        if self._depth_video_writer is None:
            raise RuntimeError(f"Failed to create first-person WebM writer at {path}")

        if not self._depth_video_atexit_registered:
            atexit.register(self._close_depth_video_writer)
            self._depth_video_atexit_registered = True

    def _record_depth_video_frame(self, depth_image: np.ndarray):
        camera = self._depth_camera_cfg
        if not camera.record_video:
            return
        if self._depth_image_frame_id <= self._depth_video_last_source_frame_id:
            return
        self._depth_video_last_source_frame_id = self._depth_image_frame_id

        stride = max(1, int(camera.record_video_every))
        self._depth_video_source_frame_count += 1
        if (self._depth_video_source_frame_count - 1) % stride != 0:
            return

        self._init_depth_video_writer()
        assert self._depth_video_writer is not None
        frame = self._depth_to_rgb_uint8(depth_image)
        self._depth_video_writer.write(frame)
        self._depth_video_frame_count += 1

    def _close_depth_video_writer(self):
        if self._depth_video_writer is None:
            return
        self._depth_video_writer.release()
        self._depth_video_writer = None
        logger.info(f"Closed first-person depth video after {self._depth_video_frame_count} frames")

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
        mujoco.mj_forward(self.model, self.data)  # pyright: ignore[reportAttributeAccessIssue]

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

    def _body_world_pos_with_offset(self, body_name: str, local_offset: np.ndarray) -> np.ndarray | None:
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)  # pyright: ignore[reportAttributeAccessIssue]
        if body_id < 0:
            logger.warning(f"Ball random init center body {body_name!r} not found; ignoring it.")
            return None
        body_pos = np.asarray(self.data.xpos[body_id], dtype=np.float64)
        body_rot = np.asarray(self.data.xmat[body_id], dtype=np.float64).reshape(3, 3)
        return body_pos + body_rot @ local_offset

    def _ball_random_init_center(self, pos: np.ndarray) -> np.ndarray:
        center_mode = self._ball_cfg.random_init_center
        center = pos.copy()

        if center_mode == "base":
            center[:2] = np.asarray(self.data.qpos[:2], dtype=np.float64)
        elif center_mode == "bodies":
            local_offset = np.asarray(self._ball_cfg.random_init_center_body_offset, dtype=np.float64).reshape(3)
            body_positions = [
                body_pos
                for body_name in self._ball_cfg.random_init_center_body_names
                if (body_pos := self._body_world_pos_with_offset(body_name, local_offset)) is not None
            ]
            if body_positions:
                center[:2] = np.mean(body_positions, axis=0)[:2]
            else:
                logger.warning("No valid ball random init center bodies found; falling back to BallCfg.init_pos.")
        elif center_mode != "config":
            raise ValueError(f"Unsupported ball random init center mode: {center_mode}")

        return center

    def _sample_ball_init_pos(self) -> np.ndarray:
        pos = np.asarray(self._ball_cfg.init_pos, dtype=np.float64).copy()
        if not self._ball_cfg.randomize_init_pos:
            return pos

        center = self._ball_random_init_center(pos)
        xy_range = np.asarray(self._ball_cfg.random_init_xy_range, dtype=np.float64).reshape(-1)
        if xy_range.shape[0] == 1:
            xy_range = np.repeat(xy_range, 2)
        if xy_range.shape[0] != 2:
            raise ValueError("BallCfg.random_init_xy_range must contain one value or [x_range, y_range].")

        xy_offset = np.random.uniform(-0.5 * xy_range, 0.5 * xy_range)
        pos[:2] = center[:2] + xy_offset
        pos[2] = max(pos[2], self._ball_cfg.radius + self._ball_cfg.margin)
        return pos

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

    def reborn(self, init_qpos=None, init_dof_pos=None):
        if init_qpos is not None:
            self.data.qpos[0:7] = init_qpos
            self.data.qvel[:] = 0.0
            self.data.ctrl[:] = 0.0
            mujoco.mj_forward(self.model, self.data)  # pyright: ignore[reportAttributeAccessIssue]
        elif self.model.nkey > 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)  # pyright: ignore[reportAttributeAccessIssue]
        else:
            mujoco.mj_resetData(self.model, self.data)  # pyright: ignore[reportAttributeAccessIssue]

        if init_dof_pos is not None:
            init_dof_pos = np.asarray(init_dof_pos, dtype=np.float64).reshape(-1)
            if init_dof_pos.shape[0] != self.num_dofs:
                raise ValueError(
                    f"init_dof_pos length {init_dof_pos.shape[0]} does not match env num_dofs {self.num_dofs}"
                )
            self.data.qpos[self._dof_qpos_indices] = init_dof_pos
            self.data.qvel[:] = 0.0
            self.data.ctrl[:] = 0.0

        if self.ball_enabled:
            self._ball_released = False
            mujoco.mj_forward(self.model, self.data)  # pyright: ignore[reportAttributeAccessIssue]
            self.set_ball_pos(self._sample_ball_init_pos())
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
        if self._depth_camera_cfg.enabled and self._depth_camera_cfg.record_video:
            self._record_depth_video_frame(self._render_depth_image())

    def get_data(self):
        env_data = super().get_data()
        if self._depth_camera_cfg.enabled:
            depth_image = self._render_depth_image()
            env_data["depth_image"] = depth_image
            env_data["image"] = depth_image
            env_data["camera_timestamp"] = self._depth_image_wall_timestamp
            env_data["camera_sim_time"] = self._depth_image_sim_time
        return env_data

    def shutdown(self):
        self._close_depth_video_writer()
        if self._depth_renderer is not None:
            self._depth_renderer.close()
            self._depth_renderer = None
        self.viewer.close()


if __name__ == "__main__":
    from robojudo.config.g1.env.g1_mujuco_env_cfg import G1MujocoEnvCfg

    mujoco_env = MujocoEnv(cfg_env=G1MujocoEnvCfg())
    mujoco_env.viewer._paused = False

    while True:
        # mujoco_env.update()
        mujoco_env.step(np.zeros(mujoco_env.num_dofs))
        time.sleep(0.02)
