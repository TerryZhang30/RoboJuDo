from typing import Literal

from pydantic import model_validator

from robojudo.config import Config
from robojudo.tools.tool_cfgs import DoFConfig, ForwardKinematicCfg, ZedOdometryCfg


class BallCfg(Config):
    """Configuration for a dynamic ball object in simulation."""
    enabled: bool = False
    radius: float = 0.1
    mass: float = 0.4
    rgba: list[float] = [0.85, 0.45, 0.15, 1.0]
    condim: int = 6
    friction: list[float] = [1.5, 0.5, 0.5]
    solref: list[float] | None = None
    solimp: list[float] = [1.0, 1.0, 0.001, 0.5, 2.0]
    margin: float = 0.0
    init_pos: list[float] = [0.0, -0.0, 0.0]
    randomize_init_pos: bool = False
    random_init_xy_range: list[float] = [0.0, 0.0]
    """Full x/y side lengths in meters. [0.15, 0.15] samples within a 15cm x 15cm square."""
    random_init_center: Literal["config", "base", "bodies"] = "config"
    random_init_center_body_names: list[str] = []
    random_init_center_body_offset: list[float] = [0.0, 0.0, 0.0]
    """Local offset added to each center body before averaging, useful for foot sole centers."""
    hand_offset_z: float = 0.20
    """Fallback: place ball this far above the midpoint of both hands."""


class DepthCameraCfg(Config):
    """Configuration for an ego depth camera injected into a MuJoCo model."""

    enabled: bool = False
    camera_name: str = "d435_depth"
    link_body_name: str = "d435_link"
    parent_body_name: str = "torso_link"
    create_link_body: bool = True

    pos: list[float] = [0.0576235, 0.01753, 0.42987]
    """Position of the synthetic d435 link in parent-body coordinates."""

    quat: list[float] = [
        0.6592524821011075,
        0.25570718574871737,
        -0.25570718574871737,
        -0.6592524821011075,
    ]
    """MuJoCo wxyz quat converted from bc_vision eval.py local_euler_zyx=(0, 0.8307767239493009, 0)."""

    width: int = 224
    height: int = 224
    fps: float = 50.0
    """Depth image update frequency in simulation. Use <= 0 to render every request."""
    fovy: float = 90.0

    depth_min: float = 0.0
    depth_max: float = 3.0

    record_video: bool = False
    record_video_path: str = "debug_depth_frames/run_pipeline_first_person.webm"
    record_video_fps: float = 30.0
    record_video_codec: str = "VP90"
    record_video_every: int = 1


class RealDepthCameraCfg(Config):
    """Configuration for a real depth camera attached to the robot compute."""

    enabled: bool = False
    camera_type: Literal["realsense"] = "realsense"

    width: int = 224
    height: int = 224
    source_width: int = 640
    source_height: int = 480
    fps: int = 60

    depth_min: float = 0.01
    depth_max: float = 3.0
    timeout_ms: int = 1000
    max_frame_age_s: float = 0.15
    serial_no: str | None = None
    use_background_thread: bool = True


class EnvCfg(Config):
    env_type: str  # name of the environment class
    is_sim: bool = False

    urdf: str | None = None
    xml: str
    body_names: list[str] | None = None

    dof: DoFConfig

    forward_kinematic: ForwardKinematicCfg | None = None
    update_with_fk: bool = False
    """Whether to update info from fk"""
    torso_name: str = "torso_link"
    """Name of the torso link, used in fk info extraction"""

    born_place_align: bool = True
    """Whether to align the born place to zero position and heading"""


class MujocoEnvCfg(EnvCfg):
    env_type: str = "MujocoEnv"
    is_sim: bool = True
    # ====== ENV CONFIGURATION ======
    sim_duration: float = 60.0
    sim_dt: float = 0.001
    sim_decimation: int = 20

    visualize_extras: bool = True  # TODO: remove

    ball: BallCfg = BallCfg()
    depth_camera: DepthCameraCfg = DepthCameraCfg()


class RobotEnvCfg(EnvCfg):
    env_type: str = "DummyEnv"
    is_sim: bool = False
    # ====== ENV CONFIGURATION ======
    act: bool = True

    odometry_type: Literal["NONE", "DUMMY", "ZED"] = "NONE"
    zed_cfg: ZedOdometryCfg | None = None
    """ZED odometry config, if odometry_type is "ZED", this must be set"""
    real_depth_camera: RealDepthCameraCfg = RealDepthCameraCfg()

    @model_validator(mode="after")
    def check_zed_config(self):
        if self.odometry_type == "ZED" and self.zed_cfg is None:
            raise ValueError("zed_cfg must be set if odometry_type is 'ZED'")
        return self


class UnitreeEnvCfg(RobotEnvCfg):
    """
    Configuration for Unitree Robot environment.
    """

    class UnitreeCfg(Config):
        """Unitree SDK configuration"""

        net_if: str = "eth0"
        """network interface to communicate with the robot"""

        robot: Literal["h1", "g1"]
        msg_type: Literal["hg", "go"]
        control_mode: str = "position"
        hand_type: Literal["Dex-3", "Inspire", "NONE"] = "NONE"

        lowcmd_topic: str = "rt/lowcmd"
        lowstate_topic: str = "rt/lowstate"

        enable_odometry: bool = False
        sport_state_topic: str = "rt/odommodestate"

        control_dt: float = 0.02
        """control command dt"""

    env_type: str = "UnitreeEnv"  # For unitree_sdk2py
    # env_type: str = "UnitreeCppEnv" # For unitree_cpp
    """UnitreeEnv for unitree_sdk2py, UnitreeCppEnv for unitree_cpp, check README for more details"""

    unitree: UnitreeCfg

    odometry_type: Literal["NONE", "DUMMY", "UNITREE", "ZED"] = "DUMMY"  # pyright: ignore[reportIncompatibleVariableOverride]

    joint2motor_idx: list[int] | None = None
    """Mapping from env dof to motor index, None for direct mapping"""
    weak_motor: list[int] = []

    hand_retarget: None = None  # TODO

    @model_validator(mode="after")
    def check_joint2motor_idx(self):
        if self.joint2motor_idx is not None and len(self.joint2motor_idx) != self.dof.num_dofs:
            raise ValueError("joint2motor_idx length must match dof.num_dofs")
        return self
