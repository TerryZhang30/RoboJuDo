from robojudo.config import ASSETS_DIR
from robojudo.policy.policy_cfgs import HumanxPolicyCfg
from robojudo.tools.tool_cfgs import DoFConfig


class G1_29HumanxDoF(DoFConfig):
    joint_names: list[str] = [
        'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint',
        'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
        'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint',
        'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint',
        'waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint',
        'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint',
        'left_elbow_joint', 'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint',
        'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint',
        'right_elbow_joint', 'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint'
    ]

    default_pos: list[float] = [
        -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
        -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ]

    stiffness: list[float] = [
        99.1, 99.1, 40.2, 99.1, 28.5, 28.5,
        99.1, 99.1, 40.2, 99.1, 28.5, 28.5,
        40.2, 28.5, 28.5,
        14.3, 14.3, 14.3, 14.3, 14.3, 8.6, 8.6,
        14.3, 14.3, 14.3, 14.3, 14.3, 8.6, 8.6,
    ]

    damping: list[float] = [
        6.3, 6.3, 2.6, 6.3, 1.8, 1.8,
        6.3, 6.3, 2.6, 6.3, 1.8, 1.8,
        2.6, 1.8, 1.8,
        0.9, 0.9, 0.9, 0.9, 0.9, 0.5, 0.5,
        0.9, 0.9, 0.9, 0.9, 0.9, 0.5, 0.5,
    ]


class G1HumanxPolicyCfg(HumanxPolicyCfg):
    robot: str = "g1"
    policy_name: str = "jumpshot"
    freq: int = 100
    disable_autoload: bool = True

    policy_file_override: str | None = None

    obs_dof: G1_29HumanxDoF = G1_29HumanxDoF()
    action_dof: G1_29HumanxDoF = G1_29HumanxDoF()

    @property
    def policy_file(self) -> str:
        if self.policy_file_override:
            return self.policy_file_override
        return (ASSETS_DIR / f"models/{self.robot}/humanx/{self.policy_name}.pt").as_posix()

    @property
    def motion_data_path(self) -> str:
        if self.motion_data_path is None:
            return (ASSETS_DIR / f"motions/{self.robot}/humanx/{self.policy_name}.pkl").as_posix()
        return self.motion_data_path

    actions_scale: float = 0.25
    action_clip: float = 100.0

    obs_hist_length: dict[str, int] = {
        "base_ang_vel": 4,
        "projected_gravity": 4,
        "actions": 4,
        "dof_pos": 4,
        "dof_vel": 4,
        # "local_curr_ball_position": 4,
    }

    obs_hist_dims: dict[str, int] = {
        "base_ang_vel": 3,
        "projected_gravity": 3,
        "actions": 29,
        "dof_pos": 29,
        "dof_vel": 29,
        # "local_curr_ball_position": 3,
    }

    actor_obs_keys: list[str] = [
        "base_ang_vel",
        "projected_gravity",
        "dof_pos",
        "dof_vel",
        "actions",
        "pd_error",
        # "local_curr_ball_position",
        "history_obs_buf",
        # "local_ref_origin_pos_xy",
        # "dif_ref_heading",
    ]

    dof_names: list[str] = G1_29HumanxDoF().joint_names

    dof_effort_limits: list[float] = [
        139.0, 139.0, 88.0, 139.0, 35.0, 35.0,
        139.0, 139.0, 88.0, 139.0, 35.0, 35.0,
        88.0, 35.0, 35.0,
        25.0, 25.0, 25.0, 25.0, 25.0, 13.4, 13.4,
        25.0, 25.0, 25.0, 25.0, 25.0, 13.4, 13.4,
    ]

    kps: list[float] = [
        99.1, 99.1, 40.2, 99.1, 28.5, 28.5,
        99.1, 99.1, 40.2, 99.1, 28.5, 28.5,
        40.2, 28.5, 28.5,
        14.3, 14.3, 14.3, 14.3, 14.3, 8.6, 8.6,
        14.3, 14.3, 14.3, 14.3, 14.3, 8.6, 8.6,
    ]

    motion_adjustments: dict[int, float] = {}
