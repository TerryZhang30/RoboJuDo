from typing import Literal

from pydantic import Field

from robojudo.config import ROOT_DIR
from robojudo.policy.policy_cfgs import PickballBCPolicyCfg

from .g1_asap_policy_cfg import G1_29AsapDoF


class G1PickballBCDoF(G1_29AsapDoF):
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


class G1PickballBCPolicyCfg(PickballBCPolicyCfg):
    robot: str = "g1"
    policy_name: str = "pickball_bc"
    freq: int = 80
    phase_dt: float = 1.0 / 60.0

    policy_file_override: str | None = (ROOT_DIR / "bc_assets/last.onnx").as_posix()
    motion_data_path_override: str | None = Field(
        default=(ROOT_DIR / "bc_assets/BMaster_pick_with_ball_cg_modified_20251020.pkl").as_posix(),
        alias="motion_data_path",
    )

    obs_dof: G1PickballBCDoF = G1PickballBCDoF()
    action_dof: G1PickballBCDoF = G1PickballBCDoF()

    @property
    def policy_file(self) -> str:
        if self.policy_file_override:
            return self.policy_file_override
        return (ROOT_DIR / f"bc_assets/{self.policy_name}.onnx").as_posix()

    actions_scale: float = 0.25
    action_clip: float = 100.0
    onnx_device: str = "auto"

    obs_hist_length: dict[str, int] = {
        "base_ang_vel": 4,
        "projected_gravity": 4,
        "actions": 4,
        "dof_pos": 4,
        "dof_vel": 4,
    }

    obs_hist_dims: dict[str, int] = {
        "base_ang_vel": 3,
        "projected_gravity": 3,
        "actions": 29,
        "dof_pos": 29,
        "dof_vel": 29,
    }

    vector_obs_keys: list[str] = [
        "dof_pos",
        "dof_vel",
        "base_ang_vel",
        "projected_gravity",
        "phase",
        "pd_error",
        "actions",
        "history_actor",
    ]

    dof_names: list[str] = G1PickballBCDoF().joint_names

    dof_effort_limits: list[float] = [
        139.0, 139.0, 88.0, 139.0, 35.0, 35.0,
        139.0, 139.0, 88.0, 139.0, 35.0, 35.0,
        88.0, 35.0, 35.0,
        25.0, 25.0, 25.0, 25.0, 25.0, 13.4, 13.4,
        25.0, 25.0, 25.0, 25.0, 25.0, 13.4, 13.4,
    ]

    image_shape: list[int] = [224, 224, 1]
    image_obs_keys: list[str] = ["depth_image", "image", "camera_image"]
    image_fill_value: float = 0.0
    track_motion_ball: bool = False
    release_ball_at_stand_frame: bool = False
    release_ball_after_init: bool = True
    apply_action_scales: bool = True

    max_joint_delta_rad: float | None = 0.6
    joint_delta_safety_mode: Literal["raise", "hold", "zero", "freeze"] = "freeze"
