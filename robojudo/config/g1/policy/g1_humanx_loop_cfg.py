from robojudo.config import ASSETS_DIR
from robojudo.policy.policy_cfgs import HumanxPolicyCfg
from robojudo.policy.policy_cfgs import HumanxLoopPolicyCfg

from .g1_asap_policy_cfg import G1_29AsapDoF


class G1_29HumanxLoopDoF(G1_29AsapDoF):
    """Humanx-specific PD parameters, inherits joint structure from ASAP"""

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


class G1HumanxLoopPolicyCfg(HumanxLoopPolicyCfg):
    robot: str = "g1"
    policy_name: str = "jumpshot"
    freq: int = 100
    disable_autoload: bool = True

    policy_file_override: str | None = None

    obs_dof: G1_29HumanxLoopDoF = G1_29HumanxLoopDoF()
    action_dof: G1_29HumanxLoopDoF = G1_29HumanxLoopDoF()

    @property
    def policy_file(self) -> str:
        if self.policy_file_override:
            return self.policy_file_override
        return (ASSETS_DIR / f"models/{self.robot}/humanx/{self.policy_name}.onnx").as_posix()

    actions_scale: float = 0.25
    action_clip: float = 100.0

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

    actor_obs_keys: list[str] = [
        "base_ang_vel",
        "projected_gravity",
        "dof_pos",
        "dof_vel",
        "actions",
        "pd_error",
        "history_obs_buf",
        "ref_motion_phase",
    ]

    dof_names: list[str] = G1_29HumanxLoopDoF().joint_names

    dof_effort_limits: list[float] = [
        139.0, 139.0, 88.0, 139.0, 35.0, 35.0,
        139.0, 139.0, 88.0, 139.0, 35.0, 35.0,
        88.0, 35.0, 35.0,
        25.0, 25.0, 25.0, 25.0, 25.0, 13.4, 13.4,
        25.0, 25.0, 25.0, 25.0, 25.0, 13.4, 13.4,
    ]

    motion_adjustments: dict[int, float] = {}
