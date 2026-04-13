from robojudo.config import ASSETS_DIR, Config
from robojudo.policy.policy_cfgs import PolicyCfg

from .g1_humanx_policy_cfg import G1_29HumanxDoF

class G1_29StandDoF(G1_29HumanxDoF):
    """Stand-specific PD parameters, inherits joint structure from Humanx"""

    stiffness: list[float] = [
        100, 100, 100, 150, 40, 40,
        100, 100, 100, 150, 40, 40,
        200, 200, 200,
        100, 100, 100, 50, 50, 50, 50,
        100, 100, 100, 50, 50, 50, 50,
    ]

    damping: list[float] = [
        2, 2, 2, 4, 2, 2,
        2, 2, 2, 4, 2, 2,
        4, 4, 4,
        2, 2, 2, 2, 2, 2, 2,
        2, 2, 2, 2, 2, 2, 2,
    ]

class G1StandPolicyCfg(PolicyCfg):
    """Anti-fall stand policy config for G1 (29-dof, ONNX, 96-dim obs)."""

    class ObsScalesCfg(Config):
        base_lin_vel: float = 2.0
        base_ang_vel: float = 0.25
        dof_pos: float = 1.0
        dof_vel: float = 0.05

    policy_type: str = "StandPolicy"
    robot: str = "g1"
    policy_name: str = "model_stand"
    freq: int = 100
    disable_autoload: bool = True

    policy_file_override: str | None = None

    obs_dof: G1_29StandDoF = G1_29StandDoF()
    action_dof: G1_29StandDoF = G1_29StandDoF()

    actions_scale: float = 0.25
    action_clip: float | None = 100.0

    obs_scales: ObsScalesCfg = ObsScalesCfg()

    dof_names: list[str] = G1_29StandDoF().joint_names

    dof_effort_limits: list[float] = [
        139.0, 139.0, 88.0, 139.0, 35.0, 35.0,
        139.0, 139.0, 88.0, 139.0, 35.0, 35.0,
        88.0, 35.0, 35.0,
        25.0, 25.0, 25.0, 25.0, 25.0, 13.4, 13.4,
        25.0, 25.0, 25.0, 25.0, 25.0, 13.4, 13.4,
    ]

    @property
    def policy_file(self) -> str:
        if self.policy_file_override:
            return self.policy_file_override
        return (ASSETS_DIR / f"models/{self.robot}/humanx/{self.policy_name}.onnx").as_posix()
