from curses import use_default_colors
from robojudo.config import cfg_registry
from robojudo.controller.ctrl_cfgs import (
    JoystickCtrlCfg,  # noqa: F401
    KeyboardCtrlCfg,  # noqa: F401
    UnitreeCtrlCfg,  # noqa: F401
)
from robojudo.pipeline.pipeline_cfgs import (
    RlLocoMimicPipelineCfg,  # noqa: F401
    RlMultiPolicyPipelineCfg,  # noqa: F401
    RlPipelineCfg,  # noqa: F401
)

from .ctrl.g1_beyondmimic_ctrl_cfg import G1BeyondmimicCtrlCfg  # noqa: F401
from .ctrl.g1_motion_ctrl_cfg import (  # noqa: F401
    G1MotionCtrlCfg,
    G1MotionH2HCtrlCfg,
    G1MotionKungfuBotCtrlCfg,
    G1MotionTwistCtrlCfg,
)
from .ctrl.g1_twist_redis_ctrl_cfg import G1TwistRedisCtrlCfg  # noqa: F401
from .env.g1_dummy_env_cfg import G1DummyEnvCfg  # noqa: F401
from .env.g1_mujuco_env_cfg import G1_12MujocoEnvCfg, G1_23MujocoEnvCfg, G1MujocoEnvCfg  # noqa: F401
from .env.g1_real_env_cfg import G1RealEnvCfg, G1UnitreeCfg  # noqa: F401
from .policy.g1_amo_policy_cfg import G1AmoPolicyCfg  # noqa: F401
from .policy.g1_asap_policy_cfg import G1AsapLocoPolicyCfg, G1AsapPolicyCfg  # noqa: F401
from .policy.g1_beyondmimic_policy_cfg import G1BeyondMimicPolicyCfg  # noqa: F401
from .policy.g1_h2h_policy_cfg import G1H2HPolicyCfg  # noqa: F401
from .policy.g1_kungfubot_policy_cfg import G1KungfuBotGeneralPolicyCfg, G1KungfuBotPolicyCfg  # noqa: F401
from .policy.g1_smooth_policy_cfg import G1SmoothPolicyCfg  # noqa: F401
from .policy.g1_twist_policy_cfg import G1TwistPolicyCfg  # noqa: F401
from .policy.g1_unitree_policy_cfg import G1UnitreePolicyCfg, G1UnitreeWoGaitDoF, G1UnitreeWoGaitPolicyCfg  # noqa: F401
from .policy.g1_humanx_policy_cfg import G1HumanxPolicyCfg  # noqa: F401
from .policy.g1_pickball_bc_policy_cfg import G1PickballBCPolicyCfg  # noqa: F401
from robojudo.environment.env_cfgs import BallCfg, DepthCameraCfg, RealDepthCameraCfg  # noqa: F401
from .policy.g1_stand_policy_cfg import G1StandPolicyCfg  # noqa: F401
from .policy.g1_humanx_loop_cfg import G1HumanxLoopPolicyCfg  # noqa: F401


# ======================== Basic Configs ======================== #
@cfg_registry.register
class g1(RlPipelineCfg):
    """
    Unitree G1 robot configuration, Unitree Policy, Sim2Sim.
    You can modify to play with other policies and controllers.
    """

    robot: str = "g1"
    env: G1MujocoEnvCfg = G1MujocoEnvCfg()
    # env: G1_23MujocoEnvCfg = G1_23MujocoEnvCfg()
    # env: G1_12MujocoEnvCfg = G1_12MujocoEnvCfg()

    ctrl: list[JoystickCtrlCfg | KeyboardCtrlCfg] = [  # note: the ranking of controllers matters
        JoystickCtrlCfg(),
        # KeyboardCtrlCfg(),
    ]

    # policy: G1UnitreePolicyCfg = G1UnitreePolicyCfg()
    # policy: G1UnitreeWoGaitPolicyCfg = G1UnitreeWoGaitPolicyCfg()
    policy: G1AmoPolicyCfg = G1AmoPolicyCfg()

    # run_fullspeed: bool = env.is_sim


@cfg_registry.register
class g1_real(g1):
    """
    Unitree G1 robot, Unitree Policy, Sim2Real.
    To extend the sim2sim config to sim2real, just need to change the env to real env.
    """

    # env: G1DummyEnvCfg = G1DummyEnvCfg()
    env: G1RealEnvCfg = G1RealEnvCfg(
        # env_type="UnitreeEnv",  # For unitree_sdk2py
        env_type="UnitreeCppEnv",  # For unitree_cpp, check README for more details
        unitree=G1UnitreeCfg(
            net_if="eth0",  # note: change to your network interface
        ),
    )

    ctrl: list[UnitreeCtrlCfg] = [
        UnitreeCtrlCfg(),
    ]

    do_safety_check: bool = True  # enable safety check for real robot


@cfg_registry.register
class g1_switch(RlMultiPolicyPipelineCfg):
    """
    Multi-policy pipeline: switch between loco, mimic, and stand policies.
    """

    robot: str = "g1"
    env: G1MujocoEnvCfg = G1MujocoEnvCfg(
        ball=BallCfg(
            enabled=False,
            radius=0.11,
            mass=0.1,
            rgba=[0.85, 0.45, 0.15, 1.0],
            init_pos=[0.19, -0.03, 1.08],
            hand_offset_z=0.01,
        ),
    )

    switch_prepare_duration_s: float = 0.05

    ctrl: list[KeyboardCtrlCfg | JoystickCtrlCfg] = [
        KeyboardCtrlCfg(
            triggers_extra={
                "Key.tab": "[POLICY_TOGGLE]",
                "1": "[POLICY_SWITCH],0",  # Shift+1 -> loco (AMO)
                "2": "[POLICY_SWITCH],1",  # Shift+2 -> mimic (humanx)
                "3": "[POLICY_SWITCH],2",  # Shift+3 -> stand (防跌倒)
                "t": "[POSE_TOGGLE]",
            }
        ),
        JoystickCtrlCfg(
            triggers_extra={
                "RB+Down": "[POLICY_SWITCH],0",
                "RB+Up": "[POLICY_SWITCH],1",
                "RB+Left": "[POLICY_SWITCH],2",
            }
        ),
    ]

    policies: list[G1AmoPolicyCfg | G1HumanxPolicyCfg | G1StandPolicyCfg] = [
        G1AmoPolicyCfg(
            motion_adjustments={},
            motion_data_path="/home/zzx/Documents/RoboJuDo/assets/motions/g1/humanx/jumpshot.pkl",
            use_motion_as_default_pose=False,
        ),
        G1HumanxPolicyCfg(
            policy_file_override="/home/zzx/Documents/RoboJuDo/assets/models/g1/humanx/model_689000.onnx",
        ),
        G1StandPolicyCfg(
            policy_file_override="/home/zzx/Documents/RoboJuDo/assets/models/g1/humanx/model_27000.onnx",
        ),
    ]


@cfg_registry.register
class g1_locomimic(RlLocoMimicPipelineCfg):
    """
    Example of loco mimic pipeline configuration.
    You can switch between loco and mimic policies during runtime, with interpolation.
    === Check more fancy locomimic examples in g1_loco_mimic_cfg.py ===
    """

    robot: str = "g1"
    env: G1MujocoEnvCfg = G1MujocoEnvCfg()

    ctrl: list[KeyboardCtrlCfg | JoystickCtrlCfg] = [
        KeyboardCtrlCfg(
            triggers_extra={
                "]": "[POLICY_LOCO]",
                "[": "[POLICY_MIMIC]",
            }
        ),
        JoystickCtrlCfg(
            triggers_extra={
                "RB+Down": "[POLICY_LOCO]",
                "RB+Up": "[POLICY_MIMIC]",
            }
        ),
    ]

    loco_policy: G1AmoPolicyCfg = G1AmoPolicyCfg()
    mimic_policies: list[G1HumanxPolicyCfg] = [
        G1HumanxPolicyCfg(),
    ]


# ======================== Configs for supported Policy ======================== #


@cfg_registry.register
class g1_h2h(RlPipelineCfg):
    """
    Human2Humanoid
    """

    robot: str = "g1"
    env: G1MujocoEnvCfg = G1MujocoEnvCfg()
    ctrl: list[KeyboardCtrlCfg | G1MotionH2HCtrlCfg] = [
        KeyboardCtrlCfg(),
        G1MotionH2HCtrlCfg(),
    ]

    policy: G1H2HPolicyCfg = G1H2HPolicyCfg()


@cfg_registry.register
class g1_beyondmimic(RlPipelineCfg):
    """
    BeyondMimic Policy, support both with and without state estimator.
    """

    robot: str = "g1"
    env: G1MujocoEnvCfg = G1MujocoEnvCfg()
    ctrl: list[KeyboardCtrlCfg] = [
        KeyboardCtrlCfg(),
    ]

    policy: G1BeyondMimicPolicyCfg = G1BeyondMimicPolicyCfg(
        policy_name="Jump_wose",
        without_state_estimator=True,
        use_modelmeta_config=True,  # use robot dof config from modelmeta
        use_motion_from_model=True,  # use motion from onnx model
        max_timestep=140,
    )


@cfg_registry.register
class g1_beyondmimic_with_ctrl(RlPipelineCfg):
    """
    BeyondMimic with External BeyondMimicCtrl as motion source.
    """

    robot: str = "g1"
    env: G1MujocoEnvCfg = G1MujocoEnvCfg()
    ctrl: list[KeyboardCtrlCfg | G1BeyondmimicCtrlCfg] = [
        KeyboardCtrlCfg(),
        G1BeyondmimicCtrlCfg(
            motion_name="dance1_subject2",  # you can put your own motion file in assets/motions/g1
        ),
    ]

    policy: G1BeyondMimicPolicyCfg = G1BeyondMimicPolicyCfg(
        policy_name="Dance_wose",
        use_motion_from_model=False,  # use motion from BeyondmimicCtrl instead of the onnx
    )


@cfg_registry.register
class g1_asap(RlPipelineCfg):
    """
    Unitree G1 robot configuration, ASAP Policy, Sim2Sim.
    You can modify to play with other policies and controllers.
    """

    robot: str = "g1"
    env: G1MujocoEnvCfg = G1MujocoEnvCfg(forward_kinematic=None, update_with_fk=False, born_place_align=True)

    ctrl: list[JoystickCtrlCfg | KeyboardCtrlCfg] = [  # note: the ranking of controllers matters
        # JoystickCtrlCfg(),
        KeyboardCtrlCfg(triggers={"i": "[SIM_REBORN]", "o": "[SHUTDOWN]", "r": "[MOTION_RESET]"}),
    ]

    policy: G1AsapPolicyCfg = G1AsapPolicyCfg()
    """You can also try other models, from ASAP, RoboMimic, KungfuBot(PBHC)"""
    # policy: G1KungfuBotPolicyCfg = G1KungfuBotPolicyCfg() # KungfuBot horse_squat
    # # fmt: off
    # policy: G1AsapPolicyCfg = G1AsapPolicyCfg(
    #     policy_name="robomimic",
    #     relative_path="dance_0605.onnx",
    #     motion_length_s=18.0,
    #     start_upper_body_dof_pos = [
    #         0, 0, 0,
    #         0.35, 0.18, 0, 0.87,
    #         0.35, -0.18, 0, 0.87,
    #     ],
    # )
    # # fmt: on


@cfg_registry.register
class g1_asap_loco(RlPipelineCfg):
    """
    Unitree G1 robot configuration, ASAP Locomotion Policy, Sim2Sim.
    You can modify to play with other policies and controllers.
    """

    robot: str = "g1"
    env: G1MujocoEnvCfg = G1MujocoEnvCfg(forward_kinematic=None, update_with_fk=False, born_place_align=False)

    ctrl: list[JoystickCtrlCfg | KeyboardCtrlCfg] = [  # note: the ranking of controllers matters
        # JoystickCtrlCfg(),
        KeyboardCtrlCfg(
            triggers={
                "i": "[SIM_REBORN]",
                "o": "[SHUTDOWN]",
            }
        ),
    ]

    policy: G1AsapLocoPolicyCfg = G1AsapLocoPolicyCfg()


@cfg_registry.register
class g1_kungfubot2(RlPipelineCfg):
    """
    PBHC KungfuBot2 General Policy
    """

    robot: str = "g1"
    env: G1MujocoEnvCfg = G1MujocoEnvCfg()
    ctrl: list[KeyboardCtrlCfg | G1MotionKungfuBotCtrlCfg] = [
        KeyboardCtrlCfg(),
        G1MotionKungfuBotCtrlCfg(
            motion_name="kungfubot/Horse-stance_pose",  # put motion files in assets/motions/g1/phc/kungfubot
        ),
    ]

    policy: G1KungfuBotGeneralPolicyCfg = G1KungfuBotGeneralPolicyCfg(
        policy_name="horse_test_43000",  # this is a test model trained with only one motion
        compatibility_old_version=True,  # for old version of kungfubot general policy (before 2025-11-13 bugfix #68)
    )


@cfg_registry.register
class g1_twist(RlPipelineCfg):
    """
    Unitree G1 robot configuration, TWIST Policy, Sim2Sim.
    TwistRedisCtrl for the original repo of high level motion stream over redis.
    MotionTwistCtrl for built-in motion control.
    """

    robot: str = "g1"
    env: G1MujocoEnvCfg = G1MujocoEnvCfg(forward_kinematic=None, update_with_fk=False, born_place_align=False)

    ctrl: list[G1TwistRedisCtrlCfg | G1MotionTwistCtrlCfg] = [  # note: the ranking of controllers matters
        G1TwistRedisCtrlCfg(redis_host="localhost"),  # with hign level motion lib through redis
        # G1MotionTwistCtrlCfg(), # with built-in motion ctrl
    ]

    policy: G1TwistPolicyCfg = G1TwistPolicyCfg()


# ======================== Fancy Example Configs ======================== #


@cfg_registry.register
class g1_switch_beyondmimic(RlMultiPolicyPipelineCfg):
    """
    Switch between multiple BeyondMimic policies. Withour Interpolation.
    """

    robot: str = "g1"
    env: G1MujocoEnvCfg = G1MujocoEnvCfg()
    ctrl: list[KeyboardCtrlCfg | JoystickCtrlCfg] = [
        KeyboardCtrlCfg(
            triggers_extra={
                "Key.tab": "[POLICY_TOGGLE]",
                "!": "[POLICY_SWITCH],0",  # note: with shift
                "@": "[POLICY_SWITCH],1",  # note: with shift
                "#": "[POLICY_SWITCH],2",  # note: with shift
                "$": "[POLICY_SWITCH],3",  # note: with shift
            }
        ),
        JoystickCtrlCfg(
            triggers_extra={
                "RB+Down": "[POLICY_SWITCH],0",
                "RB+Left": "[POLICY_SWITCH],1",
                "RB+Up": "[POLICY_SWITCH],2",
                "RB+Right": "[POLICY_SWITCH],3",
            }
        ),
    ]

    policies: list[G1AmoPolicyCfg | G1BeyondMimicPolicyCfg] = [
        G1AmoPolicyCfg(        motion_adjustments={
            -6: 0.4,       # right_shoulder_roll  → obs target
            -13: -0.2,     # left_shoulder_roll   → obs target
            -8: -0.3,      # left_wrist_yaw       → direct set
            4: 0.0,        # left_ankle_pitch      → action target
            10: 0.0,       # right_ankle_pitch     → action target
        },),
        G1BeyondMimicPolicyCfg(policy_name="Violin", without_state_estimator=False, max_timestep=500),
        G1BeyondMimicPolicyCfg(policy_name="Waltz", without_state_estimator=False, max_timestep=850),
        G1BeyondMimicPolicyCfg(policy_name="Dance_wose", without_state_estimator=True),
    ]


@cfg_registry.register
class g1_amo_adjusted(RlPipelineCfg):
    """
    AMO Policy with motion adjustments from Humanx.
    Wrist joints are set directly; other joints shift AMO's default targets.
    """

    robot: str = "g1"
    env: G1MujocoEnvCfg = G1MujocoEnvCfg()

    ctrl: list[JoystickCtrlCfg | KeyboardCtrlCfg] = [
        JoystickCtrlCfg(),
        KeyboardCtrlCfg(),
    ]

    policy: G1AmoPolicyCfg = G1AmoPolicyCfg(
        motion_adjustments={
            -6: 0.2,       # right_shoulder_roll  → obs target
            -13: -0.1,     # left_shoulder_roll   → obs target
            -8: -0.3,      # left_wrist_yaw       → direct set
            -1:0.3,        # right_wrist_yaw
            4: 0.05,        # left_ankle_pitch      → action target
            10: 0.05,       # right_ankle_pitch     → action target
        },
        motion_data_path_override="/home/zzx/Documents/RoboJuDo/assets/motions/g1/humanx/jumpshot.pkl" ,
        use_motion_as_default_pose=True,

    )


# TIPS: check g1_loco_mimic_cfg.py for more complex examples


@cfg_registry.register
class g1_humanx(RlPipelineCfg):
    """
    Unitree G1 robot configuration, Humanx Policy for Human-Object Interaction tasks.
    """

    robot: str = "g1"
    sim_slowmo_factor: float = 3.0

    env: G1MujocoEnvCfg = G1MujocoEnvCfg(
        forward_kinematic=None,
        update_with_fk=False,
        born_place_align=True,
        sim_decimation=10,
        ball=BallCfg(
            enabled=True,
            radius=0.11,
            mass=0.5,
            rgba=[0.85, 0.45, 0.15, 1.0],
            init_pos=[0.19, -0.03, 0.88],
            hand_offset_z=0.02,
        ),
    )

    ctrl: list[KeyboardCtrlCfg] = [
        KeyboardCtrlCfg(triggers={"i": "[SIM_REBORN]", "o": "[SHUTDOWN]", "r": "[MOTION_RESET]"}),
    ]

    policy: G1HumanxPolicyCfg = G1HumanxPolicyCfg(
        policy_name="fake_action",
        policy_file_override="/home/zzx/Documents/RoboJuDo/assets/models/g1/humanx/model_689000.onnx",
        motion_adjustments={
            -6: 0.4,
            -13: -0.2,
            -8: -0.3,
            4: 0.1,
            10: 0.1,
        },
        motion_data_path="/home/zzx/Documents/RoboJuDo/assets/motions/g1/humanx/BMaster_fake_action_and_shot_hoi_wsf.pkl",
    )


@cfg_registry.register
class g1_pickball_bc(RlPipelineCfg):
    """
    G1 pickball_bc task using the BC ONNX and matching motion/object data.
    """

    robot: str = "g1"
    sim_slowmo_factor: float = 1.0
    reset_to_policy_init_pose: bool = True

    env: G1MujocoEnvCfg = G1MujocoEnvCfg(
        forward_kinematic=None,
        update_with_fk=False,
        born_place_align=True,
        sim_decimation=12,
        ball=BallCfg(
            enabled=True,
            radius=0.119,
            mass=0.4,
            rgba=[0.85, 0.45, 0.15, 1.0],
            condim=6,
            friction=[3.5, 0.8, 0.05],
            solref=[0.004, 1.2],
            solimp=[0.98, 0.995, 0.001, 0.5, 2.0],
            margin=0.002,
            init_pos=[0.0, 0.0, 0.12],
            randomize_init_pos=True,
            random_init_xy_range=[0.3, 0.3],
            random_init_center="bodies",
            random_init_center_body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
            random_init_center_body_offset=[0.2, 0.0, 0.0],
            hand_offset_z=0.0,
        ),
        depth_camera=DepthCameraCfg(
            enabled=True,
            camera_name="d435_depth",
            link_body_name="d435_link",
            parent_body_name="torso_link",
            create_link_body=True,
            pos=[0.0576235, 0.01753, 0.42987],
            quat=[
                0.6592524821011075,
                0.25570718574871737,
                -0.25570718574871737,
                -0.6592524821011075,
            ],
            width=224,
            height=224,
            fps=50.0,
            fovy=90.0,
            depth_min=0.01,
            depth_max=3.0,
            record_video=True,
            record_video_path="debug_depth_frames/run_pipeline_first_person.webm",
            record_video_fps=50.0,
            record_video_codec="VP90",
            record_video_every=1,
        ),
    )

    ctrl: list[KeyboardCtrlCfg] = [
        KeyboardCtrlCfg(
            triggers={
                "i": "[SIM_REBORN]",
                "o": "[SHUTDOWN]",
                "r": "[MOTION_RESET]",
                "t": "[MOTION_REPLAY]",
            }
        ),
    ]

    policy: G1PickballBCPolicyCfg = G1PickballBCPolicyCfg(
        policy_file_override="/home/zzx/Documents/RoboJuDo/bc_assets/epoch_0050_bc_3k.onnx"
    )


@cfg_registry.register
class pickball_bc(g1_pickball_bc):
    """Alias for the G1 pickball BC MuJoCo task."""


@cfg_registry.register
class g1_pickball_bc_real(g1_pickball_bc):
    """G1 pickball BC task on the real robot with an attached RealSense depth camera."""

    env: G1RealEnvCfg = G1RealEnvCfg(
        env_type="UnitreeCppEnv",
        unitree=G1UnitreeCfg(net_if="eth0", control_dt=0.02),
        real_depth_camera=RealDepthCameraCfg(
            enabled=True,
            camera_type="realsense",
            width=224,
            height=224,
            source_width=640,
            source_height=480,
            fps=60,
            depth_min=0.01,
            depth_max=3.0,
            timeout_ms=1000,
            max_frame_age_s=0.15,
            use_background_thread=True,
        ),
    )

    ctrl: list[UnitreeCtrlCfg] = [
        UnitreeCtrlCfg(
            triggers={
                "L2": "[SHUTDOWN]",
                "Y": "[MOTION_RESET]",
            }
        )
    ]

    policy: G1PickballBCPolicyCfg = G1PickballBCPolicyCfg(
        policy_file_override="/home/zzx/Documents/RoboJuDo/bc_assets/epoch_0050_bc_5k.onnx",
        require_image=True,
        max_image_age_s=0.2,
        max_joint_delta_rad=0.6,
        joint_delta_safety_mode="hold",
    )

    do_safety_check: bool = True
    wait_for_zero_torque_start: bool = True
    prepare_duration_s: float = 2.0
    prepare_reset_before_done: bool = False
    wait_for_start_confirmation: bool = True
    start_confirm_button: str = "L1"
    shutdown_button: str = "L2"
    start_hold_stiffness_scale: float = 3.0
    start_hold_damping_scale: float = 3.0


@cfg_registry.register
class pickball_bc_real(g1_pickball_bc_real):
    """Alias for the real G1 pickball BC task."""



@cfg_registry.register
class g1_humanx_real(g1_humanx):
    """Humanx Policy on Real G1 Robot"""

    env: G1RealEnvCfg = G1RealEnvCfg(
        env_type="UnitreeCppEnv",
        unitree=G1UnitreeCfg(net_if="eth0", control_dt=0.01),
    )

    ctrl: list[UnitreeCtrlCfg] = [
        UnitreeCtrlCfg(
            triggers={
                "L2": "[SHUTDOWN]",
                "X": "[MOTION_FADE_IN]",
                "B": "[MOTION_FADE_OUT]",
                "Y": "[MOTION_RESET]",
            }
        )
    ]
    do_safety_check: bool = True
    wait_for_zero_torque_start: bool = True
    zero_torque_start_button: str = "Start"
    prepare_duration_s: float = 2.0
    prepare_reset_before_done: bool = False
    wait_for_start_confirmation: bool = True
    start_confirm_button: str = "L1"
    shutdown_button: str = "L2"
    start_hold_stiffness_scale: float = 3.0
    start_hold_damping_scale: float = 3.0

@cfg_registry.register
class g1_humanx_amo_adjusted_real(g1_humanx_real):
    """Humanx Policy with AMO-like motion adjustments, on Real G1 Robot"""

    pipeline_type: str = "RlMultiPolicyPipeline"

    switch_prepare_duration_s: float = 1.5

    policies: list[G1AmoPolicyCfg | G1HumanxPolicyCfg] = [
        G1AmoPolicyCfg(
            motion_adjustments={
                -5: 0.3,
                -12: -0.2,
                -1: -0.3,
                4: 0.05,
                10: 0.05,
            },
            use_motion_as_default_pose=False,
        ),
        G1HumanxPolicyCfg(
            policy_file_override="/home/zzx/Documents/RoboJuDo/assets/models/g1/humanx/jumpshot.pt"
            ),
    ]
    
    ctrl: list[UnitreeCtrlCfg] = [
        UnitreeCtrlCfg(
            triggers_extra={
                "X": "[POLICY_SWITCH],0",
                "Y": "[POLICY_SWITCH],1",
                "L2": "[SHUTDOWN]",
                "R2": "[POSE_TOGGLE]",
            }
        ),
    ]

    do_safety_check: bool = True
    wait_for_zero_torque_start: bool = True
    zero_torque_start_button: str = "Start"
    prepare_duration_s: float = 2.0
    prepare_reset_before_done: bool = False
    wait_for_start_confirmation: bool = True
    start_confirm_button: str = "L1"
    shutdown_button: str = "L2"
    start_hold_stiffness_scale: float = 1.0
    start_hold_damping_scale: float = 1.0

@cfg_registry.register
class g1_humanx_loop(RlPipelineCfg):
    """Humanx Loop Policy"""

    robot: str = "g1"
    env: G1MujocoEnvCfg = G1MujocoEnvCfg(
        born_place_align=True,
        sim_decimation=10,
        ball=BallCfg(
            enabled=True,
            radius=0.13,
            mass=0.2,
            rgba=[0.85, 0.45, 0.15, 1.0],
            init_pos=[0.19, -0.03, 0.88],
            hand_offset_z=0.05,
        ),
    )

    ctrl: list[KeyboardCtrlCfg] = [
        KeyboardCtrlCfg(triggers={"i": "[SIM_REBORN]", "o": "[SHUTDOWN]", "r": "[MOTION_RESET]", "t": "[MOTION_REPLAY]"}),
    ]

    policy: G1HumanxLoopPolicyCfg = G1HumanxLoopPolicyCfg(
        policy_file_override="/home/zzx/Documents/RoboJuDo/assets/models/g1/humanx/loop.onnx",
        motion_data_path="/home/zzx/Documents/RoboJuDo/assets/motions/g1/humanx/BMaster_fake_action_and_shot_hoi_loop.pkl",
        # motion_adjustments={
        #     -6: 0.4,
        #     -13: -0.2,
        #     -8: -0.3,
        #     4: 0.1,
        #     10: 0.1,
        # },
    )


@cfg_registry.register
class g1_humanx_loop_real(g1_humanx_loop):
    """Humanx Loop Policy on Real G1 Robot"""
    
    env: G1RealEnvCfg = G1RealEnvCfg(
        env_type="UnitreeCppEnv",
        unitree=G1UnitreeCfg(net_if="eth0", control_dt=0.01),
    )

    ctrl: list[UnitreeCtrlCfg] = [
        UnitreeCtrlCfg(
            triggers={
                "L2": "[SHUTDOWN]",
                "Y": "[MOTION_RESET]",
                "A": "[MOTION_REPLAY]",
            }
        )
    ]

    policy: G1HumanxLoopPolicyCfg = G1HumanxLoopPolicyCfg(
        policy_file_override="/home/zzx/Documents/RoboJuDo/assets/models/g1/humanx/loop.onnx",
        motion_data_path="/home/zzx/Documents/RoboJuDo/assets/motions/g1/humanx/BMaster_fake_action_and_shot_hoi_loop.pkl",
    )

    wait_for_zero_torque_start: bool = True
    zero_torque_start_button: str = "Start"
    prepare_duration_s: float = 2.0
    prepare_reset_before_done: bool = False
    wait_for_start_confirmation: bool = True
    start_confirm_button: str = "L1"
    shutdown_button: str = "L2"
    start_hold_stiffness_scale: float = 3.0
    start_hold_damping_scale: float = 3.0
