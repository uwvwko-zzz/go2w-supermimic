"""Go2W asset adapter: twelve position-controlled legs and four velocity wheels.

The supplied MJCF and actuator parameters come from unitree_rl_mjlab's
unitree_go2w asset. Wheel mass, inertia and gains have not been calibrated here.
"""

from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg, XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

GO2W_XML = Path(__file__).parent / "model" / "go2w.xml"
LEGS = ("FL", "FR", "RL", "RR")
LEG_JOINT_NAMES = tuple(f"{leg}_{joint}_joint" for leg in LEGS
                        for joint in ("hip", "thigh", "calf"))
WHEEL_JOINT_NAMES = tuple(f"{leg}_wheel_joint" for leg in LEGS)
WHEEL_GEOM_NAMES = tuple(f"{leg}_wheel_collision" for leg in LEGS)


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(GO2W_XML))
  spec.modelname = "go2w"
  # Scene solver options are owned by the task, not the attached robot.
  spec.option.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
  spec.option.impratio = 1.0
  # Name every collision so domain randomisation can select actual contacts.
  for i, geom in enumerate(spec.geoms):
    if geom.contype or geom.conaffinity:
      # Warp MULTICCD cannot handle nonzero margins on these cylinder/box pairs.
      geom.margin = 0.0
      if not geom.name:
        geom.name = f"body_{i}_collision"
  return spec


def get_go2w_robot_cfg() -> EntityCfg:
  """Return a fresh config using mjlab 1.6's XML velocity actuator adapter."""
  return EntityCfg(
    spec_fn=get_spec,
    init_state=EntityCfg.InitialStateCfg(
      pos=(0.0, 0.0, 0.4),
      joint_pos={
        ".*R_hip_joint": 0.1, ".*L_hip_joint": -0.1,
        ".*_thigh_joint": 0.9, ".*_calf_joint": -1.8,
        ".*_wheel_joint": 0.0,
      },
      joint_vel={".*": 0.0},
    ),
    collisions=(CollisionCfg(
      geom_names_expr=(".*_collision",), contype=1, conaffinity=1,
      condim={".*_wheel_collision": 3, "body_.*_collision": 1},
      priority={".*_wheel_collision": 1, "body_.*_collision": 0},
    ),),
    articulation=EntityArticulationInfoCfg(
      actuators=(
        BuiltinPositionActuatorCfg(
          target_names_expr=(".*_hip_joint", ".*_thigh_joint"),
          stiffness=20.0, damping=1.0, effort_limit=23.5, armature=0.01,
        ),
        BuiltinPositionActuatorCfg(
          target_names_expr=(".*_calf_joint",),
          stiffness=40.0, damping=2.0, effort_limit=45.0, armature=0.02,
        ),
        XmlActuatorCfg(target_names_expr=WHEEL_JOINT_NAMES),
      ),
      soft_joint_pos_limit_factor=0.9,
    ),
  )
