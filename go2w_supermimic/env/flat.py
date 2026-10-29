"""Small flat-ground velocity task for the Go2W mixed leg/wheel controller.

This environment is independent of the chase-specific SuperMimic learner.
"""

from copy import deepcopy

import torch

from mjlab.envs import mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg, JointVelocityActionCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp as velocity_mdp
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg

from go2w_supermimic.robot.config import (
  LEG_JOINT_NAMES, WHEEL_JOINT_NAMES, WHEEL_GEOM_NAMES, get_go2w_robot_cfg,
)

ACTOR_HISTORY = 5


def leg_posture_cost(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  d = env.scene[asset_cfg.name].data
  return (d.joint_pos[:, asset_cfg.joint_ids]
          - d.default_joint_pos[:, asset_cfg.joint_ids]).square().sum(-1)


def standstill_cost(env, wheel_cfg: SceneEntityCfg) -> torch.Tensor:
  """Extra braking at zero command, including wheels spinning in place.

  This suppresses velocity; it does not fix a world-frame position or heading.
  """
  command = env.command_manager.get_command("twist")
  standing = command.square().sum(-1) < 1e-6
  d = env.scene["robot"].data
  cost = (d.root_link_lin_vel_b.square().sum(-1)
          + d.root_link_ang_vel_b.square().sum(-1)
          + 0.01 * d.joint_vel[:, wheel_cfg.joint_ids].square().sum(-1))
  return standing * cost


def go2w_velocity_flat_env_cfg(play: bool = False, history: int = ACTOR_HISTORY, actor_wheel_vel: bool = False):
  if history < 1:
    raise ValueError("history must be at least 1")
  cfg = make_velocity_env_cfg()
  cfg.scene.entities = {"robot": get_go2w_robot_cfg()}
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None
  cfg.scene.sensors = ()
  cfg.scene.num_envs = 1  # Training entry point must opt into larger batches.
  cfg.sim.mujoco.cone = "elliptic"
  cfg.sim.mujoco.impratio = 10
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.viewer.body_name = "base_link"
  cfg.viewer.distance = 2.0

  legs = SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES, preserve_order=True)
  wheels = SceneEntityCfg("robot", joint_names=WHEEL_JOINT_NAMES, preserve_order=True)
  cfg.actions = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot", actuator_names=LEG_JOINT_NAMES,
      scale=0.25, use_default_offset=True,
    ),
    "wheel_vel": JointVelocityActionCfg(
      entity_name="robot", actuator_names=WHEEL_JOINT_NAMES,
      scale=15.0, use_default_offset=False, clip={".*": (-35.0, 35.0)},
    ),
  }
  # 49 values/frame, 245 with five frames. No measured wheel speed or oracle
  # linear velocity in the actor; the critic may observe true linear velocity.
  terms = {
    "base_ang_vel": ObservationTermCfg(
      func=mdp.base_ang_vel, noise=UniformNoiseCfg(n_min=-0.2, n_max=0.2)),
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity, noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05)),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel, params={"asset_cfg": legs, "biased": True},
      noise=UniformNoiseCfg(n_min=-0.01, n_max=0.01)),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel, params={"asset_cfg": legs}, scale=0.05,
      noise=UniformNoiseCfg(n_min=-1.5, n_max=1.5)),
    "wheel_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel, params={"asset_cfg": wheels}, scale=0.05,
      noise=UniformNoiseCfg(n_min=-1.5, n_max=1.5)),
    "actions": ObservationTermCfg(func=mdp.last_action),
    "command": ObservationTermCfg(
      func=mdp.generated_commands, params={"command_name": "twist"}),
  }
  critic = deepcopy(terms)
  critic["joint_pos"].params["biased"] = False
  critic["base_lin_vel"] = ObservationTermCfg(func=mdp.base_lin_vel)
  if not actor_wheel_vel:
    terms.pop("wheel_vel")
  cfg.observations = {
    "actor": ObservationGroupCfg(terms=terms, concatenate_terms=True,
                                 enable_corruption=not play, history_length=history),
    "critic": ObservationGroupCfg(terms=critic, concatenate_terms=True,
                                  enable_corruption=False),
  }
  command = cfg.commands["twist"]
  command.heading_command = False
  command.ranges.heading = None
  command.rel_heading_envs = 0.0
  command.rel_standing_envs = 0.2
  command.ranges.lin_vel_x = (-1.0, 1.0)
  command.ranges.lin_vel_y = (-0.2, 0.2)
  command.ranges.ang_vel_z = (-0.5, 0.5)
  cfg.curriculum = {}
  cfg.events.pop("push_robot", None)
  cfg.events["base_com"].params["asset_cfg"].body_names = ("base_link",)
  cfg.events["foot_friction"].params["asset_cfg"].geom_names = WHEEL_GEOM_NAMES
  cfg.events["encoder_bias"].params["asset_cfg"] = legs
  # Wheel rolling must not incur quadruped foot slip / swing height costs.
  cfg.rewards = {
    "track_linear_velocity": cfg.rewards["track_linear_velocity"],
    "track_angular_velocity": cfg.rewards["track_angular_velocity"],
    "upright": RewardTermCfg(func=velocity_mdp.upright, weight=1.0,
                             params={"std": 0.45, "asset_cfg": SceneEntityCfg(
                               "robot", body_names=("base_link",))}),
    "leg_posture": RewardTermCfg(func=leg_posture_cost, weight=-0.2,
                                 params={"asset_cfg": legs}),
    "standstill": RewardTermCfg(func=standstill_cost, weight=-2.0,
                                params={"wheel_cfg": wheels}),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.05),
    "joint_torques_l2": RewardTermCfg(func=mdp.joint_torques_l2, weight=-2e-5),
    "dof_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-1.0,
                                    params={"asset_cfg": legs}),
    "terminated": RewardTermCfg(func=mdp.is_terminated, weight=-20.0),
  }
  cfg.terminations.pop("out_of_terrain_bounds", None)
  if play:
    for event in ("base_com", "encoder_bias", "foot_friction"):
      cfg.events.pop(event, None)
  return cfg
