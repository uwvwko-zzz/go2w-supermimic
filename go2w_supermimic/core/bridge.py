"""Explicit name-based mapping between mjlab and canonical Go2W model tensors."""

import torch

from go2w_supermimic.robot.config import LEG_JOINT_NAMES, WHEEL_JOINT_NAMES, WHEEL_GEOM_NAMES
from .world_model import STATE_DIM, ACTION_DIM


class Go2WBridge:
  def __init__(self, env):
    self.env = env
    self.robot = env.scene["robot"]
    self.legs, _ = self.robot.find_joints(LEG_JOINT_NAMES, preserve_order=True)
    self.wheels, _ = self.robot.find_joints(WHEEL_JOINT_NAMES, preserve_order=True)
    if len(self.legs) != 12 or len(self.wheels) != 4:
      raise ValueError("Go2W requires 12 leg joints and 4 wheel joints")
    pos = env.action_manager.get_term("joint_pos")
    vel = env.action_manager.get_term("wheel_vel")
    if list(env.cfg.actions) != ["joint_pos", "wheel_vel"]:
      raise ValueError("Expected joint_pos then wheel_vel action terms")
    names = list(pos.target_names) + list(vel.target_names)
    canonical = LEG_JOINT_NAMES + WHEEL_JOINT_NAMES
    if len(names) != ACTION_DIM or set(names) != set(canonical):
      raise ValueError("Unexpected Go2W action targets")
    self.to_canonical = [names.index(n) for n in canonical]
    self.to_env = [canonical.index(n) for n in names]
    geoms, _ = self.robot.find_geoms(WHEEL_GEOM_NAMES, preserve_order=True)
    self.geoms = self.robot.indexing.geom_ids[geoms]
    body, _ = self.robot.find_bodies("base_link")
    self.base = self.robot.indexing.body_ids[body[0]]
    # XML reference, not an already-randomized simulator value after reset.
    from go2w_supermimic.robot.config import get_spec
    self.nominal_com = torch.as_tensor(get_spec().body("base_link").ipos.copy(),
                                       device=env.device, dtype=torch.float32)

  def canonical_action(self, action):
    return action[:, self.to_canonical]

  def env_action(self, action):
    return action[:, self.to_env]

  def state(self):
    d = self.robot.data
    s = torch.cat((d.root_link_pos_w[:, 2:3], d.projected_gravity_b,
                   d.root_link_lin_vel_b, d.root_link_ang_vel_b,
                   (d.joint_pos - d.default_joint_pos)[:, self.legs],
                   d.joint_vel[:, self.legs], d.joint_vel[:, self.wheels]), -1)
    assert s.shape[-1] == STATE_DIM
    return s

  def pose(self):
    d = self.robot.data
    return torch.cat((d.root_link_pos_w[:, :2], d.heading_w[:, None]), -1)

  def context(self):
    model = self.env.sim.model
    friction = model.geom_friction
    com = model.body_ipos
    mu = friction[:, self.geoms, 0] if friction.ndim == 3 else friction[self.geoms, 0].expand(self.env.num_envs, -1)
    offset = com[:, self.base] if com.ndim == 3 else com[self.base].expand(self.env.num_envs, -1)
    return torch.cat((mu, offset - self.nominal_com), -1)

  def contacts(self):
    """Endpoint targets: contact flags4, net wheel forces12 in base frame (N).

    Explicit names prevent silent FL/FR swaps in sensor expansion order.
    Terminal/reset endpoints are excluded by replay validity in the learner.
    """
    from mjlab.utils.lab_api.math import quat_apply_inverse
    sensor = self.env.scene["wheel_ground_contact"]
    names = [name.split('/')[-1] for name in sensor.primary_names]
    order = [names.index(f"{leg}_wheel_link") for leg in ("FL","FR","RL","RR")]
    found = (sensor.data.found[:,order] > 0).float().reshape(-1,4)
    force = sensor.data.force[:,order].reshape(-1,4,3)
    quat = self.robot.data.root_link_quat_w[:,None,:].expand(-1,4,-1)
    force = quat_apply_inverse(quat.reshape(-1,4),force.reshape(-1,3)).reshape(-1,12)
    return torch.cat((found,force),-1)


def pose_delta(before, after):
  dxy = after[:, :2] - before[:, :2]
  c, s = before[:, 2].cos(), before[:, 2].sin()
  yaw = after[:, 2] - before[:, 2]
  return torch.stack((c * dxy[:, 0] + s * dxy[:, 1],
                      -s * dxy[:, 0] + c * dxy[:, 1],
                      torch.atan2(yaw.sin(), yaw.cos())), -1)
