"""Shared real/imagined posture objective for the opt-in terrain v2 task."""
import torch

LEG_SCALES = (0.25, 0.6, 0.8) * 4


def posture_cost(q, scan):
  # Only the supporting/approaching ground near the body should relax posture.
  # scan is body height minus ground; its range is independent of body height.
  local = scan.reshape(-1, 21, 13)[:, 5:16, 3:10].flatten(1)
  relief = ((local.amax(-1) - local.amin(-1)) / 0.12).clamp(0., 1.)
  multiplier = 1. - 0.875 * relief
  legs = q.reshape(-1, 4, 3).square()
  # Hip abduction stays constrained even on steps; flexion is freer there.
  return 2. * legs[:, :, 0].sum(-1) + multiplier * legs[:, :, 1:].sum((1, 2))


def env_posture_cost(env, asset_cfg):
  from .terrain import TerrainField, TerrainBridge
  bridge = getattr(env, "_go2w_posture_bridge", None)
  if bridge is None:
    bridge = TerrainBridge(env, TerrainField.from_env(env))
    env._go2w_posture_bridge = bridge
  d = env.scene[asset_cfg.name].data
  q = d.joint_pos[:, asset_cfg.joint_ids] - d.default_joint_pos[:, asset_cfg.joint_ids]
  return posture_cost(q, bridge.field.scan(bridge.state(), bridge.pose()))
