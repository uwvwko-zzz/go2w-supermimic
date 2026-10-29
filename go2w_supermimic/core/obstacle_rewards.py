"""Bounded directional motion near uneven ground, shared by real and imagined steps."""
import torch
from .world_model import LIN_VEL


def obstacle_gate(scan):
  local = scan.reshape(-1, 21, 13)[:, 5:16, 3:10].flatten(1)
  return ((local.amax(-1) - local.amin(-1) - .02) / .08).clamp(0., 1.)


def obstacle_progress(state, command, scan):
  speed = command[:, :2].norm(dim=-1)
  along = (state[:, LIN_VEL][:, :2] * command[:, :2]).sum(-1) / speed.clamp_min(.1)
  # Signed and capped: no benefit for wheel spin, pure yaw, lifting or overspeed.
  return obstacle_gate(scan) * (speed > .1) * (along / speed.clamp_min(.2)).clamp(-1., 1.)


def env_obstacle_progress(env):
  from .terrain import TerrainField, TerrainBridge
  from .stair_curriculum import buffers
  bridge = getattr(env, '_go2w_posture_bridge', None)
  if bridge is None:
    bridge = TerrainBridge(env, TerrainField.from_env(env))
    env._go2w_posture_bridge = bridge
  state = bridge.state()
  scan = bridge.field.scan(state, bridge.pose())
  command = env.command_manager.get_command('twist')
  buffers(env)
  stalled = ((command[:, :2].norm(dim=-1) > .1)
             & (state[:, LIN_VEL][:, :2].norm(dim=-1) < .05)
             & (obstacle_gate(scan) > .5))
  env._stair_stall_time = torch.where(stalled, env._stair_stall_time + env.step_dt, 0.)
  env._stair_stuck |= env._stair_stall_time >= 3.
  return obstacle_progress(state, command, scan)
