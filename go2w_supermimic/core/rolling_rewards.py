"""Terrain-gated rolling preference, shared by real and imagined rewards."""
import torch
from .world_model import GRAVITY, LEG_POS, LEG_VEL


def rolling_gate(scan, command):
  local = scan.reshape(-1, 21, 13)[:, 5:16, 3:10].flatten(1)
  relief = local.amax(-1) - local.amin(-1)
  # Flat (<2 cm): favor rolling; near obstacles (>=6 cm): allow lifting.
  flat = (1. - (relief - .02) / .04).clamp(0., 1.)
  # Non-steerable wheels may require stepping for lateral/yaw maneuvers.
  straight = torch.exp(-(command[:,1]/.25).square() - (command[:,2]/1.).square())
  return flat * straight


def wheel_bottom_height(state, default_q):
  """Cylinder bottom relative to ground below the base, using Go2W XML FK.

  Only used on near-flat ground by rolling_gate, not as obstacle clearance.
  The cylinder is symmetric about its axle, so wheel phase is unnecessary.
  """
  q = (state[:,LEG_POS] + default_q).reshape(-1,4,3)
  hip, thigh, calf = q.unbind(-1)
  side = state.new_tensor([1.,-1.,1.,-1.])
  x = -.213*thigh.sin() - .2264*(thigh+calf).sin()
  z = -.213*thigh.cos() - .2264*(thigh+calf).cos()
  y = side*(.0955+.0508)
  point = torch.stack((x + state.new_tensor([.1934,.1934,-.1934,-.1934]),
                       y*hip.cos()-z*hip.sin()+side*.0465,
                       y*hip.sin()+z*hip.cos()),-1)
  up = -torch.nn.functional.normalize(state[:,GRAVITY],dim=-1,eps=1e-6)
  axle_z = up[:,1:2]*hip.cos()+up[:,2:3]*hip.sin()
  radius_z = .086*(1.-axle_z.square()).clamp_min(1e-8).sqrt()+.026*axle_z.abs()
  return state[:,:1] + (point*up[:,None,:]).sum(-1) - radius_z


def rolling_costs(state, command, scan, default_q):
  gate = rolling_gate(scan, command)
  # Tolerate 5 mm of clearance; no positive reward for lifting anywhere.
  lift = ((wheel_bottom_height(state,default_q)-.005).clamp_min(0)/.05).square().sum(-1)
  # Reference mjlab task uses mean squared speed and relaxes during recovery.
  tilt = (state[:,GRAVITY][:,:2].norm(dim=-1)-.08).div(.22).clamp(0.,1.)
  motion = state[:,LEG_VEL].square().mean(-1) * (1.-.8*tilt)
  return {'unnecessary_wheel_lift':gate*lift, 'rolling_leg_motion':gate*motion}


def env_rolling_cost(env, term):
  from .terrain import TerrainField, TerrainBridge
  bridge = getattr(env,'_go2w_posture_bridge',None)
  if bridge is None:
    bridge = TerrainBridge(env,TerrainField.from_env(env))
    env._go2w_posture_bridge = bridge
  state = bridge.state()
  return rolling_costs(state, env.command_manager.get_command('twist'),
                       bridge.field.scan(state,bridge.pose()),
                       bridge.robot.data.default_joint_pos[0,bridge.legs])[term]
