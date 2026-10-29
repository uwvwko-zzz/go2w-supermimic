"""V6 wheel-local terrain constraints; privileged scans never enter the actor.

Clearance is a geometric surrogate, not a measured persistent-contact detector.
"""
import torch
from .world_model import GRAVITY, LEG_POS, LEG_VEL
from .rolling_rewards import wheel_geometry


def yaw_frame_points(state, points):
  # Remove roll/pitch, retaining the heading frame used by TerrainField.scan.
  up = -torch.nn.functional.normalize(state[:, GRAVITY], dim=-1, eps=1e-6)
  cp = (up[:, 1:].square().sum(-1)+1e-12).sqrt()
  sr, cr, sp = up[:, 1] / cp, up[:, 2] / cp, -up[:, 0]
  x, y, z = points.unbind(-1)
  yr = cr[:, None]*y - sr[:, None]*z
  zr = sr[:, None]*y + cr[:, None]*z
  return torch.stack((cp[:, None]*x + sp[:, None]*zr, yr,
                      -sp[:, None]*x + cp[:, None]*zr), -1)


def sample_scan(scan, xy):
  """Bilinear samples on the existing 21x13, 10cm heading-frame grid."""
  shape = xy.shape[:-1]
  x = ((xy[..., 0]+1.)*10).clamp(0,20).reshape(len(scan),-1)
  y = ((xy[..., 1]+.6)*10).clamp(0,12).reshape(len(scan),-1)
  ix, iy = x.floor().long().clamp(0,19), y.floor().long().clamp(0,11)
  ax, ay = x-ix, y-iy
  def at(dx,dy): return scan.gather(1,(ix+dx)*13+iy+dy)
  return (((1-ax)*at(0,0)+ax*at(1,0))*(1-ay)
          + ((1-ax)*at(0,1)+ax*at(1,1))*ay).reshape(shape)


def wheel_terms(state, command, scan, default_q):
  point, radius = wheel_geometry(state, default_q)
  point = yaw_frame_points(state, point)
  xy = point[..., :2]
  offsets = state.new_tensor([[x,y] for x in (-.15,0.,.15) for y in (-.10,0.,.10)])
  patch_xy = xy[:,:,None,:]+offsets
  samples = sample_scan(scan, patch_xy)
  # Index-based extrema avoid division by the number of equal extrema in
  # compiled backward through interpolated, nearly-flat patches.
  relief = samples.max(dim=-1).values-samples.min(dim=-1).values
  # Outside the scan, never assume flatness and penalize a recovering leg.
  inside = ((patch_xy[...,0].abs()<=1.) & (patch_xy[...,1].abs()<=.6)).all(-1)
  flat = (1.-(relief-.02)/.04).clamp(0,1)*inside
  straight = torch.exp(-(command[:,1]/.25).square()-(command[:,2]).square())
  gate = flat*straight[:,None]
  below = sample_scan(scan,xy)
  bottom = point[...,2]+below-radius
  lift = ((bottom-.005).clamp_min(0)/.05).square()
  tilt = ((state[:,GRAVITY][:,:2].square().sum(-1)+1e-12).sqrt()-.08).div(.22).clamp(0,1)
  motion = state[:,LEG_VEL].reshape(-1,4,3).square().mean(-1)*(1.-.8*tilt[:,None])
  q = state[:,LEG_POS].reshape(-1,4,3).square()
  posture_multiplier = torch.where(inside, 1.-.875*(relief/.12).clamp(0,1), .125)
  posture = 2.*q[:,:,0].sum(-1)+(posture_multiplier*q[:,:,1:].sum(-1)).sum(-1)
  # Only approaching a higher surface: no reward for lifting on flat ground.
  speed = command[:,:2].norm(dim=-1)
  heading_velocity = yaw_frame_points(state, torch.cat((command[:,:2], torch.zeros_like(command[:,:1])), -1)[:,None,:])[:,0,:2]
  direction = heading_velocity/(heading_velocity.square().sum(-1,keepdim=True)+1e-12).sqrt().clamp_min(.1)
  ahead_xy = xy+.16*direction[:,None,:]
  ahead = sample_scan(scan,ahead_xy)
  rise = below-ahead
  approaching = ((rise-.04)/.08).clamp(0,1)*(speed>.1)[:,None]*inside
  deficit = (radius+.015-point[...,2]-ahead).clamp_min(0)/.20
  clearance = (approaching*deficit/(1.+deficit)).mean(-1)
  return dict(unnecessary_wheel_lift=(gate*lift).sum(-1),
              rolling_leg_motion=(gate*motion).mean(-1), leg_posture=posture,
              wheel_obstacle_clearance=clearance)


def env_wheel_term(env, term):
  from .terrain import TerrainBridge,TerrainField
  bridge=getattr(env,'_go2w_posture_bridge',None)
  if bridge is None:
    bridge=TerrainBridge(env,TerrainField.from_env(env));env._go2w_posture_bridge=bridge
  state=bridge.state()
  return wheel_terms(state,env.command_manager.get_command('twist'),
                     bridge.field.scan(state,bridge.pose()),
                     bridge.robot.data.default_joint_pos[0,bridge.legs])[term]
