"""Vision-only domain randomization. No mutation of blind configs or model assets."""
from dataclasses import dataclass, asdict
import torch
from mjlab.managers.event_manager import requires_model_fields, RecomputeLevel

@dataclass(frozen=True)
class RandomizationConfig:
  mass_scale: tuple = (.85,1.15)
  inertia_shape_scale: tuple = (.9,1.1)
  payload_kg: tuple = (0.,2.)
  motor_strength: tuple = (.8,1.2)
  kp_scale: tuple = (.8,1.2)
  kd_scale: tuple = (.8,1.2)
  leg_friction_nm: tuple = (0.,.15)
  wheel_friction_nm: tuple = (0.,.05)
  action_delay_steps: tuple = (0,1)  # 20 ms control steps
  push_force_n: tuple = (-35.,35.)
  push_cooldown_steps: tuple = (250,500)
  push_duration_steps: tuple = (5,10)
  camera_translation_m: float = .01
  camera_rotation_deg: float = 3.
  camera_fovy_delta_deg: float = 5.
  focal_scale: tuple = (.95,1.05)
  principal_point_px: float = 1.
  camera_period_steps: tuple = (2,5)
  camera_latency_steps: tuple = (0,2)
  frame_drop: float = .05
  blur_sigma: tuple = (.2,.8)
  edge_dropout: float = .25
  range_noise_quadratic: float = .003
  near_failure_m: float = .18

  def __post_init__(self):
    if not 0 <= self.action_delay_steps[0] <= self.action_delay_steps[1] <= 1:
      raise ValueError('Action delay supports only 0..1 control steps')
    if not 0 <= self.camera_latency_steps[0] <= self.camera_latency_steps[1] <= 2:
      raise ValueError('Camera FIFO supports only 0..2 control steps')
    if not 1 <= self.camera_period_steps[0] <= self.camera_period_steps[1]:
      raise ValueError('Camera period must be positive')
    if not 0 <= self.frame_drop < 1:
      raise ValueError('Invalid frame drop probability')

  def to_dict(self):
    return asdict(self)

DR = RandomizationConfig()

def uniform(shape,bounds,device):
  return bounds[0]+torch.rand(shape,device=device)*(bounds[1]-bounds[0])

def randint(shape,bounds,device):
  return torch.randint(bounds[0],bounds[1]+1,shape,device=device)

@requires_model_fields('body_mass','body_inertia','actuator_gainprm','actuator_biasprm',
                       'actuator_forcerange','dof_frictionloss',recompute=RecomputeLevel.set_const)
def physical_randomization(env,env_ids,config=DR):
  r=env.scene['robot']; idx=r.indexing; model=env.sim.model
  ids=torch.arange(env.num_envs,device=env.device) if env_ids is None else env_ids
  B=len(ids); bodies=idx.body_ids; ctrls=idx.ctrl_ids
  def defaults(name): return env.sim.get_default_field(name)
  density=uniform((B,len(bodies)),config.mass_scale,env.device)
  shape=uniform(density.shape,config.inertia_shape_scale,env.device)
  mass=defaults('body_mass')[bodies][None]*density
  inertia=defaults('body_inertia')[bodies][None]*density[...,None]*shape[...,None]
  base=r.find_bodies('base_link')[0][0]
  payload=uniform((B,),config.payload_kg,env.device)
  mass[:,base]+=payload
  # Small spherical load centered at the base COM; positive, physically valid inertia.
  inertia[:,base]+= (.4*payload*.05**2)[:,None]
  model.body_mass[ids[:,None],bodies]=mass
  model.body_inertia[ids[:,None],bodies]=inertia
  strength=uniform((B,len(ctrls)),config.motor_strength,env.device)
  kp=uniform(strength.shape,config.kp_scale,env.device)
  kd=uniform(strength.shape,config.kd_scale,env.device)
  gain=defaults('actuator_gainprm')[ctrls][None].repeat(B,1,1)
  bias=defaults('actuator_biasprm')[ctrls][None].repeat(B,1,1)
  position=bias[:,:,1]!=0
  gain[:,:,0]*=strength*torch.where(position,kp,kd)
  bias[:,:,1]*=strength*kp
  bias[:,:,2]*=strength*kd
  model.actuator_gainprm[ids[:,None],ctrls]=gain
  model.actuator_biasprm[ids[:,None],ctrls]=bias
  model.actuator_forcerange[ids[:,None],ctrls]=defaults('actuator_forcerange')[ctrls][None]*strength[...,None]
  names=r.joint_names
  bounds=torch.tensor([config.wheel_friction_nm if 'wheel' in n else config.leg_friction_nm for n in names],device=env.device)
  friction=bounds[:,0]+torch.rand(B,len(names),device=env.device)*(bounds[:,1]-bounds[:,0])
  model.dof_frictionloss[ids[:,None],idx.joint_v_adr]=friction


def episode_randomization(env,env_ids,config=DR):
  ids=torch.arange(env.num_envs,device=env.device) if env_ids is None else env_ids
  if not hasattr(env,'_vision_delay'):
    env._vision_delay=torch.zeros(env.num_envs,16,device=env.device,dtype=torch.long)
    env._vision_push=torch.zeros(env.num_envs,5,device=env.device)
  env._vision_delay[ids]=randint((len(ids),16),config.action_delay_steps,env.device)
  env._vision_push[ids]=0
  env._vision_push[ids,4]=randint((len(ids),),config.push_cooldown_steps,env.device).float()


def advance_push(state,config=DR):
  # [world force xyz, remaining active steps, cooldown steps].
  force,left,wait=state[:,:3],state[:,3],state[:,4]
  left=(left-1).clamp_min(0)
  force=torch.where((left>0)[:,None],force,0.)
  wait=torch.where(left>0,wait,wait-1)
  trigger=(left==0)&(wait<=0)
  sample=uniform(force.shape,config.push_force_n,state.device)
  sample=sample*torch.tensor([1.,1.,.3],device=state.device)
  force=torch.where(trigger[:,None],sample,force)
  left=torch.where(trigger,randint(left.shape,config.push_duration_steps,state.device),left)
  wait=torch.where(trigger,randint(wait.shape,config.push_cooldown_steps,state.device),wait)
  return torch.cat((force,left[:,None],wait[:,None]),-1)


def push_step(env,env_ids=None,config=DR):
  env._vision_push=advance_push(env._vision_push,config)
  r=env.scene['robot']; base=r.find_bodies('base_link')[0]
  r.write_external_wrench_to_sim(env._vision_push[:,None,:3],torch.zeros(env.num_envs,1,3,device=env.device),body_ids=base)
