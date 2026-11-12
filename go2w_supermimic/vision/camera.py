"""Camera randomization and held-frame transport, shared by collection and imagination."""
import math
import torch
import torch.nn.functional as F
from .randomization import DR,uniform,randint
from .depth import quat_matrix

PARAM_DIM=14
POSE_HISTORY=3

def camera_parameters(n,device,randomize=True):
  p=torch.zeros(n,PARAM_DIM,device=device)
  p[:,7:9]=1.; p[:,11]=1.; p[:,13]=.01
  if not randomize: return p
  p[:,:3]=uniform((n,3),(-DR.camera_translation_m,DR.camera_translation_m),device)
  p[:,3:6]=uniform((n,3),(-math.radians(DR.camera_rotation_deg),math.radians(DR.camera_rotation_deg)),device)
  p[:,6]=uniform((n,),(-DR.camera_fovy_delta_deg,DR.camera_fovy_delta_deg),device)
  p[:,7:9]=uniform((n,2),DR.focal_scale,device)
  p[:,9:11]=uniform((n,2),(-DR.principal_point_px,DR.principal_point_px),device)
  p[:,11]=randint((n,),DR.camera_period_steps,device)
  p[:,12]=randint((n,),DR.camera_latency_steps,device)
  p[:,13]=uniform((n,),DR.blur_sigma,device)
  return p


def installation_rotation(angles):
  r,p,y=angles.unbind(-1)
  cr,sr,cp,sp,cy,sy=r.cos(),r.sin(),p.cos(),p.sin(),y.cos(),y.sin()
  return torch.stack((cy*cp,cy*sp*sr-sy*cr,cy*sp*cr+sy*sr,
      sy*cp,sy*sp*sr+cy*cr,sy*sp*cr-cy*sr,-sp,cp*sr,cp*cr),-1).reshape(-1,3,3)


def corrupt_depth(image,params,config):
  """Meters BxHxW: edge artifacts, distance noise, blur, near failure, missing pixels."""
  close=image<DR.near_failure_m
  gradient=(F.max_pool2d(image[:,None],3,1,1)+F.max_pool2d(-image[:,None],3,1,1))[:,0]
  edge=(gradient.detach()>.08)&(torch.rand_like(image)<DR.edge_dropout)
  std=config.noise_std+DR.range_noise_quadratic*image.detach().square()
  image=image+torch.randn_like(image)*std
  image=torch.where(edge,config.far,image)
  # Per-image 3x3 Gaussian, small enough to avoid convolution weight expansion.
  # Replicate borders instead of blending invalid padded zero depth.
  patches=F.unfold(F.pad(image[:,None],(1,1,1,1),mode='replicate'),3).reshape(len(image),9,*image.shape[-2:])
  grid=torch.tensor([2.,1.,2.,1.,0.,1.,2.,1.,2.],device=image.device)
  weights=torch.softmax(-grid[None]/(2*params[:,13:14].square().clamp_min(1e-4)),dim=-1)
  image=(patches*weights[:,:,None,None]).sum(1)
  missing=close|(torch.rand_like(image)<config.dropout)
  return torch.where(missing,config.far,image).clamp(config.near,config.far)


def transport(renderer,pose,fifo,image,clock,params,noise=True,initial=None):
  fifo=torch.cat((pose[:,None],fifo[:,:-1]),1)
  if initial is not None:
    fifo=torch.where(initial[:,None,None],pose[:,None].expand_as(fifo),fifo)
  delayed=fifo[torch.arange(len(pose),device=pose.device),params[:,12].long()]
  capture=(clock.long()%params[:,11].long().clamp_min(1))==0
  deliver=capture & (torch.rand_like(clock)>=DR.frame_drop)
  if initial is not None: deliver|=initial
  # Render all samples to retain static shapes and gradients; delivered frames are held.
  fresh=renderer.render(delayed[:,:3],delayed[:,3:].reshape(-1,3,3),noise=noise,params=params)
  image=torch.where(deliver[:,None,None,None],fresh,image)
  return fifo,image,clock+1


class CameraRuntime:
  def __init__(self,env,renderer,randomize):
    self.env,self.renderer,self.randomize=env,renderer,randomize
    n=env.num_envs; dev=env.device
    self.params=camera_parameters(n,dev,randomize)
    self.fifo=torch.zeros(n,POSE_HISTORY,12,device=dev)
    c=renderer.config
    self.image=torch.ones(n,1,c.height,c.width,device=dev)
    self.clock=torch.zeros(n,device=dev)
    self.initial=torch.ones(n,device=dev,dtype=torch.bool)
    self.tick=None

  def reset(self,ids):
    self.params[ids]=camera_parameters(len(ids),self.env.device,self.randomize)
    self.initial[ids]=True; self.clock[ids]=0

  def observe(self,noise=True):
    tick=getattr(self.env,'_sim_step_counter',0)
    if self.tick==tick and not self.initial.any(): return self.image
    d=self.env.scene['robot'].data
    pose=torch.cat((d.root_link_pos_w,quat_matrix(d.root_link_quat_w).flatten(1)),-1)
    self.fifo,self.image,self.clock=transport(self.renderer,pose,self.fifo,self.image,
        self.clock,self.params,noise,self.initial)
    self.initial.zero_(); self.tick=tick
    return self.image

  def snapshot(self):
    return dict(camera_params=self.params.clone(),camera_fifo=self.fifo.flatten(1).clone(),
                camera_clock=self.clock[:,None].clone(),
                camera_image=(self.image.flatten(1)*255).round().byte())
