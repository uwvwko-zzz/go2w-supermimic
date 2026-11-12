"""Terrain-only pinhole depth with a local surface gradient, shared by real and imagined observations.

Private static Warp scene; no physics stepping, no robot self-occlusion in v1.
Forward hits use exact MuJoCo terrain geometry, not the critic's height atlas.
Gradients hold the selected surface fixed (visibility changes are non-differentiable).
"""
import copy
import math
import torch
from .config import DepthConfig


def quat_matrix(q):
  q=torch.nn.functional.normalize(q,dim=-1)
  w,x,y,z=q.unbind(-1)
  return torch.stack((1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),
      2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),
      2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)),-1).reshape(*q.shape[:-1],3,3)


def state_rotation(state, pose):
  # Projected gravity and heading determine roll/pitch/yaw away from pitch singularities.
  g=torch.nn.functional.normalize(state[:,1:4],dim=-1,eps=1e-6)
  roll=torch.atan2(-g[:,1],-g[:,2])
  pitch=torch.asin(g[:,0].clamp(-.99999,.99999))
  yaw=pose[:,2]
  cr,sr,cp,sp,cy,sy=roll.cos(),roll.sin(),pitch.cos(),pitch.sin(),yaw.cos(),yaw.sin()
  return torch.stack((cy*cp,cy*sp*sr-sy*cr,cy*sp*cr+sy*sr,
      sy*cp,sy*sp*sr+cy*cr,sy*sp*cr-cy*sr,-sp,cp*sr,cp*cr),-1).reshape(-1,3,3)


def surface_depth(o,d,forward,dist,normal,config):
  hit=(dist>=0)&(dist<=config.far)
  normal=torch.where(hit[...,None],normal,torch.zeros_like(normal))
  point=o.detach()+dist.clamp_min(0)[...,None]*d.detach()
  nd=(normal*d).sum(-1)
  denominator=torch.where(nd<0,-1.,1.)*nd.detach().abs().clamp_min(.05)+(nd-nd.detach())
  linear=(normal*(point-o)).sum(-1)/denominator
  distance=dist+linear-linear.detach()
  return torch.where(hit,distance*forward,config.far).clamp(config.near,config.far)


class DepthRenderer:
  def __init__(self, config, device, cast):
    self.config,self.device,self.cast=config,device,cast
    y,x=torch.meshgrid(torch.arange(config.height,device=device),torch.arange(config.width,device=device),indexing='ij')
    f=config.height/(2*math.tan(math.radians(config.fovy_deg)/2))
    # Body frame: x forward, y left, z up; image right/down have negative y/z.
    dirs=torch.stack((torch.ones_like(x),(config.width/2-x-.5)/f,(config.height/2-y-.5)/f),-1).float().reshape(-1,3)
    dirs=torch.nn.functional.normalize(dirs,dim=-1)
    self.forward=dirs[:,0].clone()
    a=math.radians(config.pitch_down_deg)
    rotation=torch.tensor([[math.cos(a),0,math.sin(a)],[0,1,0],[-math.sin(a),0,math.cos(a)]],device=device)
    self.dirs=dirs@rotation.T
    self.offset=torch.tensor(config.position,device=device,dtype=torch.float32)

  @classmethod
  def from_env(cls,env,config):
    return cls(config,env.device,StaticScene(env))

  def rays(self,position,rotation,params=None):
    """World rays and camera-forward components shared by depth and GUI overlays."""
    body_rotation=rotation
    forward=self.forward
    dirs=self.dirs
    offset=self.offset
    if params is not None:
      from .camera import installation_rotation
      rotation=rotation@installation_rotation(params[:,3:6])
      offset=offset[None]+params[:,:3]
      c=self.config
      y,x=torch.meshgrid(torch.arange(c.height,device=position.device),torch.arange(c.width,device=position.device),indexing='ij')
      focal=c.height/(2*torch.tan((params[:,6]+c.fovy_deg)*math.pi/360))
      yy=(c.width/2-x.flatten()[None]-.5+params[:,9:10])/(focal[:,None]*params[:,7:8])
      zz=(c.height/2-y.flatten()[None]-.5+params[:,10:11])/(focal[:,None]*params[:,8:9])
      dirs=torch.nn.functional.normalize(torch.stack((torch.ones_like(yy),yy,zz),-1),dim=-1)
      forward=dirs[:,:,0]
      a=math.radians(c.pitch_down_deg)
      tilt=position.new_tensor([[math.cos(a),0,math.sin(a)],[0,1,0],[-math.sin(a),0,math.cos(a)]])
      dirs=dirs@tilt.T
    origin=position+torch.einsum('bij,bj->bi',body_rotation,offset.expand(len(position),3))
    d=torch.einsum('bij,bpj->bpi',rotation,dirs.expand(len(position),-1,3))
    o=origin[:,None,:].expand_as(d)
    return o,d,forward

  def render(self,position,rotation,noise=True,params=None):
    o,d,forward=self.rays(position,rotation,params)
    dist,normal=self.cast(o.detach().contiguous(),d.detach().contiguous())
    image=surface_depth(o,d,forward,dist,normal,self.config)
    if noise and params is not None:
      from .camera import corrupt_depth
      image=corrupt_depth(image.reshape(-1,self.config.height,self.config.width),params,self.config)
    elif noise:
      image=image+torch.randn_like(image)*self.config.noise_std
      image=torch.where(torch.rand_like(image)<self.config.dropout,self.config.far,image)
    image=(image.clamp(self.config.near,self.config.far)-self.config.near)/(self.config.far-self.config.near)
    return image.reshape(-1,1,self.config.height,self.config.width)

  def from_state(self,state,pose,field,noise=True):
    z=state[:,0]+field.sample(pose[:,:2])
    return self.render(torch.cat((pose[:,:2],z[:,None]),-1),state_rotation(state,pose),noise)


class StaticScene:
  def __init__(self,env):
    import mujoco
    import mujoco_warp as mw
    import warp as wp
    self.device=env.device
    model=copy.deepcopy(env.sim.mj_model)
    # Private model, terrain in group 0; hide every robot geom in group 5.
    terrain=model.geom_bodyid==model.body('terrain').id
    model.geom_group[:]=5
    model.geom_group[terrain]=0
    data=mujoco.MjData(model)
    # Rendering needs poses only; full forward would allocate initial contacts
    # and constraints when the default robot pose overlaps the course.
    mujoco.mj_kinematics(model,data)
    with wp.ScopedDevice(self.device):
      self.m=mw.put_model(model)
      self.d=mw.put_data(model,data,nworld=1,nconmax=1,njmax=1)
      self.rc=mw.create_render_context(mjm=model,nworld=1,cam_res=None,
          render_rgb=None,render_depth=None,use_textures=False,use_shadows=False,
          enabled_geom_groups=[0],cam_active=[False]*model.ncam,render_seg=None)
      mw.kinematics(self.m,self.d)
      mw.refit_bvh(self.m,self.d,self.rc)
    self.group=wp.types.vector(length=6,dtype=float)(-1,0,0,0,0,0)
    self.exclude={}

  @torch.compiler.disable
  @torch.no_grad()
  def __call__(self,o,d):
    import warp as wp
    from mujoco_warp import rays
    B,P=o.shape[:2]
    count=B*P
    if count not in self.exclude:
      self.exclude[count]=wp.from_torch(torch.full((count,),-1,device=self.device,dtype=torch.int32),dtype=wp.int32)
    dist=torch.empty(1,count,device=self.device)
    normal=torch.empty(1,count,3,device=self.device)
    geom=torch.empty(1,count,device=self.device,dtype=torch.int32)
    def run():
      rays(m=self.m,d=self.d,pnt=wp.from_torch(o.reshape(1,count,3),dtype=wp.vec3),
          vec=wp.from_torch(d.reshape(1,count,3),dtype=wp.vec3),geomgroup=self.group,
          flg_static=True,bodyexclude=self.exclude[count],dist=wp.from_torch(dist),
          geomid=wp.from_torch(geom),normal=wp.from_torch(normal,dtype=wp.vec3),rc=self.rc)
    with wp.ScopedDevice(self.device):
      if str(self.device).startswith('cuda'):
        with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(self.device))):
          run()
      else:
        run()
    return dist.reshape(B,P),normal.reshape(B,P,3)
