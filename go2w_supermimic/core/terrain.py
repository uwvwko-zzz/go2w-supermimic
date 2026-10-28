"""Shared static height atlas and privileged, differentiable yaw-frame scans."""
import numpy as np
import torch
from .bridge import Go2WBridge

SCAN_DIM = 21 * 13

class TerrainField:
  def __init__(self, heights, origin, resolution=.04):
    self.h, self.origin, self.res = heights, origin, resolution
    x,y = torch.meshgrid(torch.linspace(-1,1,21,device=heights.device),
                         torch.linspace(-.6,.6,13,device=heights.device), indexing="ij")
    self.offsets = torch.stack((x.flatten(),y.flatten()),-1)

  def sample(self, xy):
    # Clamp *integer* neighbour indices too. NaN survives float clamp and
    # casts to INT64_MIN; on CUDA that would poison the entire device context.
    finite = torch.isfinite(xy).all(-1)
    clean = torch.where(torch.isfinite(xy), xy, torch.zeros_like(xy))
    gx = ((clean[...,0]-self.origin[0])/self.res).clamp(0,self.h.shape[1]-1)
    gy = ((clean[...,1]-self.origin[1])/self.res).clamp(0,self.h.shape[0]-1)
    x = gx.floor().long().clamp(0,self.h.shape[1]-2)
    y = gy.floor().long().clamp(0,self.h.shape[0]-2)
    a,b = gx-x,gy-y
    value = ((1-a)*self.h[y,x]+a*self.h[y,x+1])*(1-b)+((1-a)*self.h[y+1,x]+a*self.h[y+1,x+1])*b
    # Do not disguise corrupt states as valid ground: the learner will discard
    # this update, while real invalid environments are reset before querying.
    return torch.where(finite, value, torch.full_like(value, float("nan")))

  def scan(self, state, pose):
    c,s = pose[:,2:3].cos(),pose[:,2:3].sin()
    dx,dy = self.offsets[:,0],self.offsets[:,1]
    xy = pose[:,None,:2]+torch.stack((c*dx-s*dy,s*dx+c*dy),-1)
    return (state[:,:1]+self.sample(pose[:,:2])[:,None]-self.sample(xy)).clamp(-1.5,1.5)

  @classmethod
  def from_env(cls, env, resolution=.04):
    """Rasterize exact oriented boxes in small bounding rectangles, no per-pixel rays.

    Shared CPU build is ~6 MB at 4 cm for a 48 x 48 m atlas. No robot geometry.
    Only these static box courses are supported; fail if new shapes are introduced.
    """
    import mujoco
    model = env.sim.mj_model
    data = mujoco.MjData(model)
    mujoco.mj_forward(model,data)
    robot_ids = set(env.scene["robot"].indexing.geom_ids.cpu().tolist())
    ids = [i for i in range(model.ngeom) if i not in robot_ids and model.geom_contype[i]]
    corners = np.array([[a,b,c] for a in (-1,1) for b in (-1,1) for c in (-1,1)])
    boxes=[]
    for i in ids:
      if model.geom_type[i] != mujoco.mjtGeom.mjGEOM_BOX:
        raise ValueError(f"Unsupported terrain geometry: {model.geom(i).name}")
      center, rot, size = data.geom_xpos[i],data.geom_xmat[i].reshape(3,3),model.geom_size[i]
      points = (corners*size) @ rot.T+center
      boxes.append((center.copy(),rot.copy(),size.copy(),points.min(0),points.max(0)))
    lo=np.min([b[3][:2] for b in boxes],axis=0)-resolution
    hi=np.max([b[4][:2] for b in boxes],axis=0)+resolution
    shape=np.ceil((hi-lo)/resolution).astype(int)+1
    heights=np.zeros((shape[1],shape[0]),dtype=np.float32)
    for center,rot,size,low,high in boxes:
      start=np.maximum(0,np.floor((low[:2]-lo)/resolution).astype(int))
      end=np.minimum(shape,np.ceil((high[:2]-lo)/resolution).astype(int)+1)
      xs=lo[0]+np.arange(start[0],end[0])*resolution
      ys=lo[1]+np.arange(start[1],end[1])*resolution
      x,y=np.meshgrid(xs,ys)
      # Intersect vertical line p=(x,y,z) with all three oriented-box slabs.
      lower=np.full_like(x,-np.inf); upper=np.full_like(x,np.inf); valid=np.ones_like(x,dtype=bool)
      for k in range(3):
        base=rot[0,k]*(x-center[0])+rot[1,k]*(y-center[1])-rot[2,k]*center[2]
        slope=rot[2,k]
        if abs(slope)<1e-10:
          valid &= np.abs(base)<=size[k]+1e-9
        else:
          z1=(-size[k]-base)/slope; z2=(size[k]-base)/slope
          lower=np.maximum(lower,np.minimum(z1,z2)); upper=np.minimum(upper,np.maximum(z1,z2))
      surface=np.where(valid & (lower<=upper),upper,0)
      patch=heights[start[1]:end[1],start[0]:end[0]]
      np.maximum(patch,surface,out=patch)
    return cls(torch.as_tensor(heights,device=env.device),tuple(lo),resolution)


def advance_pose(pose, delta):
  c,s=pose[:,2].cos(),pose[:,2].sin()
  return pose+torch.stack((c*delta[:,0]-s*delta[:,1],s*delta[:,0]+c*delta[:,1],delta[:,2]),-1)

class TerrainBridge(Go2WBridge):
  def __init__(self, env, field):
    super().__init__(env)
    self.field=field

  def state(self):
    state=super().state()
    state[:,0]-=self.field.sample(self.robot.data.root_link_pos_w[:,:2])
    return state
