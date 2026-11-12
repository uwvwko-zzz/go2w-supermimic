"""Vision-only checkpoint contract; no fallback to a blind actor."""
import torch
from torch import nn
from go2w_supermimic.core.policy import (
  Network, FRAME_DIM, CRITIC_DIM, frame, push_history, critic_input,
)
from go2w_supermimic.core.world_model import RunningNorm
from .config import DepthConfig

POLICY_FORMAT = 'go2w-vision-policy-v2'

class Actor(nn.Module):
  def __init__(self, history=5, hidden=(256,128,64)):
    super().__init__()
    self.history, self.hidden, self.actor_wheel_vel = history, tuple(hidden), False
    self.norm = RunningNorm(FRAME_DIM * history)
    self.encoder = nn.Sequential(
      nn.Conv2d(1,16,3,stride=2,padding=1), nn.ELU(),
      nn.Conv2d(16,32,3,stride=2,padding=1), nn.ELU(),
      nn.AdaptiveAvgPool2d((3,4)), nn.Flatten(), nn.Linear(384,64), nn.ELU())
    layers=[]
    n=FRAME_DIM*history+64
    for size in hidden:
      layers.extend((nn.Linear(n,size),nn.ELU())); n=size
    layers.append(nn.Linear(n,16))
    self.net=nn.Sequential(*layers)
    nn.init.uniform_(self.net[-1].weight,-.01,.01)
    nn.init.zeros_(self.net[-1].bias)

  def forward(self, history, depth):
    if depth.ndim != 4 or depth.shape[1] != 1:
      raise ValueError('Vision actor requires NCHW depth, in addition to proprioception')
    return self.net(torch.cat((self.norm(history),self.encoder(depth)), -1)).tanh()

def checkpoint_wheel_vel(checkpoint):
  if checkpoint.get('format') != POLICY_FORMAT:
    raise ValueError('Expected vision checkpoint; blind checkpoints are not compatible')
  return False

def load_actor(checkpoint, device='cpu'):
  checkpoint_wheel_vel(checkpoint)
  actor=Actor(checkpoint['history'],checkpoint['actor_hidden']).to(device)
  actor.load_state_dict(checkpoint['actor'])
  return actor.eval()

def evaluation_env_cfg(checkpoint, level=0, kind=None):
  checkpoint_wheel_vel(checkpoint)
  from .env import go2w_vision_env_cfg
  if level not in range(6):
    raise ValueError('Terrain level must be 0..5')
  cfg=go2w_vision_env_cfg(play=True,history=checkpoint['history'],
      seed=checkpoint['config']['seed'],camera=DepthConfig(**checkpoint['camera']))
  cfg.sim.nconmax=max(cfg.sim.nconmax or 0,512)
  cfg.sim.njmax=max(cfg.sim.njmax or 0,2048)
  gen=cfg.scene.terrain.terrain_generator
  gen.num_rows=1
  gen.difficulty_range=(level/5,level/5)
  if kind is not None:
    gen.sub_terrains={kind:gen.sub_terrains[kind]}
  return cfg
