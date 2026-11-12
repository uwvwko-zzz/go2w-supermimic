"""Vision replay with privileged randomized dynamics and compressed camera state."""
import torch
from go2w_supermimic.core.replay import Replay as BaseReplay
from .camera import PARAM_DIM,POSE_HISTORY
from .config import DepthConfig

class Replay(BaseReplay):
  def __init__(self,*args,context_dim=7,**kwargs):
    super().__init__(*args,**kwargs)
    base=self.data['context']; t,n=base.shape[:2]
    self.data['context']=torch.zeros(t,n,context_dim,device=base.device)
    c=DepthConfig()
    for key,size in [('camera_params',PARAM_DIM),('camera_fifo',POSE_HISTORY*12),('camera_clock',1),('camera_image',c.width*c.height)]:
      self.data[key]=torch.zeros(t,n,size,device=base.device,dtype=torch.uint8 if key=='camera_image' else torch.float32)
