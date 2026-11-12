"""Privileged dynamics context for vision domain randomization."""
import torch
from go2w_supermimic.core.terrain import TerrainBridge

class VisionBridge(TerrainBridge):
  def context(self):
    base=super().context()
    model=self.env.sim.model; ids=self.robot.indexing
    # Actual model parameters, not merely their randomization multipliers.
    fields=(('body_mass',ids.body_ids,1),('body_inertia',ids.body_ids,2),
            ('actuator_gainprm',ids.ctrl_ids,2),('actuator_biasprm',ids.ctrl_ids,2),
            ('actuator_forcerange',ids.ctrl_ids,2),('dof_frictionloss',ids.joint_v_adr,1))
    values=[base]
    self.context_slices={}
    cursor=base.shape[-1]
    for name,index,ndim in fields:
      value=getattr(model,name)
      value=value[:,index] if value.ndim==ndim+1 else value[index][None].expand(self.env.num_envs,*value[index].shape)
      self.context_slices[name]=(cursor,cursor+value.flatten(1).shape[-1])
      cursor+=value.flatten(1).shape[-1]
      values.append(value.flatten(1))
    self.context_slices['delay']=(cursor,cursor+16)
    values.append(getattr(self.env,'_vision_delay',torch.zeros(self.env.num_envs,16,device=self.env.device)).float())
    yaw=self.robot.data.heading_w
    values.append(torch.stack((yaw.cos(),yaw.sin()),-1))
    values.append(getattr(self.env,'_vision_push',torch.zeros(self.env.num_envs,5,device=self.env.device)))
    return torch.cat(values,-1)
