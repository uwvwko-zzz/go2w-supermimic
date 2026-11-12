"""Per-joint, episode-constant control-step delay, limited to 0 or 1 step."""
from dataclasses import dataclass
import torch
from mjlab.envs.mdp.actions.actions import (JointPositionAction, JointPositionActionCfg,
                                           JointVelocityAction, JointVelocityActionCfg)
from go2w_supermimic.robot.config import LEG_JOINT_NAMES,WHEEL_JOINT_NAMES

class DelayMixin:
  def process_actions(self,actions):
    names=LEG_JOINT_NAMES+WHEEL_JOINT_NAMES
    indices=[names.index(n) for n in self.target_names]
    delay=getattr(self._env,'_vision_delay',None)
    previous=getattr(self,'_last_command',torch.zeros_like(actions))
    applied=actions if delay is None else torch.where(delay[:,indices]>0,previous,actions)
    super().process_actions(applied)
    # Public action history remains the command, not the unobservable applied target.
    self._raw_actions.copy_(actions)
    self._last_command=actions.clone()

  def reset(self,env_ids=None):
    super().reset(env_ids)
    if hasattr(self,'_last_command'): self._last_command[env_ids]=0

class DelayedPosition(DelayMixin,JointPositionAction): pass
class DelayedVelocity(DelayMixin,JointVelocityAction): pass

@dataclass(kw_only=True)
class DelayedPositionCfg(JointPositionActionCfg):
  def build(self,env): return DelayedPosition(self,env)

@dataclass(kw_only=True)
class DelayedVelocityCfg(JointVelocityActionCfg):
  def build(self,env): return DelayedVelocity(self,env)
