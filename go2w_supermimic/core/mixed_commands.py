"""Explicit straight/side/yaw command modes, with more stair traversal samples."""
from dataclasses import dataclass
import torch
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand, UniformVelocityCommandCfg


def mix_commands(values, stairs):
  result = values.clone()
  mode = torch.rand(len(values),device=values.device)
  # General terrain: 40% pure vx, 20% pure vy, 20% pure yaw, 20% combined.
  result[mode<.4,1:] = 0.
  result[(mode>=.4)&(mode<.6),0] = 0.
  result[(mode>=.4)&(mode<.6),2] = 0.
  result[(mode>=.6)&(mode<.8),:2] = 0.
  # Stairs: 80% low-speed pure forward/backward, 20% full-range commands.
  result[stairs] = values[stairs]
  traverse = stairs & (mode<.8)
  magnitude = .2 + .4*torch.rand(len(values),device=values.device)
  result[traverse,0] = torch.where(values[traverse,0]<0,-magnitude[traverse],magnitude[traverse])
  result[traverse,1:] = 0.
  return result


class MixedVelocityCommand(UniformVelocityCommand):
  def _resample_command(self, env_ids):
    super()._resample_command(env_ids)
    gen = self._env.cfg.scene.terrain.terrain_generator
    stair_columns = torch.tensor([c.kind in ('up','down') for c in gen.sub_terrains.values()],device=self.device)
    stairs = stair_columns[self._env.scene.terrain.terrain_types[env_ids]]
    mixed = mix_commands(self.vel_command_b[env_ids],stairs)
    # Stair magnitudes must obey the current curriculum bounds, including reverse.
    for i, bounds in enumerate((self.cfg.ranges.lin_vel_x, self.cfg.ranges.lin_vel_y, self.cfg.ranges.ang_vel_z)):
      mixed[:,i].clamp_(*bounds)
    self.vel_command_b[env_ids] = mixed
    if self.cfg.hold_stair_command:
      self.time_left[env_ids[stairs]] = self._env.cfg.episode_length_s + 1.



@dataclass(kw_only=True)
class MixedVelocityCommandCfg(UniformVelocityCommandCfg):
  hold_stair_command: bool = False
  def build(self, env):
    return MixedVelocityCommand(self,env)
