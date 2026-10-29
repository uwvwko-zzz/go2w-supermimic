"""V5 episode outcomes and performance-gated velocity curriculum.

All buffers are training-only metadata, never actor observations.
"""
import torch

KINDS = ('flat', 'up', 'down', 'platform', 'slope', 'rubble')
MIN_ITERATIONS = (0, 3000, 8000, 15000, 22000)


def buffers(env):
  if not hasattr(env, '_stair_stats'):
    env._stair_stats = torch.zeros(6, 6, 4, device=env.device)  # trials, pass, fall, stuck
    env._stair_stall_time = torch.zeros(env.num_envs, device=env.device)
    env._stair_stuck = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  return env._stair_stats


def stair_pass(offset, gravity, level, up):
  """Require body/rear clearance of the last riser, correct landing height and stability."""
  heights = offset.new_tensor([.02, .04, .08, .12, .16, .20])[level]
  expected = torch.where(up, 5 * heights, -5 * heights)
  return ((offset[:, 0].abs() > 2.8) & (offset[:, 1].abs() < 2.5)
          & (offset[:, 2] > expected + .20) & (offset[:, 2] < expected + .60)
          & (gravity[:, :2].norm(dim=-1) < .4))


def curriculum(env, env_ids):
  stats = buffers(env)
  terrain = env.scene.terrain
  d = env.scene['robot'].data
  offset = d.root_link_pos_w[env_ids] - env.scene.env_origins[env_ids]
  levels = terrain.terrain_levels[env_ids].clone()
  types = terrain.terrain_types[env_ids]
  # Six fixed columns, generated in KINDS order.
  stairs = (types == 1) | (types == 2)
  active = env.episode_length_buf[env_ids] > 0
  fallen = env.termination_manager.terminated[env_ids]
  success = (offset[:, 0].abs() > 5.4) & (offset[:, 1].abs() < 6.5)
  top = stair_pass(offset, d.projected_gravity_b[env_ids], levels, types == 1)
  # Landing contacts rule out simply jumping through the clearance test.
  contact = env.scene['wheel_ground_contact'].data.found[env_ids].reshape(len(env_ids), -1).gt(0).sum(-1) >= 2
  success = torch.where(stairs, top & contact, success) & active & ~fallen
  command = env.command_manager.get_command('twist')[env_ids]
  traversal = (command[:, 0].abs() > .1) & (command[:, 1:].abs().amax(-1) < .05)
  trials = (~stairs | traversal) & active & ((env.episode_length_buf[env_ids] * env.step_dt >= 5.) | fallen)
  values = torch.stack((trials, success & trials, fallen & trials,
                        env._stair_stuck[env_ids] & trials), -1).float()
  stats.view(-1, 4).index_add_(0, types * 6 + levels, values)
  terrain.terrain_levels[env_ids] = (levels + success.long() - (active & fallen).long()).clamp(0, 5)
  terrain.env_origins[env_ids] = terrain.terrain_origins[terrain.terrain_levels[env_ids], types]
  env._stair_stall_time[env_ids] = 0
  env._stair_stuck[env_ids] = False
  return terrain.terrain_levels.float().mean()


def next_stage(stage, iteration, stats, velocity_mae):
  """Minimum age AND sufficient actual stair passes, never time alone."""
  if stage >= 4 or iteration < MIN_ITERATIONS[stage + 1]:
    return stage
  # Stage increases require success at 4, 8, 12, then 16 cm respectively.
  trials, passed = stats[1, stage + 1, :2]
  accurate = max(velocity_mae[:2]) < .18 and velocity_mae[2] < .30
  return stage + int(trials >= 64 and passed / max(trials, 1) >= .60 and accurate)


def summary(env):
  stats = buffers(env).detach().cpu().tolist()
  levels, types = env.scene.terrain.terrain_levels, env.scene.terrain.terrain_types
  result = {}
  for i, name in enumerate(KINDS):
    selected = levels[types == i]
    result[name] = dict(level_counts=torch.bincount(selected, minlength=6).cpu().tolist(),
                       episodes_by_level=stats[i])
  return result
