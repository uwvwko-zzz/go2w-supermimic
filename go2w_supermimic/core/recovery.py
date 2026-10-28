"""Isolate invalid simulation rows before terrain lookup or replay insertion."""
import torch


def invalid_rows(env, obs):
  d = env.scene["robot"].data
  tensors = (d.root_link_pos_w, d.projected_gravity_b, d.root_link_lin_vel_b,
             d.root_link_ang_vel_b, d.joint_pos, d.joint_vel, obs["actor"])
  return ~torch.stack([torch.isfinite(t).flatten(1).all(-1) for t in tensors]).all(0)


def recover_invalid_envs(env, obs, reward):
  bad = invalid_rows(env, obs) | ~torch.isfinite(reward)
  ids = bad.nonzero(as_tuple=False).flatten()
  if ids.numel():
    print(f"RECOVERY reset {ids.numel()} invalid environments; first IDs={ids[:16].tolist()}", flush=True)
    obs, _ = env.reset(env_ids=ids)
    # A failed reset must be handled by the supervisor, never added to replay.
    if invalid_rows(env, obs).any():
      raise RuntimeError("Environment reset failed to restore finite state")
  return obs, bad
