"""Non-wheel collision termination and pre-reset labels for imagined survival."""
import torch


def illegal_nonwheel_contact(env, force_threshold=10.):
  data = env.scene['nonwheel_ground_contact'].data
  force = data.force_history if data.force_history is not None else data.force
  finite = torch.isfinite(force).flatten(1).all(-1)
  magnitude = torch.nan_to_num(force, nan=0., posinf=0., neginf=0.).norm(dim=-1)
  peak = magnitude.flatten(1).amax(-1)
  hit = (peak > force_threshold) & finite
  # Scene/termination buffers are cleared by automatic reset. These copies
  # survive it, so the learner retains the transition that caused termination.
  env._nonwheel_hit = hit.detach().clone()
  env._nonwheel_label_valid = finite.detach().clone()
  env._nonwheel_peak = peak.detach().clone()
  return hit


def collision_events(batch):
  valid = batch['valid'].squeeze(-1).bool()
  prefix = torch.cat((torch.ones_like(valid[:,:1]), valid.cumprod(1).bool()[:,:-1]),1)
  return ((batch['collision'].squeeze(-1) > .5)
          & batch['collision_valid'].squeeze(-1).bool() & prefix).any(-1)
