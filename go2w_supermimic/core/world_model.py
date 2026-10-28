"""Go2W residual dynamics with position legs and velocity wheels.

Canonical order: FL, FR, RL, RR; hip/thigh/calf per leg, then four wheels.
Actions are normalized policy outputs: leg targets = default + 0.25*a[:12],
wheel targets = clamp(15*a[12:], -35, 35) rad/s. Wheel phase is excluded.
This module intentionally imports nothing from the original Go2 learner.
"""

import torch
from torch import nn

STATE_DIM = 38
ACTION_DIM = 16
CONTEXT_DIM = 7  # four wheel sliding frictions, base COM xyz offset (metres)
POSE_DIM = 3
Z, GRAVITY, LIN_VEL, ANG_VEL = slice(0, 1), slice(1, 4), slice(4, 7), slice(7, 10)
LEG_POS, LEG_VEL, WHEEL_VEL = slice(10, 22), slice(22, 34), slice(34, 38)
FORMAT = "go2w-flat-dynamics-v1"


class RunningNorm(nn.Module):
  def __init__(self, dim):
    super().__init__()
    self.register_buffer("mean", torch.zeros(dim))
    self.register_buffer("var", torch.ones(dim))
    self.register_buffer("count", torch.tensor(1e-4))

  @torch.no_grad()
  def update(self, x):
    x = x.reshape(-1, self.mean.numel())
    if not len(x):
      return
    delta = x.mean(0) - self.mean
    total = self.count + len(x)
    self.var.copy_((self.var * self.count + x.var(0, unbiased=False) * len(x)
                    + delta.square() * self.count * len(x) / total) / total)
    self.mean.add_(delta * len(x) / total)
    self.count.copy_(total)

  @property
  def scale(self):
    return self.var.clamp_min(1e-6).sqrt()

  def forward(self, x):
    return ((x - self.mean) / self.scale).clamp(-10, 10)


class Go2WWorldModel(nn.Module):
  """(state38, action16, previous16, context7) -> (next state38, pose delta3).

  Flat ground only. Linear and angular velocity use the base frame. Planar
  displacement uses the previous yaw frame. No target/chase or terrain scan.
  Policy history is separate from physical state and is not fed to this model.
  """
  def __init__(self, hidden=(512, 512, 512)):
    super().__init__()
    self.hidden = tuple(hidden)
    dims = (STATE_DIM + 2 * ACTION_DIM + CONTEXT_DIM, *hidden, STATE_DIM + POSE_DIM)
    layers = []
    for i, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
      layers.append(nn.Linear(a, b))
      if i < len(dims) - 2:
        layers.append(nn.SiLU())
    self.net = nn.Sequential(*layers)
    self.input_norm = RunningNorm(dims[0])
    self.output_norm = RunningNorm(dims[-1])

  @staticmethod
  def inputs(state, action, previous_action, context):
    for name, x, dim in (("state", state, STATE_DIM), ("action", action, ACTION_DIM),
                         ("previous_action", previous_action, ACTION_DIM),
                         ("context", context, CONTEXT_DIM)):
      if x.ndim != 2 or x.shape != (state.shape[0], dim):
        raise ValueError(f"{name}: expected (batch, {dim}), got {tuple(x.shape)}")
    return torch.cat((state, action, previous_action, context), -1)

  @torch.no_grad()
  def update_stats(self, state, action, previous_action, context, next_state, pose_delta):
    self.input_norm.update(self.inputs(state, action, previous_action, context))
    self.output_norm.update(torch.cat((next_state - state, pose_delta), -1))

  def forward(self, state, action, previous_action, context):
    x = self.inputs(state, action, previous_action, context)
    delta = self.net(self.input_norm(x)) * self.output_norm.scale + self.output_norm.mean
    return state + delta[:, :STATE_DIM], delta[:, STATE_DIM:]

  def loss(self, state, actions, previous_actions, context, targets, poses, valid):
    """Autoregressive window loss; validity must be false at/reset after done.

    Tensor shapes: state[B,38], actions/previous[B,K,16], context[B,7],
    targets[B,K,38], poses[B,K,3], valid[B,K]. A reset invalidates the rest
    of that window, even if later transitions are individually valid.
    """
    mask = valid.bool().cumprod(1).bool()
    total = next(self.parameters()).sum() * 0.0
    for t in range(actions.shape[1]):
      # Do not execute invalid rows: NaN terminal/reset data must not leak
      # into either the loss or gradients through multiplication by zero.
      keep = mask[:, t]
      if not keep.any():
        break
      pred, pose = self(state[keep], actions[keep, t], previous_actions[keep, t], context[keep])
      error = torch.cat((pred - targets[keep, t], pose - poses[keep, t]), -1)
      total = total + (error / self.output_norm.scale).square().mean(-1).sum()
      state = state.clone()
      state[keep] = pred
    return total / mask.sum().clamp_min(1)

  def checkpoint(self):
    return {"format": FORMAT, "hidden": self.hidden, "model": self.state_dict()}

  @classmethod
  def from_checkpoint(cls, checkpoint, device="cpu"):
    if checkpoint.get("format") != FORMAT:
      raise ValueError("Expected a Go2W dynamics checkpoint; Go2 checkpoints are incompatible")
    model = cls(checkpoint["hidden"]).to(device)
    model.load_state_dict(checkpoint["model"])
    return model
