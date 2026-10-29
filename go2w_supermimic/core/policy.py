"""Go2W actor history and actor/critic networks, independent of the Go2 learner."""

import torch
from torch import nn
from .world_model import RunningNorm, ANG_VEL, GRAVITY, LEG_POS, LEG_VEL

TERMS = (3, 3, 12, 12, 16, 3)
LEGACY_FRAME_DIM = 53
FRAME_DIM = sum(TERMS)
CRITIC_DIM = 38 + 3 + 7 + 16
POLICY_FORMAT = "go2w-flat-policy-v1"


def frame(state, last_env_action, command, bias, noise=False):
  """Exactly the term order/scaling in env.flat (noise before scaling)."""
  values = [state[:, ANG_VEL], state[:, GRAVITY], state[:, LEG_POS] + bias,
            state[:, LEG_VEL], last_env_action, command]
  widths = (0.2, 0.05, 0.01, 1.5, 0., 0.)
  if noise:
    values = [v + (torch.rand_like(v) * 2 - 1) * w if w else v
              for v, w in zip(values, widths)]
  return torch.cat([v * (0.05 if i == 3 else 1.) for i, v in enumerate(values)], -1)


def push_history(history, latest, length):
  """Term-wise histories, oldest to newest, matching mjlab flattening."""
  result, hidx, fidx = [], 0, 0
  for dim in TERMS:
    result.append(torch.cat((history[:, hidx + dim:hidx + dim * length],
                             latest[:, fidx:fidx + dim]), -1))
    hidx += dim * length
    fidx += dim
  return torch.cat(result, -1)


def critic_input(state, command, context, previous):
  return torch.cat((state, command, context, previous), -1)


class Network(nn.Module):
  def __init__(self, inputs, outputs, hidden):
    super().__init__()
    self.norm = RunningNorm(inputs)
    layers = []
    for n in hidden:
      layers.extend((nn.Linear(inputs, n), nn.ELU()))
      inputs = n
    layers.append(nn.Linear(inputs, outputs))
    self.net = nn.Sequential(*layers)

  def forward(self, x):
    return self.net(self.norm(x))


class Actor(Network):
  def __init__(self, history=5, hidden=(256, 128, 64), actor_wheel_vel=False):
    super().__init__((LEGACY_FRAME_DIM if actor_wheel_vel else FRAME_DIM) * history, 16, hidden)
    self.actor_wheel_vel = actor_wheel_vel
    self.history, self.hidden = history, tuple(hidden)
    nn.init.uniform_(self.net[-1].weight, -0.01, 0.01)
    nn.init.zeros_(self.net[-1].bias)

  def forward(self, obs):
    # Same bounded action distribution during collection and imagination.
    return super().forward(obs).tanh()


def checkpoint_wheel_vel(checkpoint):
  """Infer legacy observation schema from weights; validate explicit metadata."""
  flag = checkpoint.get("actor_wheel_vel")
  if "actor" in checkpoint:
    dim = checkpoint["actor"]["net.0.weight"].shape[1]
    history = checkpoint["history"]
    if dim not in (FRAME_DIM * history, LEGACY_FRAME_DIM * history):
      raise ValueError("Unsupported actor observation dimension")
    inferred = dim == LEGACY_FRAME_DIM * history
    if flag is not None and flag != inferred:
      raise ValueError("Actor observation metadata disagrees with weights")
    return inferred
  return bool(flag) if flag is not None else False


def load_actor(checkpoint, device="cpu"):
  if checkpoint.get("format") not in (POLICY_FORMAT, "go2w-terrain-policy-v1"):
    raise ValueError("Expected a Go2W policy checkpoint, not Go2 or dynamics-only weights")
  actor = Actor(checkpoint["history"], checkpoint["actor_hidden"], checkpoint_wheel_vel(checkpoint)).to(device)
  actor.load_state_dict(checkpoint["actor"])
  return actor.eval()


def evaluation_env_cfg(checkpoint, level=0, kind=None):
  """Select the matching task; terrain remains privileged at deployment."""
  if checkpoint.get("format") == "go2w-terrain-policy-v1":
    from go2w_supermimic.env.terrain import go2w_velocity_terrain_env_cfg
    cfg = go2w_velocity_terrain_env_cfg(play=True, history=checkpoint["history"],
                                       seed=checkpoint["config"]["seed"],
                                       revision=checkpoint["config"].get("terrain_revision", 1),
                                       actor_wheel_vel=checkpoint_wheel_vel(checkpoint))
    # A single staircase can overlap the robot's pre-reset origin at compile
    # time. Reserve contacts for that transient pose as well as normal rolling.
    cfg.sim.nconmax = max(cfg.sim.nconmax or 0, 512)
    cfg.sim.njmax = max(cfg.sim.njmax or 0, 2048)
    if level not in range(6):
      raise ValueError("Terrain level must be 0..5")
    gen = cfg.scene.terrain.terrain_generator
    # Evaluation fixes difficulty; one row, same seeded generator.
    gen.num_rows = 1
    gen.difficulty_range = (level/5,level/5)
    if kind is not None:
      gen.sub_terrains = {kind: gen.sub_terrains[kind]}
    return cfg
  from go2w_supermimic.env.flat import go2w_velocity_flat_env_cfg
  return go2w_velocity_flat_env_cfg(play=True, history=checkpoint["history"],
                                     actor_wheel_vel=checkpoint_wheel_vel(checkpoint))
