"""Differentiable rewards, full-return actor objective and TD(lambda) critic targets."""

import math
from contextlib import contextmanager
import torch
from .world_model import GRAVITY, LIN_VEL, ANG_VEL, LEG_POS, LEG_VEL, WHEEL_VEL
from .policy import frame, push_history, critic_input


@contextmanager
def frozen(*modules):
  """Freeze weights without disabling gradients through inputs/actions."""
  parameters = [(p, p.requires_grad) for m in modules for p in m.parameters()]
  try:
    for p, _ in parameters:
      p.requires_grad_(False)
    yield
  finally:
    for p, enabled in parameters:
      p.requires_grad_(enabled)


def survival(state):
  gravity = torch.nn.functional.normalize(state[:, GRAVITY], dim=-1, eps=1e-6)
  # Smooth proxy for the environment's 70-degree fall cutoff, width ~5 deg.
  return torch.sigmoid((-gravity[:, 2] - math.cos(math.radians(70))) / 0.08)


def reward_terms(state, action, previous, command, constants, survival_prob, scan=None):
  v, w = state[:, LIN_VEL], state[:, ANG_VEL]
  q, qd, wd = state[:, LEG_POS], state[:, LEG_VEL], state[:, WHEEL_VEL]
  standing = (command.square().sum(-1) < 1e-6).float()
  # Endpoint PD torque surrogate: actual MuJoCo forces also reflect implicit
  # integration over substeps. The low-weight effort term is approximate.
  leg_torque = (constants["kp"] * (constants.get("leg_action_scale", 0.25) * action[:, :12] - q)
                - constants["kd"] * qd).clamp(-constants["effort"], constants["effort"])
  wheel_torque = (2. * ((constants.get("wheel_action_scale", 15.) * action[:, 12:]).clamp(-35., 35.) - wd)).clamp(-45., 45.)
  absolute_q = q + constants["default_q"]
  limits = ((constants["limits"][..., 0] - absolute_q).clamp_min(0)
            + (absolute_q - constants["limits"][..., 1]).clamp_min(0)).sum(-1)
  posture = q.square().sum(-1)
  if constants.get("terrain_posture_v2", False) and not constants.get("wheel_local_rewards", False):
    if scan is None:
      raise ValueError("Terrain v2 posture reward requires the privileged scan")
    from .terrain_posture import posture_cost
    posture = posture_cost(q, scan)
  terrain = constants.get("terrain_rewards", False)
  terms = {
    "track_linear_velocity": torch.exp(-((v[:, :2] - command[:, :2]).square().sum(-1)
                                          + (0. if terrain else v[:, 2].square())) / 0.25),
    "track_angular_velocity": torch.exp(-((w[:, 2] - command[:, 2]).square()
                                           + (0. if terrain else w[:, :2].square().sum(-1))) / 0.5),
    "upright": torch.exp(-state[:, GRAVITY][:, :2].square().sum(-1) / constants.get("upright_std", 0.45)**2),
    "leg_posture": posture,
    "standstill": standing * (v.square().sum(-1) + w.square().sum(-1)
                              + 0.01 * wd.square().sum(-1)),
    "action_rate_l2": (action - previous).square().sum(-1),
    "joint_torques_l2": leg_torque.square().sum(-1) + wheel_torque.square().sum(-1),
    "dof_pos_limits": limits,
    "terminated": 1. - survival_prob,
  }

  if constants.get("rolling_rewards", False) and not constants.get("wheel_local_rewards", False):
    if scan is None:
      raise ValueError("Rolling rewards require a terrain scan")
    from .rolling_rewards import rolling_costs
    terms.update(rolling_costs(state, command, scan, constants["default_q"]))
  if constants.get("obstacle_progress", False):
    from .obstacle_rewards import obstacle_progress
    terms["obstacle_progress"] = obstacle_progress(state, command, scan)
  if constants.get("wheel_local_rewards", False):
    from .wheel_terrain_rewards import wheel_terms
    terms.update(wheel_terms(state, command, scan, constants["default_q"]))
  return terms


def lambda_returns(rewards, continuation, next_values, gamma=0.99, lam=0.95):
  carry = next_values[-1]
  result = []
  for t in reversed(range(len(rewards))):
    carry = rewards[t] + gamma * continuation[t] * ((1 - lam) * next_values[t] + lam * carry)
    result.append(carry)
  return torch.stack(result[::-1])


def discounted_objective(rewards, continuation, terminal_value, gamma=.99):
  alive = torch.ones_like(terminal_value)
  total = torch.zeros_like(terminal_value)
  discount = 1.
  for reward, cont in zip(rewards, continuation):
    total = total + discount * alive * reward
    alive = alive * cont
    discount *= gamma
  return total + discount * alive * terminal_value


def make_imagined_step(model, actor, constants, weights, env_action_order,
                       dt=.02, noise=True, field=None):
  """Pure single-step graph, shared by eager and compiled execution."""
  def step(state, previous, history, pose, scan, command, context, bias):
    action = actor(history)
    model_context = context if field is None else torch.cat((context, scan), -1)
    collision_prob = None
    if constants.get('nonwheel_collision', False):
      nxt, delta, _, collision_logit = model.forward_all(state, action, previous, model_context)
      collision_prob = collision_logit.squeeze(-1).sigmoid()
    else:
      nxt, delta = model(state, action, previous, model_context)
    if field is not None:
      from .terrain import advance_pose
      pose = advance_pose(pose, delta)
    next_scan = None if field is None else field.scan(nxt, pose)
    cont = survival(nxt)
    if collision_prob is not None:
      cont = cont * (1. - collision_prob)
    terms = reward_terms(nxt, action, previous, command, constants, cont,
                         scan=next_scan)
    reward = dt * sum(weights[name] * value for name, value in terms.items())
    latest = frame(nxt, action[:, env_action_order], command, bias, noise=noise)
    history = push_history(history, latest, actor.history)
    return nxt, action, history, pose, next_scan, reward, cont
  return step


def rollout(model, actor, target_critic, seed, constants, weights, env_action_order,
            horizon=16, dt=0.02, gamma=0.99, lam=0.95, noise=True, field=None,
            step_fn=None):
  """Actor: full discounted return + terminal value. Critic: detached TD(lambda).

  Caller freezes dynamics and target critic weights, preserving state gradients.
  Reuse one terrain scan at each state, including the rollout's initial state.
  """
  state, previous, history = seed["state"], seed["previous"], seed["history"]
  command, context, bias = seed["command"], seed["context"], seed["bias"]
  pose = seed.get("world_pose")
  def value_input(s, prev, scan):
    base = critic_input(s, command, context, prev)
    return base if field is None else torch.cat((base, scan), -1)
  if step_fn is None:
    step_fn = make_imagined_step(model, actor, constants, weights, env_action_order,
                                 dt, noise, field)
  rewards, continuations, inputs, next_inputs = [], [], [], []
  scan = None if field is None else field.scan(state, pose)
  for _ in range(horizon):
    inputs.append(value_input(state, previous, scan).detach())
    state, previous, history, pose, scan, reward, cont = step_fn(
      state, previous, history, pose, scan, command, context, bias)
    rewards.append(reward)
    continuations.append(cont)
    next_inputs.append(value_input(state, previous, scan).detach())
  final_value = target_critic(value_input(state, previous, scan)).squeeze(-1)
  r, c = torch.stack(rewards), torch.stack(continuations)
  objective = discounted_objective(r, c, final_value, gamma)
  with torch.no_grad():
    next_values = target_critic(torch.stack(next_inputs)).squeeze(-1)
    targets = lambda_returns(r.detach(), c.detach(), next_values, gamma, lam)
  return -objective.mean(), torch.stack(inputs), targets
