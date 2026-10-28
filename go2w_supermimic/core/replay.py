"""Bounded Go2W transition storage with chronological multi-step sampling."""

import torch
from .world_model import STATE_DIM, ACTION_DIM, CONTEXT_DIM


class Replay:
  def __init__(self, length, num_envs, device, history_dim=0, terrain=False, contacts=False):
    if min(length, num_envs) < 1:
      raise ValueError("Replay length and environment count must be positive")
    self.length, self.size, self.cursor = length, 0, 0
    self.archive = None
    dims = dict(state=STATE_DIM, action=ACTION_DIM, previous=ACTION_DIM,
                context=CONTEXT_DIM, next=STATE_DIM, pose=3, valid=1)
    if terrain:
      dims["world_pose"] = 3
    if contacts:
      dims["next_contacts"] = 16
    if history_dim:
      dims.update(history=history_dim, command=3, bias=12)
    self.data = {k: torch.zeros(length, num_envs, dim, device=device) for k, dim in dims.items()}

  def add(self, **items):
    if items.keys() != self.data.keys():
      raise ValueError("Replay transition fields do not match configured fields")
    for k, value in items.items():
      self.data[k][self.cursor].copy_(value.detach())
    self.cursor = (self.cursor + 1) % self.length
    self.size = min(self.size + 1, self.length)

  def sample(self, batch, window, keys=None):
    if self.size < window or window < 1:
      raise ValueError("Not enough transitions for a window")
    device = self.data["state"].device
    start = torch.randint(self.size - window + 1, (batch,), device=device)
    oldest = self.cursor if self.size == self.length else 0
    times = (oldest + start[:, None] + torch.arange(window, device=device)) % self.length
    envs = torch.randint(self.data["state"].shape[1], (batch, 1), device=device)
    return {k: v[times, envs] for k, v in self.data.items() if keys is None or k in keys}


  def sample_world_model(self, batch, window, prioritized=.25, archived=.25,
                         capacity=100000, keep_prob=.02):
    keys = ('state','next','action','previous','context','pose','valid',
            'world_pose','next_contacts')
    if 'next_contacts' not in self.data:
      return self.sample(batch, window, keys)
    if self.archive is None:
      self.archive = WindowArchive(capacity)
    n_priority = int(batch * prioritized)
    n_archive = int(batch * archived) if self.archive.size else 0
    parts = [self.sample(batch-n_priority-n_archive, window, keys)]
    candidates = self.sample(4*max(n_priority,1), window, keys)
    score, rare = event_scores(candidates)
    selected = rare & (torch.rand_like(score) < keep_prob)
    self.archive.add({k:v[selected] for k,v in candidates.items()})
    if n_priority:
      indices = (score + .001*torch.rand_like(score)).topk(n_priority).indices
      parts.append({k:v[indices] for k,v in candidates.items()})
    if n_archive:
      parts.append(self.archive.sample(n_archive))
    return {k:torch.cat([p[k] for p in parts]) for k in parts[0]}


def event_scores(w):
  """Measured wheel contact transitions; ignore all labels after a reset.

  Force impacts, sustained flight, hard vertical motion and episode ends enter
  the archive. Contact changes also include normal wheel touchdown/liftoff.
  """
  valid = w['valid'].squeeze(-1).bool()
  mask = valid.cumprod(1).bool()
  contacts = w['next_contacts'][:,:,:4] > .5
  pairs = mask[:,1:] & mask[:,:-1]
  changes = ((contacts[:,1:] != contacts[:,:-1]).sum(-1)*pairs).sum(-1)
  flight = ((~contacts.any(-1)) & mask).sum(-1)
  force = torch.nan_to_num(w['next_contacts'][:,:,4:],nan=0.).reshape(*mask.shape,4,3).norm(dim=-1)
  impact = ((force.amax(-1)>200.) & mask).any(-1)
  hard_vz = ((w['state'][:,:,6].abs()>.6) & mask).any(-1)
  # A terminal endpoint is not a valid dynamics target, but its preceding
  # window is still useful. Windows starting after termination are excluded.
  ended = ((~valid) & torch.cat((torch.ones_like(mask[:,:1]),mask[:,:-1]),1)).any(-1)
  score = changes.float() + flight.float() + 4.*ended + 2.*impact
  rare = ((flight>=2) | hard_vz | impact | ended) & mask[:,0]
  return score * mask[:,0], rare


class WindowArchive:
  """Bounded GPU ring storing only world-model fields, not actor histories."""
  def __init__(self, capacity):
    self.capacity, self.size, self.cursor = capacity, 0, 0
    self.data = None

  def add(self, windows):
    n = min(len(windows['state']),self.capacity)
    if not n:return
    if self.data is None:
      self.data = {k:v.new_empty((self.capacity,*v.shape[1:])) for k,v in windows.items()}
    ids = (torch.arange(n,device=windows['state'].device)+self.cursor)%self.capacity
    for k,v in windows.items():self.data[k][ids] = v[-n:].detach()
    self.cursor = (self.cursor+n)%self.capacity
    self.size = min(self.capacity,self.size+n)

  def sample(self, batch):
    ids = torch.randint(self.size,(batch,),device=self.data['state'].device)
    return {k:v[ids] for k,v in self.data.items()}
