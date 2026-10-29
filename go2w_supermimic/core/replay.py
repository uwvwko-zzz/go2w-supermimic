"""Bounded Go2W transition storage with chronological multi-step sampling."""

import torch
from .world_model import STATE_DIM, ACTION_DIM, CONTEXT_DIM


class Replay:
  def __init__(self, length, num_envs, device, history_dim=0, terrain=False, contacts=False, courses=False, collisions=False):
    if min(length, num_envs) < 1:
      raise ValueError("Replay length and environment count must be positive")
    self.length, self.size, self.cursor = length, 0, 0
    self.archive = None
    self.success_archive = None
    dims = dict(state=STATE_DIM, action=ACTION_DIM, previous=ACTION_DIM,
                context=CONTEXT_DIM, next=STATE_DIM, pose=3, valid=1)
    if terrain:
      dims["world_pose"] = 3
    if contacts:
      dims["next_contacts"] = 16
    if collisions:
      dims.update(collision=1, collision_valid=1)
    if courses:
      dims["course"] = 4  # type, level, uphill ground transition, top crossing
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
    if 'course' in self.data:
      return self._sample_courses(batch, window, capacity, keep_prob)
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


  def _sample_courses(self, batch, window, capacity, keep_prob):
    # Total archive capacity remains bounded by --archive-size, split evenly.
    if self.success_archive is None:
      self.archive = WindowArchive(max(1, capacity // 2))
      self.success_archive = WindowArchive(max(1, capacity - capacity // 2))
    keys = tuple(k for k in self.data if k not in ('history', 'command', 'bias'))
    n_priority = batch // 4
    candidates = self.sample(max(1, n_priority * 4), window, keys)
    score, rare = event_scores(candidates)
    valid = candidates['valid'].squeeze(-1).bool().cumprod(1).bool()
    tag = candidates['course']
    moving = (candidates['pose'][:, :, :2].norm(dim=-1) > .001) & valid
    up = tag[:, :, 0] == 1
    useful = ((tag[:, :, 2:] > .5).any(-1) & valid & up).any(-1)
    # Do not fill the general archive with repeated stationary stair impacts.
    rare &= (~up.any(-1) | moving.any(-1))
    if 'collision' in candidates:
      from .collision import collision_events
      rare |= collision_events(candidates)
    for archive, chosen in ((self.archive, rare), (self.success_archive, useful)):
      chosen = chosen & (torch.rand_like(score) < keep_prob)
      archive.add({k:v[chosen] for k,v in candidates.items()})
    n_success = batch // 8 if self.success_archive.size else 0
    n_rare = batch // 8 if self.archive.size else 0
    # At least half the batch is always uniform replay, including when sparse.
    parts = [self.sample(batch - n_priority - n_success - n_rare, window, keys)]
    if n_priority:
      rank = score + 8 * useful.float() + 2 * (moving & up).any(-1).float()
      selected = (rank + .001 * torch.rand_like(rank)).topk(n_priority).indices
      parts.append({k:v[selected] for k,v in candidates.items()})
    if n_success: parts.append(self.success_archive.sample(n_success))
    if n_rare: parts.append(self.archive.sample(n_rare))
    return {k:torch.cat([part[k] for part in parts]) for k in keys}


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
  score = score * mask[:,0]
  if 'collision' in w:
    from .collision import collision_events
    collision = collision_events(w)
    score = score + 8.*collision
    rare |= collision
  return score, rare


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
