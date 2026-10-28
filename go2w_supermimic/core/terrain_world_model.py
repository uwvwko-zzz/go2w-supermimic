"""Go2W terrain dynamics. Flat model checkpoints remain a separate format."""
import torch
from torch import nn
from .world_model import Go2WWorldModel, RunningNorm
from .terrain import SCAN_DIM, advance_pose

FORMAT = "go2w-terrain-dynamics-v1"
POLICY_FORMAT = "go2w-terrain-policy-v1"

class TerrainWorldModel(Go2WWorldModel):
  def __init__(self, hidden=(512,512,512), contacts=False):
    super().__init__(hidden)
    self.net[0] = nn.Linear(77+SCAN_DIM,hidden[0])
    self.input_norm = RunningNorm(77+SCAN_DIM)
    self.predict_contacts = contacts
    if contacts:
      self.contact_head = nn.Linear(hidden[-1],16)
    self.last_contact_metrics = {}

  def forward_with_contacts(self, state, action, previous_action, context):
    """Predict endpoint state, pose, and optional [logits4, asinh(force/100)12]."""
    x = self.input_norm(self.inputs(state,action,previous_action,context))
    for layer in list(self.net.children())[:-1]:
      x = layer(x)
    delta = self.net[-1](x)*self.output_norm.scale+self.output_norm.mean
    contacts = self.contact_head(x) if self.predict_contacts else None
    return state+delta[:,:38],delta[:,38:],contacts

  def forward(self, state, action, previous_action, context):
    pred,delta,_ = self.forward_with_contacts(state,action,previous_action,context)
    return pred,delta

  @staticmethod
  def decode_contacts(prediction):
    """Contact probabilities and signed base-frame force vectors in newtons."""
    return prediction[:,:4].sigmoid(),100.*prediction[:,4:].clamp(-10,10).sinh().reshape(-1,4,3)

  @staticmethod
  def inputs(state, action, previous_action, context):
    if context.shape[-1] != 7+SCAN_DIM:
      raise ValueError("Terrain dynamics require context7 plus height scan273")
    return torch.cat((Go2WWorldModel.inputs(state,action,previous_action,context[:,:7]),context[:,7:]),-1)

  def terrain_loss(self, b, field):
    state,pose=b["state"][:,0],b["world_pose"][:,0]
    mask=b["valid"].squeeze(-1).bool().cumprod(1).bool()
    total=next(self.parameters()).sum()*0.
    contact_total=total.detach().clone(); force_total=total.detach().clone()
    for t in range(b["action"].shape[1]):
      keep=mask[:,t]
      if not keep.any(): break
      context=torch.cat((b["context"][keep,0],field.scan(state[keep],pose[keep])),-1)
      pred,delta,contact=self.forward_with_contacts(state[keep],b["action"][keep,t],b["previous"][keep,t],context)
      error=torch.cat((pred-b["next"][keep,t],delta-b["pose"][keep,t]),-1)
      total=total+(error/self.output_norm.scale).square().mean(-1).sum()
      if self.predict_contacts:
        target=b["next_contacts"][keep,t]
        finite=torch.isfinite(target).all(-1)
        # Missing/non-finite sensor labels must not poison dynamics gradients.
        prediction=contact[finite]; target=target[finite]
        classification=torch.nn.functional.binary_cross_entropy_with_logits(
          prediction[:,:4],target[:,:4],reduction='none').mean(-1)
        force=torch.nn.functional.smooth_l1_loss(prediction[:,4:],
          torch.asinh(target[:,4:]/100.),reduction='none').mean(-1)
        total=total+.1*(classification+force).sum()
        contact_total=contact_total+classification.detach().sum()
        force_total=force_total+force.detach().sum()
      next_pose=advance_pose(pose[keep],delta)
      state=state.clone(); pose=pose.clone()
      state[keep]=pred; pose[keep]=next_pose
    count=mask.sum().clamp_min(1)
    if self.predict_contacts:
      self.last_contact_metrics={'contact_bce':contact_total/count,'contact_force_loss':force_total/count}
    return total/count

  def checkpoint(self):
    return {"format":FORMAT,"hidden":self.hidden,"contacts":self.predict_contacts,"model":self.state_dict()}

  @classmethod
  def from_checkpoint(cls, checkpoint, device="cpu"):
    if checkpoint.get("format") != FORMAT: raise ValueError("Expected terrain dynamics checkpoint")
    model=cls(checkpoint["hidden"],contacts=checkpoint.get("contacts",False)).to(device)
    model.load_state_dict(checkpoint["model"])
    return model
