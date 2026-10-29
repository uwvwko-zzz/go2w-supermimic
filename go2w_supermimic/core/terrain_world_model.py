"""Go2W terrain dynamics. Flat model checkpoints remain a separate format."""
import torch
from torch import nn
from .world_model import Go2WWorldModel, RunningNorm
from .terrain import SCAN_DIM, advance_pose

FORMAT = "go2w-terrain-dynamics-v1"
POLICY_FORMAT = "go2w-terrain-policy-v1"

class TerrainWorldModel(Go2WWorldModel):
  def __init__(self, hidden=(512,512,512), contacts=False, collisions=False):
    super().__init__(hidden)
    self.net[0] = nn.Linear(77+SCAN_DIM,hidden[0])
    self.input_norm = RunningNorm(77+SCAN_DIM)
    self.predict_contacts = contacts
    if contacts:
      self.contact_head = nn.Linear(hidden[-1],16)
    self.predict_collisions = collisions
    if collisions:
      self.collision_head = nn.Linear(hidden[-1],1)
      nn.init.zeros_(self.collision_head.weight)
      nn.init.constant_(self.collision_head.bias, -4.6)
    self.last_contact_metrics = {}

  def forward_all(self, state, action, previous_action, context):
    """Predict endpoint state, pose, and optional [logits4, asinh(force/100)12]."""
    x = self.input_norm(self.inputs(state,action,previous_action,context))
    for layer in list(self.net.children())[:-1]:
      x = layer(x)
    delta = self.net[-1](x)*self.output_norm.scale+self.output_norm.mean
    contacts = self.contact_head(x) if self.predict_contacts else None
    collision = self.collision_head(x) if self.predict_collisions else None
    return state+delta[:,:38],delta[:,38:],contacts,collision

  def forward_with_contacts(self, state, action, previous_action, context):
    pred,delta,contacts,_ = self.forward_all(state,action,previous_action,context)
    return pred,delta,contacts

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
    # Include the first terminal transition for collision classification only.
    prefix=torch.cat((torch.ones_like(mask[:,:1]),mask[:,:-1]),1)
    total=next(self.parameters()).sum()*0.
    contact_total=total.detach().clone(); force_total=total.detach().clone()
    collision_total=total.clone(); collision_count=total.detach().clone()
    positives=total.detach().clone(); predicted=total.detach().clone()
    for t in range(b["action"].shape[1]):
      keep=prefix[:,t] if self.predict_collisions else mask[:,t]
      if not keep.any(): break
      context=torch.cat((b["context"][keep,0],field.scan(state[keep],pose[keep])),-1)
      pred,delta,contact,collision=self.forward_all(state[keep],b["action"][keep,t],b["previous"][keep,t],context)
      current=mask[keep,t]
      # Never regress the reset state as the endpoint of a collision.
      if current.any():
        error=torch.cat((pred[current]-b["next"][keep,t][current],delta[current]-b["pose"][keep,t][current]),-1)
        total=total+(error/self.output_norm.scale).square().mean(-1).sum()
        if self.predict_contacts:
          target=b["next_contacts"][keep,t][current]
          finite=torch.isfinite(target).all(-1)
          prediction=contact[current][finite]; target=target[finite]
          classification=torch.nn.functional.binary_cross_entropy_with_logits(
            prediction[:,:4],target[:,:4],reduction='none').mean(-1)
          force=torch.nn.functional.smooth_l1_loss(prediction[:,4:],
            torch.asinh(target[:,4:]/100.),reduction='none').mean(-1)
          total=total+.1*(classification+force).sum()
          contact_total=contact_total+classification.detach().sum()
          force_total=force_total+force.detach().sum()
      if self.predict_collisions:
        label=b['collision'][keep,t]
        usable=b['collision_valid'][keep,t].squeeze(-1).bool() & torch.isfinite(label).all(-1)
        logits=collision[usable]; labels=label[usable]
        loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,labels,reduction='sum')
        collision_total=collision_total+loss
        collision_count=collision_count+usable.sum()
        positives=positives+labels.detach().sum()
        predicted=predicted+logits.detach().sigmoid().sum()
      next_pose=advance_pose(pose[keep],delta)
      state=state.clone(); pose=pose.clone()
      state[keep]=pred; pose[keep]=next_pose
    count=mask.sum().clamp_min(1)
    if self.predict_contacts:
      self.last_contact_metrics={'contact_bce':contact_total/count,'contact_force_loss':force_total/count}
    result=total/count
    if self.predict_collisions:
      denom=collision_count.clamp_min(1)
      result=result+collision_total/denom
      self.last_contact_metrics.update(collision_bce=collision_total.detach()/denom,
        collision_label_rate=positives/denom,collision_prediction_mean=predicted/denom,
        collision_label_count=collision_count)
    return result

  def checkpoint(self):
    return {"format":FORMAT,"hidden":self.hidden,"contacts":self.predict_contacts,"collisions":self.predict_collisions,"model":self.state_dict()}

  @classmethod
  def from_checkpoint(cls, checkpoint, device="cpu"):
    if checkpoint.get("format") != FORMAT: raise ValueError("Expected terrain dynamics checkpoint")
    model=cls(checkpoint["hidden"],contacts=checkpoint.get("contacts",False),collisions=checkpoint.get("collisions",False)).to(device)
    model.load_state_dict(checkpoint["model"])
    return model
