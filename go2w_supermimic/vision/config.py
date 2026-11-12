"""Camera contract saved in every vision checkpoint (provisional mount, not calibrated hardware)."""
from dataclasses import asdict, dataclass
import math

@dataclass(frozen=True)
class DepthConfig:
  width: int = 32
  height: int = 18
  fovy_deg: float = 60.
  pitch_down_deg: float = 20.
  position: tuple = (.32, 0., .08)
  near: float = .1
  far: float = 4.
  noise_std: float = .01
  dropout: float = .01

  def __post_init__(self):
    if self.width < 4 or self.height < 4 or not 0 < self.fovy_deg < 170:
      raise ValueError('Invalid depth resolution/FOV')
    if not 0 < self.near < self.far or self.noise_std < 0 or not 0 <= self.dropout < 1:
      raise ValueError('Invalid depth range/noise')
    if len(self.position) != 3 or not all(math.isfinite(x) for x in (*self.position, self.pitch_down_deg)):
      raise ValueError('Invalid camera mount')

  def to_dict(self):
    return asdict(self)
