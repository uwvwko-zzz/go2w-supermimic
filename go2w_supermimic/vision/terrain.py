"""Vision-only high platforms; keep the six-column blind course ABI intact."""
from dataclasses import dataclass
import mujoco
import numpy as np
from mjlab.terrains.terrain_generator import TerrainOutput, TerrainGeometry
from go2w_supermimic.env.terrain import CourseCfg

PLATFORM_HEIGHTS = (.10, .20, .30, .40, .50, .60)
TASK_CONTRACT = 'vision-high-platform-60cm-base-contact-100N-v1'
BASE_CONTACT_THRESHOLD = 100.


@dataclass
class HighPlatformCfg(CourseCfg):
  kind: str = 'platform'

  def function(self, difficulty, spec, rng):
    h = PLATFORM_HEIGHTS[min(5, max(0, round(difficulty * 5)))]
    sx, sy = self.size
    body = spec.body('terrain')
    geoms = []
    def box(pos, size):
      geom = body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=pos, size=size)
      geoms.append(TerrainGeometry(geom=geom, color=(.45,.45,.40,1.)))
    box((sx/2, sy/2, -.1), (sx/2, sy/2, .1))
    # Two broad solid landings around a 3 m central approach. No intermediate
    # risers: 60 cm means one vertical face, not a sum of smaller stairs.
    spawn = sx/2 if self.bidirectional else 1.
    start = spawn + 1.5
    box(((start+sx)/2, sy/2, h/2), ((sx-start)/2, sy/2, h/2))
    if self.bidirectional:
      end = spawn - 1.5
      box((end/2, sy/2, h/2), (end/2, sy/2, h/2))
    return TerrainOutput(origin=np.array([spawn, sy/2, 0.]), geometries=geoms)
