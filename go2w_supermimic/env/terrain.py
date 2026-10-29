"""Static Go2W courses. Independent from the original Go2 terrain preset."""
from dataclasses import dataclass
import math
import mujoco
import numpy as np
import torch
from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainOutput, TerrainGeometry, TerrainGeneratorCfg
from mjlab.managers import CurriculumTermCfg, TerminationTermCfg, RewardTermCfg
from .flat import go2w_velocity_flat_env_cfg

STEP_HEIGHTS = (0.02, 0.04, 0.08, 0.12, 0.16, 0.20)

@dataclass
class CourseCfg(SubTerrainCfg):
  kind: str = "rubble"
  bidirectional: bool = False
  stair_tread: float = .45

  def function(self, difficulty, spec, rng):
    level = min(5, max(0, round(difficulty * 5)))
    h = STEP_HEIGHTS[level]
    sx, sy = self.size
    if self.bidirectional:
      sx = 8.  # Mirror the original course into a 16 m square.
    body, geoms = spec.body("terrain"), []
    def box(x, y, z, a, b, c, yaw=0., pitch=0., shade=.45):
      # Trim the source half at its spawn plane before mirroring, avoiding
      # coincident contact surfaces in the central floor/down-stair landing.
      if self.bidirectional and yaw == 0. and pitch == 0.:
        left, right = max(1., x-a/2), x+a/2
        if right <= left:
          return
        x, a = (left+right)/2, right-left
      # Composition Rz(yaw) Ry(pitch); z is centre, dimensions are full lengths.
      cy, ss, cp, sp = math.cos(yaw/2), math.sin(yaw/2), math.cos(pitch/2), math.sin(pitch/2)
      placements = [(x, (cy*cp,-ss*sp,cy*sp,ss*cp))]
      if self.bidirectional:
        # Reflect about x=8: yaw and pitch change sign, roll stays unchanged.
        placements = [(x+7., (cy*cp,-ss*sp,cy*sp,ss*cp)),
                      (9.-x, (cy*cp,-ss*sp,-cy*sp,-ss*cp))]
      for xx, quat in placements:
        geom = body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=(xx,y,z),
                             size=(a/2,b/2,c/2), quat=quat)
        geoms.append(TerrainGeometry(geom=geom, color=(shade,shade,.9*shade,1.)))
    box(sx/2,sy/2,-.1,sx,sy,.2,shade=.6)
    origin = np.array([1., sy/2, 0.])
    if self.kind in ("up", "down"):
      n, tread = 5, self.stair_tread
      if self.kind == "up":
        # Overlapping foundation boxes make solid risers and a long top landing.
        for k in range(n):
          start = 2. + k*tread
          box((start+sx)/2,sy/2,(k+1)*h/2,sx-start,sy,(k+1)*h)
      else:
        origin[2] = n*h
        for k in range(n):
          end = 2. + k*tread
          height = (n-k)*h
          box(end/2,sy/2,height/2,end,sy,height)
    elif self.kind == "platform":
      for x in (2.6,4.6):
        box(x,sy/2,h/2,.9,sy,h)
    elif self.kind == "slope":
      # A continuous wedge made by a tilted slab with its low edge buried.
      angle = math.radians(3+12*difficulty)
      length = 3.
      thickness = .16
      center_z = length/2*math.sin(angle)-thickness/2*math.cos(angle)
      box(3.5,sy/2,center_z,length,sy,thickness,pitch=-angle)
      end = 3.5+length/2*math.cos(angle)
      top = length*math.sin(angle)
      box((end+sx)/2,sy/2,top/2,sx-end,sy,top)
    elif self.kind == "rubble":
      # Staggered rows cover the lane. Static overlapping rocks, never loose bodies.
      # Heights of all layers together are capped by h (2--20 cm).
      for ix, x in enumerate(np.arange(2.15,5.9,.48)):
        for y in np.arange(.35,sy-.2,.52):
          a,b = rng.uniform(.35,.65,2)
          height = h*rng.uniform(.35,.7)
          xx,yy = x+rng.uniform(-.12,.12),y+rng.uniform(-.12,.12)
          yaw = rng.uniform(-math.pi,math.pi)
          box(xx,yy,height/2,a,b,height,yaw=yaw,shade=rng.uniform(.3,.55))
          if rng.random() < .25+.35*difficulty:
            cap = h-height
            # Deliberately overlapping cap; bounded surface height even in stacks.
            pitch = rng.uniform(-.15,.15)*difficulty
            half_z = (cap*math.cos(pitch)+a*.65*abs(math.sin(pitch)))/2
            box(xx+.04,yy-.03,h-half_z,a*.65,b*.65,cap,
                yaw=yaw+.6,pitch=pitch,shade=.55)
    elif self.kind != "flat":
      raise ValueError(self.kind)
    if self.bidirectional:
      origin[0] = 8.
    return TerrainOutput(origin=origin, geometries=geoms)


def terrain_linear_tracking(env):
  velocity = env.scene["robot"].data.root_link_lin_vel_b
  command = env.command_manager.get_command("twist")
  return torch.exp(-(velocity[:,:2]-command[:,:2]).square().sum(-1)/.25)


def terrain_yaw_tracking(env):
  velocity = env.scene["robot"].data.root_link_ang_vel_b
  command = env.command_manager.get_command("twist")
  return torch.exp(-(velocity[:,2]-command[:,2]).square()/.5)


def course_exit(env):
  offset = env.scene["robot"].data.root_link_pos_w - env.scene.env_origins
  return (offset[:,0] > 5.4) | (offset[:,0] < -.65) | (offset[:,1].abs() > 2.7)


def course_curriculum(env, env_ids):
  terrain = env.scene.terrain
  offset = env.scene["robot"].data.root_link_pos_w[env_ids] - env.scene.env_origins[env_ids]
  active = env.episode_length_buf[env_ids] > 0
  fallen = env.termination_manager.terminated[env_ids]
  success = active & ~fallen & (offset[:,0] > 5.4) & (offset[:,1].abs() < 2.7)
  failure = active & (fallen | (offset[:,0] < -.65) | (offset[:,1].abs() > 2.7))
  terrain.terrain_levels[env_ids] = (terrain.terrain_levels[env_ids] + success.long() - failure.long()).clamp(0,5)
  terrain.env_origins[env_ids] = terrain.terrain_origins[terrain.terrain_levels[env_ids], terrain.terrain_types[env_ids]]
  return terrain.terrain_levels.float().mean()


def bidirectional_exit(env):
  offset = env.scene["robot"].data.root_link_pos_w - env.scene.env_origins
  return (offset[:,0].abs() > 6.5) | (offset[:,1].abs() > 6.5)


def bidirectional_curriculum(env, env_ids):
  terrain = env.scene.terrain
  offset = env.scene["robot"].data.root_link_pos_w[env_ids] - env.scene.env_origins[env_ids]
  active = env.episode_length_buf[env_ids] > 0
  fallen = env.termination_manager.terminated[env_ids]
  # Both course directions count. Lateral exits are neutral, not evidence
  # of clearing steps; negative commands no longer count as failure.
  success = active & ~fallen & (offset[:,0].abs() > 5.4) & (offset[:,1].abs() < 6.5)
  failure = active & fallen
  terrain.terrain_levels[env_ids] = (terrain.terrain_levels[env_ids] + success.long() - failure.long()).clamp(0,5)
  terrain.env_origins[env_ids] = terrain.terrain_origins[terrain.terrain_levels[env_ids], terrain.terrain_types[env_ids]]
  return terrain.terrain_levels.float().mean()


def go2w_velocity_terrain_env_cfg(play=False, history=5, seed=0, revision=1, actor_wheel_vel=False):
  if revision not in (1, 2, 3, 4, 5, 6, 7):
    raise ValueError("Terrain revision must be 1, 2, 3, 4, 5, 6 or 7")
  cfg = go2w_velocity_flat_env_cfg(play=play, history=history, actor_wheel_vel=actor_wheel_vel)
  cfg.sim.nconmax = 128
  cfg.sim.njmax = 512
  cfg.scene.terrain.terrain_type = "generator"
  cfg.scene.terrain.max_init_terrain_level = 0
  cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(
    seed=seed, size=(8.,8.), num_rows=6, curriculum=True, color_scheme="none",
    sub_terrains={kind: CourseCfg(kind=kind, proportion=weight) for kind,weight in
                 (("flat",.10),("up",.25),("down",.20),("platform",.15),("slope",.10),("rubble",.20))})
  cfg.events["reset_base"].params["pose_range"].update(x=(-.1,.1),y=(-.1,.1),yaw=(-.08,.08))
  command = cfg.commands["twist"]
  command.ranges.lin_vel_x = (.2,.45)
  command.ranges.lin_vel_y = (0.,0.)
  command.ranges.ang_vel_z = (-.05,.05)
  command.rel_standing_envs = .1
  cfg.episode_length_s = 35.
  cfg.rewards["leg_posture"].weight = -.05
  cfg.rewards["track_linear_velocity"].func = terrain_linear_tracking
  cfg.rewards["track_linear_velocity"].params = {}
  cfg.rewards["track_angular_velocity"].func = terrain_yaw_tracking
  cfg.rewards["track_angular_velocity"].params = {}
  cfg.rewards["upright"].params["std"] = .6
  if revision >= 2:
    from go2w_supermimic.core.terrain_posture import env_posture_cost
    cfg.actions["joint_pos"].scale = {".*_hip_joint": .25, ".*_thigh_joint": .6, ".*_calf_joint": .8}
    cfg.rewards["leg_posture"].func = env_posture_cost
    cfg.rewards["leg_posture"].weight = -.4
  cfg.curriculum = {} if play else {"course": CurriculumTermCfg(func=course_curriculum)}
  cfg.terminations["course_exit"] = TerminationTermCfg(func=course_exit, time_out=True)
  if revision >= 3:
    command.ranges.lin_vel_x = (-2., 2.)
    command.ranges.lin_vel_y = (-1., 1.)
    command.ranges.ang_vel_z = (-3., 3.)
    command.rel_forward_envs = 0.
    command.rel_world_envs = 0.
    cfg.actions["wheel_vel"].scale = 35.
    gen = cfg.scene.terrain.terrain_generator
    gen.size = (16., 16.)
    for course in gen.sub_terrains.values():
      course.bidirectional = True
    cfg.curriculum = {} if play else {"course": CurriculumTermCfg(func=bidirectional_curriculum)}
    cfg.terminations["course_exit"] = TerminationTermCfg(func=bidirectional_exit, time_out=True)
  if revision >= 4:
    from mjlab.sensor import ContactMatch, ContactSensorCfg
    cfg.scene.sensors = tuple(cfg.scene.sensors or ()) + (ContactSensorCfg(
      name="wheel_ground_contact", primary=ContactMatch(mode="body",
        pattern=tuple(f"{leg}_wheel_link" for leg in ("FL","FR","RL","RR")), entity="robot"),
      secondary=ContactMatch(mode="body", pattern="terrain"),
      fields=("found","force"), reduce="netforce", num_slots=1),)
    from go2w_supermimic.core.rolling_rewards import env_rolling_cost
    for course in cfg.scene.terrain.terrain_generator.sub_terrains.values():
      course.stair_tread = .30
    for name, weight in (("unnecessary_wheel_lift", -.2), ("rolling_leg_motion", -.12)):
      cfg.rewards[name] = RewardTermCfg(func=env_rolling_cost, weight=weight, params={"term":name})
    if not play:
      from go2w_supermimic.core.mixed_commands import MixedVelocityCommandCfg
      cfg.commands["twist"] = MixedVelocityCommandCfg(**vars(command))
  if revision >= 5:
    from go2w_supermimic.core.obstacle_rewards import env_obstacle_progress
    from go2w_supermimic.core.stair_curriculum import curriculum
    cfg.rewards['obstacle_progress'] = RewardTermCfg(func=env_obstacle_progress, weight=.5)
    if not play:
      cfg.curriculum = {'course': CurriculumTermCfg(func=curriculum)}
      cfg.commands['twist'].hold_stair_command = True
    for kind, weight in zip(('flat','up','down','platform','slope','rubble'), (.1,.35,.15,.1,.1,.2)):
      cfg.scene.terrain.terrain_generator.sub_terrains[kind].proportion = weight
  if revision == 6:
    from go2w_supermimic.core.wheel_terrain_rewards import env_wheel_term
    for name in ('leg_posture', 'unnecessary_wheel_lift', 'rolling_leg_motion'):
      cfg.rewards[name].func = env_wheel_term
      cfg.rewards[name].params = {'term': name}
    cfg.rewards['wheel_obstacle_clearance'] = RewardTermCfg(
      func=env_wheel_term, weight=-.25, params={'term':'wheel_obstacle_clearance'})
  if revision >= 7:
    from go2w_supermimic.core.collision import illegal_nonwheel_contact
    # V7 = V5 + non-wheel collisions; experimental V6 shaping is independent.
    cfg.scene.sensors += (ContactSensorCfg(
      name='nonwheel_ground_contact',
      primary=ContactMatch(mode='body', pattern='.*', entity='robot',
                           exclude=tuple(f'{leg}_wheel_link' for leg in ('FL','FR','RL','RR'))),
      secondary=ContactMatch(mode='body', pattern='terrain'),
      fields=('found','force'), reduce='netforce', num_slots=1,
      history_length=cfg.decimation),)
    cfg.terminations['illegal_nonwheel_contact'] = TerminationTermCfg(
      func=illegal_nonwheel_contact, params={'force_threshold':10.})
  return cfg
