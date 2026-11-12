"""Fresh task configs built on v7; never modifies an existing blind config or model asset."""
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from go2w_supermimic.env.terrain import go2w_velocity_terrain_env_cfg
from .config import DepthConfig


def depth_observation(env, camera, noise=True,randomize=True):
  from .depth import DepthRenderer, quat_matrix
  if not hasattr(env,'_go2w_depth_renderer'):
    env._go2w_depth_renderer=DepthRenderer.from_env(env,camera)
  from .camera import CameraRuntime
  if not hasattr(env,'_vision_camera_runtime'):
    env._vision_camera_runtime=CameraRuntime(env,env._go2w_depth_renderer,randomize)
  return env._vision_camera_runtime.observe(noise=noise)


def reset_camera(env,env_ids):
  camera=getattr(env,'_vision_camera_runtime',None)
  if camera is not None: camera.reset(env_ids)


def go2w_vision_env_cfg(play=False,history=5,seed=0,revision=7,camera=None,randomize=True):
  if revision != 7:
    raise ValueError('Vision v1 uses the v7 task contract')
  cfg=go2w_velocity_terrain_env_cfg(play=play,history=history,seed=seed,revision=7)
  from .terrain import HighPlatformCfg, BASE_CONTACT_THRESHOLD
  from .curriculum import curriculum
  from mjlab.managers import CurriculumTermCfg
  from mjlab.sensor import ContactMatch
  gen = cfg.scene.terrain.terrain_generator
  gen.sub_terrains['platform'] = HighPlatformCfg(proportion=.1, bidirectional=True)
  if not play:
    cfg.curriculum['course'] = CurriculumTermCfg(func=curriculum)
  # Preserve the replay label key, but redefine it consistently as BASE contact.
  # Legs may support a climb; touching a riser must not immediately end it.
  for sensor in cfg.scene.sensors:
    if sensor.name == 'nonwheel_ground_contact':
      sensor.primary = ContactMatch(mode='body', pattern='base_link', entity='robot')
  cfg.terminations['illegal_nonwheel_contact'].params['force_threshold'] = BASE_CONTACT_THRESHOLD
  from mjlab.managers import EventTermCfg
  from .randomization import physical_randomization,episode_randomization,push_step
  from .actions import DelayedPositionCfg,DelayedVelocityCfg
  if randomize and not play:
    cfg.events['vision_physics']=EventTermCfg(func=physical_randomization,mode='startup')
    cfg.events['vision_episode']=EventTermCfg(func=episode_randomization,mode='reset')
    cfg.events['vision_push']=EventTermCfg(func=push_step,mode='step')
    cfg.actions['joint_pos']=DelayedPositionCfg(**vars(cfg.actions['joint_pos']))
    cfg.actions['wheel_vel']=DelayedVelocityCfg(**vars(cfg.actions['wheel_vel']))
  cfg.events['vision_camera_reset']=EventTermCfg(func=reset_camera,mode='reset')
  cfg.observations['depth']=ObservationGroupCfg(
      terms={'image':ObservationTermCfg(func=depth_observation,
          params={'camera':camera or DepthConfig(),'noise':True,'randomize':randomize and not play})},
      concatenate_terms=True,flatten_history_dim=False,enable_corruption=False)
  return cfg
