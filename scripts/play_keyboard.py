"""Keyboard velocity control of the trained Go2W policy (one environment)."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import dataclass, field
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / "checkpoints/go2w_flat_40000/model_40000.pt"


@dataclass
class KeyboardInput:
  speed: float = 0.6
  lateral_speed: float = 0.2
  turn_speed: float = 0.4
  keys: set[str] = field(default_factory=set)

  def clear(self):
    self.keys.clear()

  def command(self):
    if "space" in self.keys:
      return (0., 0., 0.)
    x = float("up" in self.keys) - float("down" in self.keys)
    y = float("left" in self.keys) - float("right" in self.keys)
    scale = max(1., math.hypot(x, y))
    yaw = float("q" in self.keys) - float("e" in self.keys)
    return (self.speed * x / scale, self.lateral_speed * y / scale, self.turn_speed * yaw)


def configure_rendering(headless, backend="auto"):
  backend = ("egl" if headless else "glfw") if backend == "auto" else backend
  os.environ["MUJOCO_GL"] = backend
  os.environ["SDL_FRAMEBUFFER_ACCELERATION"] = "0"
  os.environ["SDL_RENDER_DRIVER"] = "software"
  os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
  return backend


def keyboard_command_cfg(controls):
  from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand, UniformVelocityCommandCfg

  class KeyboardVelocity(UniformVelocityCommand):
    def _resample_command(self, env_ids):
      self._update_command(env_ids)

    def _update_command(self, env_ids=None):
      # Commands are in the robot body frame, exactly as during training.
      for axis, value in enumerate(controls.command()):
        self.vel_command_b[:, axis] = value

    def reset(self, env_ids=None):
      controls.clear()
      return super().reset(env_ids)

  @dataclass(kw_only=True)
  class KeyboardVelocityCfg(UniformVelocityCommandCfg):
    def build(self, env):
      return KeyboardVelocity(self, env)

  return KeyboardVelocityCfg(
    entity_name="robot", resampling_time_range=(1e9, 1e9),
    heading_command=False, rel_heading_envs=0., rel_standing_envs=0.,
    rel_forward_envs=0., debug_vis=True,
    ranges=UniformVelocityCommandCfg.Ranges(
      lin_vel_x=(-controls.speed, controls.speed),
      lin_vel_y=(-controls.lateral_speed, controls.lateral_speed),
      ang_vel_z=(-controls.turn_speed, controls.turn_speed)),
  )


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--terrain-level", type=int, default=0, choices=range(6))
  p.add_argument("--terrain-kind", choices=("flat","up","down","platform","slope","rubble"), default="rubble")
  p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
  p.add_argument("--device", default="cpu", help="One CPU environment by default to limit GPU memory.")
  p.add_argument("--speed", type=float, default=0.6)
  p.add_argument("--lateral-speed", type=float, default=0.2)
  p.add_argument("--turn-speed", type=float, default=0.4)
  p.add_argument("--width", type=int, default=800)
  p.add_argument("--height", type=int, default=600)
  p.add_argument("--headless", action="store_true")
  p.add_argument("--steps", type=int, default=0, help="0: interactive until Esc; otherwise exit after N steps.")
  p.add_argument("--demo", action="store_true", help="Automatically exercise forward/stop/left/turn/stop.")
  p.add_argument("--video", type=Path, help="Stream rendered frames to MP4.")
  p.add_argument("--gl-backend", choices=("auto", "glfw", "egl"), default="auto")
  args = p.parse_args()
  if not args.checkpoint.is_file():
    p.error(f"Checkpoint not found: {args.checkpoint}")
  if args.steps < 0 or (args.headless and args.steps == 0):
    p.error("Headless mode requires --steps > 0; steps cannot be negative")
  if min(args.width, args.height) < 320 or args.width % 2 or args.height % 2:
    p.error("width and height must be even and at least 320")
  for name, value, maximum in (("speed", args.speed, 1.), ("lateral-speed", args.lateral_speed, 0.2),
                               ("turn-speed", args.turn_speed, 0.5)):
    if not math.isfinite(value) or not 0 < value <= maximum:
      p.error(f"{name} must be in (0, {maximum}] to stay within the training command range")
  backend = configure_rendering(args.headless, args.gl_backend)

  import torch
  from mjlab.envs import ManagerBasedRlEnv
  from go2w_supermimic.core.bridge import Go2WBridge
  from go2w_supermimic.core.policy import load_actor

  pg = None
  if not args.headless:
    try:
      import pygame as pg
    except ImportError:
      p.error("Keyboard window requires pygame: .venv/bin/python -m pip install 'pygame>=2.5'")
  checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
  actor = load_actor(checkpoint, args.device)
  controls = KeyboardInput(args.speed, args.lateral_speed, args.turn_speed)
  from go2w_supermimic.core.policy import evaluation_env_cfg
  cfg = evaluation_env_cfg(checkpoint, args.terrain_level, args.terrain_kind)
  cfg.scene.num_envs = 1
  # Interactive playback continues through falls, contacts and course exits.
  # mjlab disallows further steps after a termination with auto_reset=False.
  cfg.auto_reset = False
  cfg.terminations = {}
  cfg.seed = 0
  cfg.episode_length_s = 1e9  # No automatic timed teleport during keyboard play.
  cfg.commands["twist"] = keyboard_command_cfg(controls)
  cfg.viewer.width, cfg.viewer.height = args.width, args.height
  cfg.viewer.distance, cfg.viewer.elevation, cfg.viewer.azimuth = 2.5, -20., 140.

  with ExitStack() as stack:
    env = ManagerBasedRlEnv(cfg=cfg, device=args.device,
                            render_mode="rgb_array" if pg or args.video else None)
    stack.callback(env.close)
    obs, _ = env.reset()
    bridge = Go2WBridge(env)
    if checkpoint["env_action_order"] != bridge.to_env or abs(checkpoint["dt"] - env.step_dt) > 1e-8:
      raise ValueError("Checkpoint action order/control timestep differs from this environment")
    if pg:
      stack.callback(pg.quit)
      # Open SDL only after MuJoCo initialization; SDL displays RGB in software.
      pg.display.init()
      pg.font.init()
      screen = pg.display.set_mode((args.width, args.height))
      pg.display.set_caption("Go2W | Arrows move | Q/E turn | Space stop | R reset | Esc quit")
      font = pg.font.Font(None, 22)
      clock = pg.time.Clock()
      key_names = {pg.K_UP: "up", pg.K_DOWN: "down", pg.K_LEFT: "left", pg.K_RIGHT: "right",
                   pg.K_q: "q", pg.K_e: "e", pg.K_SPACE: "space"}
    video = None
    if args.video:
      import mediapy
      args.video.parent.mkdir(parents=True, exist_ok=True)
      video = stack.enter_context(mediapy.VideoWriter(str(args.video), shape=(args.height, args.width),
                                                       fps=round(1 / env.step_dt)))
    iteration = checkpoint["step"]
    del checkpoint  # Drop optimizer/world-model tensors; inference uses actor only.
    print(f"Loaded Go2W iteration {iteration}: {args.checkpoint}", flush=True)
    print(f"One environment | {args.device} | {backend} | {env.step_dt:.3f}s control period", flush=True)
    print("Arrows: forward/back/strafe | Q/E: turn | release/Space: zero command | R: reset | Esc: quit", flush=True)
    print("Keyboard changes enter the next observation (one control-step latency).", flush=True)
    step, running = 0, True
    with torch.inference_mode():
      while running and (args.steps == 0 or step < args.steps):
        if pg:
          for event in pg.event.get():
            if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE):
              running = False
            elif event.type == pg.WINDOWFOCUSLOST:
              controls.clear()
            elif event.type == pg.KEYDOWN:
              if event.key == pg.K_r:
                controls.clear()
                obs, _ = env.reset()
              elif event.key in key_names:
                controls.keys.add(key_names[event.key])
            elif event.type == pg.KEYUP and event.key in key_names:
              controls.keys.discard(key_names[event.key])
        if not running:
          break
        if args.demo:
          phases = ({"up"}, set(), {"left"}, {"q"}, set())
          controls.keys = set(phases[(step // 100) % len(phases)])
        action = actor(obs["actor"])
        if not torch.isfinite(action).all():
          raise RuntimeError("Non-finite policy action")
        obs, _, terminated, _, _ = env.step(bridge.env_action(action))
        if not torch.isfinite(obs["actor"]).all():
          raise RuntimeError("Non-finite observation")
        if pg or video:
          rgb = env.render()
          if video:
            video.add_image(rgb)
          if pg:
            pg.surfarray.blit_array(screen, rgb.swapaxes(0, 1))
            vx, vy, yaw = controls.command()
            actual = bridge.robot.data
            lines = [f"Go2W model {iteration} | Arrows move | Q/E turn | Space stop | R reset | Esc exit",
                     f"Command: vx {vx:+.2f}  vy {vy:+.2f} m/s  yaw {yaw:+.2f} rad/s",
                     f"Actual: vx {actual.root_link_lin_vel_b[0, 0]:+.2f}  vy {actual.root_link_lin_vel_b[0, 1]:+.2f}"
                     f"  yaw {actual.root_link_ang_vel_b[0, 2]:+.2f}"]
            for i, line in enumerate(lines):
              screen.blit(font.render(line, True, (240, 240, 240), (20, 24, 30)), (8, 8 + i * 22))
            pg.display.flip()
            clock.tick(round(1 / env.step_dt))
        step += 1
    print(f"Completed {step} control steps.", flush=True)


if __name__ == "__main__":
  main()
