"""Go2W browser sliders with one CPU environment and bounded JPEG storage."""
import argparse
import gc
from io import BytesIO
import os
from pathlib import Path
import threading
import time

from play_keyboard import DEFAULT_CHECKPOINT, keyboard_command_cfg
from go2w_supermimic.core.gui import ControlState, make_server


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
  parser.add_argument("--port", type=int, default=8765)
  parser.add_argument("--device", default="cpu")
  parser.add_argument("--terrain-kind", choices=("flat", "up", "down", "platform", "slope", "rubble"), default="rubble")
  parser.add_argument("--terrain-level", type=int, choices=range(6), default=0)
  parser.add_argument("--fps", type=int, default=50)
  parser.add_argument("--shadows", action="store_true", help="Enable costly shadows/reflections")
  parser.add_argument("--steps", type=int, default=0, help="Exit after N simulation steps; 0 means until Ctrl+C")
  args = parser.parse_args()
  if not args.checkpoint.is_file():
    parser.error(f"Checkpoint not found: {args.checkpoint}")
  if not 1 <= args.fps <= 50 or not 1 <= args.port <= 65535 or args.steps < 0:
    parser.error("Require fps 1..50, port 1..65535 and nonnegative steps")
  os.environ.setdefault("MUJOCO_GL", "egl")
  os.environ.setdefault("OMP_NUM_THREADS", "1")
  import torch
  from PIL import Image
  from mjlab.envs import ManagerBasedRlEnv
  from go2w_supermimic.core.bridge import Go2WBridge
  from go2w_supermimic.core.policy import load_actor, evaluation_env_cfg

  torch.set_num_threads(1)
  state = ControlState()
  server = make_server(state, args.port)
  env = None
  worker = None
  try:
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    actor = load_actor(checkpoint, args.device)
    # Retain only metadata required to rebuild a scene, not training tensors.
    checkpoint = {k: checkpoint[k] for k in ("format", "history", "config", "env_action_order", "dt", "step")}
    checkpoint["actor_wheel_vel"] = actor.actor_wheel_vel
    def build_scene(kind, level):
      cfg = evaluation_env_cfg(checkpoint, level, kind)
      cfg.scene.num_envs = 1
      # Interactive playback continues through falls, contacts and course exits.
      # mjlab disallows further steps after a termination with auto_reset=False.
      cfg.auto_reset = False
      cfg.terminations = {}
      # Playback only needs the actor: no training rewards, terrain reward scans,
      # critic observations or training metrics are consumed by the GUI.
      cfg.rewards = {}
      cfg.metrics = {}
      cfg.observations.pop("critic", None)
      cfg.episode_length_s = 1e9
      cfg.commands["twist"] = keyboard_command_cfg(state)
      cfg.viewer.width, cfg.viewer.height = 640, 480
      cfg.viewer.enable_shadows = args.shadows
      cfg.viewer.enable_reflections = args.shadows
      cfg.viewer.distance, cfg.viewer.elevation, cfg.viewer.azimuth = 2.5, -20., 140.
      new_env = ManagerBasedRlEnv(cfg=cfg, device=args.device, render_mode="rgb_array")
      bridge = Go2WBridge(new_env)
      if checkpoint["env_action_order"] != bridge.to_env or abs(checkpoint["dt"] - new_env.step_dt) > 1e-8:
        new_env.close()
        raise ValueError("Checkpoint action order/control timestep mismatch")
      obs, _ = new_env.reset()
      return new_env, bridge, obs

    kind, level = args.terrain_kind, args.terrain_level
    env, bridge, obs = build_scene(kind, level)
    metadata = dict(checkpoint=args.checkpoint.name, iteration=checkpoint["step"],
                    terrain=checkpoint["format"] == "go2w-terrain-policy-v1", device=args.device,
                    terrain_revision=checkpoint["config"].get("terrain_revision", 1))
    metadata.update(scene=f"{kind} · 等级 {level}", terrain_kind=kind, terrain_level=level)
    state.publish([0., 0., 0.], 0, **metadata)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    print(f"Go2W GUI ready: http://127.0.0.1:{server.server_port}/", flush=True)
    print(f"One environment on {args.device}. Ctrl+C exits. Commands expire after 2 seconds without a browser heartbeat.", flush=True)
    step, next_render = 0, 0.
    stats_since = time.monotonic()
    stats_step, frames = 0, 0
    render_fps = realtime_factor = 0.
    with torch.inference_mode():
      while not args.steps or step < args.steps:
        began = time.monotonic()
        selection = state.take_scene()
        if selection is not None:
          # Release the old simulator before constructing the new one.
          state.clear()
          env.close()
          del bridge, obs
          d = None
          env = None
          gc.collect()
          if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
          kind, level = selection
          env, bridge, obs = build_scene(kind, level)
          metadata.update(scene=f"{kind} · 等级 {level}", terrain_kind=kind, terrain_level=level)
          state.finish_scene()
          stats_since = next_render = time.monotonic()
          stats_step, frames = step, 0
        if state.take_reset():
          obs, _ = env.reset()
        with state.lock:
          paused = state.paused
        if not paused:
          action = actor(obs["actor"])
          if not torch.isfinite(action).all():
            raise RuntimeError("Non-finite policy action")
          obs, _, terminated, truncated, _ = env.step(bridge.env_action(action))
          if not torch.isfinite(obs["actor"]).all():
            raise RuntimeError("Non-finite observation")
          step += 1
        d = bridge.robot.data
        actual = [d.root_link_lin_vel_b[0, 0].item(), d.root_link_lin_vel_b[0, 1].item(),
                  d.root_link_ang_vel_b[0, 2].item()]
        state.publish(actual, step, render_fps=render_fps, realtime_factor=realtime_factor, **metadata)
        if began >= next_render:
          # Only the simulation thread touches the native renderer camera.
          with state.lock:
            azimuth, elevation, distance = state.camera
          camera = env._offline_renderer._cam
          camera.azimuth, camera.elevation, camera.distance = azimuth, elevation, distance
          buffer = BytesIO()
          Image.fromarray(env.render()).save(buffer, format="JPEG", quality=80)
          with state.lock:
            state.jpeg = buffer.getvalue()
          frames += 1
          # Schedule from frame start, not completion: rendering time is part
          # of the frame budget rather than an extra delay on every frame.
          next_render += 1 / args.fps
          if next_render < began:
            next_render = began + 1 / args.fps
        now = time.monotonic()
        if now - stats_since >= 1.:
          render_fps = frames / (now - stats_since)
          realtime_factor = (step - stats_step) * env.step_dt / (now - stats_since)
          frames, stats_step, stats_since = 0, step, now
        time.sleep(max(0., env.step_dt - (time.monotonic() - began)))
    print(f"Render: {render_fps:.1f} FPS | realtime: {realtime_factor:.2f}x", flush=True)
  except KeyboardInterrupt:
    print("Closing Go2W GUI.", flush=True)
  finally:
    state.clear()
    if worker is not None:
      server.shutdown()
      worker.join(timeout=2.)
    server.server_close()
    if env is not None:
      env.close()


if __name__ == "__main__":
  main()
