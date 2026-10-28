"""Local browser controls. Latest-frame storage only; simulation owns MuJoCo."""
import errno
import json
import math
from pathlib import Path
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class ControlState:
  def __init__(self, timeout=2.):
    self.speed, self.lateral_speed, self.turn_speed = 2., 1., 3.
    self.lock = threading.RLock()
    self.values = (0., 0., 0.)
    self.last_seen = time.monotonic()
    self.timeout = timeout
    self.paused = False
    self.reset_requested = False
    self.camera = [140., -20., 2.5]
    self.scene_requested = None
    self.loading = False
    self.jpeg = b""
    self.status = {"ready": False, "actual": [0., 0., 0.]}

  def clear(self):
    with self.lock:
      self.values = (0., 0., 0.)

  def command(self):
    with self.lock:
      if time.monotonic() - self.last_seen > self.timeout:
        self.clear()
      return (0., 0., 0.) if self.paused or self.loading else self.values

  def update(self, request):
    if not isinstance(request, dict):
      raise ValueError("Expected an object")
    action = request.get("action", "command")
    if action not in ("command", "heartbeat", "stop", "pause", "reset", "camera_reset", "scene"):
      raise ValueError("Unknown action")
    values = None
    if action == "command":
      values = request.get("values")
      if not isinstance(values, list) or len(values) != 3:
        raise ValueError("Expected vx, vy, yaw_rate")
      if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
        raise ValueError("Commands must be finite numbers")
      values = tuple(max(-limit, min(limit, value)) for value, limit in
                     zip(values, (self.speed, self.lateral_speed, self.turn_speed)))
    camera_delta = request.get("camera_delta", [0., 0., 0.])
    if (not isinstance(camera_delta, list) or len(camera_delta) != 3
        or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in camera_delta)):
      raise ValueError("Invalid camera delta")
    scene = None
    if action == "scene":
      kind, level = request.get("kind"), request.get("level")
      if kind not in ("flat", "up", "down", "platform", "slope", "rubble") or type(level) is not int or level not in range(6):
        raise ValueError("Invalid terrain selection")
      if not self.status.get("terrain", False):
        raise ValueError("Terrain selection requires a terrain policy")
      scene = (kind, level)
    with self.lock:
      self.camera[0] = (self.camera[0] + camera_delta[0]) % 360
      self.camera[1] = max(-89., min(0., self.camera[1] + camera_delta[1]))
      self.camera[2] = max(.5, min(12., self.camera[2] * math.exp(max(-2., min(2., camera_delta[2])))))
      if action == "camera_reset":
        self.camera = [140., -20., 2.5]
      if scene is not None:
        self.scene_requested = scene
        self.loading = True
        self.clear()
      # Expire stale commands before a late heartbeat can revive them.
      self.command()
      self.last_seen = time.monotonic()
      if values is not None and not self.loading:
        self.values = values
      elif action in ("stop", "reset", "pause"):
        self.clear()
        if action == "pause":
          self.paused = not self.paused
        elif action == "reset":
          self.reset_requested = True
      return self.snapshot()

  def snapshot(self):
    with self.lock:
      return dict(self.status, command=list(self.command()), paused=self.paused, loading=self.loading, camera=list(self.camera))

  def take_scene(self):
    with self.lock:
      scene, self.scene_requested = self.scene_requested, None
      return scene

  def finish_scene(self):
    with self.lock:
      self.loading = self.scene_requested is not None
      self.clear()

  def take_reset(self):
    with self.lock:
      reset, self.reset_requested = self.reset_requested, False
      return reset

  def publish(self, actual, step, **metadata):
    with self.lock:
      self.status.update(ready=True, actual=actual, step=step, **metadata)


def make_server(state, port=8765):
  html = Path(__file__).with_name("gui.html").read_bytes()

  class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
      pass

    def send(self, status, data, kind):
      self.send_response(status)
      self.send_header("Content-Type", kind)
      self.send_header("Content-Length", str(len(data)))
      self.send_header("Cache-Control", "no-store")
      self.end_headers()
      try:
        self.wfile.write(data)
      except (BrokenPipeError, ConnectionResetError):
        pass

    def do_GET(self):
      route = self.path.split("?", 1)[0]
      if route == "/":
        self.send(200, html, "text/html; charset=utf-8")
      elif route == "/api/state":
        self.send(200, json.dumps(state.snapshot()).encode(), "application/json")
      elif route == "/frame.jpg":
        with state.lock:
          jpeg = state.jpeg
        self.send(200 if jpeg else 503, jpeg, "image/jpeg")
      else:
        self.send(404, b"Not found", "text/plain")

    def do_POST(self):
      if self.path != "/api/control":
        self.send(404, b"Not found", "text/plain")
        return
      # Local-only API: browser requests must come from this page's origin.
      origin = self.headers.get("Origin")
      if origin and origin != "http://" + self.headers.get("Host", ""):
        self.send(403, b"Wrong origin", "text/plain")
        return
      try:
        length = int(self.headers.get("Content-Length", "0"))
        if not 0 < length <= 2048:
          raise ValueError("Invalid body length")
        result = state.update(json.loads(self.rfile.read(length)))
        self.send(200, json.dumps(result).encode(), "application/json")
      except (ValueError, TypeError) as error:
        self.send(400, json.dumps({"error": str(error)}).encode(), "application/json")

  requested_port = port
  while True:
    try:
      server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
      break
    except OSError as error:
      if error.errno != errno.EADDRINUSE or port >= 65535:
        raise
      port += 1
  if port != requested_port:
    print(f"Port {requested_port} is occupied; using http://127.0.0.1:{server.server_port}/", flush=True)
  server.daemon_threads = True
  return server
