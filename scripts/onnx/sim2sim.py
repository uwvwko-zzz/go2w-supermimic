"""Native MuJoCo + ONNX Runtime sim2sim playback (no mjlab)."""
import argparse
import json
import math
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import sys
import threading
from io import BytesIO
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[2]
LEGS = ('FL', 'FR', 'RL', 'RR')
LEG_NAMES = [f'{leg}_{joint}_joint' for leg in LEGS for joint in ('hip', 'thigh', 'calf')]
WHEEL_NAMES = [f'{leg}_wheel_joint' for leg in LEGS]


def main(vision):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--onnx', required=True, type=Path)
  parser.add_argument('--vx', type=float, default=0.)
  parser.add_argument('--vy', type=float, default=0.)
  parser.add_argument('--yaw', type=float, default=0.)
  parser.add_argument('--port', type=int, default=8765)
  parser.add_argument('--fps', type=int, default=30)
  parser.add_argument('--steps', type=int, default=0, help='Control steps; 0 runs until viewer closes')
  display = parser.add_mutually_exclusive_group()
  display.add_argument('--gui', action='store_true', help='Open the native MuJoCo viewer (default)')
  display.add_argument('--headless', action='store_true', help='Run without a viewer')
  args = parser.parse_args()
  meta = json.loads(args.onnx.with_suffix('.json').read_text())
  if (meta['format'] == 'go2w-vision-policy-v2') != vision:
    parser.error('Model type does not match this script')
  if meta.get('actor_wheel_vel', False):
    parser.error('Legacy wheel-velocity observation is not supported')
  if args.headless and args.steps <= 0:
    parser.error('--headless requires --steps')
  session = ort.InferenceSession(str(args.onnx), providers=['CPUExecutionProvider'])
  xml_path = ROOT/'go2w_supermimic/robot/model/go2w.xml'
  def build_model(kind='flat', level=0):
    tree = ET.parse(xml_path)
    tree.find('compiler').set('meshdir', str(xml_path.parent/'assets'))
    visual = tree.find('visual')
    if visual is None:
      visual = ET.SubElement(tree.getroot(), 'visual')
    ET.SubElement(visual, 'headlight', ambient='.25 .25 .25', diffuse='.5 .5 .5', specular='.1 .1 .1')
    world = tree.find('worldbody')
    ET.SubElement(world, 'light', name='scene_sun', directional='true', pos='0 0 8', dir='-.3 -.4 -1',
                  ambient='.15 .15 .15', diffuse='.45 .45 .45', specular='.1 .1 .1', castshadow='false')
    for geom in world.iter('geom'):
      if geom.get('class') != 'visual': geom.set('margin','0')
    actuators=tree.find('actuator')
    for name in LEG_NAMES:
      calf='_calf_' in name
      ET.SubElement(actuators,'position',name='native_'+name,joint=name,
                    kp='40' if calf else '20',kv='2' if calf else '1',
                    ctrllimited='false',forcelimited='true',forcerange='-45 45' if calf else '-23.5 23.5')
    ET.SubElement(world, 'geom', name='ground', type='box', pos='0 0 -.1', size='50 50 .1', group='0', friction='1 .005 .0001', rgba='.42 .46 .42 1')
    h=(.02,.04,.08,.12,.16,.20)[level]
    def box(x,y,z,sx,sy,sz,angle=0.):
      ET.SubElement(world,'geom',type='box',pos=f'{x} {y} {z}',size=f'{sx/2} {sy/2} {sz/2}',
                    quat=f'{math.cos(angle/2)} 0 {math.sin(angle/2)} 0',group='0',friction='0.8 0.02 0.001',rgba='.48 .48 .42 1')
    if kind in ('up','down'):
      tread=.30 if meta.get('config',{}).get('terrain_revision',1)>=4 else .45
      for direction in (-1,1):
        for k in range(5):
          height=(k+1)*h if kind=='up' else (5-k)*h
          x=direction*(1.+k*tread+tread/2)
          if kind=='up':
            length=6.-k*tread; x=direction*(1.+k*tread+length/2)
          else: length=tread
          box(x,0,height/2,length,8,height)
      if kind=='down': box(0,0,2.5*h,2,8,5*h)
    elif kind=='platform':
      height=(.1,.2,.3,.4,.5,.6)[level] if vision else h
      for direction in (-1,1):
        if vision: box(direction*4.75,0,height/2,6.5,8,height)
        else:
          for x in (1.6,3.6): box(direction*x,0,height/2,.9,8,height)
    elif kind=='slope':
      a=math.radians(3+12*level/5)
      for direction in (-1,1):
        box(direction*2.5,0,1.5*math.sin(a)-.08*math.cos(a),3,8,.16,-direction*a)
        height=3*math.sin(a); box(direction*5.3,0,height/2,2.6,8,height)
    elif kind=='rubble':
      rng=np.random.default_rng(meta.get('config',{}).get('seed',0))
      for direction in (-1,1):
        for x in np.arange(1.15,5.,.48):
          for y in np.arange(-3.5,3.5,.52):
            height=h*rng.uniform(.35,.7); sx,sy=rng.uniform(.35,.65,2)
            box(direction*(x+rng.uniform(-.12,.12)),y,height/2,sx,sy,height)
    return mujoco.MjModel.from_xml_string(ET.tostring(tree.getroot(), encoding='unicode'))
  model = build_model()
  model.opt.timestep = .005
  model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
  model.opt.iterations = 10; model.opt.ls_iterations = 20
  model.opt.impratio = 10
  model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
  data = mujoco.MjData(model)
  joint_ids = np.array([model.joint(name).id for name in LEG_NAMES])
  wheel_ids = np.array([model.joint(name).id for name in WHEEL_NAMES])
  qidx = model.jnt_qposadr[joint_ids]; vidx = model.jnt_dofadr[joint_ids]
  wheel_ctrl = np.array([model.actuator(name).id for name in WHEEL_NAMES])
  leg_ctrl = np.array([model.actuator('native_'+name).id for name in LEG_NAMES])
  for idx, name in enumerate(LEG_NAMES):
    model.dof_armature[vidx[idx]] = .02 if '_calf_' in name else .01
  nominal = np.array([v for leg in LEGS for v in ((-.1 if leg.endswith('L') else .1), .9, -1.8)])
  kp = np.array([v for _ in LEGS for v in (20., 20., 40.)])
  kd = np.array([v for _ in LEGS for v in (1., 1., 2.)])
  limits = np.array([v for _ in LEGS for v in (23.5, 23.5, 45.)])
  data.qpos[2] = .44; data.qpos[qidx] = nominal
  mujoco.mj_forward(model, data)
  history_len = meta['history']; widths = (3, 3, 12, 12, 16, 3)
  histories = [np.zeros((history_len, n), np.float32) for n in widths]
  first_frame = True
  last = np.zeros(16, np.float32); command = np.array((args.vx,args.vy,args.yaw),np.float32)
  control_dt = float(meta['dt']); substeps = round(control_dt/model.opt.timestep)
  if abs(substeps*model.opt.timestep-control_dt)>1e-8: raise ValueError('Unsupported control timestep')
  camera = meta.get('camera', {})
  def depth():
    h,w = camera['height'],camera['width']; f=h/(2*math.tan(math.radians(camera['fovy_deg'])/2))
    yy,xx=np.meshgrid(np.arange(h),np.arange(w),indexing='ij')
    rays=np.stack((np.ones_like(xx),(w/2-xx-.5)/f,(h/2-yy-.5)/f),-1).reshape(-1,3)
    rays=rays/np.linalg.norm(rays,axis=1,keepdims=True)
    a=math.radians(camera['pitch_down_deg']); tilt=np.array([[math.cos(a),0,math.sin(a)],[0,1,0],[-math.sin(a),0,math.cos(a)]])
    rot=data.xmat[model.body('base_link').id].reshape(3,3); pos=data.xpos[model.body('base_link').id]+rot@np.array(camera['position'])
    directions=(rays@tilt.T)@rot.T; gids=np.empty(1,dtype=np.int32); groups=np.array([1,0,0,0,0,0],dtype=np.uint8)
    distances=np.array([mujoco.mj_ray(model,data,pos,d,groups,1,-1,gids) for d in directions])
    values=np.where((distances>=0)&(distances<=camera['far']), distances*rays[:,0], camera['far'])
    return ((np.clip(values,camera['near'],camera['far'])-camera['near'])/(camera['far']-camera['near'])).astype(np.float32).reshape(1,1,h,w)
  def step():
    nonlocal last, first_frame
    base=model.body('base_link').id; rot=data.xmat[base].reshape(3,3)
    # Free-joint angular qvel is expressed in the base link frame.
    pieces=(data.qvel[3:6],rot.T@np.array([0.,0.,-1.]),data.qpos[qidx]-nominal,data.qvel[vidx]*.05,last,command)
    for hist, val in zip(histories,pieces):
      if first_frame: hist[:]=val
      else: hist[:-1]=hist[1:]; hist[-1]=val
    first_frame=False
    obs=np.concatenate([h.flatten() for h in histories])[None,:].astype(np.float32)
    feeds={'proprioception':obs}
    if vision: feeds['depth']=depth()
    last=session.run(['action'],feeds)[0][0]
    if not np.isfinite(last).all(): raise RuntimeError('Non-finite action')
    revision=meta.get('config',{}).get('terrain_revision',1)
    scales=np.tile([.25,.6,.8] if revision>=2 else [.25,.25,.25],4)
    target=nominal+last[:12]*scales; data.ctrl[leg_ctrl]=target; data.ctrl[wheel_ctrl]=np.clip(last[12:]*(35 if revision>=3 else 15),-35,35)
    for _ in range(substeps):
      mujoco.mj_step(model,data)
    mujoco.mj_forward(model,data)
  if args.gui:
    from PIL import Image
    sys.path.insert(0, str(ROOT))
    if vision:
      from go2w_supermimic.vision.gui import ControlState, make_server
    else:
      from go2w_supermimic.core.gui import ControlState, make_server
    state=ControlState(); server=make_server(state,args.port)
    worker=threading.Thread(target=server.serve_forever,daemon=True); worker.start()
    renderer=mujoco.Renderer(model, height=480, width=640)
    cam=mujoco.MjvCamera(); mujoco.mjv_defaultFreeCamera(model,cam)
    print(f'Native MuJoCo GUI ready: http://127.0.0.1:{server.server_port}/',flush=True)
    try:
      kind,level='flat',0
      count=0; next_render=0.; last_image=None
      while not args.steps or count<args.steps:
        start=time.monotonic()
        selection=state.take_scene()
        if selection is not None:
          kind,level=selection
          renderer.close()
          model=build_model(kind,level)
          model.opt.timestep=.005; model.opt.integrator=mujoco.mjtIntegrator.mjINT_IMPLICITFAST; model.opt.iterations=10; model.opt.ls_iterations=20; model.opt.impratio=10; model.opt.cone=mujoco.mjtCone.mjCONE_ELLIPTIC
          data=mujoco.MjData(model)
          for idx,name in enumerate(LEG_NAMES): model.dof_armature[vidx[idx]]=.02 if '_calf_' in name else .01
          data.qpos[2]=.44+(5*(.02,.04,.08,.12,.16,.20)[level] if kind=='down' else 0)
          data.qpos[qidx]=nominal; mujoco.mj_forward(model,data)
          last.fill(0); first_frame=True
          for hist in histories: hist.fill(0)
          renderer=mujoco.Renderer(model,height=480,width=640)
          state.finish_scene()
        if state.take_reset():
          mujoco.mj_resetData(model,data); data.qpos[2]=.44+(5*(.02,.04,.08,.12,.16,.20)[level] if kind=='down' else 0); data.qpos[qidx]=nominal
          mujoco.mj_forward(model,data); last.fill(0); first_frame=True
          for hist in histories: hist.fill(0)
        with state.lock:
          paused=state.paused; azimuth,elevation,distance=state.camera
        command[:]=state.command()
        if not paused:
          step(); count+=1
        base=model.body('base_link').id
        linear_b=data.xmat[base].reshape(3,3).T @ data.qvel[:3]
        state.publish([float(linear_b[0]),float(linear_b[1]),float(data.qvel[5])],count,
                      checkpoint=args.onnx.name,iteration=meta['step'],terrain=True,terrain_kind=kind,terrain_level=level,terrain_revision=meta.get('config',{}).get('terrain_revision',1),
                      scene=f'原生 MuJoCo {kind} · 等级 {level}',device='MuJoCo / ONNX CPU')
        if start>=next_render:
          cam.lookat[:]=data.xpos[base]; cam.azimuth=azimuth; cam.elevation=elevation; cam.distance=distance
          renderer.update_scene(data,camera=cam)
          frame=Image.fromarray(renderer.render())
          if vision:
            depth_image=(depth()[0,0]*255).astype(np.uint8)
            frame.paste(Image.fromarray(depth_image).resize((192,108)),(8,8))
          buffer=BytesIO(); frame.save(buffer,format='JPEG',quality=80)
          with state.lock: state.jpeg=buffer.getvalue()
          next_render=start+1/max(1,args.fps)
        time.sleep(max(0.,control_dt-(time.monotonic()-start)))
    except KeyboardInterrupt:
      print('Closing native MuJoCo GUI.',flush=True)
    finally:
      server.shutdown(); worker.join(timeout=2.); server.server_close(); renderer.close()
  elif args.headless:
    for i in range(args.steps):
      step()
      if os.environ.get('GO2W_TRACE') and i % 50 == 0:
        print('TRACE', i, float(data.qpos[2]), float(data.qvel[5]), flush=True)
    print(f'Completed {args.steps} native MuJoCo steps; base height={data.qpos[2]:.3f} m')
  else:
    from mujoco import viewer as mjviewer
    with mjviewer.launch_passive(model,data) as viewer:
      count=0
      while viewer.is_running() and (not args.steps or count<args.steps):
        start=time.monotonic(); step(); viewer.sync(); count+=1
        time.sleep(max(0.,control_dt-(time.monotonic()-start)))
