"""ONNX Runtime inference and exported environment metadata."""
import json
from pathlib import Path


def load_onnx(path, vision):
  import onnxruntime as ort
  import torch

  path = Path(path)
  metadata_path = path.with_suffix('.json')
  if not metadata_path.is_file():
    raise FileNotFoundError(f'Missing metadata {metadata_path}; re-export using scripts/export_onnx.py')
  metadata = json.loads(metadata_path.read_text())
  if 'config' not in metadata:
    raise ValueError(f'{metadata_path} lacks config; re-export using scripts/export_onnx.py')
  is_vision = metadata.get('format') == 'go2w-vision-policy-v2'
  if is_vision != vision:
    raise ValueError('ONNX policy type does not match this GUI')
  session = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
  expected = ['proprioception', 'depth'] if vision else ['proprioception']
  if [i.name for i in session.get_inputs()] != expected:
    raise ValueError(f'Unexpected ONNX inputs; expected {expected}')

  def actor(*inputs):
    feeds = {name: value.detach().cpu().numpy() for name, value in zip(expected, inputs)}
    return torch.from_numpy(session.run(['action'], feeds)[0]).to(inputs[0].device)

  return metadata, actor
