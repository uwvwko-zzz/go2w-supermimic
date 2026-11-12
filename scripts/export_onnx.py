"""Export Go2W terrain or vision actor checkpoints to ONNX."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from go2w_supermimic.core.policy import load_actor as load_blind
from go2w_supermimic.vision.policy import load_actor as load_vision


def export(path, output=None):
  import onnx
  import onnxruntime as ort

  checkpoint = torch.load(path, map_location="cpu", weights_only=True)
  vision = checkpoint.get("format") == "go2w-vision-policy-v2"
  actor = (load_vision if vision else load_blind)(checkpoint, "cpu")
  dim = actor.norm.mean.numel()
  inputs = (torch.randn(2, dim),)
  names = ["proprioception"]
  if vision:
    camera = checkpoint["camera"]
    inputs += (torch.rand(2, 1, camera["height"], camera["width"]),)
    names.append("depth")
  if vision:
    # Adaptive pooling 5x8 -> 3x4 has overlapping bins and cannot be
    # represented by ONNX AveragePool with a single fixed stride.
    class AdaptivePool(torch.nn.Module):
      def forward(self, x):
        rows = []
        for i in range(3):
          cols = []
          for j in range(4):
            cols.append(x[:, :, i * 5 // 3:((i + 1) * 5 + 2) // 3, j * 2:(j + 1) * 2].mean((2, 3), keepdim=True))
          rows.append(torch.cat(cols, dim=3))
        return torch.cat(rows, dim=2)
    actor.encoder[4] = AdaptivePool()
  output = output or path.with_suffix(".onnx")
  output.parent.mkdir(parents=True, exist_ok=True)
  dynamic = {name: {0: "batch"} for name in names + ["action"]}
  with torch.no_grad():
    expected = actor(*inputs).numpy()
    torch.onnx.export(actor, inputs, str(output), input_names=names,
                      output_names=["action"], dynamic_axes=dynamic,
                      opset_version=17, dynamo=False)
  model = onnx.load(str(output))
  onnx.checker.check_model(model)
  session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
  actual = session.run(None, {n: x.numpy() for n, x in zip(names, inputs)})[0]
  np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
  metadata = {k: checkpoint[k] for k in ("format", "history", "dt", "step", "env_action_order", "camera", "config") if k in checkpoint}
  metadata.update(inputs={n: list(x.shape[1:]) for n, x in zip(names, inputs)}, output_action_dim=16,
                  actor_wheel_vel=actor.actor_wheel_vel)
  output.with_suffix('.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + '\n')
  print(f"{output}: inputs {metadata['inputs']}, max error {np.max(np.abs(actual-expected)):.2g}")


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--model', type=Path, required=True, help='Path to the .pt checkpoint')
  args = parser.parse_args()
  export(args.model)

if __name__ == '__main__':
  main()
