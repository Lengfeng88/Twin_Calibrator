import torch
import numpy as np
import onnxruntime as ort
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from train.model import LightRegressor

PT_PATH   = "models/light_regressor.pt"
ONNX_PATH = "models/light_regressor.onnx"

# ── Load the trained weights ────────────────────────────
model = LightRegressor()
model.load_state_dict(torch.load(PT_PATH, map_location="cpu"))
model.eval()

# ── Export ONNX ───────────────────────────────────
dummy = torch.randn(1, 3, 64, 64)
torch.onnx.export(
    model, dummy, ONNX_PATH,
    input_names=["image"],
    output_names=["light_pred"],
    dynamic_axes={"image": {0: "batch"}, "light_pred": {0: "batch"}},
    opset_version=17,
    do_constant_folding=True
)
print(f"Export complete: {ONNX_PATH}")

# ── Numerical consistency verification ──────────────────────────────
sess = ort.InferenceSession(ONNX_PATH,
           providers=["CPUExecutionProvider"])

test_input = np.random.rand(4, 3, 64, 64).astype(np.float32)

with torch.no_grad():
    pt_out = model(torch.from_numpy(test_input)).numpy()

ort_out = sess.run(None, {"image": test_input})[0]

max_diff = float(np.max(np.abs(pt_out - ort_out)))
print(f"PyTorch vs ONNX Maximum error: {max_diff:.2e}")
assert max_diff < 1e-4, "Inconsistent values, check the exported parameters."
print("The verification has passed, and the model can be deployed securely.")

# ── Print model information ────────────────────────────────
import onnx
m = onnx.load(ONNX_PATH)
size_mb = os.path.getsize(ONNX_PATH) / 1024**2
print(f"Model size: {size_mb:.2f} MB")
print(f"Input:  {[d.dim_value for d in m.graph.input[0].type.tensor_type.shape.dim]}")
print(f"Output:  {[d.dim_value for d in m.graph.output[0].type.tensor_type.shape.dim]}")
