import torch
import numpy as np
import onnxruntime as ort
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from train.model import LightRegressor

PT_PATH   = "models/light_regressor.pt"
ONNX_PATH = "models/light_regressor.onnx"

# ── 加载训练好的权重 ────────────────────────────
model = LightRegressor()
model.load_state_dict(torch.load(PT_PATH, map_location="cpu"))
model.eval()

# ── 导出 ONNX ───────────────────────────────────
dummy = torch.randn(1, 3, 64, 64)
torch.onnx.export(
    model, dummy, ONNX_PATH,
    input_names=["image"],
    output_names=["light_pred"],
    dynamic_axes={"image": {0: "batch"}, "light_pred": {0: "batch"}},
    opset_version=17,
    do_constant_folding=True
)
print(f"导出完成: {ONNX_PATH}")

# ── 数值一致性验证 ──────────────────────────────
sess = ort.InferenceSession(ONNX_PATH,
           providers=["CPUExecutionProvider"])

test_input = np.random.rand(4, 3, 64, 64).astype(np.float32)

with torch.no_grad():
    pt_out = model(torch.from_numpy(test_input)).numpy()

ort_out = sess.run(None, {"image": test_input})[0]

max_diff = float(np.max(np.abs(pt_out - ort_out)))
print(f"PyTorch vs ONNX 最大误差: {max_diff:.2e}")
assert max_diff < 1e-4, "数值不一致，检查导出参数"
print("验证通过，模型可安全部署。")

# ── 打印模型信息 ────────────────────────────────
import onnx
m = onnx.load(ONNX_PATH)
size_mb = os.path.getsize(ONNX_PATH) / 1024**2
print(f"模型大小: {size_mb:.2f} MB")
print(f"输入:  {[d.dim_value for d in m.graph.input[0].type.tensor_type.shape.dim]}")
print(f"输出:  {[d.dim_value for d in m.graph.output[0].type.tensor_type.shape.dim]}")