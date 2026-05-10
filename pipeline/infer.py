import onnxruntime as ort
import numpy as np, time

class OrtInferencer:
    def __init__(self, model_path: str, threads: int = 4):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL)
        self.session = ort.InferenceSession(
            model_path, sess_options=opts,
            providers=["CPUExecutionProvider"])
        self.inp = self.session.get_inputs()[0].name
        self._warmup()

    def _warmup(self, n=3):
        d = np.random.rand(1,3,64,64).astype(np.float32)
        for _ in range(n): self.session.run(None, {self.inp: d})

    def run(self, tensor: np.ndarray) -> dict:
        t0 = time.perf_counter()
        out = self.session.run(None, {self.inp: tensor})
        return {"logits": out[0],
                "latency_ms": round((time.perf_counter()-t0)*1000, 2)}