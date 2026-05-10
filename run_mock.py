"""
run_mock.py — 模拟验证，无需摄像头
用法: python run_mock.py --scene warm_lamp --frames 120
"""
import argparse, time
from mock.sensor       import MockSensor, PRESETS
from pipeline.preprocess import preprocess_frame
from pipeline.infer      import OrtInferencer
from pipeline.calibrate  import metrics_from_model, calibrate_from_metrics
from twin.state          import TwinState
from twin.synchronizer   import LightSynchronizer
from twin.report         import SyncReport, ReportStore

def run(scene, n_frames):
    sensor  = MockSensor(scene=scene)
    infer   = OrtInferencer("models/light_regressor.onnx")
    state   = TwinState()
    syncer  = LightSynchronizer()
    store   = ReportStore("twin_sync_mock.db")

    sensor.start()
    print(f"\n场景: {PRESETS[scene].name}  帧数: {n_frames}")
    print(f"{'帧':>4}  {'lum误差':>8}  {'cct误差':>8}  "
          f"{'置信':>6}  {'质量分':>6}  {'延迟ms':>7}")
    print("-" * 50)

    for i in range(1, n_frames+1):
        t0     = time.time()
        frame  = sensor.capture_array()
        tensor = preprocess_frame(frame)
        result = infer.run(tensor)
        metrics= metrics_from_model(result["logits"])
        calib  = calibrate_from_metrics(metrics)

        state.update_physical(calib)
        corr = syncer.step(state)

        if not corr.get("skipped"):
            report = SyncReport.from_state(state)
            store.write(report)
            print(f"{i:>4}  "
                  f"{report.lum_error:>+8.1f}  "
                  f"{report.cct_error:>+8.0f}  "
                  f"{report.confidence:>6.3f}  "
                  f"{report.sync_score:>6.3f}  "
                  f"{result['latency_ms']:>7.1f}")

        time.sleep(max(0, 0.5-(time.time()-t0)))

    sensor.stop()
    print(f"\n完成。数据库: twin_sync_mock.db")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scene",  default="office",
                   choices=list(PRESETS.keys()))
    p.add_argument("--frames", type=int, default=120)
    args = p.parse_args()
    run(args.scene, args.frames)