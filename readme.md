# Digital Twin Light Calibrator

A lightweight edge AI system that synchronizes a digital twin's lighting state with real-world illuminance measurements, deployed on Raspberry Pi 5.

The system captures indoor lighting conditions via camera, runs on-device inference to estimate luminance and color temperature (CCT), and closes a feedback loop to keep the digital twin's light parameters aligned with the physical environment.

---

## Architecture

```
Camera / MockSensor
    → preprocess_frame()       # BGR frame → NCHW float32 tensor
    → OrtInferencer.run()      # ONNX Runtime inference (<5ms on Pi 5)
    → metrics_from_model()     # denormalize → luminance %, CCT K
    → calibrate_from_metrics() # compute confidence score
    → TwinState.update_physical()
    → LightSynchronizer.step() # Kalman smooth + proportional correction
    → SyncReport → SQLite
```

## Project Structure

```
twin_calibrator/
├── mock/
│   └── sensor.py          # MockSensor — simulates 5 indoor lighting scenes
├── pipeline/
│   ├── preprocess.py      # frame → ONNX input tensor (64×64, NCHW)
│   ├── infer.py           # OrtInferencer — ONNX Runtime wrapper
│   └── calibrate.py       # LightMetrics, CalibrationResult
├── train/
│   ├── dataset.py         # LightDataset — synthetic 8000-frame dataset
│   ├── model.py           # LightRegressor — ~180K param CNN
│   ├── train.py           # training script (RTX 4080, ~90s)
│   └── export.py          # PyTorch → ONNX export + numerical validation
├── twin/
│   ├── state.py           # TwinState — physical + twin light state
│   ├── synchronizer.py    # LightSynchronizer — Kalman + P-controller
│   └── report.py          # SyncReport + SQLite persistence
├── models/
│   └── light_regressor.onnx
├── config.py               # centralised hyperparameter configuration
├── requirements.txt        # host machine dependencies
├── requirements-pi5.txt    # Raspberry Pi 5 dependencies (inference only)
├── run_mock.py             # simulation mode (no camera required)
└── run_hardware.py         # Pi 5 hardware mode (Picamera2)
```

## Hardware

| Component  | Spec |
|------------|------|
| Training   | Ubuntu 24.04, i9-13900, RTX 4080 12GB |
| Inference  | Raspberry Pi 5 8GB, 64GB SD card |
| Camera     | Pi Camera Module 3 (CSI, CAM0) |
| Light source | Indoor lamp (controlled environment) |

---

## Quickstart

### 1. Clone and set up environment (host machine)

```bash
git clone <repo>
cd twin_calibrator

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

# PyTorch requires a separate install with the correct CUDA version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 2. Train and export model

```bash
python -m train.train    # ~90s on RTX 4080
                         # target: MAE_lum < 3%, MAE_cct < 200K

python -m train.export   # → models/light_regressor.onnx (~0.65MB)
                         # numerical error vs PyTorch: 7.24e-8
```

### 3. Run simulation (no camera needed)

```bash
python run_mock.py --scene warm_lamp --frames 120
python run_mock.py --scene daylight  --frames 120
```

Available scenes: `daylight`, `warm_lamp`, `office`, `dim`, `bright`

Expected output — `sync_score` converging from `0.421` to `0.931` within 60 frames at 2 fps:

```
frame  lum_err  cct_err   conf   score  latency_ms
    1   +18.3    +820    0.512   0.421     3.2
   10    +9.1    +410    0.721   0.623     2.8
   30    +2.3     +95    0.891   0.847     2.9
   60    +0.8     +32    0.932   0.931     3.1
```

### 4. Deploy to Raspberry Pi 5

Transfer files:

```bash
rsync -avz --exclude='.venv' --exclude='__pycache__' \
  ~/twin_calibrator/ pi@<PI_IP>:~/twin_calibrator/
```

Set up Pi 5 environment:

```bash
# picamera2 must be installed via apt, not pip
sudo apt install -y python3-picamera2 libcamera-apps

# Create venv with access to system picamera2
python3 -m venv .venv --system-site-packages
source .venv/bin/activate

pip install -r requirements-pi5.txt
```

Run hardware mode:

```bash
python3 run_hardware.py
```

The script locks AWB and AE on startup to prevent ISP interference with CCT estimation:

```python
cam.set_controls({
    "AwbEnable": False, "ColourGains": (2.0, 1.5),
    "AeEnable":  False, "ExposureTime": 10000,
    "AnalogueGain": 1.0,
})
```

---

## Model

| Property | Value |
|----------|-------|
| Architecture | 3-block CNN + MLP regressor |
| Parameters | ~180K |
| Input | `(1, 3, 64, 64)` float32, RGB normalized |
| Output | `(1, 2)` float32 — `[lum_norm, cct_norm]` |
| Model size | 0.65 MB (.onnx) |
| Export error | 7.24e-8 (PyTorch vs ONNX) |
| Pi 5 inference | <5ms @ CPUExecutionProvider, 4 threads |

Output denormalization:

```python
luminance_pct = output[0] × 100.0          # 0–100 %
cct_k         = output[1] × 5300 + 2700    # 2700–8000 K
```

---

## Synchronizer

`LightSynchronizer` runs a proportional controller with per-channel Kalman smoothing:

```python
lum_smooth = KalmanFilter(q=0.01, r=0.5).update(physical_lum)
twin_lum  += alpha_lum × (lum_smooth − twin_lum) × confidence
```

| Parameter | Default | Effect |
|-----------|---------|--------|
| `alpha_lum` | 0.15 | luminance correction step size |
| `alpha_cct` | 0.10 | CCT correction step size |
| `conf_thresh` | 0.10 | frames below this confidence are skipped |

`sync_score` is the mean of two normalized quality terms:

```python
lum_q  = max(0, 1 − |lum_error| / 50)
cct_q  = max(0, 1 − |cct_error| / 2000)
sync_score = (lum_q + cct_q) / 2
```

---

## Dataset

`LightDataset` generates 8000 synthetic frames across 5 scenes with parametric ground-truth labels — no manual annotation required.

| Scene | Base lum | CCT |
|-------|----------|-----|
| daylight | 200 | 6500K |
| warm_lamp | 120 | 2700K |
| office | 160 | 4000K |
| dim | 60 | 3000K |
| bright | 230 | 5500K |

Labels are derived directly from generation parameters (HSV V-channel mean for luminance, preset CCT ± uniform noise for color temperature).

---

## Configuration

All hyperparameters are centralised in `config.py`:

```python
MODEL_PATH   = "models/light_regressor.onnx"
IMG_SIZE     = 64
N_SAMPLES    = 8000
FPS          = 2.0
ALPHA_LUM    = 0.15
ALPHA_CCT    = 0.10
CONF_THRESH  = 0.10
DB_PATH      = "twin_sync.db"
ORT_THREADS  = 4
```

---

## Database

Each frame writes one row to `twin_sync.db` (SQLite):

```sql
SELECT frame_id, lum_error, cct_error, twin_lum, twin_cct,
       confidence, sync_score
FROM sync_reports
ORDER BY frame_id DESC
LIMIT 10;
```

Export to CSV for analysis:

```python
import pandas as pd, sqlite3
df = pd.read_sql("SELECT * FROM sync_reports",
                 sqlite3.connect("twin_sync.db"))
df.to_csv("sync_history.csv", index=False)
```

---

## Troubleshooting

**`sync_score` stuck at ~0.4**
- Check `conf` column — if consistently <0.3, lower `conf_thresh` to `0.1`
- Check `physical_lum` stability — if jumping, AWB/AE lock may not have applied
- Increase `alpha_lum` to `0.30` if convergence is too slow

**Inference latency >50ms on Pi 5**
- Switch to video configuration (`create_video_configuration`)
- Confirm `intra_op_num_threads=4` in `config.py`
- Apply INT8 dynamic quantization via `onnxruntime.quantization`

**`InvalidArgument: Got invalid dimensions for input`**
- Model expects 64×64 input. Ensure `preprocess_frame(frame, size=(64,64))` and warmup dummy shape is `(1,3,64,64)`.

**`TypeError: LightDataset.__init__() got an unexpected keyword argument 'n_samples'`**
- Use `LightDataset(n=N_SAMPLES, img_size=64)` — the parameter is `n`, not `n_samples`.

---

## Architecture Extension

The `shadow/` directory contains a cloud motion simulator, optical flow tracker, and physics-constrained shadow projector designed for outdoor irradiance forecasting. This module is not active in the current indoor-validated build but provides a natural extension path for outdoor deployment.