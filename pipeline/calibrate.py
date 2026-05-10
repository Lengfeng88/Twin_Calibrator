import numpy as np
from dataclasses import dataclass

LUM_RANGE, CCT_MIN, CCT_RANGE = 240.0, 2700.0, 5300.0

@dataclass
class LightMetrics:
    luminance_pct: float
    cct_k: float

@dataclass
class CalibrationResult:
    metrics: LightMetrics
    confidence: float

def metrics_from_model(logits: np.ndarray) -> LightMetrics:
    lum = float(logits[0,0]) * 100.0
    cct = float(logits[0,1]) * CCT_RANGE + CCT_MIN
    return LightMetrics(
        luminance_pct=round(lum, 2),
        cct_k=round(cct))

def calibrate_from_metrics(
        metrics: LightMetrics,
        twin_target_lum: float = 70.0,
        twin_target_cct: float = 4500.0) -> CalibrationResult:
    lum_err = abs(metrics.luminance_pct - twin_target_lum)
    cct_err = abs(metrics.cct_k - twin_target_cct)
    conf = float(np.clip(
        1.0 - lum_err/50.0 - cct_err/3000.0, 0, 1))
    return CalibrationResult(metrics=metrics,
                             confidence=round(conf,3))