import numpy as np
from .state import TwinState

class KalmanScalar:
    def __init__(self, q=0.01, r=0.5):
        self.q=q; self.r=r; self.x=None; self.p=1.0
    def update(self, z):
        if self.x is None: self.x=z; return z
        self.p += self.q
        k = self.p / (self.p + self.r)
        self.x += k * (z - self.x)
        self.p *= (1 - k)
        return self.x

class LightSynchronizer:
    def __init__(self, alpha_lum=0.15, alpha_cct=0.10,
                 conf_thresh=0.3):
        self.alpha_lum   = alpha_lum
        self.alpha_cct   = alpha_cct
        self.conf_thresh = conf_thresh
        self.kf_lum = KalmanScalar(q=0.01, r=0.5)
        self.kf_cct = KalmanScalar(q=10.0, r=200.0)

    def step(self, state: TwinState) -> dict:
        if state.physical_confidence < self.conf_thresh:
            return {"skipped": True}

        lum_s = self.kf_lum.update(state.physical_lum_pct)
        cct_s = self.kf_cct.update(state.physical_cct_k)
        conf  = state.physical_confidence

        state.twin_lum_pct += self.alpha_lum * (lum_s - state.twin_lum_pct) * conf
        state.twin_cct_k   += self.alpha_cct * (cct_s - state.twin_cct_k)   * conf

        state.twin_lum_pct = float(np.clip(state.twin_lum_pct,  0,    100))
        state.twin_cct_k   = float(np.clip(state.twin_cct_k,    2700, 8000))

        return {
            "frame_id":  state.frame_id,
            "lum_err":   state.lum_error,
            "cct_err":   state.cct_error,
            "twin_lum":  round(state.twin_lum_pct, 2),
            "twin_cct":  round(state.twin_cct_k),
            "confidence": round(conf, 3),
            "skipped":   False,
        }