import numpy as np
import time
from dataclasses import dataclass

@dataclass
class ScenePreset:
    name: str
    base_lum: float   # 0–255 Basic brightness
    cct_k: float      # Color temperature K
    noise_std: float  # Gaussian noise

PRESETS = {
    "daylight":  ScenePreset("daylight",  200, 6500, 8),
    "warm_lamp": ScenePreset("warm_lamp",  120, 2700, 6),
    "office":    ScenePreset("office",  160, 4000, 7),
    "dim":       ScenePreset("dim",     60, 3000, 12),
    "bright":    ScenePreset("bright",    230, 5500, 5),
}

class MockSensor:
    """Simulates an indoor camera with an interface compatible with Picamera2."""
    def __init__(self, scene="office", size=(480,640), drift=0.05):
        self.preset = PRESETS[scene]
        self.h, self.w = size
        self.drift = drift
        self._tick = 0

    def configure(self, config=None): pass
    def create_still_configuration(self, main=None): return {}
    def start(self): pass
    def stop(self): pass

    def capture_array(self) -> np.ndarray:
        self._tick += 1
        p = self.preset
        drift = np.sin(self._tick * self.drift) * 15
        lum = float(np.clip(p.base_lum + drift, 10, 250))
        r, g, b = self._cct_to_rgb(p.cct_k)
        frame = np.zeros((self.h, self.w, 3), dtype=np.float32)
        frame[:,:,0] = lum * r
        frame[:,:,1] = lum * g
        frame[:,:,2] = lum * b
        noise = np.random.normal(0, p.noise_std, (self.h, self.w, 3))
        return np.clip(frame + noise, 0, 255).astype(np.uint8)

    @staticmethod
    def _cct_to_rgb(cct_k):
        t = np.clip((cct_k - 2700) / 5300, 0, 1)
        r = 1.0 - t * 0.25
        g = 0.85 + t * 0.05
        b = 0.6  + t * 0.40
        s = r + g + b
        return r/s, g/s, b/s
