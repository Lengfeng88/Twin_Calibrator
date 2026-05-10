import time
from dataclasses import dataclass, field

@dataclass
class TwinState:
    physical_lum_pct:    float = 50.0
    physical_cct_k:      float = 5000.0
    physical_confidence: float = 0.5
    target_lum_pct:      float = 70.0
    target_cct_k:        float = 4500.0
    twin_lum_pct:        float = 70.0
    twin_cct_k:          float = 4500.0
    timestamp:           float = field(default_factory=time.time)
    frame_id:            int   = 0

    def update_physical(self, calib):
        self.physical_lum_pct    = calib.metrics.luminance_pct
        self.physical_cct_k      = calib.metrics.cct_k
        self.physical_confidence = calib.confidence
        self.timestamp = time.time()
        self.frame_id += 1

    @property
    def lum_error(self): return round(self.physical_lum_pct - self.twin_lum_pct, 2)
    @property
    def cct_error(self): return round(self.physical_cct_k   - self.twin_cct_k)