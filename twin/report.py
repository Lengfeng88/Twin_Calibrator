import sqlite3, time
from dataclasses import dataclass
from .state import TwinState
import numpy as np

@dataclass
class SyncReport:
    frame_id:   int
    timestamp:  float
    lum_error:  float
    cct_error:  float
    twin_lum:   float
    twin_cct:   float
    confidence: float
    sync_score: float

    @staticmethod
    def from_state(state: TwinState) -> "SyncReport":
        lq = max(0, 1 - abs(state.lum_error) / 50)
        cq = max(0, 1 - abs(state.cct_error) / 2000)
        return SyncReport(
            frame_id   = state.frame_id,
            timestamp  = state.timestamp,
            lum_error  = state.lum_error,
            cct_error  = state.cct_error,
            twin_lum   = round(state.twin_lum_pct, 2),
            twin_cct   = round(state.twin_cct_k),
            confidence = state.physical_confidence,
            sync_score = round((lq + cq) / 2, 3),
        )

class ReportStore:
    def __init__(self, db="twin_sync.db", history=300):
        self._buf = []
        self.history = history
        self.conn = sqlite3.connect(db, check_same_thread=False)
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS sync_reports (
            frame_id   INTEGER PRIMARY KEY,
            timestamp  REAL,
            lum_error  REAL, cct_error  REAL,
            twin_lum   REAL, twin_cct   REAL,
            confidence REAL, sync_score REAL
        )""")
        self.conn.commit()

    def write(self, r: SyncReport):
        self.conn.execute(
            "INSERT OR REPLACE INTO sync_reports VALUES (?,?,?,?,?,?,?,?)",
            (r.frame_id, r.timestamp, r.lum_error, r.cct_error,
             r.twin_lum, r.twin_cct, r.confidence, r.sync_score))
        self.conn.commit()
        self._buf.append(r)
        if len(self._buf) > self.history: self._buf.pop(0)

    def recent(self, n=60): return self._buf[-n:]