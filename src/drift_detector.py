# drift_detector.py
import numpy as np
import json
from pathlib import Path
from datetime import datetime

class DriftDetector:
    """
    Population Stability Index (PSI) — standard industry drift metric.
    PSI < 0.1  → no drift
    PSI < 0.25 → moderate drift, investigate  
    PSI >= 0.25 → significant drift, retrain
    """

    def __init__(self, reference_data: np.ndarray, feature_names: list, n_bins=10):
        self.feature_names = feature_names
        self.n_bins        = n_bins
        # Build reference histograms from training data
        self.ref_histograms = []
        for i in range(reference_data.shape[1]):
            counts, edges = np.histogram(reference_data[:, i], bins=n_bins)
            self.ref_histograms.append((counts / counts.sum(), edges))

    def psi(self, ref_pct, cur_pct, eps=1e-8):
        ref_pct = np.clip(ref_pct, eps, None)
        cur_pct = np.clip(cur_pct, eps, None)
        return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))

    def check(self, current_data: np.ndarray) -> dict:
        results = {}
        for i, name in enumerate(self.feature_names):
            ref_pct, edges = self.ref_histograms[i]
            cur_counts, _  = np.histogram(current_data[:, i], bins=edges)
            cur_pct        = cur_counts / cur_counts.sum()
            score          = self.psi(ref_pct, cur_pct)
            results[name]  = {
                "psi":    round(score, 4),
                "status": "ok" if score < 0.1 else "warn" if score < 0.25 else "DRIFT"
            }

        # Log it
        record = {"timestamp": datetime.utcnow().isoformat(), "features": results}
        with open("logs/drift_log.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")

        drifted = [k for k, v in results.items() if v["status"] == "DRIFT"]
        if drifted:
            print(f"[DRIFT ALERT] Features drifted: {drifted}")

        return results