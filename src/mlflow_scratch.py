import json, shutil, hashlib
from datetime import datetime
from pathlib import Path

RUNS_DIR = Path("mlruns")

class Run:
    def __init__(self, experiment="default"):
        self.run_id   = hashlib.md5(datetime.utcnow().isoformat().encode()).hexdigest()[:8]
        self.run_dir  = RUNS_DIR / experiment / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "artifacts").mkdir()
        self.params  = {}
        self.metrics = {}

    def log_param(self, key, value):
        self.params[key] = value

    def log_metric(self, key, value):
        self.metrics[key] = value
        
        with open(self.run_dir / f"metric_{key}.tsv", "a") as f:
            f.write(f"{datetime.utcnow().isoformat()}\t{value}\n")

    def log_artifact(self, filepath):
        shutil.copy(filepath, self.run_dir / "artifacts")

    def save(self):
        with open(self.run_dir / "run.json", "w") as f:
            json.dump({
                "run_id":    self.run_id,
                "timestamp": datetime.utcnow().isoformat(),
                "params":    self.params,
                "metrics":   self.metrics,
            }, f, indent=2)
        print(f"[Run {self.run_id}] saved → {self.run_dir}")