import json
import shutil
import hashlib
from datetime import datetime
from pathlib import Path

RUNS_DIR = Path("mlruns")

class Run:
    """Basic MLflow-inspired logger for tracking parameters, metrics, and artifacts of a training run."""
    def __init__(self, experiment="default"):
        self.run_id   = hashlib.md5(datetime.utcnow().isoformat().encode()).hexdigest()[:8] # Unique run ID based on timestamp
        self.run_dir  = RUNS_DIR / experiment / self.run_id 
        self.run_dir.mkdir(parents=True, exist_ok=True) 
        (self.run_dir / "artifacts").mkdir() 
        self.params  = {} # Dictionary to store hyperparameters and other run parameters
        self.metrics = {} # Dictionary to store metrics like loss, accuracy, etc.

    def log_param(self, key, value):
        """Store parameter key-value pair for run."""
        self.params[key] = value

    def log_metric(self, key, value):
        """Store metric key-value pair for run, and append to a TSV file for tracking over time."""
        self.metrics[key] = value
        
        with open(self.run_dir / f"metric_{key}.tsv", "a") as f:
            f.write(f"{datetime.utcnow().isoformat()}\t{value}\n")

    def log_artifact(self, filepath):
        """Copy file to run's artifact directory for later reference."""
        shutil.copy(filepath, self.run_dir / "artifacts")

    def save(self):
        """Save run metadata (parameters and latest metrics) to a JSON file for easy retrieval and analysis."""
        with open(self.run_dir / "run.json", "w") as f:
            json.dump({
                "run_id":    self.run_id,
                "timestamp": datetime.utcnow().isoformat(),
                "params":    self.params,
                "metrics":   self.metrics,
            }, f, indent=2)
        print(f"[Run {self.run_id}] saved → {self.run_dir}")