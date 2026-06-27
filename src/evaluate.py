import json
from datetime import datetime
from pathlib import Path

Path("logs").mkdir(exist_ok=True)
METRICS_LOG = Path("logs/evaluation_log.jsonl")
METRICS_LOG.parent.mkdir(exist_ok=True)

def evaluate_and_log(model, scaler_mean, scaler_std, X_test, y_true):
    """Run after every training cycle and on a schedule against held-out data."""
    
    X_scaled = (X_test - scaler_mean) / scaler_std
    probs  = model.predict_proba(X_scaled)
    preds  = (probs >= 0.5).astype(int)

    tp = int(((preds == 1) & (y_true == 1)).sum())
    fp = int(((preds == 1) & (y_true == 0)).sum())
    fn = int(((preds == 0) & (y_true == 1)).sum())
    tn = int(((preds == 0) & (y_true == 0)).sum())

    precision  = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall     = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1         = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy   = (tp + tn) / len(y_true)

    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "accuracy":  round(accuracy, 4),
        "f1":        round(f1, 4),
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "n_samples": len(y_true),
    }

    # Append to log — one JSON record per line, easy to parse later
    with open(METRICS_LOG, "a") as f:
        f.write(json.dumps(record) + "\n")

    # Alert if F1 drops below threshold
    if f1 < 0.80:
        print(f"[ALERT] F1 dropped to {f1:.3f} — consider retraining")

    return record