# Raisin Classification — Logistic Regression from Scratch

A learning project to explore data science and ML engineering techniques hands-on —
from model mathematics through to serving, monitoring, and explainability.

Binary classification on the UCI Raisin dataset (Kecimen vs Besni) using logistic
regression implemented in pure Python + NumPy, served via a homemade HTTP API.
No sklearn, no Flask, no MLflow — everything is built from scratch.

## Project structure

```
ml_from_scratch/
├── data/
│   └──Raisin_Dataset.csv
├── src/
│   ├── constants.py          # Shared constants: FEATURE_NAMES, CLASS_LABELS, CLASS_ENCODING
│   ├── utils.py              # Shared utilities: scale()
│   ├── model.py              # Logistic regression: sigmoid, BCE loss, gradient descent, L2 regularisation
│   ├── train.py              # Training pipeline: load, preprocess, split, train, evaluate, save
│   ├── server.py             # HTTP API server (http.server, no frameworks)
│   ├── evaluate.py           # Standalone evaluation utilities: accuracy, F1, ROC-AUC from scratch
│   ├── drift_detector.py     # PSI-based data drift detection against training distribution
│   ├── explain.py            # Feature contribution scoring (logit decomposition) + waterfall plot
│   ├── mlflow_scratch.py     # Experiment tracker: params, metrics, artifacts — MLflow from scratch
│   └── mlruns/               # Auto-generated run data (gitignored)
├── models/                   # Saved weights, scaler params, reference data (gitignored)
├── tests/
│   ├── test_model.py
│   └── test_server.py
├── logs/
│   └── drift_log.jsonl       # Per-request drift scores (auto-generated)
├── requirements.txt
└── README.md
```

## Quickstart
If using venv:
```bash
pip install -r requirements.txt
```
If using conda:
```bash
conda install --yes --file requirements.txt
```
Train the model (saves weights to models/).
```bash
python src/train.py
```
Start the API server.
```bash
python src/server.py
```

## API endpoints

| Endpoint     | Method | Description                                     |
|--------------|--------|-------------------------------------------------|
| `/health`    | GET    | Model load status, drift detector, buffer size  |
| `/schema`    | GET    | Feature names, training means and std devs      |
| `/predict`   | POST   | Predict class + probability + confidence        |
| `/explain`   | POST   | Per-feature contribution to the prediction      |
| `/waterfall` | GET    | Last explain request as a waterfall chart (PNG) |
| `/train`     | POST   | Retrain model in background (returns 202)       |
| `/loss`      | GET    | Training vs validation loss curve (PNG)         |
| `/debug`     | GET    | Raw weights, bias, and scaler values            |

## Example requests
In a separate terminal run the below endpoints.
```bash
# Check server status
curl http://localhost:8080/health

# Trigger retraining (non-blocking)
curl -X POST http://localhost:8080/train

# Single prediction
curl -s -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [45928, 286.54, 208.76, 0.684989, 47336, 0.699599, 844.162]}' | jq .

# Explain a prediction
curl -s -X POST http://localhost:8080/explain \
  -H "Content-Type: application/json" \
  -d '{"features": [45928, 286.54, 208.76, 0.684989, 47336, 0.699599, 844.162]}' | jq .

# Simulate 100 requests to trigger drift check
for i in $(seq 1 100); do
  curl -s -X POST http://localhost:8080/predict \
    -H "Content-Type: application/json" \
    -d '{"features": [87524, 442.25, 253.29, 0.819, 98270, 0.651, 1184.0]}' > /dev/null
done


# Demo PSI with different values, run from project root
tail -n +2 data/Raisin_Dataset.csv | sort -R | head -100 | while IFS=',' read -r area maj min ecc conv ext per class; do
  curl -s -X POST http://localhost:8080/predict \
    -H "Content-Type: application/json" \
    -d "{\"features\": [$area, $maj, $min, $ecc, $conv, $ext, $per]}" > /dev/null
done


# GET endpoints — fetch data
curl http://localhost:8080/loss      > loss_curve.png  && open loss_curve.png
curl http://localhost:8080/waterfall > waterfall.png   && open waterfall.png
curl http://localhost:8080/debug  | jq .
curl http://localhost:8080/schema | jq .
```

## Experiment tracking

Training runs are logged automatically to `src/mlruns/` using a from-scratch MLflow
implementation. Each run records hyperparameters, per-epoch loss, final test metrics,
and saved model artifacts. Inference runs log confidence and drift PSI scores, flushing
to disk every 100 predictions.

```
src/mlruns/
├── raisin_logistic_regression/   # one folder per training run
│   └── <run_id>/
│       ├── run.json              # params + final metrics
│       ├── metric_train_loss.tsv
│       ├── metric_val_loss.tsv
│       └── artifacts/            # weights, scaler, loss curve
└── inference/                    # live prediction logging
    └── <run_id>/
        ├── metric_confidence.tsv
        ├── metric_psi_*.tsv      # per-feature drift scores (written every 100 requests)
        └── run.json
```

## Data drift detection

Incoming prediction requests are buffered. Every 100 requests, PSI (Population Stability
Index) is computed per feature against the training distribution saved in
`models/reference_data.npy`. PSI > 0.25 signals significant drift and is logged to the
console, the inference run, and `logs/drift_log.jsonl`.

## Next steps

- **Wire up evaluate.py**: the standalone evaluation utilities in `evaluate.py` (accuracy,
  F1, ROC-AUC) duplicate logic currently in `train.py`. The next step is to have
  `train.py` import from `evaluate.py` directly, making it the single source of truth for
  all metrics and removing the duplication.
- **Tests**: unit tests cover explain and plot logic. Integration tests that spin up the
  server and hit real endpoints would give end-to-end coverage of the full request path.
- **Docker**: package the server into a minimal container. Given the runtime dependencies
  are NumPy and matplotlib, a `python:3.11-slim` base should stay under ~120MB.
- **Regularisation**: L2 is implemented via `lambda_` in `LogisticRegression`. Next would
  be a hyperparameter sweep across `lambda_` values, logged as separate runs in `mlruns/`,
  to show its effect on val loss and generalisation.
- **Continuous evaluation**: the current setup evaluates once post-training. A natural
  extension is scheduled re-evaluation against a held-out slice as new predictions
  accumulate, with metrics written back to the inference run.
