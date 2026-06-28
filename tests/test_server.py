import numpy as np
import os
import sys
import tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from explain import explain_prediction, plot_waterfall
from constants import FEATURE_NAMES


def test_explain_predictions():
    weights      = np.array([0.5, -0.3, 0.2, -0.1, 0.4, 0.8, -0.6])
    bias         = -0.25
    scaler_mean  = np.array([87524.0, 442.2, 253.3, 0.82, 90546.0, 0.76, 1184.0])
    scaler_std   = np.array([10000.0,  50.0,  30.0, 0.05, 10000.0, 0.05,  100.0])
    X_raw        = np.array([87524.0, 442.2, 253.3, 0.82, 90546.0, 0.76, 1184.0])

    result = explain_prediction(weights, bias, scaler_mean, scaler_std, X_raw)

    # 2. Check the return dict has all expected keys
    for key in ("predicted_class", "probability", "logit", "bias", "top_features"):
        assert key in result, f"Missing key: {key}"

    # 3. Check value ranges
    assert result["predicted_class"] in ("Besni", "Kecimen")
    assert 0.0 <= result["probability"] <= 1.0

    # 4. Check top_features structure
    assert len(result["top_features"]) == len(weights)
    for f in result["top_features"]:
        assert f["feature"] in FEATURE_NAMES
        assert f["direction"] in ("→ Besni", "→ Kecimen")

    # 5. Check sorted by absolute contribution descending
    contribs = [abs(f["contribution"]) for f in result["top_features"]]
    assert contribs == sorted(contribs, reverse=True)

    # 6. Check contributions + bias == logit  (the core invariant of the formula)
    total = sum(f["contribution"] for f in result["top_features"])
    assert abs(total + bias - result["logit"]) < 1e-3, "Contributions don't sum to logit"

    # 7. At the mean, X_scaled is all zeros so all contributions must be 0
    result_at_mean = explain_prediction(weights, bias, scaler_mean, scaler_std, scaler_mean)
    assert all(f["contribution"] == 0.0 for f in result_at_mean["top_features"])


def test_plot_waterfall():
    # 1. Construct a synthetic result dict — same shape as explain_prediction returns
    result = {
        "predicted_class": "Besni",
        "probability": 0.72,
        "logit": 0.59,
        "bias": -0.25,
        "top_features": [
            {"feature": "Extent",       "raw_value": 0.76, "contribution":  0.34, "direction": "→ Besni"},
            {"feature": "Eccentricity", "raw_value": 0.82, "contribution": -0.28, "direction": "→ Kecimen"},
            {"feature": "Area",         "raw_value": 87524, "contribution":  0.05, "direction": "→ Besni"},
        ],
    }

    # 2. Test it writes a PNG file and the file is non-empty
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "waterfall.png")
        plot_waterfall(result, save_path)
        assert os.path.exists(save_path)
        assert os.path.getsize(save_path) > 1000  # a real PNG, not an empty file

    # 3. Test it handles all-positive contributions without crashing
    result_all_pos = {**result, "top_features": [
        {"feature": "Extent", "raw_value": 0.76, "contribution": 0.5, "direction": "→ Besni"},
    ]}
    with tempfile.TemporaryDirectory() as tmpdir:
        plot_waterfall(result_all_pos, os.path.join(tmpdir, "w.png"))

    # 4. Test it handles all-negative contributions without crashing
    result_all_neg = {**result, "top_features": [
        {"feature": "Eccentricity", "raw_value": 0.82, "contribution": -0.5, "direction": "→ Kecimen"},
    ]}
    with tempfile.TemporaryDirectory() as tmpdir:
        plot_waterfall(result_all_neg, os.path.join(tmpdir, "w.png"))



if __name__ == "__main__":
    test_plot_waterfall()
    test_explain_predictions()
    print("All tests passed.")