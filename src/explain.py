import numpy as np
import matplotlib.pyplot as plt

from constants import FEATURE_NAMES, CLASS_LABELS
from utils import scale


def explain_prediction(weights, bias, scaler_mean, scaler_std, X_raw):
    X_scaled      = scale(X_raw, scaler_mean, scaler_std)
    contributions = weights * X_scaled
    logit         = contributions.sum() + bias
    probability   = 1 / (1 + np.exp(-logit))

    explanation = sorted([
        {
            "feature":      FEATURE_NAMES[i],
            "raw_value":    round(float(X_raw[i]), 3),
            "contribution": round(float(contributions[i]), 4),
            "direction":    "→ Besni" if contributions[i] > 0 else "→ Kecimen",
        }
        for i in range(len(weights))
    ], key=lambda x: abs(x["contribution"]), reverse=True)

    return {
        "predicted_class": CLASS_LABELS[int(probability >= 0.5)],
        "probability":     round(float(probability), 4),
        "logit":           round(float(logit), 4),
        "bias":            round(float(bias), 4),
        "top_features":    explanation,
    }


def plot_waterfall(result, save_path):
    features        = result["top_features"]
    bias            = result["bias"]
    logit           = result["logit"]
    predicted_class = result["predicted_class"]
    probability     = result["probability"]

    labels   = ["bias"] + [f["feature"] for f in features]
    contribs = [bias]   + [f["contribution"] for f in features]

    bottoms, heights, colors = [], [], []
    running = 0
    for i, c in enumerate(contribs):
        bottoms.append(running if c >= 0 else running + c)
        heights.append(abs(c))
        colors.append("#888888" if i == 0 else ("#4CAF50" if c >= 0 else "#F44336"))
        running += c

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(labels))

    ax.bar(x, heights, bottom=bottoms, color=colors, edgecolor="white", linewidth=0.5, width=0.6)

    # connector lines between bars
    cumulative = 0
    for i, c in enumerate(contribs):
        cumulative += c
        if i < len(contribs) - 1:
            ax.plot([x[i] + 0.3, x[i + 1] - 0.3], [cumulative, cumulative],
                    color="black", linewidth=0.8, linestyle="--")

    # value labels
    for i, (b, h, c) in enumerate(zip(bottoms, heights, contribs)):
        y_pos = b + h if c >= 0 else b
        va    = "bottom" if c >= 0 else "top"
        ax.text(x[i], y_pos, f"{c:+.4f}", ha="center", va=va, fontsize=8)

    ax.axhline(y=logit, color="navy", linestyle="--", linewidth=1.2,
               label=f"logit = {logit:.4f}")
    ax.axhline(y=0, color="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Logit contribution")
    ax.set_title(f"Prediction: {predicted_class}  (p = {probability})")
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()