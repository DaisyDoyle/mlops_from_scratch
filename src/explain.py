import numpy as np

FEATURE_NAMES = [
    "Area", "MajorAxisLength", "MinorAxisLength",
    "Eccentricity", "ConvexArea", "Extent", "Perimeter"
]

def explain_prediction(weights, bias, scaler_mean, scaler_std, X_raw):
    """
    For logistic regression: contribution of each feature to the logit score.
    logit = w1*x1 + w2*x2 + ... + b
    Each term w_i * x_i_scaled is the contribution of feature i.
    """
    X_scaled       = (X_raw - scaler_mean) / scaler_std
    contributions  = weights * X_scaled   # element-wise — each feature's push on the logit
    logit          = contributions.sum() + bias
    probability    = 1 / (1 + np.exp(-logit))

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
        "predicted_class": "Besni" if probability >= 0.5 else "Kecimen",
        "probability":     round(float(probability), 4),
        "logit":           round(float(logit), 4),
        "top_features":    explanation,
    }