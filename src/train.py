import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import logging
from mlflow_scratch import Run  

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

sys.path.insert(0, BASE_DIR)  

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s [%(filename)s:%(lineno)d]',
    handlers=[logging.FileHandler("pipeline.log"), logging.StreamHandler()]
    )
logger = logging.getLogger(__name__)


def load_data():
    logger.info("Loading dataset")
    df = pd.read_csv("data/Raisin_Dataset.csv")
    return df

def preprocess_data(df):
    logger.info("Preprocessing data")
    X = df.drop(columns=["Class"])
    X_mean = X.mean(axis=0)
    X_std  = X.std(axis=0)
    np.save(os.path.join(MODELS_DIR, "reference_data.npy"), X.values)
    X_normalized = (X - X_mean) / X_std
    y = df["Class"].apply(lambda x: 1 if x == "Kecimen" else 0).values
    return X_mean, X_std, X_normalized, y

def split_dataset(X_normalized, y):
    logger.info("Splitting dataset into train/val/test")
    indices = np.arange(X_normalized.shape[0])
    np.random.seed(42)
    np.random.shuffle(indices)
    n = len(indices)
    train_end = int(0.7 * n)
    val_end   = int(0.9 * n)
    train_idx, val_idx, test_idx = indices[:train_end], indices[train_end:val_end], indices[val_end:]
    
    return (X_normalized[train_idx], y[train_idx],
            X_normalized[val_idx],   y[val_idx],
            X_normalized[test_idx],  y[test_idx])

def train_model(X_train, y_train, X_val, y_val, run: Run):
    from model import LogisticRegression

    logger.info("Training model with hyperparameters: learning_rate=0.1, n_iters=300, lambda_=0")
    params = dict(learning_rate=0.1, n_iters=300, lambda_=0.0)
    for k, v in params.items():
        run.log_param(k, v)

    model = LogisticRegression(**params)
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    
    for i, (tl, vl) in enumerate(zip(model.losses, model.val_losses)):
        run.log_metric("train_loss", tl)
        run.log_metric("val_loss",   vl)
        logger.info("Loss at epoch %d: train_loss=%.4f, val_loss=%.4f", i+1, tl, vl)

    return model

def calculate_roc_auc_score(y_true, y_scores):
    pos_scores = y_scores[y_true == 1]
    neg_scores = y_scores[y_true == 0]
    correct = sum(
        1 if p > n else 0.5 if p == n else 0
        for p in pos_scores for n in neg_scores
    )
    return correct / (len(pos_scores) * len(neg_scores))

def calculate_f1_score(predictions, y_test):
    tp = np.sum((predictions == 1) & (y_test == 1))
    fp = np.sum((predictions == 1) & (y_test == 0))
    fn = np.sum((predictions == 0) & (y_test == 1))
    precision = tp / (tp + fp)
    recall    = tp / (tp + fn)
    return 2 * (precision * recall) / (precision + recall)

def evaluate_model(model, X_test, y_test, run: Run):
    predictions  = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    accuracy = np.mean(predictions == y_test)
    f1       = calculate_f1_score(predictions, y_test)
    roc_auc  = calculate_roc_auc_score(y_test, probabilities)

    
    run.log_metric("test_accuracy", accuracy)
    run.log_metric("test_f1",       f1)
    run.log_metric("test_roc_auc",  roc_auc)
    logger.info(f"Evaluation results - Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")

    return accuracy, f1, roc_auc

def plot_loss_curves(train_losses, val_losses, save_path):
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses,   label="Val Loss")   
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curves")
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def save_model(model, X_mean, X_std, save_dir, run: Run):
    np.save(f"{save_dir}/model_weights.npy", model.weights)
    np.save(f"{save_dir}/model_bias.npy",    model.bias)
    np.save(f"{save_dir}/scaler_mean.npy",   X_mean)
    np.save(f"{save_dir}/scaler_std.npy",    X_std)
    
    for fname in ["model_weights.npy", "model_bias.npy", "scaler_mean.npy", "scaler_std.npy"]:
        run.log_artifact(f"{save_dir}/{fname}")

    logging.info(f"Model saved to {save_dir}")


if __name__ == "__main__":
    run = Run(experiment="raisin_logistic_regression")  

    df = load_data()
    X_mean, X_std, X_normalized, y = preprocess_data(df)
    X_train, y_train, X_val, y_val, X_test, y_test = split_dataset(X_normalized.values, y)

    X_test_raw = (X_test * X_std.values) + X_mean.values
    for row in X_test_raw[:5]:
        logging.info(f"Sample features (raw): {row}")

    model = train_model(X_train, y_train, X_val, y_val, run)

    accuracy, f1, roc_auc = evaluate_model(model, X_test, y_test, run)
    logging.info(f"Accuracy: {accuracy:.4f} | F1: {f1:.4f} | ROC-AUC: {roc_auc:.4f}")

    curve_path = os.path.join(MODELS_DIR, "loss_curves.png")
    plot_loss_curves(model.losses, model.val_losses, curve_path)
    run.log_artifact(curve_path) 

    save_model(model, X_mean, X_std, MODELS_DIR, run)

    run.save()  
    logging.info("Training pipeline completed successfully")