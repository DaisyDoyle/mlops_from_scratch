import numpy as np
import json
import os
import subprocess
from http.server import HTTPServer, BaseHTTPRequestHandler

from explain import explain_prediction, plot_waterfall
from constants import FEATURE_NAMES
from drift_detector import DriftDetector
from mlflow_scratch import Run
from model import LogisticRegression
from utils import scale
import threading

import logging 


logging.basicConfig(level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s [%(filename)s:%(lineno)d]',
    handlers=[logging.FileHandler("server.log"), logging.StreamHandler()]
    )
logger = logging.getLogger(__name__)


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

model = LogisticRegression()
scalar_mean = None
scalar_std = None

_model_lock = threading.Lock()
_is_training = False

_request_buffer = []
_BUFFER_SIZE    = 100

detector = None

_inference_run = Run(experiment="inference")
   
FEATURE_SCHEMA = [
    {"index": 0, "name": "Area",            "description": "Number of pixels in the raisin boundary"},
    {"index": 1, "name": "MajorAxisLength", "description": "Length of the major axis (pixels)"},
    {"index": 2, "name": "MinorAxisLength", "description": "Length of the minor axis (pixels)"},
    {"index": 3, "name": "Eccentricity",    "description": "Eccentricity of the ellipse (0–1)"},
    {"index": 4, "name": "ConvexArea",      "description": "Convex hull pixel count"},
    {"index": 5, "name": "Extent",          "description": "Ratio of pixels to bounding box (0–1)"},
    {"index": 6, "name": "Perimeter",       "description": "Raisin boundary length (pixels)"},
]


def get_schema():
    schema = []
    for i, feat in enumerate(FEATURE_SCHEMA):
        entry = {**feat}
        if scalar_mean is not None and scalar_std is not None:
            entry["mean"] = round(float(scalar_mean[i]), 4)
            entry["std"]  = round(float(scalar_std[i]), 4)
        else:
            entry["mean"] = "model not loaded"
            entry["std"]  = "model not loaded"
        schema.append(entry)
    return schema


def load_trained_parameters():
    global model, scalar_mean, scalar_std, detector
    try:

        
        weights = np.load(os.path.join(MODELS_DIR, "model_weights.npy"), allow_pickle=True)
        bias    = float(np.load(os.path.join(MODELS_DIR, "model_bias.npy"), allow_pickle=True))

        raw_mean = np.load(os.path.join(MODELS_DIR, "scaler_mean.npy"), allow_pickle=True)
        raw_std  = np.load(os.path.join(MODELS_DIR, "scaler_std.npy"), allow_pickle=True)

        new_mean = np.array(getattr(raw_mean, "values", raw_mean), dtype=float).flatten()
        new_std  = np.array(getattr(raw_std,  "values", raw_std),  dtype=float).flatten()

        ref_data = np.load(os.path.join(MODELS_DIR, "reference_data.npy"), allow_pickle=True)
        new_detector = DriftDetector(ref_data, FEATURE_NAMES)

        with _model_lock:   # atomic swap — no request reads half-updated state
            model.weights = weights
            model.bias    = bias
            scalar_mean   = new_mean
            scalar_std    = new_std
            detector      = new_detector

        
        _inference_run.log_param("model_weights_shape", str(model.weights.shape))
        _inference_run.log_param("bias", round(model.bias, 6))

        print("[SUCCESS] Model, scaler, and drift detector loaded.")
    except FileNotFoundError as e:
        print(f"[WARNING] Missing file: {e}")
        scalar_mean, scalar_std, detector = None, None, None



# def drift_detection(new_data):
#     global detector
#     detector = DriftDetector(new_data, FEATURE_NAMES)
#     logger.info(f"detector loaded new ref data: {new_data[:1]}")
#     # return detector
    


class ModelTrainingRequestHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        
        def _serve_image(self, path, not_found_message):
            image_path = os.path.join(MODELS_DIR, f"{path}.png")
            if os.path.exists(image_path):
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.end_headers()
                with open(image_path, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_error_response(404, not_found_message)


        if self.path == "/loss" or self.path == "/":
            _serve_image(self, "loss_curves", "Loss curve not found. Run /train first.")

        elif self.path == "/waterfall":
            _serve_image(self, "waterfall", "No waterfall plot yet. POST /explain first.")

            
        elif self.path == "/health":
            self.send_success_response(200, {
                "status":          "ok",
                "model_loaded":    model.weights is not None,
                "scaler_loaded":   scalar_mean is not None,
                "drift_detector":  detector is not None,
                "buffer_size":     len(_request_buffer),
                "buffer_capacity": _BUFFER_SIZE,
            })

        elif self.path == "/debug":
            self.send_success_response(200, {
                "weights":      model.weights.tolist() if model.weights is not None else None,
                "bias":         model.bias,
                "scaler_mean":  scalar_mean.tolist() if scalar_mean is not None else None,
                "scaler_std":   scalar_std.tolist() if scalar_std is not None else None,
            })

        elif self.path == "/schema":
            try:
                self.send_success_response(200, {
                    "model":   "LogisticRegression",
                    "classes": {"0": "Kecimen", "1": "Besni"},
                    "features": get_schema(),
                    "note":    "Send raw unscaled values. Scaling applied server-side."
                })
            except Exception as e:
                self.send_error_response(500, f"Schema error: {str(e)}")

        else:
            self.send_error_response(404, "Endpoint not found.")

    def do_POST(self):
        if self.path == "/train":
            self.handle_train()
        elif self.path == "/predict":
            self.handle_predict()
        elif self.path == "/explain":        
            self.handle_explain()
        else:
            self.send_error_response(404, "Endpoint not found. Use POST /train or POST /predict")

    def handle_explain(self):
        try:

            with _model_lock:
                weights     = model.weights
                bias        = model.bias
                mean        = scalar_mean
                std         = scalar_std

            if weights is None or mean is None:
                self.send_error_response(503, "Model not loaded.")
                return

            payload = self.parse_json_payload()
            if payload is None or 'features' not in payload:
                self.send_error_response(422, "Missing required parameter: features")
                return

            X_raw = np.array(payload['features'], dtype=float).flatten()

            if X_raw.shape != mean.shape:
                self.send_error_response(422,
                    f"Shape mismatch. Expected {mean.shape[0]} features, got {X_raw.shape[0]}.")
                return

            result = explain_prediction(
                weights, bias,
                mean, std, X_raw
            )

            self.send_success_response(200, result)

            try:

                waterfall_path = os.path.join(MODELS_DIR, "waterfall.png")
                plot_waterfall(result, waterfall_path)
            except Exception as e:
                logger.error(f"plot_waterfall failed: {e}")

        except Exception as e:
            self.send_error_response(500, f"Explain error: {str(e)}")

    

    def handle_train(self):
        try:
            def run_training():
                global _is_training
                _is_training = True
                try:
                    script_path = os.path.join(BASE_DIR, "src", "train.py")
                    result = subprocess.run(
                        ["python", script_path],
                        capture_output=True,
                        text=True
                    )
                    print(f"[TRAIN] Return code: {result.returncode}")
                    print(f"[TRAIN] stdout: {result.stdout}")
                    print(f"[TRAIN] stderr: {result.stderr}")

                    if result.returncode == 0:
                        load_trained_parameters()
                        print(f"[TRAIN] Reload complete. Weights: {model.weights is not None}")
                    else:
                        print("[TRAIN] Training script failed — not reloading")
                except Exception as e:
                    print(f"[TRAIN] Exception in thread: {e}")
                finally:
                    _is_training = False


            thread = threading.Thread(target=run_training, daemon=True)
            thread.start()

            self.send_success_response(202, {
                "status": "training_initiated",
                "message": "Training running in background. Poll /health for completion."
            })
        except Exception as e:
            self.send_error_response(500, f"Failed to start training: {str(e)}")

    def handle_predict(self):
        try:

            with _model_lock:
                weights     = model.weights
                bias        = model.bias
                mean        = scalar_mean
                std         = scalar_std

            if weights is None or mean is None:
                self.send_error_response(503, "Model parameters uninitialized.")
                return

            payload = self.parse_json_payload()
            if payload is None or 'features' not in payload:
                self.send_error_response(422, "Missing required parameter: features")
                return

            X_raw = np.array(payload['features'], dtype=float).flatten()

            if X_raw.shape != mean.shape:
                self.send_error_response(422, 
                    f"Shape mismatch. Expected {mean.shape[0]} features, got {X_raw.shape[0]}.")
                return


            print(f"[DEBUG] Raw input:    {X_raw}")
            print(f"[DEBUG] Scaler mean:  {mean}")
            print(f"[DEBUG] Scaler std:   {std}")

            X_scaled = scale(X_raw, mean, std)
            print(f"[DEBUG] Scaled input: {X_scaled}")


            if np.any(np.abs(X_scaled) > 10):
                print("[WARNING] Scaled features contain extreme values — "
                    "check whether input is already normalised")

            prob_matrix = model.predict_proba(X_scaled.reshape(1, -1))
            prob = float(np.round(prob_matrix.item(), decimals=3))

            print(f"[DEBUG] Probability:  {prob}")

            predicted_class = "Besni" if prob >= 0.5 else "Kecimen"

            self.send_success_response(200, {
                "probability":      prob,
                "predicted_class":  predicted_class,
                "confidence":       round(max(prob, 1 - prob), 3),
            })

            global _request_buffer

            _request_buffer.append(X_raw.tolist())
            print(f"[BUFFER] Added request to buffer. Current size: {len(_request_buffer)}")

            _inference_run.log_metric("confidence", round(max(prob, 1 - prob), 3))
            _inference_run.log_metric("prediction", 1 if predicted_class == "Besni" else 0)

            if len(_request_buffer) >= _BUFFER_SIZE and detector is not None:
                batch = np.array(_request_buffer)
                drift_results = detector.check(batch)
                _request_buffer.clear()

                for feature, result in drift_results.items():
                    _inference_run.log_metric(f"psi_{feature}", result.get("psi", 0))

                _inference_run.save()  

                print(f"[DRIFT CHECK] {json.dumps(drift_results, indent=2)}")
            elif detector is None:
                print("[WARNING] Drift detector not initialised — skipping check")

        except Exception as e:
            print(f"[ERROR] Inference error: {str(e)}")
            import traceback
            traceback.print_exc()
            self.send_error_response(500, f"Inference error: {str(e)}")

    def parse_json_payload(self):
        content_length = int(self.headers.get('Content-Length', 0))
        if content_length == 0: 
            return None
        return json.loads(self.rfile.read(content_length).decode('utf-8'))

    def send_success_response(self, code, payload):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode('utf-8'))

    def send_error_response(self, code, message):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"error": message}).encode('utf-8'))



if __name__ == "__main__":
    load_trained_parameters() # new line
    addr = ('', 8080)
    server = HTTPServer(addr, ModelTrainingRequestHandler)
    print("Integrated Server online. Dashboard: http://localhost:8080/loss")
    server.serve_forever()
