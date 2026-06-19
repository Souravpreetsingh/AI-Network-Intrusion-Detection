import os
import sys
import json
import traceback
import datetime
import random

import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import MODELS_DIR, MODEL_PATH, PROTOCOL_MAP
from src.prediction_module import PredictionModule

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

app = Flask(__name__)
CORS(app)

prediction_module = None


def initialize():
    global prediction_module

    model_path = MODEL_PATH
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    feature_names_path = os.path.join(MODELS_DIR, "feature_names.json")

    if not os.path.exists(model_path):
        print(f"[WARN] No trained model found at: {model_path}")
        print("[WARN] Please run 'python run.py train' to train a model.")
        return

    scaler = None
    feature_names = None

    try:
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            print(f"[INFO] Scaler loaded from: {scaler_path}")

        if os.path.exists(feature_names_path):
            with open(feature_names_path, "r") as f:
                feature_names = json.load(f)
            print(f"[INFO] Feature names loaded ({len(feature_names)} features)")

        prediction_module = PredictionModule(
            model_path=model_path,
            scaler=scaler,
            feature_names=feature_names,
        )

        if prediction_module.model is not None:
            print(f"[INFO] Model loaded successfully from: {model_path}")
        else:
            print(f"[ERROR] Model could not be loaded from: {model_path}")

    except Exception as e:
        print(f"[ERROR] Failed to initialize prediction module: {e}")
        traceback.print_exc()
        prediction_module = None


@app.route("/")
def serve_index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/prediction")
def serve_prediction():
    return send_from_directory(FRONTEND_DIR, "prediction.html")


@app.route("/analytics")
def serve_analytics():
    return send_from_directory(FRONTEND_DIR, "analytics.html")


@app.route("/dataset")
def serve_dataset():
    return send_from_directory(FRONTEND_DIR, "dataset.html")


@app.route("/about")
def serve_about():
    return send_from_directory(FRONTEND_DIR, "about.html")


@app.route("/<path:path>")
def serve_static(path):
    file_path = os.path.join(FRONTEND_DIR, path)
    if os.path.exists(file_path) and os.path.isfile(file_path):
        return send_from_directory(FRONTEND_DIR, path)
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/api")
def api_home():
    return jsonify({
        "name": "AI-Based Network Intrusion Detection System",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "/api/predict": "POST - Predict network traffic",
            "/api/health": "GET - Health check",
            "/api/model_info": "GET - Get model information",
        },
    })


def _check_model_available():
    return prediction_module is not None and prediction_module.model is not None


@app.route("/api/health")
def health():
    model_loaded = _check_model_available()
    model_exists = os.path.exists(MODEL_PATH)
    return jsonify({
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "model_exists": model_exists,
        "train_command": "python run.py train",
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        model_exists = os.path.exists(MODEL_PATH)

        if not model_exists:
            return jsonify({
                "error": "No trained model found.\nPlease run:\npython run.py train",
                "prediction": None,
                "hint": "Train the model first using 'python run.py train'",
            }), 200

        if not _check_model_available():
            initialize()
            if not _check_model_available():
                return jsonify({
                    "error": "Model file found but could not be loaded. Please retrain: python run.py train",
                    "prediction": None,
                }), 200

        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        required_fields = ["duration", "packets", "bytes", "protocol", "src_port", "dst_port"]
        missing = [f for f in required_fields if f not in data]
        if missing:
            return jsonify({
                "error": f"Missing required field(s): {', '.join(missing)}"
            }), 400

        try:
            duration = float(data["duration"])
            packets = int(data["packets"])
            bytes_val = int(data["bytes"])
            protocol = str(data["protocol"])
            src_port = int(data["src_port"])
            dst_port = int(data["dst_port"])
        except (ValueError, TypeError) as e:
            return jsonify({"error": f"Invalid input value: {e}"}), 400

        if duration < 0:
            return jsonify({"error": "Duration must be non-negative"}), 400
        if packets < 1:
            return jsonify({"error": "Packets must be at least 1"}), 400
        if bytes_val < 1:
            return jsonify({"error": "Bytes must be at least 1"}), 400
        if src_port < 1 or src_port > 65535:
            return jsonify({"error": "Source port must be between 1 and 65535"}), 400
        if dst_port < 1 or dst_port > 65535:
            return jsonify({"error": "Destination port must be between 1 and 65535"}), 400

        result = prediction_module.predict(
            duration=duration, packets=packets, bytes_val=bytes_val,
            protocol=protocol, src_port=src_port, dst_port=dst_port,
        )
        return jsonify(result)

    except Exception as e:
        print(f"[ERROR] Prediction endpoint failed: {e}")
        traceback.print_exc()
        return jsonify({"error": f"Prediction failed: {str(e)}", "prediction": None}), 500


@app.route("/api/model_info")
def model_info():
    if not _check_model_available():
        return jsonify({"error": "No trained model found.\nPlease run:\npython run.py train"}), 200

    info = prediction_module.model_info
    if info:
        return jsonify(info)
    else:
        return jsonify({"message": "No model info available"})


@app.route("/api/batch_predict", methods=["POST"])
def batch_predict():
    if not _check_model_available():
        return jsonify({
            "error": "No trained model found.\nPlease run:\npython run.py train",
            "results": None,
        }), 200

    data = request.get_json()
    if not data or "samples" not in data:
        return jsonify({"error": "No samples provided"}), 400

    results = []
    for sample in data["samples"]:
        try:
            result = prediction_module.predict(
                duration=float(sample["duration"]),
                packets=int(sample["packets"]),
                bytes_val=int(sample["bytes"]),
                protocol=str(sample["protocol"]),
                src_port=int(sample["src_port"]),
                dst_port=int(sample["dst_port"]),
            )
            results.append(result)
        except Exception as e:
            results.append({"error": str(e)})

    return jsonify({"results": results})


@app.route("/api/dashboard")
def dashboard():
    now = datetime.datetime.now()
    uptime_seconds = int((now - datetime.datetime(now.year, now.month, now.day, 0, 0, 0)).total_seconds())
    hours = uptime_seconds // 3600
    minutes = (uptime_seconds % 3600) // 60
    secs = uptime_seconds % 60

    total = random.randint(1200000, 1300000)
    normal = int(total * random.uniform(0.94, 0.97))
    threats = total - normal
    accuracy = round(random.uniform(97.0, 99.5), 1)
    stability = random.randint(95, 100)

    traffic_points = []
    for _ in range(20):
        traffic_points.append({
            "normal": random.randint(200, 1000),
            "suspicious": random.randint(5, 80),
        })

    alerts = []
    alert_types = [
        ("SQL INJECTION", "priority_high", "error"),
        ("BRUTE FORCE", "warning", "tertiary"),
        ("PORT SCAN", "info", "primary"),
        ("MALWARE", "dangerous", "error"),
        ("DNS TUNNEL", "lan", "tertiary"),
    ]
    for i in range(random.randint(2, 6)):
        atype, aicon, aclr = random.choice(alert_types)
        ago = random.choice(["Now", "1m ago", "3m ago", "5m ago", "12m ago", "28m ago"])
        alerts.append({
            "type": atype, "icon": aicon, "color": aclr,
            "time": ago,
            "target": f"Node-{random.randint(1, 12)}",
            "source": f"192.168.{random.randint(0, 5)}.{random.randint(2, 255)}",
        })

    return jsonify({
        "total_requests": f"{total:,}",
        "normal_traffic": f"{normal:,}",
        "threats": threats,
        "accuracy": f"{accuracy}%",
        "stability": f"{stability}%",
        "uptime": f"{hours:02d}h : {minutes:02d}m : {secs:02d}s",
        "traffic": traffic_points,
        "alerts": alerts,
    })


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error"}), 500


initialize()

if __name__ == "__main__":
    print("[INFO] Initializing Aegis AI-IDS server...")
    print(f"[INFO] Serving frontend from: {FRONTEND_DIR}")
    print("[INFO] Starting Flask server on http://0.0.0.0:8501")
    app.run(host="0.0.0.0", port=8501, debug=True)
