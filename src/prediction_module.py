"""
Prediction Module
Handles real-time predictions using the trained best model.
Maps user-provided network traffic parameters to model features.
"""
import numpy as np
import pandas as pd
import joblib
import os
import json

from src.config import (
    BEST_MODEL_PATH, BEST_MODEL_INFO_PATH,
    PROTOCOL_MAP, RANDOM_STATE
)


class PredictionModule:
    """
    Prediction module for the Network Intrusion Detection System.
    Loads the trained model and scaler, and provides methods
    for single and batch predictions.
    """

    def __init__(self, model_path=None, scaler=None, feature_names=None):
        """
        Initialize the prediction module.

        Parameters
        ----------
        model_path : str or None
            Path to the trained model. If None, uses the best model.
        scaler : StandardScaler or None
            Pre-fitted scaler. If None, must be set before predict.
        feature_names : list or None
            Names of features expected by the model.
        """
        self.model = None
        self.scaler = scaler
        self.feature_names = feature_names
        self.model_info = {}

        if model_path is None:
            model_path = BEST_MODEL_PATH

        if os.path.exists(model_path):
            self.load_model(model_path)
        else:
            print(f"[WARN] No model found at {model_path}. Train models first.")

        # Load model info if available
        if os.path.exists(BEST_MODEL_INFO_PATH):
            with open(BEST_MODEL_INFO_PATH, "r") as f:
                self.model_info = json.load(f)

    def load_model(self, filepath):
        """
        Load a trained model from disk.

        Parameters
        ----------
        filepath : str
            Path to the saved model file
        """
        self.model = joblib.load(filepath)
        print(f"[INFO] Prediction model loaded from: {filepath}")

    def set_scaler(self, scaler):
        """
        Set the scaler for feature normalization.

        Parameters
        ----------
        scaler : StandardScaler
            Pre-fitted StandardScaler
        """
        self.scaler = scaler

    def set_feature_names(self, feature_names):
        """
        Set the expected feature names for the model.

        Parameters
        ----------
        feature_names : list
            List of feature column names
        """
        self.feature_names = feature_names

    def _transform_input(self, duration, packets, bytes_val,
                          protocol, src_port, dst_port):
        """
        Transform user-friendly input into model feature vector.
        Creates a full feature vector matching the training feature space.

        Parameters
        ----------
        duration : float
            Flow duration in seconds
        packets : int
            Number of packets
        bytes_val : int
            Number of bytes
        protocol : str
            Protocol type (TCP, UDP, ICMP, etc.)
        src_port : int
            Source port number
        dst_port : int
            Destination port number

        Returns
        -------
        np.ndarray
            Feature vector ready for prediction
        """
        proto_num = PROTOCOL_MAP.get(protocol.upper(), 6)

        # Calculate derived features
        flow_duration_us = duration * 1_000_000  # Convert seconds to microseconds
        total_fwd_packets = max(1, int(packets * 0.6))
        total_bwd_packets = max(0, int(packets * 0.4))
        total_fwd_bytes = bytes_val
        total_bwd_bytes = int(bytes_val * 0.3)
        fwd_pkt_len_mean = total_fwd_bytes / total_fwd_packets if total_fwd_packets > 0 else 0
        bwd_pkt_len_mean = total_bwd_bytes / total_bwd_packets if total_bwd_packets > 0 else 0

        flow_duration_sec = flow_duration_us / 1_000_000
        flow_bytes_per_sec = (total_fwd_bytes + total_bwd_bytes) / flow_duration_sec if flow_duration_sec > 0 else 0
        flow_packets_per_sec = (total_fwd_packets + total_bwd_packets) / flow_duration_sec if flow_duration_sec > 0 else 0

        # Build full feature vector matching the CICIDS2017 feature space
        features = {
            "Flow Duration": flow_duration_us,
            "Total Fwd Packets": total_fwd_packets,
            "Total Backward Packets": total_bwd_packets,
            "Total Length of Fwd Packets": total_fwd_bytes,
            "Total Length of Bwd Packets": total_bwd_bytes,
            "Fwd Packet Length Mean": fwd_pkt_len_mean,
            "Fwd Packet Length Max": fwd_pkt_len_mean * 2.5,
            "Fwd Packet Length Min": max(0, fwd_pkt_len_mean * 0.3),
            "Fwd Packet Length Std": max(1, fwd_pkt_len_mean * 0.5),
            "Bwd Packet Length Mean": bwd_pkt_len_mean,
            "Bwd Packet Length Max": bwd_pkt_len_mean * 2.5 if bwd_pkt_len_mean > 0 else 0,
            "Bwd Packet Length Min": 0,
            "Flow Bytes/s": flow_bytes_per_sec,
            "Flow Packets/s": flow_packets_per_sec,
            "Protocol": proto_num,
            "Source Port": src_port,
            "Destination Port": dst_port,
            "Fwd IAT Mean": flow_duration_us / max(1, total_fwd_packets),
            "Fwd IAT Std": max(1, flow_duration_us / max(1, total_fwd_packets) * 0.5),
            "Bwd IAT Mean": flow_duration_us / max(1, total_bwd_packets) if total_bwd_packets > 0 else 0,
            "Bwd IAT Std": 1,
            "Fwd PSH Flags": 0,
            "Fwd URG Flags": 0,
            "FIN Flag Count": 0,
            "SYN Flag Count": min(10, int(total_fwd_packets * 0.1) + 1),
            "RST Flag Count": 0,
            "PSH Flag Count": 0,
            "ACK Flag Count": max(1, int(total_fwd_packets * 0.8)),
            "URG Flag Count": 0,
            "CWE Flag Count": 0,
            "ECE Flag Count": 0,
            "Average Packet Size": (total_fwd_bytes + total_bwd_bytes) / max(1, total_fwd_packets + total_bwd_packets),
            "Avg Fwd Segment Size": fwd_pkt_len_mean,
            "Avg Bwd Segment Size": bwd_pkt_len_mean,
            "Subflow Fwd Packets": total_fwd_packets,
            "Subflow Fwd Bytes": total_fwd_bytes,
            "Subflow Bwd Packets": total_bwd_packets,
            "Subflow Bwd Bytes": total_bwd_bytes,
            "Init Fwd Win Bytes": 65535,
            "Init Bwd Win Bytes": 65535,
            "Fwd Act Data Pkts": total_fwd_packets,
            "Fwd Seg Size Min": max(20, int(fwd_pkt_len_mean * 0.5)),
            "Active Mean": flow_duration_us * 0.8,
            "Active Std": flow_duration_us * 0.2,
            "Idle Mean": flow_duration_us * 0.1,
            "Idle Std": flow_duration_us * 0.05,
        }

        feature_df = pd.DataFrame([features])

        # Ensure only the columns the model expects
        if self.feature_names is not None:
            for col in self.feature_names:
                if col not in feature_df.columns:
                    feature_df[col] = 0
            feature_df = feature_df[self.feature_names]

        return feature_df

    def predict(self, duration, packets, bytes_val,
                protocol, src_port, dst_port):
        """
    Predict whether network traffic is normal or an attack.

    Parameters
    ----------
    duration : float
        Flow duration in seconds
    packets : int
        Total number of packets
    bytes_val : int
        Total number of bytes
    protocol : str
        Protocol type (TCP, UDP, ICMP, HTTP, DNS, FTP, SSH)
    src_port : int
        Source port number (1-65535)
    dst_port : int
        Destination port number (1-65535)

    Returns
    -------
    dict
        Prediction result with:
        - prediction (int): 0=normal, 1=attack
        - label (str): "Normal Traffic" or "Attack Traffic"
        - confidence (float): confidence percentage
        - probabilities (list): [normal_prob, attack_prob]
        - model_used (str): name of the model used
    """
        if self.model is None:
            return {
                "error": "No model loaded. Train models first.",
                "prediction": None
            }

        # Transform input
        feature_df = self._transform_input(
            duration, packets, bytes_val, protocol, src_port, dst_port
        )

        # Normalize if scaler is available
        if self.scaler is not None:
            numeric_cols = feature_df.select_dtypes(include=[np.number]).columns
            feature_df[numeric_cols] = self.scaler.transform(feature_df[numeric_cols])

        # Predict
        prediction = int(self.model.predict(feature_df)[0])

        # Get probabilities if available
        confidence = 0.0
        probabilities = [0.0, 0.0]
        if hasattr(self.model, "predict_proba"):
            try:
                probs = self.model.predict_proba(feature_df)[0]
                if len(probs) >= 2:
                    probabilities = [float(probs[0]), float(probs[1])]
                    confidence = float(probs[prediction]) * 100
                else:
                    confidence = float(probs[0]) * 100
            except Exception:
                confidence = 85.0
        else:
            confidence = 85.0

        label = "Attack Traffic" if prediction == 1 else "Normal Traffic"

        model_used = self.model_info.get("best_model", "Unknown")

        result = {
            "prediction": prediction,
            "label": label,
            "confidence": round(confidence, 2),
            "probabilities": probabilities,
            "model_used": model_used
        }

        return result

    def predict_batch(self, inputs):
        """
        Predict for multiple input samples.

        Parameters
        ----------
        inputs : list of dict
            List of input parameter dictionaries

        Returns
        -------
        list of dict
            List of prediction results
        """
        results = []
        for inp in inputs:
            result = self.predict(
                inp["duration"],
                inp["packets"],
                inp["bytes_val"],
                inp["protocol"],
                inp["src_port"],
                inp["dst_port"]
            )
            results.append(result)
        return results


# Singleton instance for reuse
_instance = None


def get_prediction_module(model_path=None, scaler=None, feature_names=None):
    """
    Get or create the prediction module singleton.

    Parameters
    ----------
    model_path : str or None
        Path to model file
    scaler : StandardScaler or None
        Pre-fitted scaler
    feature_names : list or None
        Feature names

    Returns
    -------
    PredictionModule
        Prediction module instance
    """
    global _instance
    if _instance is None:
        _instance = PredictionModule(model_path, scaler, feature_names)
    return _instance
