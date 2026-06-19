"""
Training Pipeline for AI-Based Network Intrusion Detection System.

Loads the CICIDS2017 dataset, preprocesses it (handles missing values,
removes duplicates, encodes categorical columns), trains a Random Forest
classifier, evaluates accuracy, and saves the trained model to
models/model.pkl.

Usage:
    python train_pipeline.py
    python run.py train
"""

import os
import sys
import json
import traceback

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Project paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "model.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
FEATURE_NAMES_PATH = os.path.join(MODELS_DIR, "feature_names.json")
PROTOCOL_MAP_PATH = os.path.join(MODELS_DIR, "protocol_map.json")

os.makedirs(MODELS_DIR, exist_ok=True)

# Label column name in the dataset
LABEL_COL = "Binary Label"
# Columns to exclude from features
EXCLUDE_COLS = ["Label", LABEL_COL]

# Protocol mapping used for encoding
PROTOCOL_MAP = {
    "TCP": 6,
    "UDP": 17,
    "ICMP": 1,
    "HTTP": 6,
    "DNS": 17,
    "FTP": 6,
    "SSH": 6,
}


def find_dataset():
    """Find a CSV dataset in the dataset directory."""
    if not os.path.isdir(DATASET_DIR):
        raise FileNotFoundError(f"Dataset directory not found: {DATASET_DIR}")
    for f in sorted(os.listdir(DATASET_DIR)):
        if f.endswith(".csv"):
            path = os.path.join(DATASET_DIR, f)
            print(f"[INFO] Found dataset: {f}")
            return path
    raise FileNotFoundError(
        f"No CSV dataset found in {DATASET_DIR}. "
        "Please place a CICIDS2017 CSV file in the dataset/ folder."
    )


def load_dataset(filepath=None):
    """Load the dataset from CSV."""
    if filepath is None:
        filepath = find_dataset()
    print(f"[INFO] Loading dataset from: {filepath}")
    df = pd.read_csv(filepath)
    print(f"[INFO] Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def preprocess(df):
    """Clean and preprocess the dataset.

    Steps:
    1. Remove duplicates
    2. Handle missing values (drop rows with any NaN)
    3. Encode categorical columns
    4. Ensure the label column exists
    """
    # Remove duplicates
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    if before > after:
        print(f"[INFO] Removed {before - after} duplicate rows")

    # Handle missing values
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        print(f"[INFO] Dropping {missing_count} missing values")
        df = df.dropna()

    # Ensure label column exists
    if LABEL_COL not in df.columns:
        # Try to find the last numeric column as the label
        if "Binary Label" in df.columns:
            pass
        else:
            raise KeyError(f"Label column '{LABEL_COL}' not found in dataset. "
                           f"Available columns: {list(df.columns)}")

    # Encode categorical columns (object dtype columns, except labels)
    for col in df.columns:
        if col in EXCLUDE_COLS:
            continue
        if df[col].dtype == "object":
            print(f"[INFO] Encoding categorical column: {col}")
            df[col] = pd.factorize(df[col])[0]

    # Separate features and target
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    X = df[feature_cols]
    y = df[LABEL_COL].astype(int)

    print(f"[INFO] Features: {X.shape[1]} columns")
    print(f"[INFO] Target distribution:\n{y.value_counts().to_string()}")

    return X, y, feature_cols


def train_model(X, y):
    """Train a Random Forest classifier.

    Splits data into train/test sets, trains the model, evaluates
    accuracy, and returns the trained model and test accuracy.
    """
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[INFO] Training set: {X_train.shape[0]} samples")
    print(f"[INFO] Test set: {X_test.shape[0]} samples")

    # Scale features
    print("[INFO] Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Random Forest
    print("[INFO] Training Random Forest classifier...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )
    model.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"[INFO] Accuracy: {acc:.4f} ({acc * 100:.2f}%)")
    print(f"\n[INFO] Classification Report:\n{classification_report(y_test, y_pred, target_names=['Normal', 'Attack'])}")

    return model, scaler, acc


def save_artifacts(model, scaler, feature_names, protocol_map):
    """Save trained model, scaler, and metadata to the models/ directory."""
    # Save model
    print(f"[INFO] Saving model to: {MODEL_PATH}")
    joblib.dump(model, MODEL_PATH)

    # Save scaler
    print(f"[INFO] Saving scaler to: {SCALER_PATH}")
    joblib.dump(scaler, SCALER_PATH)

    # Save feature names
    print(f"[INFO] Saving feature names to: {FEATURE_NAMES_PATH}")
    with open(FEATURE_NAMES_PATH, "w") as f:
        json.dump(feature_names, f, indent=2)

    # Save protocol map
    print(f"[INFO] Saving protocol map to: {PROTOCOL_MAP_PATH}")
    with open(PROTOCOL_MAP_PATH, "w") as f:
        json.dump(protocol_map, f, indent=2)

    print(f"[INFO] Model saved successfully")


def run_training_pipeline(filepath=None):
    """Execute the complete training pipeline end-to-end.

    Returns a dict with training results on success, or None on failure.
    """
    try:
        print("\n" + "=" * 60)
        print("  AI-BASED NETWORK INTRUSION DETECTION SYSTEM")
        print("  Training Pipeline")
        print("=" * 60)

        # Step 1: Load
        print("\n[STEP 1/4] Loading dataset...")
        df = load_dataset(filepath)
        print("  Dataset loaded")

        # Step 2: Preprocess
        print("\n[STEP 2/4] Preprocessing data...")
        X, y, feature_names = preprocess(df)

        # Step 3: Train
        print("\n[STEP 3/4] Training started...")
        model, scaler, accuracy = train_model(X, y)

        # Step 4: Save
        print("\n[STEP 4/4] Saving artifacts...")
        save_artifacts(model, scaler, feature_names, PROTOCOL_MAP)

        print("\n" + "=" * 60)
        print("  TRAINING COMPLETE")
        print("=" * 60)
        print(f"  Model:       {MODEL_PATH}")
        print(f"  Accuracy:    {accuracy * 100:.2f}%")
        print(f"  Features:    {len(feature_names)}")
        print("=" * 60)

        return {
            "success": True,
            "accuracy": accuracy,
            "model_path": MODEL_PATH,
            "features": len(feature_names),
        }

    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        print("\nPlease ensure a CICIDS2017 CSV dataset is present in the dataset/ folder.")
        print("You can generate a synthetic dataset with: python run.py generate")
        return {"success": False, "error": str(e)}

    except KeyError as e:
        print(f"\n[ERROR] {e}")
        return {"success": False, "error": str(e)}

    except ImportError as e:
        print(f"\n[ERROR] Missing library: {e}")
        print("Please install required packages: pip install -r requirements.txt")
        return {"success": False, "error": str(e)}

    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        traceback.print_exc()
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    run_training_pipeline()
