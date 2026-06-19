"""
Model Training Module
Trains KNN, Decision Tree, and Random Forest classifiers on the preprocessed data
and saves the trained models to disk.
"""
import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from src.config import (
    MODELS_DIR, RANDOM_STATE, N_JOBS,
    MODEL_FILES, BEST_MODEL_PATH, BEST_MODEL_INFO_PATH
)


def train_knn(X_train, y_train, n_neighbors=5):
    """
    Train a K-Nearest Neighbors classifier.

    Parameters
    ----------
    X_train : pd.DataFrame or np.ndarray
        Training features
    y_train : pd.Series or np.ndarray
        Training labels
    n_neighbors : int
        Number of neighbors

    Returns
    -------
    KNeighborsClassifier
        Trained KNN model
    """
    print("[INFO] Training K-Nearest Neighbors (KNN)...")
    model = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights='distance',
        metric='minkowski',
        n_jobs=N_JOBS
    )
    model.fit(X_train, y_train)
    print(f"[INFO] KNN training complete. k={n_neighbors}")
    return model


def train_decision_tree(X_train, y_train, max_depth=None):
    """
    Train a Decision Tree classifier.

    Parameters
    ----------
    X_train : pd.DataFrame or np.ndarray
        Training features
    y_train : pd.Series or np.ndarray
        Training labels
    max_depth : int or None
        Maximum depth of the tree

    Returns
    -------
    DecisionTreeClassifier
        Trained Decision Tree model
    """
    print("[INFO] Training Decision Tree...")
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)
    print(f"[INFO] Decision Tree training complete. Depth={model.tree_.max_depth}")
    return model


def train_random_forest(X_train, y_train, n_estimators=100):
    """
    Train a Random Forest classifier.

    Parameters
    ----------
    X_train : pd.DataFrame or np.ndarray
        Training features
    y_train : pd.Series or np.ndarray
        Training labels
    n_estimators : int
        Number of trees in the forest

    Returns
    -------
    RandomForestClassifier
        Trained Random Forest model
    """
    print("[INFO] Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS
    )
    model.fit(X_train, y_train)
    print(f"[INFO] Random Forest training complete. {n_estimators} trees.")
    return model


def train_all_models(X_train, y_train):
    """
    Train all three models (KNN, Decision Tree, Random Forest).

    Parameters
    ----------
    X_train : pd.DataFrame or np.ndarray
        Training features
    y_train : pd.Series or np.ndarray
        Training labels

    Returns
    -------
    dict
        Dictionary mapping model names to trained model objects
    """
    models = {}

    models["KNN"] = train_knn(X_train, y_train)
    models["Decision Tree"] = train_decision_tree(X_train, y_train)
    models["Random Forest"] = train_random_forest(X_train, y_train)

    return models


def save_model(model, filepath):
    """
    Save a trained model to disk using joblib.

    Parameters
    ----------
    model : sklearn estimator
        Trained model
    filepath : str
        Path to save the model
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"[INFO] Model saved to: {filepath}")


def save_best_model(model, model_name, metrics):
    """
    Save the best performing model with metadata.

    Parameters
    ----------
    model : sklearn estimator
        Best trained model
    model_name : str
        Name of the best model
    metrics : dict
        Performance metrics of the best model
    """
    save_model(model, BEST_MODEL_PATH)

    info = {
        "best_model": model_name,
        "metrics": metrics
    }
    with open(BEST_MODEL_INFO_PATH, "w") as f:
        json.dump(info, f, indent=2)
    print(f"[INFO] Best model info saved to: {BEST_MODEL_INFO_PATH}")


def load_model(filepath):
    """
    Load a trained model from disk.

    Parameters
    ----------
    filepath : str
        Path to the saved model

    Returns
    -------
    sklearn estimator
        Loaded model
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model not found at: {filepath}")
    model = joblib.load(filepath)
    print(f"[INFO] Model loaded from: {filepath}")
    return model


def load_best_model():
    """
    Load the best performing model.

    Returns
    -------
    sklearn estimator
        Best model
    dict
        Model info (name, metrics)
    """
    model = load_model(BEST_MODEL_PATH)

    info = {}
    if os.path.exists(BEST_MODEL_INFO_PATH):
        with open(BEST_MODEL_INFO_PATH, "r") as f:
            info = json.load(f)

    return model, info


def save_all_models(models):
    """
    Save all trained models to disk.

    Parameters
    ----------
    models : dict
        Dictionary mapping model names to trained model objects

    Returns
    -------
    dict
        Mapping of model names to their saved file paths
    """
    saved_paths = {}
    for name, model in models.items():
        filepath = MODEL_FILES.get(name)
        if filepath:
            save_model(model, filepath)
            saved_paths[name] = filepath
    return saved_paths
