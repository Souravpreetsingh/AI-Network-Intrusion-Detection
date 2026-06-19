"""
Data Preprocessing Module
Handles loading, cleaning, encoding, and normalizing the CICIDS2017 dataset.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import os

from src.config import DATASET_PATH, FEATURE_COLUMNS, TEST_SIZE, RANDOM_STATE


def load_dataset(filepath=None):
    """
    Load the CICIDS2017 dataset from CSV.

    Parameters
    ----------
    filepath : str or None
        Path to the dataset CSV. If None, uses default path from config.

    Returns
    -------
    pd.DataFrame
        Raw loaded dataset
    """
    if filepath is None:
        filepath = DATASET_PATH

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found at {filepath}. "
            "Run generate_sample_data.py first to create a synthetic dataset."
        )

    print(f"[INFO] Loading dataset from: {filepath}")
    df = pd.read_csv(filepath)
    print(f"[INFO] Dataset shape: {df.shape}")
    return df


def handle_missing_values(df):
    """
    Handle missing values in the dataset.

    - Drop columns with >80% missing values
    - Fill numeric columns with median
    - Fill categorical columns with mode

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset

    Returns
    -------
    pd.DataFrame
        Dataset with missing values handled
    """
    print("[INFO] Handling missing values...")

    # Check for missing values
    missing_counts = df.isnull().sum()
    missing_cols = missing_counts[missing_counts > 0]

    if len(missing_cols) == 0:
        print("[INFO] No missing values found.")
        return df

    print(f"[INFO] Found {len(missing_cols)} columns with missing values.")

    # Drop columns with >80% missing
    high_missing = missing_cols[missing_cols > 0.8 * len(df)].index
    if len(high_missing) > 0:
        print(f"[INFO] Dropping {len(high_missing)} columns with >80% missing: {list(high_missing)}")
        df = df.drop(columns=high_missing)

    # Fill remaining missing values
    for col in df.columns:
        if df[col].dtype in [np.float64, np.int64, np.float32, np.int32]:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else "unknown")

    print(f"[INFO] Remaining missing values: {df.isnull().sum().sum()}")
    return df


def remove_duplicates(df):
    """
    Remove duplicate rows from the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset

    Returns
    -------
    pd.DataFrame
        Dataset with duplicates removed
    """
    initial_len = len(df)
    df = df.drop_duplicates()
    removed = initial_len - len(df)
    if removed > 0:
        print(f"[INFO] Removed {removed} duplicate rows.")
    else:
        print("[INFO] No duplicates found.")
    return df


def remove_infinite_values(df):
    """
    Replace infinite values with NaN, then fill with column medians.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset

    Returns
    -------
    pd.DataFrame
        Cleaned dataset
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    return df


def encode_categorical_features(df, label_col="Label", binary_label_col="Binary Label"):
    """
    Encode categorical features using Label Encoding.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset
    label_col : str
        Name of the multi-class label column (e.g., "DoS", "BENIGN")
    binary_label_col : str
        Name of the binary label column (0=normal, 1=attack)

    Returns
    -------
    pd.DataFrame
        Dataset with categorical features encoded
    dict
        Dictionary of LabelEncoders used
    """
    print("[INFO] Encoding categorical features...")
    encoders = {}

    if "Protocol" in df.columns:
        le = LabelEncoder()
        df["Protocol"] = le.fit_transform(df["Protocol"].astype(str))
        encoders["Protocol"] = le

    for col in df.select_dtypes(include=["object"]).columns:
        if col not in [label_col, binary_label_col]:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le

    print(f"[INFO] Encoded {len(encoders)} categorical columns.")
    return df, encoders


def normalize_numerical_features(df, exclude_cols=None, scaler=None):
    """
    Normalize numerical features using StandardScaler.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset
    exclude_cols : list or None
        Columns to exclude from normalization (e.g., labels)
    scaler : StandardScaler or None
        If provided, use this scaler (for transforming new data).
        If None, fit a new scaler.

    Returns
    -------
    pd.DataFrame
        Dataset with numerical features normalized
    StandardScaler
        The scaler used for normalization
    """
    if exclude_cols is None:
        exclude_cols = ["Label", "Binary Label"]

    print("[INFO] Normalizing numerical features...")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cols_to_normalize = [c for c in numeric_cols if c not in exclude_cols]

    if scaler is None:
        scaler = StandardScaler()
        df[cols_to_normalize] = scaler.fit_transform(df[cols_to_normalize])
        print(f"[INFO] Fitted new StandardScaler on {len(cols_to_normalize)} features.")
    else:
        df[cols_to_normalize] = scaler.transform(df[cols_to_normalize])
        print(f"[INFO] Applied existing StandardScaler on {len(cols_to_normalize)} features.")

    return df, scaler


def split_data(df, target_col="Binary Label"):
    """
    Split dataset into training and testing sets.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed dataset
    target_col : str
        Name of the target column

    Returns
    -------
    X_train, X_test, y_train, y_test : pd.DataFrame, pd.DataFrame, pd.Series, pd.Series
    """
    print(f"[INFO] Splitting data (test_size={TEST_SIZE})...")

    exclude = ["Label", "Binary Label"]
    feature_cols = [c for c in df.columns if c not in exclude]

    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    print(f"[INFO] Train set: {X_train.shape[0]} samples")
    print(f"[INFO] Test set: {X_test.shape[0]} samples")
    print(f"[INFO] Attack ratio in train: {y_train.mean():.3f}")
    print(f"[INFO] Attack ratio in test: {y_test.mean():.3f}")

    return X_train, X_test, y_train, y_test


def preprocess_pipeline(filepath=None, for_training=True):
    """
    Run the complete preprocessing pipeline.

    Parameters
    ----------
    filepath : str or None
        Path to dataset
    for_training : bool
        If True, split into train/test. If False, return full processed data.

    Returns
    -------
    tuple or pd.DataFrame
        If for_training: (X_train, X_test, y_train, y_test, scaler, encoders)
        If not: (df_processed, scaler, encoders)
    """
    # Step 1: Load
    df = load_dataset(filepath)

    # Step 2: Handle missing values
    df = handle_missing_values(df)

    # Step 3: Remove duplicates
    df = remove_duplicates(df)

    # Step 4: Remove infinite values
    df = remove_infinite_values(df)

    # Step 5: Encode categorical features
    df, encoders = encode_categorical_features(df)

    # Step 6: Drop or keep the binary label column
    if "Binary Label" not in df.columns and "Label" in df.columns:
        df["Binary Label"] = (df["Label"] != "BENIGN").astype(int)
        print("[INFO] Created Binary Label column from Label column.")

    # Step 7: Normalize
    df, scaler = normalize_numerical_features(
        df, exclude_cols=["Label", "Binary Label"]
    )

    # Step 8: Split if training
    if for_training:
        X_train, X_test, y_train, y_test = split_data(df)
        return X_train, X_test, y_train, y_test, scaler, encoders
    else:
        return df, scaler, encoders
