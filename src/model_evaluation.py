"""
Model Evaluation Module
Evaluates trained models, generates comparison reports,
confusion matrices, accuracy graphs, and precision/recall reports.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)

from src.config import REPORTS_DIR, RANDOM_STATE


plt.rcParams.update({
    "figure.facecolor": "#0a0e17",
    "axes.facecolor": "#111827",
    "axes.edgecolor": "#2a3040",
    "axes.labelcolor": "#94a3b8",
    "text.color": "#e8edf5",
    "xtick.color": "#64748b",
    "ytick.color": "#64748b",
    "grid.color": "#1a1f2e",
    "legend.facecolor": "#1a1f2e",
    "legend.edgecolor": "#2a3040",
    "legend.labelcolor": "#e8edf5",
    "figure.dpi": 150
})


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluate a trained model on test data.

    Parameters
    ----------
    model : sklearn estimator
        Trained model
    X_test : pd.DataFrame or np.ndarray
        Test features
    y_test : pd.Series or np.ndarray
        True labels
    model_name : str
        Name for display

    Returns
    -------
    dict
        Dictionary of evaluation metrics
    """
    print(f"\n[INFO] Evaluating {model_name}...")

    y_pred = model.predict(X_test)

    # Try predict_proba for AUC
    y_prob = None
    try:
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
    except Exception:
        y_prob = None

    metrics = {
        "model_name": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
    }

    if y_prob is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_test, y_prob)
        except Exception:
            metrics["roc_auc"] = None
    else:
        metrics["roc_auc"] = None

    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall   : {metrics['recall']:.4f}")
    print(f"  F1 Score : {metrics['f1_score']:.4f}")
    if metrics["roc_auc"]:
        print(f"  AUC-ROC  : {metrics['roc_auc']:.4f}")

    return metrics, y_pred


def compare_models(models, X_test, y_test):
    """
    Evaluate and compare multiple models.

    Parameters
    ----------
    models : dict
        Dictionary mapping model names to trained model objects
    X_test : pd.DataFrame or np.ndarray
        Test features
    y_test : pd.Series or np.ndarray
        True labels

    Returns
    -------
    pd.DataFrame
        Comparison dataframe of all model metrics
    dict
        Dictionary of model_name -> (metrics, y_pred)
    """
    results = {}
    all_metrics = []

    for name, model in models.items():
        metrics, y_pred = evaluate_model(model, X_test, y_test, name)
        results[name] = {"metrics": metrics, "y_pred": y_pred}
        all_metrics.append(metrics)

    import pandas as pd
    comparison_df = pd.DataFrame(all_metrics)
    comparison_df = comparison_df.set_index("model_name")

    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print(comparison_df.to_string())

    return comparison_df, results


def select_best_model(comparison_df, metric="f1_score"):
    """
    Select the best model based on a given metric.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        Comparison dataframe with model metrics
    metric : str
        Metric to use for selection (accuracy, f1_score, etc.)

    Returns
    -------
    str
        Name of the best model
    """
    best_model_name = comparison_df[metric].idxmax()
    best_score = comparison_df[metric].max()
    print(f"\n[INFO] Best model: {best_model_name} ({metric}={best_score:.4f})")
    return best_model_name


def generate_confusion_matrix(y_test, y_pred, model_name, save_path=None):
    """
    Generate and save a confusion matrix plot.

    Parameters
    ----------
    y_test : pd.Series or np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    model_name : str
        Name of the model (for title)
    save_path : str or None
        Path to save the figure

    Returns
    -------
    matplotlib.figure.Figure
        The confusion matrix figure
    """
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="RdYlGn",
        xticklabels=["Normal", "Attack"],
        yticklabels=["Normal", "Attack"],
        ax=ax, cbar=True,
        linewidths=1, linecolor="#2a3040",
        annot_kws={"color": "#e8edf5", "size": 14}
    )
    ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=12, fontweight="bold")
    ax.set_title(f"Confusion Matrix - {model_name}", fontsize=14, fontweight="bold", color="#e8edf5")

    tn, fp, fn, tp = cm.ravel()
    stats_text = f"TN: {tn}  FP: {fp}\nFN: {fn}  TP: {tp}"
    ax.text(
        0.5, -0.15, stats_text,
        transform=ax.transAxes,
        ha="center", fontsize=11,
        color="#94a3b8",
        fontfamily="monospace"
    )
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"[INFO] Confusion matrix saved to: {save_path}")

    return fig


def generate_accuracy_comparison(comparison_df, save_path=None):
    """
    Generate a bar chart comparing model accuracies.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        Comparison dataframe with model metrics
    save_path : str or None
        Path to save the figure

    Returns
    -------
    matplotlib.figure.Figure
        The accuracy comparison figure
    """
    metrics_to_plot = ["accuracy", "precision", "recall", "f1_score"]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(comparison_df.index))
    width = 0.2
    colors = ["#00d4ff", "#00ff88", "#8b5cf6", "#ff8c00"]

    for i, metric in enumerate(metrics_to_plot):
        values = comparison_df[metric].values * 100  # Convert to percentage
        bars = ax.bar(
            x + i * width, values, width,
            label=metric.replace("_", " ").title(),
            color=colors[i], alpha=0.85,
            edgecolor="white", linewidth=0.5
        )
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:.1f}%",
                ha="center", va="bottom",
                fontsize=9, fontweight="bold",
                color="#e8edf5"
            )

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(comparison_df.index, fontsize=12)
    ax.set_ylabel("Score (%)", fontsize=12, fontweight="bold")
    ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold", color="#e8edf5")
    ax.legend(loc="lower right", fontsize=10)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"[INFO] Accuracy comparison saved to: {save_path}")

    return fig


def generate_classification_report_text(y_test, y_pred, model_name, save_path=None):
    """
    Generate a detailed classification report.

    Parameters
    ----------
    y_test : pd.Series or np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    model_name : str
        Name of the model
    save_path : str or None
        Path to save the report

    Returns
    -------
    str
        Classification report text
    """
    report = classification_report(y_test, y_pred, target_names=["Normal", "Attack"])
    header = f"{'='*60}\nClassification Report - {model_name}\n{'='*60}\n"
    full_report = header + report

    print(full_report)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            # Also include confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            cm_str = f"\nConfusion Matrix:\n{cm}\n"
            f.write(full_report + cm_str)
        print(f"[INFO] Classification report saved to: {save_path}")

    return full_report


def generate_roc_curves(models, X_test, y_test, save_path=None):
    """
    Generate ROC curves for all models that support predict_proba.

    Parameters
    ----------
    models : dict
        Dictionary mapping model names to trained model objects
    X_test : pd.DataFrame or np.ndarray
        Test features
    y_test : pd.Series or np.ndarray
        True labels
    save_path : str or None
        Path to save the figure

    Returns
    -------
    matplotlib.figure.Figure or None
        The ROC curve figure, or None if no model supports predict_proba
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#00d4ff", "#00ff88", "#8b5cf6", "#ff8c00"]
    color_idx = 0

    has_any = False
    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            try:
                y_prob = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                auc_score = roc_auc_score(y_test, y_prob)
                ax.plot(
                    fpr, tpr,
                    color=colors[color_idx % len(colors)],
                    linewidth=2.5,
                    label=f"{name} (AUC={auc_score:.3f})"
                )
                has_any = True
                color_idx += 1
            except Exception:
                continue

    if not has_any:
        plt.close(fig)
        return None

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1, color="#64748b")
    ax.set_xlabel("False Positive Rate", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Positive Rate", fontsize=12, fontweight="bold")
    ax.set_title("ROC Curves", fontsize=14, fontweight="bold", color="#e8edf5")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"[INFO] ROC curves saved to: {save_path}")

    return fig


def generate_all_reports(models, X_test, y_test, model_results=None):
    """
    Generate all evaluation reports including confusion matrices,
    accuracy comparison, and classification reports.

    Parameters
    ----------
    models : dict
        Dictionary mapping model names to trained model objects
    X_test : pd.DataFrame or np.ndarray
        Test features
    y_test : pd.Series or np.ndarray
        True labels
    model_results : dict or None
        Pre-computed model results (metrics, y_pred) for each model

    Returns
    -------
    pd.DataFrame
        Comparison dataframe
    dict
        Best model info
    """
    # Compare models
    comparison_df, results = compare_models(models, X_test, y_test)

    # Select best model
    best_model_name = select_best_model(comparison_df, metric="f1_score")
    best_model = models[best_model_name]
    best_metrics = comparison_df.loc[best_model_name].to_dict()

    # Generate confusion matrix for each model
    for name, result in results.items():
        cm_path = os.path.join(REPORTS_DIR, f"confusion_matrix_{name.lower().replace(' ', '_')}.png")
        generate_confusion_matrix(y_test, result["y_pred"], name, save_path=cm_path)

        report_path = os.path.join(REPORTS_DIR, f"classification_report_{name.lower().replace(' ', '_')}.txt")
        generate_classification_report_text(y_test, result["y_pred"], name, save_path=report_path)

    # Generate accuracy comparison chart
    acc_path = os.path.join(REPORTS_DIR, "accuracy_comparison.png")
    generate_accuracy_comparison(comparison_df, save_path=acc_path)

    # Generate ROC curves
    roc_path = os.path.join(REPORTS_DIR, "roc_curves.png")
    generate_roc_curves(models, X_test, y_test, save_path=roc_path)

    best_info = {
        "best_model_name": best_model_name,
        "best_model": best_model,
        "best_metrics": best_metrics
    }

    return comparison_df, best_info
