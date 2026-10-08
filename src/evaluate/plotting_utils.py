
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sb
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import label_binarize

from typing import Iterable

sb.set_theme(style="darkgrid")

ROOT = Path(__file__).resolve().parents[2]   # adjust so this is the folder containing src/
output_folder = ROOT / "graphs"
output_folder.mkdir(parents=True, exist_ok=True)


def plot_roc_curve(y_test, y_probs, class_names=None, filename="ROC_curve.png"):
    """
    y_test:  (n_samples,) integer labels
    y_probs: (n_samples, n_classes) predicted probabilities 
    """
    n_classes = y_probs.shape[1]
    y_bin = label_binarize(y_test, classes=range(n_classes))

    fig, ax = plt.subplots(figsize=(8, 8))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
        name = class_names[i] if class_names else f"Class {i}"
        ax.plot(fpr, tpr, linewidth=1.5, label=f"{name} (AUC = {auc(fpr, tpr):.2f})")

    ax.plot([0, 1], [0, 1], "k--", label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(fontsize=7, loc="lower right")

    fig.savefig(output_folder / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)


## metric graphs
def plot_metric_graphs(epoch, train_loss, val_loss, filename="metrics.png"):
    """
    epoch: list of epoch numbers
    train_loss: list of training losses
    val_loss: list of validation losses
    """
    for i in epoch:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(epoch[:i], train_loss[:i], label="Train Loss", color="blue")
        ax.plot(epoch[:i], val_loss[:i], label="Validation Loss", color="orange")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(f"Training and Validation Loss (Epoch {i})")
        ax.legend()
        fig.savefig(output_folder / f"{filename}_epoch_{i}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


def compare_metrics(metric_dfs: dict[str, pl.DataFrame]):
    if len(metric_dfs) == 0:
        print("Cannot supply plots for metrics of length 0.")
        return

    for metric_name in next(iter(metric_dfs.values())).columns:
        if metric_name == "epoch":
            continue

        fig, ax = plt.subplots(figsize=(8, 8))

        for model_name, metric_df in metric_dfs.items():
            values = metric_df[metric_name]
            epochs = range(1, len(values) + 1)
            ax.plot(epochs, values, marker="o", label=f"{model_name}")

        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric_name)
        ax.set_title(f"{metric_name} Comparison")
        ax.legend()
        fig.savefig(output_folder / f"{metric_name}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
