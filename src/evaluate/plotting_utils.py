
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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


def compare_metric_histograms(
    data: dict[str, pl.DataFrame],
    filename: str = "metric_histograms.png",
    bins: int = 20,
    exclude: tuple[str, ...] = ("epoch", "step"),
) -> None:
    """
    data: {model_name: metrics DataFrame}, e.g. the per-epoch training CSVs
    Draws one subplot per metric shared by all models, with one overlaid
    histogram per model (shared bin edges so models are comparable).
    If every model has a single row (e.g. the testing CSVs), a histogram
    is meaningless, so a bar per model is drawn instead.
    """
    # metrics = numeric columns present in every model's dataframe
    shared = None
    for df in data.values():
        cols = {c for c, dt in df.schema.items() if dt.is_numeric() and c not in exclude}
        shared = cols if shared is None else shared & cols
    metrics = sorted(shared or [])
    if not metrics:
        raise ValueError("No shared numeric metric columns across models.")

    single_row = all(df.height <= 1 for df in data.values())

    ncols = min(3, len(metrics))
    nrows = math.ceil(len(metrics) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

    for ax, metric in zip(axes.flat, metrics):
        values = {name: df[metric].drop_nulls().to_numpy() for name, df in data.items()}

        if single_row:
            ax.bar(list(values), [v[0] for v in values.values()])
            ax.set_ylabel("Value")
            ax.tick_params(axis="x", rotation=30)
        else:
            all_vals = np.concatenate(list(values.values()))
            edges = np.histogram_bin_edges(all_vals, bins=bins)  # shared bins
            for name, vals in values.items():
                ax.hist(vals, bins=edges, alpha=0.5, label=name)
            ax.set_xlabel("Value")
            ax.set_ylabel("Count")
            ax.legend(fontsize=7)

        ax.set_title(metric)

    for ax in axes.flat[len(metrics):]:  # hide unused subplots
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_folder / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
