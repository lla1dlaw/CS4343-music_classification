import math

import matplotlib.pyplot as plt
import numpy as np
import polars as pl


def compare_metric_histograms(
    data: dict[str, pl.DataFrame],
    filename: str | None = None,
    bins: int = 20,
    exclude: tuple[str, ...] = ("epoch", "step"),
) -> None:
    """Overlay per-model histograms, one subplot per shared numeric metric."""
    # metrics = numeric columns present in every model's dataframe
    shared = None
    for df in data.values():
        cols = {c for c, dt in df.schema.items() if dt.is_numeric() and c not in exclude}
        shared = cols if shared is None else shared & cols
    metrics = sorted(shared or [])
    if not metrics:
        raise ValueError("No shared numeric metric columns across models.")

    ncols = min(3, len(metrics))
    nrows = math.ceil(len(metrics) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

    for ax, metric in zip(axes.flat, metrics):
        values = {name: df[metric].drop_nulls().to_numpy() for name, df in data.items()}
        all_vals = np.concatenate(list(values.values()))
        edges = np.histogram_bin_edges(all_vals, bins=bins)  # shared bins

        for name, vals in values.items():
            ax.hist(vals, bins=edges, alpha=0.5, label=name)

        ax.set_title(metric)
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")
        ax.legend()

    for ax in axes.flat[len(metrics):]:  # hide unused subplots
        ax.axis("off")

    fig.tight_layout()
    if filename:
        fig.savefig(filename, dpi=150)
        plt.close(fig)
    else:
        plt.show()