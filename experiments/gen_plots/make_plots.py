from pathlib import Path
import polars as pl
from evaluate import plotting_utils as pu

results_dir = Path(__file__).parent / "results"
latest_train = max(results_dir.glob("GarbageClassifier_train_*.csv"))
metric_data = pl.read_csv(latest_train)

pu.compare_metrics(metric_data, filename="metrics_comparison.png")