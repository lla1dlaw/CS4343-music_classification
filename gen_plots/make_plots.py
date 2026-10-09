from pathlib import Path
import polars as pl
from evaluate import plotting_utils as pu

from rich import print
import pretty_errors

results_dir = Path(__file__).parent.parent / "results"

# collect train and test files
train_files = [file for file in results_dir.glob("*training*.csv")]
test_files = [file for file in results_dir.glob("*testing*.csv")]

print("-"*20)

print("Found training datafiles:")
for file in train_files:
    print(file)

print()

print("Found testing datafiles:")
for file in test_files:
    print(file)

print("-"*20)

associated_train_data = {}
associated_test_data = {}

# iterate over files, storing model names and training data
for file in train_files:
    train_model_name = file.name.split("_")[0]
    print(f"Collecting metrics for train model: {train_model_name}")
    train_data = pl.read_csv(file)
    associated_train_data[train_model_name] = train_data

for test_file in test_files:
    test_model_name  = test_file.name.split("_")[0]
    print(f"Collecting metrics for test model: {test_model_name}")
    test_data  = pl.read_csv(test_file)
    associated_test_data[test_model_name] = test_data

print("Collected Training_Data")
print(associated_train_data)

print("Collected Testing Data")
print(associated_test_data)
    
pu.compare_metric_histograms(associated_train_data, filename="train_histograms.png")
pu.compare_metric_histograms(associated_test_data, filename="test_histograms.png")
