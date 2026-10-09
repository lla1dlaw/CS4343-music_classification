# Deep Learning Final Project 
### Liam Laidlaw, Joeseph Tully, Joshua Bearfield

## Dependencies
### uv:
Install on Windows:

``` console
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Install on macOS & Linux:
``` console
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Usage
Clone this repository on HPC:
``` console
git clone https://github.com/lla1dlaw/CS4343-music_classification.git
cd CS4343-music_classification
```

Submit all slurm jobs on HPC:
``` console
./run.sh 
```

Run specific experiments in the experiments/ directory:
``` console
./run.sh alexnet_control resnet_control
```
Generate plots of the results in the results/ directory with:
``` console
uv run gen_plots/make_plots.py
```
Plots are saved to the generated graphs/ directory.
