# Deep Learning Final Project 
### Liam Laidlaw, Joeseph Tully, Joshua Bearfield

## Usage
Submit Slurm Jobs Locally (if you are already SSH'd into the HPC):

``` console
./run.sh alexnet_control resnet_control
```
Trigger Remote Execution (from your personal computer):

``` console
./run.sh -a turing.wpi.edu alexnet_control
```

Sync Code and Trigger Remote Execution (from your personal computer):

``` console
./run.sh -a turing.wpi.edu -s alexnet_control
```

If you don't provide an address with -a, run.sh simply assumes it's already on the correct machine and will immediately
try to start the experiments via sbatch (or run them locally if -l is provided).
