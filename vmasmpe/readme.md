# VMAS vs MPE Speed Benchmark

This is a Python script to compare the performance of the **VMAS** and **MPE** environments when running multiple parallel environments for reinforcement learning. The code benchmarks the execution time of both frameworks under various configurations and generates a comparison graph.
<br>
<br>

## Contents
1. [Features](#features)
2. [How It Works](#how-it-works)
3. [Customization](#customization)
<br>
<br>

## Features
- **VMAS** benchmarking on CPU and GPU.
- **MPE** benchmarking on CPU.
- Comparison of execution times across multiple parallel environments.
- Graphical representation of results.
- GPU compatibility for VMAS (tested with Nvidia 4080).
<br>
<br>


## How It Works

### Benchmarked Environments
1. **MPE - CPU**: Runs the `simple_spread` environment from PettingZoo on the CPU.
2. **VMAS - CPU**: Runs the `simple_spread` environment from VMAS on the CPU.
3. **VMAS - GPU**: Runs the `simple_spread` environment from VMAS on the GPU.

### Steps
1. The script initializes multiple parallel environments (`n_envs`) for each configuration.
2. Each environment is run for a fixed number of steps (`n_steps`).
3. Execution time is recorded and plotted against the number of environments.

### Output
- A graph comparing the execution times for the three configurations is saved as `comparison_graph.png`.
<br>
<br>

## Customization

- **Number of Steps**: Modify the `n_steps` parameter in the `run_comparison()` function to adjust the number of steps per environment.
- **Number of Parallel Environments**: The range and granularity of `n_envs` can be adjusted in the `list_n_envs` variable.

