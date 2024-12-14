# BenchMarl VMAS Benchmark

This folder contains two Python scripts to benchmark and visualize results for multi-agent reinforcement learning (MARL) algorithms on VMAS tasks.

We used [BenchMarl](https://github.com/facebookresearch/BenchMARL) tool for this task.
<br>
<br>

## Contents
1. [Features](#features)
2. [How It Works](#how-it-works)
3. [Customization](#customization)
<br>
<br>

## Features
1. **`run_bmarl.py`**:
   - Benchmarks MARL algorithms (MAPPO, IPPO, MASAC, MADDPG) on the Balance and Reverse Transport tasks from VMAS.
   - Saves results as JSON files in the `runs` folder.
   - Configured to run on GPU (CUDA).

2. **`plot_bmarl.py`**:
   - Plots benchmark results saved by `run_bmarl.py`.
   - Generates graphs comparing the performance of algorithms across tasks.
<br>
<br>

## How It Works

### Benchmark Script (`run_bmarl.py`)
1. Loads configurations for tasks (Balance and Reverse Transport) and algorithms from YAML files.
2. Runs benchmarks for MAPPO, IPPO, MASAC, and MADDPG algorithms sequentially.
3. Evaluates algorithms periodically and saves results to the `runs` folder.
4. Uses GPU for training and buffer operations for faster execution.

### Plot Script (`plot_bmarl.py`)
1. Reads JSON results saved in the `runs` folder by the benchmark script.
2. Extracts and organizes data (e.g., reward means, agent returns).
3. Generates plots showing algorithm performance over training iterations and steps.
4. Saves graphs in the `graphs` folder.
<br>
<br>

## Customization

- **Number of Training Iterations**:
  Modify `experiment_config.max_n_iters` in `run_bmarl.py`.

- **Number of Agents**:
  Update the `balance_task.config["n_agents"]` or `reverse_transport_task.config["n_agents"]`.

- **GPU/CPU Execution**:
  Change `experiment_config.train_device` and `experiment_config.buffer_device` from `"cuda"` to `"cpu"` in `run_bmarl.py` to run the benchmark on CPU instead of GPU.