'''
Code for comparing MPE and VMAS speed while having multiple parallel environments.
Adapted from the VMAS git.
Highly modified due to the fact that the original code was outdated.
Added GPU compatability.
Tested with Nvidia 4080 and intel i9 CPU.
'''

import argparse
import os
import time
import numpy as np
import torch
from tqdm import tqdm
from matplotlib import pyplot as plt
from pettingzoo.mpe import simple_spread_v3
import vmas

def mpe_make_env():
    """
    Create an instance of the PettingZoo simple_spread environment (v3).
    """
    env = simple_spread_v3.env(render_mode=None)
    env.reset()
    return env

def run_mpe_simple_spread(n_envs: int, n_steps: int):
    """
    Run the PettingZoo MPE simple_spread environment for benchmarking.
    """
    n_envs = int(n_envs)
    n_steps = int(n_steps)

    envs = [mpe_make_env() for _ in range(n_envs)]

    init_time = time.time()

    for _ in range(n_steps):
        for env_idx in range(n_envs):
            env = envs[env_idx]
            for agent in env.agent_iter():
                observation, reward, termination, truncation, info = env.last()
                if termination or truncation:
                    action = None
                else:
                    action = env.action_space(agent).sample()  # Sample valid action
                env.step(action)

    total_time = time.time() - init_time
    return total_time

def run_vmas_simple_spread(n_envs: int, n_steps: int, device: str):
    """
    Run the VMAS simple_spread environment for benchmarking.
    """
    n_envs = int(n_envs)
    n_steps = int(n_steps)
    n_agents = 2
    env = vmas.make_env(
        "simple_spread",
        device=device,
        num_envs=n_envs,
        continuous_actions=False,
        n_agents=n_agents,
    )
    simple_shared_action = [2]

    env.reset()
    init_time = time.time()

    for _ in range(n_steps):
        actions = []
        for _ in range(n_agents):
            actions.append(
                torch.tensor(
                    simple_shared_action,
                    device=device,
                ).repeat(n_envs, 1)
            )
        env.step(actions)

    total_time = time.time() - init_time
    return total_time

def run_comparison(n_steps: int = 100):
    """
    Run the comparison for MPE (CPU), VMAS (CPU), and VMAS (GPU).
    """
    configurations = {
        "MPE - CPU": {"env_type": "mpe", "device": "cpu"},
        "VMAS - CPU": {"env_type": "vmas", "device": "cpu"},
        "VMAS - GPU": {"env_type": "vmas", "device": "cuda"},
    }

    results = {}
    low = 1
    high = 10000
    num = 20
    list_n_envs = np.linspace(low, high, num)

    # Outer loop for configurations
    for label, config in configurations.items():
        print(f"Running benchmark for {label}...")
        times = []

        # Start timing the entire configuration
        config_start_time = time.time()

        # Progress bar for number of parallel environments
        for n_envs in tqdm(list_n_envs, desc=f"{label}", unit="envs"):
            if config["env_type"] == "mpe":
                times.append(run_mpe_simple_spread(n_envs=n_envs, n_steps=n_steps))
            elif config["env_type"] == "vmas":
                times.append(run_vmas_simple_spread(n_envs=n_envs, n_steps=n_steps, device=config["device"]))

        # End timing the configuration
        config_total_time = time.time() - config_start_time
        print(f"Total time for {label}: {config_total_time:.2f}s\n")  # Display total time for the configuration

        results[label] = times

    # Plotting
    fig, ax = plt.subplots()
    for label, times in results.items():
        ax.plot(list_n_envs, times, label=label)
    plt.xlabel("Number of parallel environments", fontsize=14)
    plt.ylabel("Seconds", fontsize=14)
    ax.legend(loc="upper left")

    fig.suptitle("VMAS vs MPE", fontsize=16)
    ax.set_title(f"Execution time comparison for {n_steps} steps", fontsize=10)

    save_folder = os.path.dirname(os.path.realpath(__file__)) # + "/vmas_vs_mpe_graphs"
    os.makedirs(save_folder, exist_ok=True)
    plt.savefig(f"{save_folder}/comparison_graph.png")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a time comparison between VMAS and MPE")
    args = parser.parse_args()
    run_comparison(n_steps=100)
