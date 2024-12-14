'''
Code for ploting the benchmalr results (run_bmarl.py).
Benchmalr results are saved as json files.
Outputs graphs comparing the used algorithms.
'''

import os
import json
import numpy as np
import matplotlib.pyplot as plt


def find_json_files(main_folder):
    """
    Find all JSON files in subfolders.
    """
    json_files = []
    for root, _, files in os.walk(main_folder):
        for file in files:
            if file.endswith(".json"):
                json_files.append(os.path.join(root, file))
    return json_files


def extract_data_for_plotting(file_path):
    """
    Extract relevant data for plotting.
    """
    with open(file_path, "r") as f:
        data = json.load(f)

    episode_reward_mean_data = []
    agent_return_data = []

    for task, task_data in data["vmas"].items():
        for algo, algo_data in task_data.items():
            for seed, seed_data in algo_data.items():
                for step_key, step_data in seed_data.items():
                    if step_key.startswith("step_"):
                        training_iteration = int(step_key.split("_")[1])
                        step_count = step_data["step_count"]
                        agents_return = step_data["agents_return"]
                        mean_return = np.mean(agents_return)
                        std_dev_return = np.std(agents_return)

                        episode_reward_mean_data.append({
                            "task": task,
                            "algorithm": algo,
                            "training_iteration": training_iteration,
                            "episode_reward_mean": mean_return,
                            "reward_std_dev": std_dev_return,})

                        for agent_return in agents_return:
                            agent_return_data.append({
                                "task": task,
                                "algorithm": algo,
                                "step_count": step_count,
                                "agent_return": agent_return,})

    return episode_reward_mean_data, agent_return_data


def plot_episode_reward_mean(data, task_filter, output_path):
    """
    Plot episode reward mean over training iterations for a specific task.
    """
    filtered_data = [entry for entry in data if entry["task"] == task_filter]

    grouped_data = {}
    for entry in filtered_data:
        algo = entry["algorithm"]
        if algo not in grouped_data:
            grouped_data[algo] = []
        grouped_data[algo].append(entry)

    plt.figure(figsize=(10, 6))
    for algo, entries in grouped_data.items():
        entries = sorted(entries, key=lambda x: x["training_iteration"])
        iterations = [entry["training_iteration"] for entry in entries]
        means = [entry["episode_reward_mean"] for entry in entries]
        std_devs = [entry["reward_std_dev"] for entry in entries]

        plt.plot(
            iterations, means, label=f"{algo.upper()}", linewidth=2
        )
        plt.fill_between(
            iterations,
            np.array(means) - np.array(std_devs),
            np.array(means) + np.array(std_devs),
            alpha=0.2,
        )

    plt.title(f"{task_filter.capitalize()} - Episode Reward Mean vs Training Iteration", fontsize=16)
    plt.xlabel("Training Iteration", fontsize=14)
    plt.ylabel("Episode Reward Mean", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_agent_return_over_steps(data, task_filter, output_path):
    """
    Plot agent return over steps for a specific task.
    """
    filtered_data = [entry for entry in data if entry["task"] == task_filter]

    grouped_data = {}
    for entry in filtered_data:
        algo = entry["algorithm"]
        if algo not in grouped_data:
            grouped_data[algo] = []
        grouped_data[algo].append(entry)

    plt.figure(figsize=(10, 6))
    for algo, entries in grouped_data.items():
        entries = sorted(entries, key=lambda x: x["step_count"])
        steps = [entry["step_count"] for entry in entries]
        returns = [entry["agent_return"] for entry in entries]

        plt.plot(
            steps, returns, label=f"{algo.upper()}", alpha=0.7, linestyle="--"
        )

    plt.title(f"{task_filter.capitalize()} - Agent Return Over Steps", fontsize=16)
    plt.xlabel("Step Count", fontsize=14)
    plt.ylabel("Agent Return", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


if __name__ == "__main__":
    # Set the main folder where JSON files are located
    workspace = str(os.path.dirname(os.path.abspath(__file__)))
    main_folder = os.path.join(workspace, "runs")
    output_folder = os.path.join(workspace, "graphs")
    os.makedirs(output_folder, exist_ok=True)

    # Find and process all JSON files
    json_files = find_json_files(main_folder)
    episode_reward_mean_data = []
    agent_return_data = []

    for file_path in json_files:
        erm_data, ar_data = extract_data_for_plotting(file_path)
        episode_reward_mean_data.extend(erm_data)
        agent_return_data.extend(ar_data)

    # Plot Episode Reward Mean vs Training Iteration for Balance
    plot_episode_reward_mean(
        episode_reward_mean_data,
        "balance",
        os.path.join(output_folder, "balance_episode_reward_mean.png"),
    )

    # Plot Episode Reward Mean vs Training Iteration for Reverse Transport
    plot_episode_reward_mean(
        episode_reward_mean_data,
        "reverse_transport",
        os.path.join(output_folder, "reverse_transport_episode_reward_mean.png"),
    )

    # Plot Agent Return Over Steps for Balance
    plot_agent_return_over_steps(
        agent_return_data,
        "balance",
        os.path.join(output_folder, "balance_agent_return_over_steps.png"),
    )

    # Plot Agent Return Over Steps for Reverse Transport
    plot_agent_return_over_steps(
        agent_return_data,
        "reverse_transport",
        os.path.join(output_folder, "reverse_transport_agent_return_over_steps.png"),
    )

    print(f"Graphs saved in {output_folder}")
