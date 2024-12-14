'''
Code for ploting the vmas rllib results (vmas_rllib_perf.py).
Outputs graphs comparing PPO algorithm for both vmas balance and reverse_transport scenarios.
'''

import pandas as pd
import matplotlib.pyplot as plt

# Load the CSVs
balance_csv = "graph_bl.csv" 
reverse_transport_csv = "graph_rt.csv"

# Read data
balance_data = pd.read_csv(balance_csv)
reverse_transport_data = pd.read_csv(reverse_transport_csv)

# Plot the data
plt.figure(figsize=(10, 6))

# Balance data plot
plt.plot(
    balance_data["training_iteration"],
    balance_data["episode_reward_mean"],
    label="Balance Environment",
    linewidth=2,
    color="blue",
)

# Reverse Transport data plot
plt.plot(
    reverse_transport_data["training_iteration"],
    reverse_transport_data["episode_reward_mean"],
    label="Reverse Transport Environment",
    linewidth=2,
    color="green",
)

# Adding graph details
plt.title("Episode Reward Mean vs Training Iteration", fontsize=16)
plt.xlabel("Training Iteration", fontsize=14)
plt.ylabel("Episode Reward Mean", fontsize=14)
plt.legend(fontsize=12)
plt.grid(alpha=0.5)
plt.tight_layout()

# Save and show the plot
plt.savefig("episode_reward_mean_comparison.png")
plt.show()
