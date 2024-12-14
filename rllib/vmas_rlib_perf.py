'''
Code for running VMAS on RLlib.
Performance version of the "vmas_rllib.py" file.
No WanDB logging.
No video saving.
Focused on performance.
'''

import os
import csv
import ray
import torch
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune import register_env, tune
from vmas import make_env
from vmas.simulator.environment import Wrapper
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from pathlib import Path
import logging
logging.getLogger("ray").setLevel(logging.ERROR)

# Configuration
scenario_name = "reverse_transport"   # or "balance"
n_agents = 4
continuous_actions = True
max_steps = 200
num_vectorized_envs = 10
num_workers = 10
vmas_device = "cuda"  # or "cpu"

# Environment creator
def env_creator(config):
    env = make_env(
        scenario=config["scenario_name"],
        num_envs=config["num_envs"],
        device=config["device"],
        continuous_actions=config["continuous_actions"],
        wrapper=Wrapper.RLLIB,
        max_steps=config["max_steps"],
        **config["scenario_config"],
    )
    return env

# Initialize ray for parallel computting
if not ray.is_initialized():
    ray.init(ignore_reinit_error=True)
register_env(scenario_name, lambda config: env_creator(config))

# Custom Callbacks for Logging `episode_reward_mean`
class TrainingCallbacks(DefaultCallbacks):

    def on_train_result(self, *, algorithm, result, **kwargs):
        training_iteration = result["training_iteration"]
        episode_reward_mean = result["episode_reward_mean"]

        # Check file existence and write
        file_path = str(Path(__file__).parent / "rllib_perf/graph_rt.csv")

        file_exists = os.path.exists(file_path)
        with open(file_path, 'a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(['training_iteration', 'episode_reward_mean'])
            writer.writerow([training_iteration, episode_reward_mean])    

# Training
def train():
    # GPU Allocation
    if torch.cuda.is_available():
        num_gpus = 0.001
        print("GPU is available!")
    else:
        num_gpus = 0
        print("GPU not available!")
    num_gpus_per_worker = ((1 - num_gpus) / (num_workers + 1) if (num_gpus > 0 and vmas_device == "cuda") else 0)

    callbacks = TrainingCallbacks
    workspace = str(os.path.dirname(os.path.abspath(__file__)))
    save_folder = os.path.join(workspace, "rllib_perf")
    stdout = os.path.join(workspace, "rllib_perf/stdout.log")
    stderr = os.path.join(workspace, "rllib_perf/stderr.log")

    # Run training
    tune.run(
        PPOTrainer,
        verbose=1,
        stop={"training_iteration": 200},
        local_dir=save_folder,
        log_to_file=(stdout, stderr),
        checkpoint_at_end=True,
        checkpoint_score_attr="episode_reward_mean",
        config={
            "log_level": "ERROR",
            "disable_env_checking": True,
            "framework": "torch",
            "env": scenario_name,
            "train_batch_size": 2000,
            "rollout_fragment_length": 20,
            "sgd_minibatch_size": 256,
            "num_sgd_iter": 10,
            "num_gpus": num_gpus,
            "num_workers": num_workers,
            "num_gpus_per_worker": num_gpus_per_worker,
            "num_envs_per_worker": num_vectorized_envs,
            "lr": 5e-4,
            "gamma": 0.99,
            "use_gae": True,
            "clip_param": 0.2,
            "vf_loss_coeff": 1.0,
            "env_config": {
                "device": vmas_device,
                "num_envs": num_vectorized_envs,
                "scenario_name": scenario_name,
                "continuous_actions": continuous_actions,
                "max_steps": max_steps,
                "scenario_config": {
                    "n_agents": n_agents,
                },
            },
            "callbacks": callbacks,  # Use custom callbacks for logging
        },
    )


if __name__ == "__main__":
    train()
