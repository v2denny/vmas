'''
Code for running VMAS on RLlib.
Modified the usage of GPU, now gets the num_gpus from pytorch.
Now saves a video every training iteration so we can see the training progress.
Adapted from VMAS git:
#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
'''

import os
import cv2
from typing import Dict, Optional
import numpy as np
import ray
from ray import tune
from ray.rllib import BaseEnv, Policy, RolloutWorker
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.algorithms.callbacks import DefaultCallbacks, MultiCallbacks
from ray.rllib.evaluation import Episode, MultiAgentEpisode
from ray.rllib.utils.typing import PolicyID
from ray.tune import register_env
from ray.air.callbacks.wandb import WandbLoggerCallback
from vmas import make_env
from vmas.simulator.environment import Wrapper
import torch


'''
Setting up VMAS scenario.
'''
scenario_name = "reverse_transport"     #scenario_name = "balance"
n_agents = 4
continuous_actions = True
max_steps = 300
num_vectorized_envs = 10
num_workers = 10
vmas_device = "cuda"  # or cpu

def env_creator(config: Dict):
    env = make_env(
        scenario=config["scenario_name"],
        num_envs=config["num_envs"],
        device=config["device"],
        continuous_actions=config["continuous_actions"],
        wrapper=Wrapper.RLLIB,
        max_steps=config["max_steps"],
        # Scenario specific variables
        **config["scenario_config"],
    )
    return env


'''
Starting ray for parallel operations.
'''
if not ray.is_initialized():
    ray.init()
    print("Ray init!")
    print("Creating", str(scenario_name), "environment...")
register_env(scenario_name, lambda config: env_creator(config))


'''
Custom callback functions.
'''
class EvaluationCallbacks(DefaultCallbacks):
    def on_episode_step(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        episode: MultiAgentEpisode,
        **kwargs,
    ):
        info = episode.last_info_for()
        for a_key in info.keys():
            for b_key in info[a_key]:
                try:
                    episode.user_data[f"{a_key}/{b_key}"].append(info[a_key][b_key])
                except KeyError:
                    episode.user_data[f"{a_key}/{b_key}"] = [info[a_key][b_key]]

    def on_episode_end(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: MultiAgentEpisode,
        **kwargs,
    ):
        info = episode.last_info_for()
        for a_key in info.keys():
            for b_key in info[a_key]:
                metric = np.array(episode.user_data[f"{a_key}/{b_key}"])
                episode.custom_metrics[f"{a_key}/{b_key}"] = np.sum(metric).item()

class VideoSavingCallbacks(DefaultCallbacks):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.frames = []
        self.iteration_counter = 0  # Counter to track the iteration number for video saving

    def on_episode_step(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        episode: Episode,
        **kwargs,
    ) -> None:
        frame = base_env.vector_env.try_render_at(mode="rgb_array")
        if frame is not None:
            self.frames.append(frame)

    def on_episode_end(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[PolicyID, Policy],
        episode: Episode,
        **kwargs,
    ) -> None:
        # Save video after an evaluation episode ends
        if len(self.frames) > 0:
            os.makedirs("rllib_videos", exist_ok=True)
            video_path = os.path.join("rllib_videos", f"iteration_{self.iteration_counter}.mp4")
            height, width, _ = self.frames[0].shape
            video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (width, height))

            for frame in self.frames:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                video_writer.write(frame_bgr)

            video_writer.release()
            self.frames = []
            self.iteration_counter += 1
            print(f"Saved video for iteration {self.iteration_counter} to {video_path}")


'''
Training script.
'''
def train():
    if torch.cuda.is_available():
        num_gpus = 0.001
        print("GPU is available!")
    else:
        num_gpus = 0
        print("GPU not available!")
    num_gpus_per_worker = ((1 - num_gpus) / (num_workers + 1) if (num_gpus > 0 and vmas_device == "cuda") else 0)

    checkpoint_path = "Checkpoint folder path"
    # If you load a checkpoint change self.iteration_counter at VideoSavingCallbacks so the videos are saved in the correct order
    
    if not os.path.exists(checkpoint_path):
        checkpoint_path = None
        print("Checkpoint not found, starting from 0")
    else: print(f"Checkpoint found at {checkpoint_path}")

    tune.run(
        PPOTrainer,
        stop={"training_iteration": 150},
        checkpoint_freq=1,
        keep_checkpoints_num=2,
        checkpoint_at_end=True,
        local_dir="rllib_logdir",
        checkpoint_score_attr="episode_reward_mean",
        restore=checkpoint_path,
        callbacks=[
            WandbLoggerCallback(
                project=f"{scenario_name}",
                api_key=os.environ.get("WANDB_API_KEY", ""),
            )
        ],
        config={
            "seed": 0,
            "framework": "torch",
            "env": scenario_name,
            "kl_coeff": 0.01,
            "kl_target": 0.01,
            "lambda": 0.9,
            "clip_param": 0.2,
            "vf_loss_coeff": 1,
            "vf_clip_param": float("inf"),
            "entropy_coeff": 0,
            "train_batch_size": 60000,
            "rollout_fragment_length": 125,
            "sgd_minibatch_size": 4096,
            "num_sgd_iter": 40,
            "num_gpus": num_gpus,
            "num_workers": num_workers,
            "num_gpus_per_worker": num_gpus_per_worker,
            "num_envs_per_worker": num_vectorized_envs,
            "lr": 5e-5,
            "gamma": 0.99,
            "use_gae": True,
            "use_critic": True,
            "batch_mode": "truncate_episodes",
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
            "evaluation_interval": 1,
            "evaluation_duration": 1,
            "evaluation_num_workers": 1,
            "evaluation_parallel_to_training": False,
            "evaluation_config": {
                "num_envs_per_worker": 1,
                "env_config": {
                    "num_envs": 1,
                },
                "callbacks": MultiCallbacks([VideoSavingCallbacks, EvaluationCallbacks]),
            },
            "callbacks": EvaluationCallbacks,
        },
    )

if __name__ == "__main__":
    train()
