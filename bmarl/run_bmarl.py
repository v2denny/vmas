from benchmarl.algorithms import MappoConfig, IppoConfig, MasacConfig, MaddpgConfig
from benchmarl.benchmark import Benchmark
from benchmarl.environments import VmasTask
from benchmarl.experiment import ExperimentConfig
from benchmarl.models.mlp import MlpConfig
from pathlib import Path

if __name__ == "__main__":
    # Experiment Configuration
    experiment_config = ExperimentConfig.get_from_yaml()
    experiment_config.max_n_iters = 200  # Total training iterations
    experiment_config.on_policy_collected_frames_per_batch = 2000
    experiment_config.off_policy_collected_frames_per_batch = 2000
    experiment_config.train_device = "cuda"  # Use CUDA for training
    experiment_config.buffer_device = "cuda"  # Use CUDA for buffer operations
    experiment_config.evaluation = True
    experiment_config.evaluation_interval = 2000  # Evaluate every 2000 frames
    experiment_config.evaluation_episodes = 10  # 10 episodes per evaluation
    experiment_config.lr = 5e-4  # Learning rate
    experiment_config.gamma = 0.99  # Discount factor
    experiment_config.adam_eps = 1e-5
    experiment_config.clip_grad_val = 0.5  # Gradient clipping

    # Define and ensure save folder exists
    save_folder = Path(__file__).parent / "runs"
    save_folder.mkdir(parents=True, exist_ok=True)  # Ensure the "runs" folder exists
    experiment_config.save_folder = str(save_folder)

    # Load Tasks
    balance_task = VmasTask.BALANCE.get_from_yaml()
    reverse_transport_task = VmasTask.REVERSE_TRANSPORT.get_from_yaml()

    # Update Task Configurations
    balance_task.config["max_steps"] = 200
    balance_task.config["n_agents"] = 4
    reverse_transport_task.config["max_steps"] = 200
    reverse_transport_task.config["n_agents"] = 4

    # Tasks list
    tasks = [balance_task, reverse_transport_task]

    # Algorithms: Use MAPPO and IPPO
    algorithm_configs = [
        #MappoConfig.get_from_yaml(),
        #IppoConfig.get_from_yaml(),
        MasacConfig.get_from_yaml(),
        MaddpgConfig.get_from_yaml(),
    ]

    # Models
    model_config = MlpConfig.get_from_yaml()
    critic_model_config = MlpConfig.get_from_yaml()

    # Benchmark
    benchmark = Benchmark(
        algorithm_configs=algorithm_configs,
        tasks=tasks,
        seeds={42},
        experiment_config=experiment_config,
        model_config=model_config,
        critic_model_config=critic_model_config,
    )
    benchmark.run_sequential()
