import optuna
import yaml
import numpy as np
import torch
from torch import nn as nn

from stable_baselines3 import PPO, DDPG, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.utils import set_random_seed

import robosuite as suite
from robosuite.wrappers import GymWrapper
from robosuite.controllers import load_controller_config

from optuna.visualization import plot_optimization_history


# Set name of study
study_name = "study_sawyer_pickplace"
storage_name = "sqlite:///{}.db".format(study_name)

# Load configuration
with open("config_hyperparams.yaml") as stream:
    config = yaml.safe_load(stream)

# Define the environment setup
# Make robosuite environment into a gym environment as stable baselines only supports gym environments
def make_env(env_id, options, rank, seed=0):
    def _init():
        env = GymWrapper(suite.make(env_id, **options))
        env.render_mode = 'mujoco'
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init

# Method to evaluate the policy
def evaluate_policy(model, env, n_eval_episodes=5):
    all_episode_rewards = []
    for _ in range(n_eval_episodes):
        episode_rewards = []
        done = np.array([False])
        obs = env.reset()
        while not done.all():
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_rewards.append(reward)
        all_episode_rewards.append(np.sum(episode_rewards))
    mean_reward = np.mean(all_episode_rewards)
    return mean_reward

def objective(trial):
    # Suggest hyperparameters
    #learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    learning_rate = 0.001
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    gamma = trial.suggest_categorical('gamma', [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    n_steps = trial.suggest_categorical('n_steps', [512, 1024, 2048])
    #horizon = trial.suggest_categorical('horizon', [512, 1024, 2048])
    horizon = 512
    control_freq = trial.suggest_uniform('control_freq', 100, 150)
    #total_timesteps = trial.suggest_categorical('total_timesteps', [1e5, 2e5, 5e5, 1e6, 2e6])
    total_timesteps = 3e5
    ent_coef = trial.suggest_float("ent_coef", 0.00000001, 0.1, log=True)
    clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3, 0.4])
    #n_epochs = trial.suggest_categorical("n_epochs", [1, 5, 10, 20])
    gae_lambda = trial.suggest_categorical("gae_lambda", [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0])
    max_grad_norm = trial.suggest_categorical("max_grad_norm", [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 5])
    vf_coef = trial.suggest_float("vf_coef", 0, 1)
    net_arch_type = trial.suggest_categorical("net_arch", ["tiny", "small", "medium"])

    print(
        f"Learning rate: {learning_rate}, "
        f"Batch size: {batch_size}, "
        f"Gamma: {gamma}, "
        f"N steps: {n_steps}, "
        f"Horizon: {horizon}, "
        f"Control freq: {control_freq}, "
        f"Total timesteps: {total_timesteps}, "
        f"Entropy coefficient: {ent_coef}, "
        f"Clip range: {clip_range}, "
        f"GAE lambda: {gae_lambda}, "
        f"Max grad norm: {max_grad_norm}, "
        f"Value function coefficient: {vf_coef}, "
        f"Network architecture: {net_arch_type}")

    # Set controller configuration
    controller_config = load_controller_config(default_controller=config["controller"])

    # Setup environment
    # Define environment parameters for specific environment "PickPlace"
    env_options = {
        "robots": config["robot_name"],
        "controller_configs": controller_config,
        "gripper_types": config["gripper"],
        "has_renderer": False,
        "has_offscreen_renderer": True,
        "single_object_mode": 2,
        "object_type": "milk",
        "use_camera_obs": True,         # provide image observations to agent
        "use_object_obs": False,        # don't provide object observations to agent
        "camera_names": "agentview",    # use "agentview" camera for observations
        "camera_heights": 128,          # image height
        "camera_widths": 128,           # image width
        "reward_shaping": True,         # use a dense reward signal for learning
        "horizon": horizon,
        "control_freq": control_freq,
    }    
    
    # Setup environment
    env = SubprocVecEnv([make_env("PickPlace", env_options, i, config["seed"]) for i in range(config["num_envs"])], start_method='spawn')
    env = VecNormalize(env)

    # TODO: account when using multiple envs
    if batch_size > n_steps:
        batch_size = n_steps

    # Check if cuda(linux) or mps(mac) is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Cuda backend is available.")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Mps backend is available.")
    else:
        device = torch.device("cpu")
        print("Cuda backend is not available, using CPU.")

    # Orthogonal initialization
    ortho_init = False

    activation_fn_name = trial.suggest_categorical("activation_fn", ["tanh", "relu"])

    # Independent networks usually work best when not working with images
    net_arch = {
        "tiny": dict(pi=[64], vf=[64]),
        "small": dict(pi=[64, 64], vf=[64, 64]),
        "medium": dict(pi=[256, 256], vf=[256, 256]),
    }[net_arch_type]

    activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU, "leaky_relu": nn.LeakyReLU}[activation_fn_name]

    # Initialize model
    if config["algorithm"] == "PPO":
        model = PPO(config["policy"],
                    env,
                    learning_rate=learning_rate,
                    batch_size=batch_size,
                    gamma=gamma,
                    n_steps=n_steps,
                    ent_coef=ent_coef,
                    clip_range=clip_range,
                    gae_lambda=gae_lambda,
                    max_grad_norm=max_grad_norm,
                    vf_coef=vf_coef,
                    policy_kwargs=dict(
                                    net_arch=net_arch,
                                    activation_fn=activation_fn,
                                    ortho_init=ortho_init,
                                    ),
                    verbose=0,
                    tensorboard_log=None,
                    device=device
                    )
    elif config["algorithm"] == "DDPG":
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
        model = DDPG(config["policy"], env, action_noise=action_noise, learning_rate=learning_rate, batch_size=batch_size, gamma=gamma, verbose=0, tensorboard_log=None, device=device)
    elif config["algorithm"] == "SAC":
        model = SAC(config["policy"], env, learning_rate=learning_rate, batch_size=batch_size, gamma=gamma, verbose=0, tensorboard_log=None, device=device)
    
    # Train the model
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    env.close()

    # Setup evaluation environment
    # Define environment parameters for specific environment "PickPlace"
    eval_env = SubprocVecEnv([make_env("PickPlace", env_options, i, config["seed"]) for i in range(config["num_eval_envs"])], start_method='spawn')
    eval_env = VecNormalize(eval_env)

    # Evaluate the model
    mean_reward = evaluate_policy(model, eval_env, config["n_eval_episodes"])
    print("Mean reward: ", mean_reward)
    eval_env.close()

    trial.report(mean_reward, step=total_timesteps)

    #Handle pruning based on the intermediate value
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()

    return mean_reward

# Optimize hyperparameters
if __name__ == "__main__":
    study = optuna.create_study(direction='maximize', study_name=study_name, storage=storage_name, load_if_exists=True)
    study.optimize(objective, config["n_trials"])

    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    print('Study statistics: ')
    print('  Number of finished trials: ', len(study.trials))
    print('  Number of pruned trials: ', len(pruned_trials))
    print('  Number of complete trials: ', len(complete_trials))

    print('Best trial: ')
    trial = study.best_trial

    print('  Value: ', trial.value)
    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))

    print('Best hyperparameters: ', study.best_params)
    
    plot_optimization_history(study)
