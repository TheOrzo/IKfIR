import optuna
import yaml
import os
import numpy as np
import torch
from torch import nn as nn
from stable_baselines3 import PPO, DDPG, SAC
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.utils import set_random_seed
from robosuite.wrappers import GymWrapper
import robosuite as suite
from robosuite.controllers import load_controller_config
from optuna.visualization import plot_optimization_history, plot_slice


#optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
study_name = "PPO_Sawyer_OSC_POSE"  # Unique identifier of the study.
storage_name = "sqlite:///{}.db".format(study_name)

# Define the environment setup
def make_env(env_id, options, rank, seed=0):
    def _init():
        env = GymWrapper(suite.make(env_id, **options))
        env.render_mode = 'mujoco'
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init


def save_model(model_path, model, vec_env):
    model.save(model_path + ".zip")
    vec_env.save(model_path + ".env")

def objective(trial):
    # Suggest hyperparameters
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    gamma = trial.suggest_categorical('gamma', [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    n_steps = trial.suggest_categorical('n_steps', [128, 256, 512, 1024, 2048])
    horizon = trial.suggest_categorical('horizon', [500, 1000, 1500])
    control_freq = trial.suggest_uniform('control_freq', 20, 150)
    #total_timesteps = trial.suggest_categorical('total_timesteps', [1e5, 2e5, 5e5, 1e6, 2e6])
    total_timesteps = 5e5
    ent_coef = trial.suggest_float("ent_coef", 0.00000001, 0.1, log=True)
    clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3, 0.4])
    #n_epochs = trial.suggest_categorical("n_epochs", [1, 5, 10, 20])
    gae_lambda = trial.suggest_categorical("gae_lambda", [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0])
    max_grad_norm = trial.suggest_categorical("max_grad_norm", [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 5])
    vf_coef = trial.suggest_float("vf_coef", 0, 1)
    net_arch_type = trial.suggest_categorical("net_arch", ["tiny", "small", "medium"])

    # Load configuration
    with open("config_hyperparams.yaml") as stream:
        config = yaml.safe_load(stream)

    controller_config = load_controller_config(default_controller=config["robot_controller"])

    env_options = {
        "robots": config["robot_name"],
        "controller_configs": controller_config,
        "has_renderer": False,
        "has_offscreen_renderer": True,
        "single_object_mode": 2,
        "object_type": "milk",
        "use_camera_obs": True,
        "use_object_obs": False,
        "camera_names": "agentview",
        "camera_heights": 128,
        "camera_widths": 128,
        "reward_shaping": True,
        "horizon": horizon,
        "control_freq": control_freq,
    }    
    
    # Setup environment
    if config["multiprocessing"]:
        env = SubprocVecEnv([make_env("PickPlace", env_options, i, config["seed"]) for i in range(config["num_envs"])], start_method='spawn')
        eval_env = SubprocVecEnv([make_env("PickPlace", env_options, i, config["seed"]) for i in range(config["num_envs"])], start_method='spawn')
        eval_env = VecNormalize(eval_env)
        # TODO: account when using multiple envs
        if batch_size > n_steps:
            batch_size = n_steps
    else:
        env = DummyVecEnv(make_env("PickPlace", env_options, 0, config["seed"]))
        eval_env = DummyVecEnv([make_env("PickPlace", env_options, 0, config["seed"])])
        eval_env = VecNormalize(eval_env)

    if config["normalize"]:
        env = VecNormalize(env)

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    # Orthogonal initialization
    ortho_init = False

    activation_fn_name = trial.suggest_categorical("activation_fn", ["tanh", "relu"])

    # Independent networks usually work best
    # when not working with images
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
    
    '''
    # Evaluation callback
    eval_callback = EvalCallback(eval_env, best_model_save_path="./logs/",
                                 log_path="./logs/", eval_freq=2048,
                                 deterministic=True, render=False)
    '''
    '''
    # this gives somewhat different (higher) results to the rollout/ep_re_mean. discounting effect?
    # one would expect to be able to read the rollout/ep_re_mean directly but I just cant find a way
    class SaveBestModelCallback(BaseCallback):
        def __init__(self, save_path: str, verbose=1):
            super(SaveBestModelCallback, self).__init__(verbose)
            # self.check_freq = check_freq
            self.save_path = save_path
            self.best_mean_reward = 0
            self.not_first_run = False

        def _on_rollout_start(self) -> None:
            if self.not_first_run:
                episode_rewards = []
                for i in range(len(self.locals["infos"])):
                    episode_rewards.append(self.locals["infos"][i]["episode"]["r"])
                # print(episode_rewards)
                mean_reward = sum(episode_rewards) / len(episode_rewards)
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    if self.verbose > 0:
                        print(
                            f"Saving new best model to {self.save_path} with mean reward {mean_reward}"
                        )
                    save_model(self.save_path, self.model, self.training_env)
            self.not_first_run = True

        def _on_step(self) -> bool:
            return True
    
    callback = SaveBestModelCallback(save_path="./logs/optimization/")
    '''

    # Train the model
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    
    # Evaluate the model
    #mean_reward = eval_callback.last_mean_reward
    mean_reward, _ = model.evaluate_policy(eval_env, n_eval_episodes=5, deterministic=True)
    
    trial.report(mean_reward, step=total_timesteps)

    #Handle pruning based on the intermediate value
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()

    return mean_reward

# Optimize hyperparameters
if __name__ == "__main__":
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50, timeout=600)

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
