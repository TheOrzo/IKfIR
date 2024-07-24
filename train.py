import numpy as np
import os
import robosuite as suite
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from typing import Any
from typing import Dict


from robosuite import load_controller_config
from robosuite.environments.base import register_env
from robosuite.controllers import load_controller_config
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.save_util import save_to_zip_file, load_from_zip_file
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from robosuite.wrappers import GymWrapper

from stable_baselines3 import PPO, DDPG


N_TRIALS = 100
N_STARTUP_TRIALS = 5
N_EVALUATIONS = 2
N_TIMESTEPS = int(2e4)
EVAL_FREQ = int(N_TIMESTEPS / N_EVALUATIONS)
N_EVAL_EPISODES = 3

DEFAULT_HYPERPARAMS = {
    "policy": "MlpPolicy",
    "env": "PickPlace",
}


controller_config = load_controller_config(default_controller="JOINT_POSITION")

def make_env(options, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param options: (dict) additionala arguments to pass to the specific environment class initializer
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = GymWrapper(suite.make("PickPlace", **options))
        env.render_mode = 'mujoco'
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init

def make_training_env():
    env = SubprocVecEnv([make_env(
        dict(
            robots=["Sawyer"],                        # load a Sawyer robot and a Panda robot
            gripper_types="default",                # use default grippers per robot arm
            controller_configs=controller_config,   # each arm is controlled using OSC
            has_renderer=False,                     # no on-screen rendering
            has_offscreen_renderer=True,            # off-screen rendering needed for image obs
            control_freq=20,                        # 20 hz control for applied actions
            horizon=1024,                            # each episode terminates after 600 steps
            use_object_obs=False,                   # don't provide object observations to agent
            use_camera_obs=True,                   # provide image observations to agent
            camera_names="agentview",               # use "agentview" camera for observations
            camera_heights=84,                      # image height
            camera_widths=84,                       # image width
            reward_shaping=True),                    # use a dense reward signal for learning
        i,
        7183485 + i
        ) for i in range(6)])

    return VecNormalize(env)

def make_evaluation_env():
    env = make_env(
        dict(
            robots=["Sawyer"],                        # load a Sawyer robot and a Panda robot
            gripper_types="default",                # use default grippers per robot arm
            controller_configs=controller_config,   # each arm is controlled using OSC
            has_renderer=True,                     # no on-screen rendering
            has_offscreen_renderer=True,            # off-screen rendering needed for image obs
            control_freq=20,                        # 20 hz control for applied actions
            horizon=1000,                            # each episode terminates after 200 steps
            use_object_obs=False,                   # don't provide object observations to agent
            use_camera_obs=True,                   # provide image observations to agent
            camera_names="agentview",               # use "agentview" camera for observations
            camera_heights=84,                      # image height
            camera_widths=84,                       # image width
            reward_shaping=True),                    # use a dense reward signal for learning
        0,
        435878152
        )

    return VecNormalize(env)

def sample_ppo_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Sampler for PPO hyperparameters."""
    learning_rate = trial.suggest_float("lr", 1e-5, 1, log=True)
    n_steps = trial.suggest_int("n_steps", 100, 2000, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256])
    n_epochs = trial.suggest_int("e_epochs", 1, 10)
    gamma = 1.0 - trial.suggest_float("gamma", 0.0001, 0.1, log=True)
    gae_lambda = trial.suggest_float("gae_lambda", 0.9, 1, log=True)
    clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3], log=True)
    #normalize_advantage = trial.suggest_categorical("normalize_advantage", [True, False])
    ent_coef = trial.suggest_float("ent_coef", 0, 0.01, log=True)
    vf_coef = trial.suggest_float("vf_coef", 0.5, 1, log=True)
    target_kl = trial.suggest_float("target_kl", 0.003, 0.3, log=True)

    # Display true values.
    trial.set_user_attr("lr_", learning_rate)
    trial.set_user_attr("n_steps_", n_steps)
    trial.set_user_attr("batch_size_", batch_size)
    trial.set_user_attr("n_epochs_", n_epochs)
    trial.set_user_attr("gamma_", gamma)
    trial.set_user_attr("gae_lambda_", gae_lambda)
    trial.set_user_attr("clip_range_", clip_range)
    trial.set_user_attr("ent_coef_", ent_coef)
    trial.set_user_attr("vf_coef_", vf_coef)
    trial.set_user_attr("taget_kl_", target_kl)

    return {
        "learning_rate": learning_rate,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "n_epochs": n_epochs,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "clip_range": clip_range,
        "ent_coef": ent_coef,
        "vf_coef" : vf_coef,
        "target_kl": target_kl,
        },


class TrialEvalCallback(EvalCallback):
    """Callback used for evaluating and reporting a trial."""

    def __init__(
        self,
        eval_env: make_evaluation_env,
        trial: optuna.Trial,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        deterministic: bool = True,
        verbose: int = 0,
    ):
        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose,
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super()._on_step()
            self.eval_idx += 1
            self.trial.report(self.last_mean_reward, self.eval_idx)
            # Prune trial if need.
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True


def objective(trial: optuna.Trial) -> float:
    kwargs = DEFAULT_HYPERPARAMS.copy()
    # Sample hyperparameters.
    kwargs.update(sample_a2c_params(trial))
    # Create the RL model.
    model = A2C(**kwargs)
    # Create env used for evaluation.
    eval_env = Monitor(gymnasium.make(ENV_ID))
    # Create the callback that will periodically evaluate and report the performance.
    eval_callback = TrialEvalCallback(
        eval_env, trial, n_eval_episodes=N_EVAL_EPISODES, eval_freq=EVAL_FREQ, deterministic=True
    )

    nan_encountered = False
    try:
        model.learn(N_TIMESTEPS, callback=eval_callback)
    except AssertionError as e:
        # Sometimes, random hyperparams can generate NaN.
        print(e)
        nan_encountered = True
    finally:
        # Free memory.
        model.env.close()
        eval_env.close()

    # Tell the optimizer that the trial failed.
    if nan_encountered:
        return float("nan")

    if eval_callback.is_pruned:
        raise optuna.exceptions.TrialPruned()

    return eval_callback.last_mean_reward



if __name__ == '__main__':
    env = SubprocVecEnv([make_env(
        dict(
            robots=["Sawyer"],                        # load a Sawyer robot and a Panda robot
            gripper_types="default",                # use default grippers per robot arm
            controller_configs=controller_config,   # each arm is controlled using OSC
            has_renderer=False,                     # no on-screen rendering
            has_offscreen_renderer=True,            # off-screen rendering needed for image obs
            control_freq=20,                        # 20 hz control for applied actions
            horizon=1024,                            # each episode terminates after 600 steps
            use_object_obs=False,                   # don't provide object observations to agent
            use_camera_obs=True,                   # provide image observations to agent
            camera_names="agentview",               # use "agentview" camera for observations
            camera_heights=84,                      # image height
            camera_widths=84,                       # image width
            reward_shaping=True),                    # use a dense reward signal for learning
        i,
        7183485 + i
        ) for i in range(6)])
    
    test_name = "Sawyer_freq20_hor1000_learn001_steps3M_controlJP"

    env = VecNormalize(env)
    model = PPO("MlpPolicy", env, verbose=1, gamma=0.99, learning_rate=1e-3, n_steps=1024, tensorboard_log="./" + test_name + "/tensorboard", )

    #env = VecNormalize.load('./' + test_name + '/env.pkl', env)
    #model = PPO.load("./" + test_name + "/model.zip", env=env)

    model.learn(total_timesteps=384_000, progress_bar=True)
    model.save("./" + test_name + "/model.zip")
    env.save('./' + test_name + '/env.pkl')

if __name__ == "__main__":
    # Set pytorch num threads to 1 for faster training.
    torch.set_num_threads(1)

    sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS)
    # Do not prune before 1/3 of the max budget is used.
    pruner = MedianPruner(n_startup_trials=N_STARTUP_TRIALS, n_warmup_steps=N_EVALUATIONS // 3)

    study = optuna.create_study(sampler=sampler, pruner=pruner, direction="maximize")
    try:
        study.optimize(objective, n_trials=N_TRIALS, timeout=600)
    except KeyboardInterrupt:
        pass

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    print("  User attrs:")
    for key, value in trial.user_attrs.items():
        print("    {}: {}".format(key, value))