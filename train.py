import numpy as np
import os
import robosuite as suite


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

controller_config = load_controller_config(default_controller="JOINT_POSITION")

def make_env(env_id, options, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param options: (dict) additionala arguments to pass to the specific environment class initializer
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = GymWrapper(suite.make(env_id, **options))
        env.render_mode = 'mujoco'
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init


if __name__ == '__main__':
    env = SubprocVecEnv([make_env(
        "PickPlace",
        dict(
            robots=["Sawyer"],                        # load a Sawyer robot and a Panda robot
            gripper_types="default",                # use default grippers per robot arm
            controller_configs=controller_config,   # each arm is controlled using OSC
            has_renderer=False,                     # no on-screen rendering
            has_offscreen_renderer=True,            # off-screen rendering needed for image obs
            control_freq=20,                        # 20 hz control for applied actions
            horizon=1000,                            # each episode terminates after 600 steps
            use_object_obs=False,                   # don't provide object observations to agent
            use_camera_obs=True,                   # provide image observations to agent
            camera_names="agentview",               # use "agentview" camera for observations
            camera_heights=84,                      # image height
            camera_widths=84,                       # image width
            reward_shaping=True),                    # use a dense reward signal for learning
        i,
        1837812
        ) for i in range(6)])
    
    test_name = "Sawyer_freq20_hor1000_learn001_steps768K_controlJP"

    env = VecNormalize(env)
    model = PPO("MlpPolicy", env, verbose=1, gamma=0.99, learning_rate=1e-3, n_steps=1000, tensorboard_log="./" + test_name + "/tensorboard", )

    #env = VecNormalize.load('./' + test_name + '/env.pkl', env)
    #model = PPO.load("./" + test_name + "/model.zip", env=env)

    model.learn(total_timesteps=768_000, progress_bar=True)
    model.save("./" + test_name + "/model.zip")
    env.save('./' + test_name + '/env.pkl')
 