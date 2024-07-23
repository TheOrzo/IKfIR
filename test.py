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

controller_config = load_controller_config(default_controller="OSC_POSE")

def make_env(env_id, options, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param options: (dict) additional arguments to pass to the specific environment class initializer
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
            i,
            3815046
            ) for i in range(1)])


    test_name = "Sawyer_freq20_hor1000_learn001_steps768K_controlOSC"

    #env.render_mode = 'mujoco'
    env = VecNormalize.load('./' + test_name + '/env.pkl', env)
    model = PPO.load("./" + test_name + "/model.zip", env=env)

    def get_policy_action(obs):
        # a trained policy could be used here, but we choose a random action
        #low, high = env.action_spec
        #return np.random.uniform(low, high)
        action, _states = model.predict(obs)
        return action

    # reset the environment to prepare for a rollout
    env.training = False
    env.norm_reward = False
    obs = env.reset()

    runs = 3
    for i in range(runs * 1024):
        action = get_policy_action(obs)         # use observation to decide on an action
        obs, reward, done, info = env.step(action) # play action
        env.render()