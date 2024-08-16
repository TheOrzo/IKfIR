import numpy as np
import torch
import os
import robosuite as suite

from robosuite import load_controller_config
from robosuite.controllers import load_controller_config
from robosuite.wrappers import GymWrapper

from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3 import PPO, DDPG, SAC

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

def main():
    parameters = dict(
        # Environment
        robot="Panda",
        gripper="default",
        controller="OSC_POSE",
        seed=1297111683,
        control_freq=20,
        horizon=2048,
        camera_size=128,
        episodes=200,
        eval_episodes=5,
        #n_processes=8,
        n_eval_processes=4,
        # Algorithm 
        algorithm="PPO",
        policy="MlpPolicy",
        gamma=0.99,
        learning_rate=1e-3,
        n_steps=2048,
    )

    test_name = str(parameters["robot"]) + "_freq" + str(parameters["control_freq"]) + "_hor" + str(parameters["horizon"]) + "_learn" + str(parameters["learning_rate"]) + "_episodes" + str(parameters["episodes"]) + "_control" + str(parameters["controller"])

    if not os.path.isdir('./' + test_name):
        print("No model found for this configuration. Train a model first!")
    else:
        print('Using model ' + test_name)

        # Set controller configuration
        controller_config = load_controller_config(default_controller=parameters["controller"])

        # Define the environment setup
        # Make robosuite environment into a gym environment as stable baselines only supports gym environments
        def make_env(env_id, options, rank, seed):
            def _init():
                env = GymWrapper(suite.make(env_id, **options))
                env.render_mode = 'mujoco'
                env = Monitor(env)
                env.reset(seed=seed + rank)
                return env
            set_random_seed(seed)
            return _init

        # Setup environment
        # Define environment parameters for specific environment "PickPlace"
        env = SubprocVecEnv([make_env(
            "PickPlace",
            dict(
                robots=[parameters["robot"]],                      
                gripper_types=parameters["gripper"],                
                controller_configs=controller_config,   
                has_renderer=True,
                has_offscreen_renderer=True,
                control_freq=parameters["control_freq"],
                horizon=parameters["horizon"],
                use_object_obs=False,                       # don't provide object observations to agent
                use_camera_obs=True,                        # provide image observations to agent
                camera_names="agentview",                   # use "agentview" camera for observations
                camera_heights=parameters["camera_size"],   # image height
                camera_widths=parameters["camera_size"],    # image width
                reward_shaping=True),                       # use a dense reward signal for learning
                i,
                parameters["seed"]
                ) for i in range(parameters["n_eval_processes"])], start_method='spawn')
        
        env = VecNormalize(env)
        env.load("./" + test_name + '/env.pkl', env)

        if parameters["algorithm"] == "PPO":
            model = PPO.load("./" + test_name + "/model.zip", env=env, device=device)
        elif parameters["algorithm"] == "DDPG":
            model = DDPG.load("./" + test_name + "/model.zip", env=env, device=device)
        elif parameters["algorithm"] == "SAC":
            model = SAC.load("./" + test_name + "/model.zip", env=env, device=device)
        else:
            raise ValueError("Invalid algorithm specified in the configuration.")
        
        def get_policy_action(obs):
            action, _states = model.predict(obs, deterministic=True)
            return action

        # reset the environment to prepare for a rollout
        env.training = False
        env.norm_reward = False
        episode_rewards = []
        eval_episodes = parameters["eval_episodes"]
        for i_episode in range(eval_episodes):
            obs = env.reset()
            total_reward = 0
            for t in range(parameters["horizon"]):
                env.render()
                action, _states = get_policy_action(obs)   # use observation to decide on an action
                obs, reward, done, info = env.step(action) # play action
                total_reward += reward
                if done.all():
                    print("Episode finished after {} timesteps".format(t+1))
                    break
            episode_rewards.append(total_reward)
        average_reward_per_environment = sum(episode_rewards) / len(episode_rewards)
        average_reward = np.mean(average_reward_per_environment)
        print(f"Iteration {i_episode+1}/{eval_episodes}, Average Reward per Environment: {average_reward_per_environment}, Average Reward: {average_reward}")
        
        # Close environment
        env.close()
if __name__ == "__main__":
     main()