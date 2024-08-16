import numpy as np
import torch
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
        camera_size=84,
        episodes=200,
        #eval_episodes=5,
        n_processes=8,
        #n_eval_processes=4,
        # Algorithm 
        algorithm="PPO",
        policy="MlpPolicy",
        gamma=0.99,
        learning_rate=1e-3,
        n_steps=2048,
    )

    test_name = str(parameters["robot"]) + "_freq" + str(parameters["control_freq"]) + "_hor" + str(parameters["horizon"]) + "_learn" + str(parameters["learning_rate"]) + "_episodes" + str(parameters["episodes"]) + "_control" + str(parameters["controller"])

    #Set up TensorBoard logger
    tensor_logger = "./" + test_name + "/tensorboard"
    print("TensorBoard logging to", tensor_logger)

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
            has_renderer=False,                     
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
            ) for i in range(parameters["n_processes"])], start_method='spawn')
            
    env = VecNormalize(env)

    # Initialize model for training:
    if parameters["algorithm"] == "PPO":
        model = PPO("MlpPolicy", env, verbose=1, gamma=parameters["gamma"], learning_rate=parameters["learning_rate"], n_steps=parameters["n_steps"], tensorboard_log=tensor_logger, device=device)
    elif parameters["algorithm"] == "DDPG":
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
        model = DDPG(parameters["policy"], env, action_noise=action_noise, verbose=1, batch_size=parameters["batch_size"], tensorboard_log=tensor_logger, device=device)
    elif parameters["algorithm"] == "SAC":
        model = SAC(parameters["policy"], env, verbose=1, batch_size=parameters["batch_size"], train_freq=(2500, "step"), learning_rate=0.001, gradient_steps=1000, learning_starts=3300, tensorboard_log=tensor_logger, device=device)	
    else:
        raise ValueError("Invalid algorithm specified in the configuration.")
        
    '''
    # Load existing model to train
    # Comment out the above model initialization and uncomment the following code to load an existing model
    env.load("./" + test_name + '/env.pkl', env)
    if parameters["algorithm"] == "PPO":
        model = PPO.load("./" + test_name + "/model.zip", env=env, tensorboard_log=tensor_logger, device=device)
    elif config["algorithm"] == "DDPG":
        model = DDPG.load("./" + test_name + "/model.zip", env=env,tensorboard_log=tensor_logger, device=device)
    elif config["algorithm"] == "SAC":
        model = SAC.load("./" + test_name + "/model.zip", env=env, tensorboard_log=tensor_logger, device=device)
    else:
        raise ValueError("Invalid algorithm specified in the configuration.")
    '''

    # Train the model and save it
    model.learn(total_timesteps=parameters["horizon"]*parameters["episodes"], progress_bar=True)
    model.save("./" + test_name + "/model.zip")
    env.save('./' + test_name + '/env.pkl')

    env.close()

if __name__ == "__main__":
	main()