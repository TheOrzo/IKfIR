# Solving the Pick-and-Place Environment in Robosuite
<img src="https://robosuite.ai/docs/images/env_pick_place.png" align="middle" width="100%"/>

Welcome to the "Project Assignment: Solving the Pick-and-Place Environment in Robosuite" repository! This repository is intended to allow

## Table of Contents
 This repository holds the source code framework for training and evaluating the policy in the pick-and-place environments as well as a configuration file to set the different robosuite modules (robots, controllers, etc.) and tune hyperparameters
- [Project Description](#project-description)
	 - [Course Description](#course-description)
	 - [Task Description](#task-description)
- [Installation and Setup](#installation-and-setup)
- [Getting Started](#getting-started)
	- [Initial parameters](#initial-parameters)
	- [Train an Agent](#train-an-agent)
	- [Employ an Agent](#employ-an-agent)
	- [Insights and further testing](#insights-and-further-testing)
- [Hyperparameter Tuning with Optuna](#hyperparameter-tuning-with-optuna)
	- [Installing Optuna](#installing-optuna)
	- [Running Optuna](#running-optuna)
	- [Optuna Dashboard](#optuna-dashboard)
	- [Analysis](#analysis)
	- [Testing optimized parameters](#testing-optimized-parameters)
- [Conclusion](#conclusion)
- [Sources](#sources)
	- [Benchmarks and Implementations](#benchmarks-and-implementations)
	- [Papers](#papers)
- [Contributors](#contributors)

## Project Description
### Course description
**[Innovative Konzepte zur Programmierung von Industrierobotern](https://ipr.iar.kit.edu/lehrangebote_3804.php)** is an interactive course at the Karlsruhe Institute of Technology, supervised by Prof. Björn Hein, dealing with new methods of programming industrial robots. The topics covered in this lecture include collision-detection, collision-free path planning, path optimization and the emerging field of Reinforcement Learning. As the conclusion of the lecture, a final project related to one of these topics must be implemented by a team of two course participants.
### Task Description
Our team's task is to solve the **[Pick-and-Place Environment](https://robosuite.ai/docs/modules/environments.html#pick-and-place)** from Robosuite using Reinforcement Learning. In this simulated environment, a robot arm needs to place four objects from a bin into their designated container. At every initialization of the environment, the location of the objects are randomized and the task is considered successful is the robot arm manages to place every object into their corresponding container. 

#### Subtasks:
The task (for each object) can be subdivided into the following subtasks:

 1. Reaching: Move to nearest object
 2. Grasping: Pick up the object
 3. Lifting: Carry object to container
 4. Hovering: Drop object into corresponding container
 5. Repeat starting at 1. until all objects are placed in their corresponding containers

#### Reward function:
The reward function is essential to understanding the behaviour of the robot while interacting with the environment. In robosuite each environment has implemented two different kinds of reward functions. A binary reward rewards the robot only in the case if the object is placed in its corresponding container. We employed the dense reward function which uses reward shaping and rewards the robot for each subtask (like reaching & grasping), these rewards are then added successively. The image below taken from the [python code for the pick-and-place task](https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/environments/manipulation/pick_place.py#L260) describes the additional rewards for each subtask:

![](https://github.com/TheOrzo/IKfIR/blob/main/.assets/img/reward_function.png)

## Installation and Setup
Employing robosuite on windows is possible (e.g. by using a VM or WSL), but it leads to complications during installation, which is why using a linux or mac computer is highly recommended. Our repository also uses the RL algorithm implementations from the stable release of the stable baselines 3 repository.

Install all dependencies (robosuite, SB3,..) needed for this repository by running the following commmand.
More information about their installation can be found in the [robosuite installation guide](https://robosuite.ai/docs/installation.html) and the [SB3 installation guide](https://stable-baselines3.readthedocs.io/en/master/guide/install.html).

On debian the non free cuda driver has to be installed as a kernel level module in order to use the GPU for calculations. This change resulted in crashes of wayland DSP so a X11 has to be used as a fallback.
Our code is writen for python3.11. The following python packages are needed: numpy (below version 2), robosuite, stable-baselines3[extra], libhdf5, h5py
```
!python3  -m  pip  install  ipywidgets
!TMPDIR='/var/tmp'  python3  -m  pip  install  -r  requirements.txt
```

## Getting started
### Jupyter Notebook Documentation
The project documentation, code, insights as well as our results are documented in the file main.ipynb which this readme is referring to.

### Initial parameters
To get a feel of how different parameters of the model affect the model performance in a specific environment, we train the model subsequently with different parameters. The following script is a config file defining all parameters that can be adjusted for these subsequent runs.
```
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
    eval_episodes=5
    n_processes=6,
    n_eval_processes=4
    # Algorithm 
    algorithm="PPO",
    policy="MlpPolicy",
    gamma=0.99,
    learning_rate=1e-3,
    n_steps=2048,
)

test_name = str(parameters["robot"]) + "_freq" + str(parameters["control_freq"]) + "_hor" + str(parameters["horizon"]) + "_learn" + str(parameters["learning_rate"]) + "_episodes" + str(parameters["episodes"]) + "_control" + str(parameters["controller"])
```
Initial parameters seen in this dict are taken from multiple sources (Benchmarks, Implementations & Papers) referred to under [Sources](#Sources). By initial exploring, we discovered that changing the robot model, batch_size, as well as the learning rate have the greatest impact on the model performance.

### Train an Agent
In the notebook run the cell corresponding to the training of the model to train a model with the previously specified parameters. The model and tensorboard logs will be stored in the "tests" folder named according to the specified parameters.

```
#Set up TensorBoard logger
tensor_logger = "./" + test_name + "/tensorboard"
print("TensorBoard logging to", tensor_logger)

# Set controller configuration
controller_config = load_controller_config(default_controller=parameters["controller"])

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
		use_object_obs=False, # don't provide object observations to agent
		use_camera_obs=True, # provide image observations to agent
		camera_names="agentview", # use "agentview" camera for observations
		camera_heights=parameters["camera_size"], # image height
		camera_widths=parameters["camera_size"], # image width
		reward_shaping=True), # use a dense reward signal for learning
		i,
		parameters["seed"]
		) for i in range(parameters["n_processes"])], start_method='spawn') #remove	start_method='spawn' if you are not training on MPS

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
```
#### Tensorboard
The following command will open a locally hosted http server for the tensorboard. Navigate to [http://localhost:6006](http://localhost:6006/) to view the data logged during training.
```
!python  -m  tensorboard.main  --logdir=tensor_logger
```

### Employ an Agent
Running the cell corresponding to the employment of the trained model , the trained model defined by the specified parameters will be used for the task execution. If the trained agent exists, you can run it in the specified environment.

```
if not os.path.isdir('./' + test_name):
    print("No model found for this configuration. Train a model first!")
else:
    print('Using model ' + test_name)

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
            ) for i in range(parameters["n_eval_processes"])], start_method='spawn') #remove start_method='spawn' if you are not training on MPS
    
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
            action = get_policy_action(obs)            # use observation to decide on an action
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
```

### Insights and further testing

With these initial tests, we tested a variety of robot configurations and parameters, evaluating them based on visual critic and the total collected reward per episode.
We identified the Sawyer robot with its default gripper and the PPO algorithm as our most promising candidate. An additional insight is that changing the parameters responsible for steps taken until a policy update, the horizon and the control frequency of the robot influences the performance of the agent significantly. 

Further tests were conducted, but the lack of computing performance and the parameters being highly correlated with each other served as a strong bottleneck in solving this high-level task. It is easy to overlook configurations which would enable better performance when the parameters correlate with each other. In most cases, changing a parameter requires adapting the other parameters, otherwise the agent might even perform worse. The success of trying random combinations of parameters manually is very limited, since there is a very high number of possible parameter configurations.

todo some simulation examples with tensorboard graphs and everything:

## Hyperparameter Tuning with Optuna
### Installing Optuna
To bridge the gap of achieving a higher performance of the agent despite correlating parameters, a wider field of parameters needs to be evaluated.

The [Optuna hyperparameter optimization](https://optuna.org) framework make this task feasible by automating the hyperparameter search. By sampling for each run, called trial, a value for each parameter from a specified range and training a model with these parameters, the model performance can be evaluated based on the mean reward. Optuna then provides after a specified number of trials which hyperparameters lead to the best performance, have the highest influence on model performance and how they correlate to each other.

Install it by running the following command:

```
!pip install optuna
```

### Running Optuna
Running the optuna cell executes 100 optima trials. Parameter ranges for the PPO algorithm are taken from the [RL3 baselines zoo repository](https://github.com/DLR-RM/rl-baselines3-zoo/blob/726e2f1d3f1a6ea58ad4ae61c02a4ba71d241e4b/rl_zoo3/hyperparams_opt.py#L11C5-L11C22). To reduce the hyperparameter search space, i.e. limit the number of trials, we either kept certain parameters fixed or reduced their range based on gathered insights from previous tests.

```
import yaml
from torch import nn as nn
import optuna
from optuna.visualization import plot_optimization_history


# Set name of study
study_name = "study_sawyer_pickplace"
storage_name = "sqlite:///{}.db".format(study_name)

# Load configuration
with open("config_hyperparams.yaml") as stream:
    config = yaml.safe_load(stream)

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
    env = SubprocVecEnv([make_env("PickPlace", env_options, i, config["seed"]) for i in range(config["num_envs"])], start_method='spawn') #remove start_method='spawn' if you are not training on MPS
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
    eval_env = SubprocVecEnv([make_env("PickPlace", env_options, i, config["seed"]) for i in range(config["num_eval_envs"])], start_method='spawn') #remove start_method='spawn' if you are not training on MPS
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
```

We let optuna run for 40 trials. The logs are also uploaded to this repository. See the section [Analysis](#Analysis) for the analysis of the results.

### Optuna Dashboard
Optuna dashboard visualizes the logged results of the optuna execution.

The optuna dashboard can be accessed by executing the following command and opening up https://localhost:port in your browser.

```
!optuna-dashboard sqlite:///study_sawyer_pickplace.db
```

### Analysis
Following are the results of the optuna hyperparameter optimization:
#### Trial History:
<img src="https://github.com/TheOrzo/IKfIR/blob/main/.assets/img/trial_history.png" alt="Trial History" width="500">

#### Hyperparameter Importance
<img src="https://github.com/TheOrzo/IKfIR/blob/main/.assets/img/hyperparameter_importance.png" alt="Hyperparameter Importance" width="500">

#### Parallel Coordinate (Combinations of all trialed parameters):
<img src="https://github.com/TheOrzo/IKfIR/blob/main/.assets/img/parallel_coordinate.png" alt="Parallel Coordinate" width="500">

### Testing optimized parameters

## Conclusion
As reinforcement learning is still an emerging field, many questions such as which model to use, how much training is needed and how the reward function can be designed or evaluated for a certain task need to be answered. Therefore solving a high-level RL task, such as pick-and-place, requires a structured training and evaluation approach.  New implementations and papers dealing with this topic are published frequently which we made use of while working on this project.

Initial tests helped us to decide e.g. what robot to use, how many training steps are needed and what  horizon range, leads to the best results based on visual (simulation) and tensor board evaluation. Researching papers and implementations helped us reduce the dimension of the parameter space. Even with initial training and tests, certain hyperparameters we previously glossed over had a stronger influence on the model performance than assumed, such as ent_coeff and ...  which we gathered from using the Optuna hyperparameter optimization framework.

Evaluating our results we gathered that the subtask "reaching" is achievable, the robot moves towards objects and touches them (even in the single-object mode). The main problems are reliable grasping and lifting. It seems that the robot attempts to grasp objects, but has not yet understood how its gripper works, the gripper often gets stuck or "broken" during attempts. The objects often get moved around, where they even phase through the walls of the bin. As previously mentioned the bottleneck was the limited computational power in the short time frame of two weeks.

## Sources
### Benchmarks and Implementations:
- [robosuite Benchmark](https://robosuite.ai/docs/algorithms/benchmarking.html)
- [RL Baselines3 Zoo](https://github.com/DLR-RM/rl-baselines3-zoo)
### Papers
- [Cross-Embodiment Robot Manipulation Skill Transfer using Latent Space Alignment (Wang et al. 2024](https://arxiv.org/abs/2406.01968)
- [The Task Decomposition and Dedicated Reward-System-Based Reinforcement Learning Algorithm for Pick-and-Place (Kim et al. 2023)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10296071/pdf/biomimetics-08-00240.pdf)
- [Reinforcement Learning with Task Decomposition and Task-Specific Reward System for Automation of High-Level Tasks (Kwon et al. 2024)](https://www.mdpi.com/2313-7673/9/4/196)

## Contributors
The contributors of this project are: [@Enes1097](https://github.com/Enes1097) and [@TheOrzo](https://github.com/TheOrzo) 
