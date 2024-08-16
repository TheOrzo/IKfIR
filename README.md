# Solving the Pick-and-Place Environment in Robosuite
<img src="https://robosuite.ai/docs/images/env_pick_place.png" align="middle" width="100%"/>

Welcome to the "Project Assignment: Solving the Pick-and-Place Environment in Robosuite" repository! This repository is intended to allow for the replication of our project results and documents its progress including insights as well as tests.

## Table of Contents
 This repository holds the source code framework for training and evaluating the policy in the pick-and-place environments as well as a configuration file to set the different robosuite modules (robots, controllers, etc.) and tune hyperparameters
- [Project Description](#project-description)
	 - [Course Description](#course-description)
	 - [Task Description](#task-description)
- [Installation and Setup](#installation-and-setup)
	- [Installing robosuite and stable baselines 3](#installing-robosuite-and-stable-baselines-3)
	- [Installing our repository](#installing-our-repository)
- [Getting Started](#getting-started)
	- [Initial parameters](#initial-parameters)
	- [Train an Agent](#train-an-agent)
	- [Employ an Agent](#employ-an-agent)
	- [Insights and further testing](#insights-and-further-testing)
- [Hyperparameter Tuning with Optuna](#hyperparameter-tuning-with-optuna)

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

### Installing robosuite and stable baselines 3
Employing robosuite on windows is possible (e.g. by using a VM or WSL), but it leads to complications during installing, which is why using a linux or mac computer is highly recommended. Before being able to use our repository, you need to install robosuite following the [installation guide](https://robosuite.ai/docs/installation.html) from the robosuite documentation. We installed it from source:

```
$ git clone https://github.com/ARISE-Initiative/robosuite.git
$ cd robosuite
$ pip3 install -r requirements.txt
```
Our repository uses the stable release of the stable baselines 3 for RL algorithm implementations which you can install by following the [installation guide](https://stable-baselines3.readthedocs.io/en/master/guide/install.html):

```
pip  install  stable-baselines3[extra]
```

### Installing our repository
On debian the non free cuda driver has to be installed as a kernel level module in order to use the GPU for calculations. This change resulted in crashes of wayland DSP so a X11 has to be used as a fallback.

Our code is writen for python3.11. The following python packages are needed: numpy (below version 2), robosuite, stable-baselines3[extra], libhdf5, h5py
```
python3  -m  pip  install  ipywidgets
TMPDIR='/var/tmp'  python3  -m  pip  install  -r  requirements.txt
```

## Getting started
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
```
Initial parameters seen in this dict are taken from multiple sources (Benchmarks, Implementations & Papers) referred to under [Sources](#Sources). By initial exploring, we discovered that changing the robot model, batch_size, as well as the learning rate have the greatest impact on the model performance.

### Train an Agent
Run the following command to train a model with the previously specified parameters. The model and tensorboard logs will be stored in the "tests" folder named according to the specified parameters.
```
python train.py
```

#### Tensorboard
The following command will open a locally hosted http server for the tensorboard. Navigate to [http://localhost:6006](http://localhost:6006/) to view the data logged during training.
```
python  -m  tensorboard.main  --logdir=tensor_logger
```

### Employ an Agent
With the following command, the trained model defined by the specified parameters will be used for the task execution. If the trained agent exists, you can run it in the specified environment by:

```
python employ.py
```

### Insights and further testing

With these initial tests, we tested a variety of robot configurations and parameters, evaluating them based on visual critic and the total collected reward per episode.
We identified the Sawyer robot with its default gripper and the PPO algorithm as our most promising candidate. An additional insight is that changing the parameters responsible for steps taken until a policy update, the horizon and the control frequency of the robot influences the performance of the agent significantly. 

Further tests were conducted, but the lack of computing performance and the parameters being highly correlated with each other served as a strong bottleneck in solving this high-level task. It is easy to overlook configurations which would enable better performance when the parameters correlate with each other. In most cases, changing a parameter requires adapting the other parameters, otherwise the agent might even perform worse. The success of trying random combinations of parameters manually is very limited, since there is a very high number of possible parameter configurations.

todo some simulation examples with tensorboard graphs and everything:

## Hyperparameter Tuning with Optuna
To bridge the gap of achieving a higher performance of the agent despite correlating parameters, a wider field of parameters needs to be evaluated.

The [Optuna hyperparameter optimization](https://optuna.org) framework make this task feasible by automating the hyperparameter search. By sampling for each run, called trial, a value for each parameter from a specified range and training a model with these parameters, the model performance can be evaluated based on the mean reward. Optuna then provides after a specified number of trials which hyperparameters lead to the best performance, have the highest influence on model performance and how they correlate to each other.

Running the following command executes 200 optima trials. Parameter ranges for the PPO algorithm are taken from the [RL3 baselines zoo repository](https://github.com/DLR-RM/rl-baselines3-zoo/blob/726e2f1d3f1a6ea58ad4ae61c02a4ba71d241e4b/rl_zoo3/hyperparams_opt.py#L11C5-L11C22). To reduce the hyperparameter search space, i.e. limit the number of trials, we either kept certain parameters fixed or reduced their range based on gathered insights from previous tests. Try it out by running the following:

```
python hyperparameter_optimization.py
```

We let optuna run for 48 hours. The logs are also uploaded to this repository. See the next section for the access to the dashboard and our analysis of the results.

 
## Sources
### Benchmarks and Implementations:
- [robosuite Benchmark](https://robosuite.ai/docs/algorithms/benchmarking.html)
- [RL Baselines3 Zoo](https://github.com/DLR-RM/rl-baselines3-zoo)
- [Cross-Embodiment Robot Manipulation Skill Transfer using Latent Space Alignment (Wang et al. 2024](https://arxiv.org/abs/2406.01968)
- [The Task Decomposition and Dedicated Reward-System-Based Reinforcement Learning Algorithm for Pick-and-Place (Kim et al. 2023)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10296071/pdf/biomimetics-08-00240.pdf)
- [Reinforcement Learning with Task Decomposition and Task-Specific Reward System for Automation of High-Level Tasks (Kwon et al. 2024)](https://www.mdpi.com/2313-7673/9/4/196)

## Contributors
The contributors of this project are: [@Enes1097](https://github.com/Enes1097) and [@TheOrzo](https://github.com/TheOrzo) 

