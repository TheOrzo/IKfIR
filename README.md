# Innovative Konzepte zur Programmierung von Industrierobotern
<img src="https://robosuite.ai/docs/images/env_pick_place.png" align="middle" width="100%"/>

## Project Assignment: Solving the Pick-and-Place Environment in Robosuite
Welcome to the "Solving the Pick-and-Place Environment in Robosuite" repository! This repo is intended for ease of replication of our project results, as well as documenting the progress of our project.

**[Innovative Konzepte zur Programmierung von Industrierobotern](https://ipr.iar.kit.edu/lehrangebote_3804.php)** is a interactive course at the Karlsruhe Insitute of Technology supervised by Prof. Björn Hein dealing with new ways of programming industrial robots. The topics covered in this lecture include collsion-detection, collision-free path planning, path optimization and a fairly new advancement in robot programmin, Reinforcement Learning. The final task of the course is a sub-topic of one of the covered topics and has to be implemented in a jupyter notebook by a team of two course participants.

This repository holds the following contents:

* source code framework for training and evaluating the policy in the pick-and-place environments;
* a configuration file to set the different robosuite modules (robots, controllers, etc.) and tune hyperparameters;

## Installation

### Minimal installation

From source:
```
pip install -e .
```

### Full installation (with extra envs and test dependencies)

```
apt-get install swig cmake ffmpeg
pip install -r requirements.txt
pip install -e .[plots,tests]
```

Please see [Stable Baselines3 documentation](https://stable-baselines3.readthedocs.io/en/master/) for alternatives to install stable baselines3.

## Train an Agent

The hyperparameters for each environment are defined in `hyperparameters/algo_name.yml`.

If the environment exists in this file, then you can train an agent using:
```
python train.py --algo algo_name --env env_id
```

Evaluate the agent every 10000 steps using 10 episodes for evaluation (using only one evaluation env):
```
python train.py --algo sac --env HalfCheetahBulletEnv-v0 --eval-freq 10000 --eval-episodes 10 --n-eval-envs 1
```

## Enjoy a Trained Agent

**Note: to download the repo with the trained agents, you must use `git clone --recursive https://github.com/DLR-RM/rl-baselines3-zoo`** in order to clone the submodule too.


If the trained agent exists, then you can see it in action using:
```
python enjoy.py --algo algo_name --env env_id
```

For example, enjoy A2C on Breakout during 5000 timesteps:
```
python enjoy.py --algo a2c --env BreakoutNoFrameskip-v4 --folder rl-trained-agents/ -n 5000
```

## Hyperparameters Tuning

Please see the [dedicated section](https://rl-baselines3-zoo.readthedocs.io/en/master/guide/tuning.html) of the documentation.

## Contributors

The contributors of this project are: [@TheOrzo](https://github.com/TheOrzo) and [@Enes1097](https://github.com/Enes1097)
