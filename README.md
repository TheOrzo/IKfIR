# Innovative Konzepte zur Programmierung von Industrierobotern
-------
## Project Assignment: Solving the Pick-and-Place Environment in Robosuite
-------
**Innovative Konzepte zur Programmierung von Industrierobotern**[IKzPvI]([http://mujoco.org/](https://ipr.iar.kit.edu/lehrangebote_3804.php)) is a interactive course at the Karlsruhe Insitute of Technology supervised by Prof. Björn Hein dealing with new ways of programming industrial robots. The topics covered in this lecture include collsion-detection, collision-free path planning, path optimization and a fairly new advancement in robot programmin, Reinforcement Learning. The final task of the course is a sub-topic of one of the covered topics and has to be implemented in a jupyter notebook by a team of two course participants.

Data-driven algorithms, such as reinforcement learning and imitation learning, provide a powerful and generic tool in robotics. These learning paradigms, fueled by new advances in deep learning, have achieved some exciting successes in a variety of robot control problems. However, the challenges of reproducibility and the limited accessibility of robot hardware (especially during a pandemic) have impaired research progress. The overarching goal of **robosuite** is to provide researchers with:

* a standardized set of benchmarking tasks for rigorous evaluation and algorithm development;
* a modular design that offers great flexibility to design new robot simulation environments;
* a high-quality implementation of robot controllers and off-the-shelf learning algorithms to lower the barriers to entry.

This framework was originally developed since late 2017 by researchers in [Stanford Vision and Learning Lab](http://svl.stanford.edu) (SVL) as an internal tool for robot learning research. Now it is actively maintained and used for robotics research projects in SVL and the [UT Robot Perception and Learning Lab](http://rpl.cs.utexas.edu) (RPL). We welcome community contributions to this project. For details please check out our [contributing guidelines](CONTRIBUTING.md).

This release of **robosuite** contains seven robot models, eight gripper models, six controller modes, and nine standardized tasks. It also offers a modular design of APIs for building new environments with procedural generation. We highlight these primary features below:

* **standardized tasks**: a set of standardized manipulation tasks of large diversity and varying complexity and RL benchmarking results for reproducible research;
* **procedural generation**: modular APIs for programmatically creating new environments and new tasks as combinations of robot models, arenas, and parameterized 3D objects;
* **robot controllers**: a selection of controller types to command the robots, such as joint-space velocity control, inverse kinematics control, operational space control, and 3D motion devices for teleoperation;
* **multi-modal sensors**: heterogeneous types of sensory signals, including low-level physical states, RGB cameras, depth maps, and proprioception;
* **human demonstrations**: utilities for collecting human demonstrations, replaying demonstration datasets, and leveraging demonstration data for learning. Check out our sister project [robomimic](https://arise-initiative.github.io/robomimic-web/);
* **photorealistic rendering**: integration with advanced graphics tools that provide real-time photorealistic renderings of simulated scenes.

## Citation
Please cite [**robosuite**](https://robosuite.ai) if you use this framework in your publications:
```bibtex
@inproceedings{robosuite2020,
  title={robosuite: A Modular Simulation Framework and Benchmark for Robot Learning},
  author={Yuke Zhu and Josiah Wong and Ajay Mandlekar and Roberto Mart\'{i}n-Mart\'{i}n and Abhishek Joshi and Soroush Nasiriany and Yifeng Zhu},
  booktitle={arXiv preprint arXiv:2009.12293},
  year={2020}
}
```
