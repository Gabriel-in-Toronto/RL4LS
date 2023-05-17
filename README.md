# RL4LS
ASPDAC23' Reinforcement learning for logic synthesis.

This is the source codes for our paper "Area-Driven FPGA Logic Synthesis Using Reinforcement Learning", published at 28th  Asia and South Pacific Design Automation Conference, Jan 2023.

The authors include Guanglei Zhou, Jason H.Anderson.

# Abstract
Logic synthesis involves a rich set of optimization algorithms applied in a specific sequence to a circuit netlist prior to technology mapping. A conventional approach is to apply a fixed “recipe” of such algorithms deemed to work well for a wide range of different circuits. We apply reinforcement learning (RL) to determine a unique recipe of algorithms for each circuit. Feature-importance analysis is conducted using a random-forest classifier to prune the set of features visible to the RL agent. We demonstrate conclusive learning by the RL agent and show significant FPGA area reductions
vs. the conventional approach (resyn2). In addition to circuit-by-circuit training and inference, we also train an RL agent on multiple circuits, and then apply the agent to optimize: 
1) the same set of circuits on which it was trained
2) an alternative set of “unseen” circuits. 

In both scenarios, we observe that the RL agent produces higher-quality implementations than the conventional approach. This shows that the RL agent is able to generalize, and perform beneficial logic synthesis optimizations across a variety of circuits.

# Pre-requisite
[abc](https://github.com/berkeley-abc/abc),[Stable-baseline3](https://github.com/DLR-RM/stable-baselines3),[Ccirc](https://www.eecg.toronto.edu/~jayar/software/Cgen/Cgen.html)

# Installation
1. Recursively clone this repo and its submodules: git clone --recurse-submodules git@github.com:Gabriel-in-Toronto/RL4LS.git
2. Go to submodules/abc and install the logic synthesis tool [abc](https://github.com/berkeley-abc/abc)
3. Please email me (guanglei.zhou@mail.utoronto.ca) to obtain our ccirc implementation. We modified and implemented the reconvergence feature in the ccirc release. After you obtained the copy, type "make" and build the binary. 
4. Install the Stable Baselines3 package and you are ready to go!
```
pip install stable-baselines3[extra]
```

# Usage
Our model can either be used as circuit-by-circuit model where you tained a specific circuit for multiple sample points and gather the best result. Alternatively, the model can be pre-trained and still performs better than a commonly used heuristic script "resyn2". Please refer to our paper for more details. 

# Citation
```
@INPROCEEDINGS{10044776,
  author={Zhou, Guanglei and Anderson, Jason H.},
  booktitle={2023 28th Asia and South Pacific Design Automation Conference (ASP-DAC)}, 
  title={Area-Driven FPGA Logic Synthesis Using Reinforcement Learning}, 
  year={2023},
  volume={},
  number={},
  pages={159-165},
  doi={}}
  ```
