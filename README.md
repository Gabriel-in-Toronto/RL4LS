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
