# Copyright (c) 2021, Programmable digital systems group, University of Toronto
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. 
from Env import ABCEnv
import gym 
import numpy as np
import torch
torch.use_deterministic_algorithms(True)

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv
from subprocess import check_output
import os

#set the environment seed
Experiment_Seed = 256
EXP_LR = 0.0008
num_cpu = 6  # Number of processes to use
def make_env(rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = ABCEnv()
        env.seed(seed + rank)
        env._env_init(rank)
        return env
    return _init

def ReplaceYAML(file_path,replacement_line,line_number):
    with open(file_path, 'r+') as file:
        data = file.readlines()
        file.close()
    with open(file_path, 'w') as file:
        data[int(line_number)] = replacement_line
        for line in data:
            file.write(line)        
        file.close()

if __name__ == '__main__':
    # Create the vectorized environment
    env = SubprocVecEnv([make_env(i,Experiment_Seed) for i in range(num_cpu)])

    # Train the agent
    total_timesteps = 5000 + 400
    #Pick RL algorithms A2C, PPO etc.
    model = A2C('MlpPolicy', env, verbose=1, tensorboard_log="./tensorboard/",learning_rate=EXP_LR)
    model.learn(total_timesteps)
    
    model_name = "A2C_timesteps_" + str(total_timesteps)+"_Seed"+str(Experiment_Seed)
    model.save(model_name)
    
    model.load(model_name, env=env)
    
    #evaluate policy
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes = 5)
    print("Evaluated Policy Performance for circuit performance is")
    print("mean_reward is: ", mean_reward)
    print("std_reward is: ", std_reward)

