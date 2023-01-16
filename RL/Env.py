# Copyright (c) 2021, Programmable digital systems group, University of Toronto
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. 

import gym
from gym import spaces
import time
from datetime import datetime
from icecream import ic #for debugging

#Inferent from DRiLLS import
import os
import re
#import datetime
import numpy as np
from subprocess import check_output
from Feature_extractor import extract_features
from collections import defaultdict

#import the yaml file directly
import yaml

class ABCEnv(gym.Env):
    """
    Custom Environment that follows gym interface.
    This is the env created to run the logic synthesis tool abc environment
    """

    metadata = {'render.modes': ['console']}
        
    def __init__(self):
        super(ABCEnv, self).__init__()

        #Define the action space
        self.Action_space = ['rewrite','rewrite -z','refactor','refactor -z','resub','resub -z','balance','undo']
        self.action_space_length = len(self.Action_space)
        self.observation_space_size = 21     # number of features + action histogram
        #options
        self.store_mapped = True
        self.use_mappedfeature = False
        self.action_histogram_enable = True
        self.print_all = False
        self.LevelBestnotLUTBest = True


        if self.action_histogram_enable: 
            self.observation_space_size += self.action_space_length

        #import the yaml file
        data_file = 'params.yml'

        with open(data_file, 'r') as f:
             options = yaml.load(f, Loader=yaml.FullLoader)
        self.params = options

        self.iteration = -1
        self.episode = 0
        self.sequence = ['strash']
        self.fullsequence = ['strash']
        self.lut_6, self.levels = float('inf'), float('inf')
        self.Ep_best_lut_6 = (float('inf'), float('inf'))
        self.best_known_lut_6 = (float('inf'), float('inf'), -1, -1)
        self.best_known_sequence_lut_6 = ['strash']
        self.best_known_levels = (float('inf'), float('inf'), -1, -1)
        self.best_known_sequence_levels = ['strash']
        self.action_histogram = np.zeros(self.action_space_length)
        self.minR,self.maxR = float('inf'), 0

        #logging
        self.log = None
        self.log_enable = True

        #
        self.Exp_Name = self.setExpName(filepath='log')
        self.log_file_path = "log/" + self.Exp_Name #+ dt_string

        #Define file name
        self.output_design_file = self.log_file_path + '_getFeature.blif'
        self.output_design_file_mapped = self.log_file_path + '_getFeature_mapped.blif'

        #Due to the adjusted reward function, we also need to store the initial value of lut_count and level
        #self.initial_lut_6, self.initial_levels = self._run()
        self.base = 10
        Area_mapping = '-a'
        if self.LevelBestnotLUTBest:
            Area_mapping = ''

        #Run one abc iteration to get the initial lut_6 and levels
        abc_command = 'read ' + self.params['design_file'] + ';'
        abc_command += ';'.join(self.sequence) + '; '
        abc_command += 'if ' + str(Area_mapping)+' -K ' + str(self.params['fpga_mapping']['lut_inputs']) + '; '
        abc_command += 'print_stats;'
    
        try:
            proc = check_output([self.params['abc_binary'], '-c', abc_command])
            # get reward
            self.initial_lut_6, self.initial_levels = self._get_metrics(proc)
        except Exception as e:
            print(e)


        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions, we have 7 optimization commands in Action_space
        n_actions = self.action_space_length
        self.action_space = spaces.Discrete(n_actions)
        # The observation will be the coordinate of the agent
        # this can be described both by Discrete and Box space
        self.observation_space = spaces.Box(0.0,1.0,shape=(self.observation_space_size,),dtype = np.float32)
        # self.observation_space = spaces.Box(0.0,1000000,shape=(self.observation_space_size,),dtype = np.int32)
        if self.print_all:
            print("Finish initialization ")

    def reset(self):
        """
        Important: the observation must be a numpy array
        :return: (np.array) 
        """
        self.iteration = -1
        self.episode += 1
       
        #For easier testing
        self.update()

        self.lut_6, self.levels = float('inf'), float('inf')
        self.sequence = ['strash']
        self.fullsequence = ['strash']
        self.Ep_best_lut_6 = (float('inf'), float('inf'))
        self.minR,self.maxR = float('inf'), 0

        # logging
        if self.log_enable is True:
            csv_name = 'log'+ str(self.episode) + '.csv' 
            log_file = os.path.join(self.log_file_path, csv_name)
            if not os.path.exists(self.log_file_path):
                os.makedirs(self.log_file_path)
            if self.log:
                self.log.close()
            self.log = open(log_file, 'w')
            self.log.write('iteration, optimization, LUT-6, Levels, best LUT-6, best levels,obs,reward,act\n')

        state, _ = self._run()

        # logging
        self.log.write(', '.join([str(self.iteration),self.sequence[-1], str(int(self.lut_6)), str(int(self.levels))]) + '\n')
        self.log.flush()
        return np.array(state)

    def _run(self):
        """
        run ABC on the given design file with the sequence of commands
        """
        
        output_design_file = self.output_design_file
        output_design_file_mapped = self.output_design_file_mapped
        abc_command = 'read '
        if self.iteration <= 0:
            #Intialize the circuit and save as 'getFeature.blif'
            abc_command += self.params['design_file'] + ';strash;'
        else:
            abc_command += output_design_file + ';strash;'
        abc_command += ';'.join(self.sequence) + '; '
        
        Area_mapping = '-a'
        if self.LevelBestnotLUTBest:
            Area_mapping = ''

        #if the last action is undo, read origianl file and remove last 2 element from full command sequence
        if self.sequence[-1] == 'undo':
            abc_command = 'read ' + self.params['design_file'] + ';strash;' 
            abc_command += ';'.join(self.fullsequence) + '; '
        abc_command += 'write ' + output_design_file + '; '
        abc_command += 'if ' + str(Area_mapping)+' -K ' + str(self.params['fpga_mapping']['lut_inputs']) + '; '
        abc_command += 'print_stats;'
        if self.store_mapped is True :
            abc_command += 'write ' + output_design_file_mapped + '; '
    
        self.iteration += 1

        try:
            proc = check_output([self.params['abc_binary'], '-c', abc_command])
            if self.print_all:
                print(proc)
            # get reward
            lut_6, levels = self._get_metrics(proc)
            reward = self._get_reward(lut_6, levels) 
            self.lut_6, self.levels = lut_6, levels
            # get new state of the circuit
            state = self._get_state(output_design_file)
            if self.use_mappedfeature:
                state = self._get_state(output_design_file_mapped)
            return state, reward
        except Exception as e:
            print(e)
            return None, None

    def _get_state(self, design_file):
        abc_features = defaultdict(list)
        abc_features['Ori_Level'] = self.initial_levels
        abc_features['Ori_LUTCount'] = self.initial_lut_6
        abc_features['Current_Level'] = self.levels
        abc_features['Current_LUTCount'] = self.lut_6
        features = extract_features(design_file,abc_features,self.log_file_path,self.params['ccirc_binary'])
        if self.print_all:
            ic(features)
        norm_abc_histogram = self.action_histogram/self.params['iterations']
        return np.concatenate((features, norm_abc_histogram))

    def _get_metrics(self, stats):
        """
        parse LUT count and levels from the stats command of ABC
        """
        line = stats.decode("utf-8").split('\n')[-2].split(':')[-1].strip()
        ob = re.search(r'lev *= *[0-9]+', line)
        levels = int(ob.group().split('=')[1].strip())
        
        ob = re.search(r'nd *= *[0-9]+', line)
        lut_6 = int(ob.group().split('=')[1].strip())

        return lut_6, levels

    def _get_reward(self, lut_6, levels):
        """
        """
        if (self.initial_lut_6 - lut_6) <= 0:
            return 0
        # Calculate the area difference
        reward = self.initial_lut_6 - lut_6
        
        if self.Ep_best_lut_6[0] < self.initial_lut_6:
            reward = self.Ep_best_lut_6[0] - lut_6
        
        if self.LevelBestnotLUTBest:
            # Calculate the area difference
            reward = max(self.initial_levels - levels,0)
            
            if self.best_known_levels[1] < self.initial_levels:
                reward = self.best_known_levels[1] - levels
        
        # now calculate the reward
        return max(reward,0)

    def step(self, action):
        """
        accepts optimization index and returns (new state, reward, done, info)
        """

        self.sequence = ['strash']
        self.sequence.append(self.Action_space[action])
        self.fullsequence.append(self.Action_space[action])
        if self.Action_space[action] == 'undo':
            self.fullsequence = self.fullsequence[:-2]
        self.action_histogram[action] += 1
        new_state, reward = self._run()

        if self.print_all:
            print("@iteration:" + str(self.iteration) + "  || Predicted Action is " + str(self.Action_space[action]))
            print("Applied full sequence is [" + str(';'.join(self.fullsequence) + ';] '))

        # logging
        if self.log_enable is True:
            self.minR,self.maxR = min(self.minR,new_state[20]), max(self.maxR,new_state[20])
            if self.lut_6 <= self.Ep_best_lut_6[0]:
                self.Ep_best_lut_6 = (int(self.lut_6), int(self.levels))
            if self.lut_6 <= self.best_known_lut_6[0]:
                self.best_known_lut_6 = (int(self.lut_6), int(self.levels), self.episode, self.iteration)
                self.best_known_sequence_lut_6 = str(';'.join(self.fullsequence) + ';] ')
            if self.levels <= self.best_known_levels[1]:
                self.best_known_levels = (int(self.lut_6), int(self.levels), self.episode, self.iteration)
                self.best_known_sequence_levels = str(';'.join(self.fullsequence) + ';] ')
            self.log.write(', '.join([str(self.iteration), self.sequence[-1], str(int(self.lut_6)), str(int(self.levels))]) + ', ' +
                '; '.join(list(map(str, self.best_known_lut_6))) + ', ' + 
                '; '.join(list(map(str, self.best_known_levels))) + ', ' + '; '.join(list(map(str, new_state)))+ ', ' + str(reward)+ ', ' + str(action) + '\n')
            self.log.flush()

        # Optionally we can pass additional info, we are not using that for now
        info = {'episodes': self.episode,'iterations':self.iteration,'Best_LUT':self.best_known_lut_6,'LUT':self.lut_6,'Level':self.levels,'MinR':self.minR,'MaxR':self.maxR}
        
        #Done definition
        done = bool(self.iteration == (self.params['iterations']))
        if done: 
            #remove getfeature.blif used for current exploration
            os.remove(self.output_design_file)
            if self.store_mapped:
                os.remove(self.output_design_file_mapped)

        return np.array(new_state), reward, done, info

    def render(self, mode='console'):
        if mode != 'console':
           raise NotImplementedError()

    def close(self):
        if os.path.exists(self.output_design_file):
            os.remove(self.output_design_file)
        if os.path.exists(self.output_design_file_mapped):
            os.remove(self.output_design_file_mapped)
        statsfile = str(self.modeldatetime) + '_netlist.stats'
        if os.path.exists(statsfile):
            os.remove(statsfile)

    def update(self):
        print("Episode ", (self.episode -1) ,"Current Episode Best Area is: ",self.Ep_best_lut_6,"Best knwon Area is",self.best_known_lut_6)
        print("The corresponding opt command sequence is ",self.best_known_sequence_lut_6)
        print('MinR: ' + str(self.minR) + ' MaxR: ' + str(self.maxR))

    def setDesignFile(self, design_file):
        self.params['design_file'] = design_file
    
    def DisableLogging(self):
        self.log_enable = False

    def removeFeatureFile(self,filename):
        os.remove(filename)


    def setExpName(self,filepath):
        """
        Setting Experiment name under the log directory
        If Exp_1 exist, then name current experiment as Exp_2 , if Exp_1 exist, then name current experiment as Exp_3 etc..
        """
        Exp_Num = 1
        Exp_name = 'Exp_'+str(int(Exp_Num))
        while os.path.exists(os.path.join(filepath, Exp_name)):
            Exp_Num += 1
            Exp_name = 'Exp_'+str(int(Exp_Num))
        print("logging to log/" + Exp_name)
        return Exp_name
    
    def _env_init(self,Env_id = 0):
        """
        Initilize environment by input argument 
        Inputs:
        Env_id: initialize env by input arguments, if Env_id is not 0 by default, it will by default
        """
        if Env_id != 0:
            self.Exp_Name = self.Exp_Name + '_' +str(Env_id)
        self.log_file_path = "log/" + self.Exp_Name 

        #Define file name
        self.output_design_file = self.log_file_path + '_getFeature.blif'
        self.output_design_file_mapped = self.log_file_path + '_getFeature_mapped.blif'

