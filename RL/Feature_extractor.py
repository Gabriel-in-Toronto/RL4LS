# Copyright (c) 2021, Programmable digital systems group, University of Toronto
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. 


import re
import numpy as np
import datetime
from multiprocessing import Process, Manager
from subprocess import check_output
from collections import defaultdict
from icecream import ic #for debugging


#design_file = "../DRiLLS/benchmarks/arithmetic/multiplier.blif"
#ccirc_binary = "Cgen/ccirc/ccirc"
# yosys_binary = 
# abc_binary = 


def shape_classifier(datas):
    
    gradients=np.diff(datas)
    maxima_num=0
    max_locations=[]
    count=0
    Threshold = 2* np.mean(datas)
    #print("Threshold is " , Threshold)
    type_choice = ['Decreasing','Cornical', 'Others']
    for i in gradients[:-1]:
        count+=1

        if ((i> Threshold/2 ) & (gradients[count]< Threshold/2 ) & (i != gradients[count]) ):
            maxima_num+=1
            max_locations.append(count)
    turning_points = {'maxima_number':maxima_num,'maxima_locations':max_locations}
    # Deciding the types
    if maxima_num == 0: 
        shape_type = 0#type_choice[0]
    elif maxima_num <= 5:
        shape_type = 1#type_choice[1]
    else:
        shape_type = 2#type_choice[2]

    return shape_type


def ccirc_stats(design_file, ccirc_binary, stats,Exp_name = "Exp0"):
    
    stats_file = str(Exp_name)+'_netlist.stats'
    #ic(stats_file)
    #ic(ccirc_binary)
    #ic(design_file)
    
    try:
        proc = check_output([ccirc_binary, design_file, "--partitions" ," 1","--out",stats_file])
        #ic(proc)

        readfile = stats_file
        with open(readfile) as f:
            lines = f.readlines()   
        
        for line in lines:
            #Basic characteristic of Circuit (Feature count: 5)
            if 'Number_of_Nodes' in line:
                stats['Number_of_Nodes'] = int(line.strip().split()[-1])
            if 'Number_of_Edges' in line:
                stats['Number_of_Edges'] = int(line.strip().split()[-1])
            if 'Maximum_Delay' in line:
                stats['Maximum_Delay'] = float(line.strip().split()[-1])
            if 'Number_of_Combinational_Nodes' in line:
                stats['Number_of_Combinational_Nodes'] = float(line.strip().split()[-1])
            if ('Number_of_DFF' in line) and ('Number_of_DFF' not in stats):
                #Not useful in combination circuits, will always be zero
                stats['Number_of_DFF'] = float(line.strip().split()[-1])
            
            #Characteristic of Delay structure (Feature count: 7 * ？)
            if 'Node_shape' in line:
                stats['Node_shape'] = shape_classifier(np.array(line.strip().split()[2:-1],dtype=float))
            if 'Input_shape:' in line:
                stats['Input_shape:'] = shape_classifier(np.array(line.strip().split()[2:-1],dtype=float))
            if 'Output_shape' in line:
                stats['Output_shape'] = shape_classifier(np.array(line.strip().split()[2:-1],dtype=float))
            if 'Latched_shape' in line:
                #Not useful in combination circuits, will always be zero
                stats['Latched_shape'] = shape_classifier(np.array(line.strip().split()[2:-1],dtype=float))
            if 'POshape' in line:
                stats['POshape'] = shape_classifier(np.array(line.strip().split()[2:-1],dtype=float))
            if 'Edge_length_distribution' in line:
                # The same as Intra_cluster_edge_length_distribution for cluster count = 1
                stats['Edge_length_distribution'] = shape_classifier(np.array(line.strip().split()[2:-1],dtype=float))
            # if 'Fanout_distribution' in line:
            #     #Should use a different way to analyze the graph -- to be implemented
            #     stats['Fanout_distribution'] = np.array(line.strip().split()[2:-1])
            
            # #Characteristic of connections, (Feature count: 1 * ?)
            # same as Number of Edges 
            # if 'Intra_cluster_edge_length_distribution' in line:
            #     stats['Intra_cluster_edge_length_distribution'] = np.array(line.strip().split()[2:-1])

            #Characteristic of Fanout from Nodes (Feature Count: 14)
            if 'Maximum_fanout' in line:
                stats['Maximum_fanout'] = float(line.strip().split()[-1])
            if 'Number_of_high_degree_comb' in line:
                stats['Number_of_high_degree_comb'] = float(line.strip().split()[-1])
            if 'Number_of_high_degree_pi' in line:
                stats['Number_of_high_degree_pi'] = float(line.strip().split()[-1])
            if 'Number_of_high_degree_dff' in line:
                stats['Number_of_high_degree_dff'] = float(line.strip().split()[-1])
            if 'Number_of_10plus_degree_comb' in line:
                stats['Number_of_10plus_degree_comb'] = float(line.strip().split()[-1])
            if 'Number_of_10plus_degree_pi' in line:
                stats['Number_of_10plus_degree_pi'] = float(line.strip().split()[-1])
            if 'Avg_fanin' in line:
                stats['Avg_fanin' ] = float(line.strip().split()[-2])
                s = line.strip().split()[-1]
                stats['Std_fanin' ] = float(s[s.find("(")+1:s.find(")")])
            if 'Avg_fanout' in line:
                stats['Avg_fanout' ] = float(line.strip().split()[-2])
                s = line.strip().split()[-1]
                stats['Std_fanout' ] = float(s[s.find("(")+1:s.find(")")])
            if 'Avg_fanout_comb' in line:
                stats['Avg_fanout_comb' ] = float(line.strip().split()[-2])
                s = line.strip().split()[-1]
                stats['Std_fanout_comb' ] = float(s[s.find("(")+1:s.find(")")])
            if 'Avg_fanout_pi' in line:
                stats['Avg_fanout_pi' ] = float(line.strip().split()[-2])
                s = line.strip().split()[-1]
                stats['Std_fanout_pi' ] = float(s[s.find("(")+1:s.find(")")])
            if 'Avg_fanout_dff' in line:
                stats['Avg_fanout_dff' ] = float(line.strip().split()[-2])
                s = line.strip().split()[-1]
                stats['Std_fanout_dff' ] = float(s[s.find("(")+1:s.find(")")])
            if 'Reconvergence' in line and ('Reconvergence' not in stats):
                stats['Reconvergence' ] = float(line.strip().split()[-1])
            if 'Reconvergence_max' in line:
                stats['Reconvergence_max'] = float(line.strip().split()[-1])
            if 'Reconvergence_min' in line:
                stats['Reconvergence_min' ] = float(line.strip().split()[-1])
                        
            
    except Exception as e:
        print(e)
        return None
    return stats

def abc_stats(design_file, abc_binary, stats):    
    abc_command = "read_verilog " + design_file + "; print_stats"
    try:
        proc = check_output([abc_binary, '-c', abc_command])
        lines = proc.decode("utf-8").split('\n')
        for line in lines:
            if 'i/o' in line:
                ob = re.search(r'i/o *= *[0-9]+ */ *[0-9]+', line)
                stats['input_pins'] = int(ob.group().split('=')[1].strip().split('/')[0].strip())
                stats['output_pins'] = int(ob.group().split('=')[1].strip().split('/')[1].strip())
        
                ob = re.search(r'edge *= *[0-9]+', line)
                stats['edges'] = int(ob.group().split('=')[1].strip())

                ob = re.search(r'lev *= *[0-9]+', line)
                stats['levels'] = int(ob.group().split('=')[1].strip())

                ob = re.search(r'lat *= *[0-9]+', line)
                stats['latches'] = int(ob.group().split('=')[1].strip())
    except Exception as e:
        print(e)
        return None
    
    return stats

def extract_features(design_file,abc_features,Exp_name = "Exp0" ,ccirc_binary = "Cgen/ccirc/ccirc"):
    '''
    Returns features of a given circuit as a tuple.
    Features are listed below
    '''
    stats = defaultdict(list)    
    stats = ccirc_stats(design_file, ccirc_binary, stats,Exp_name)

    # normalized features -- Total Num: 21
    features = defaultdict(float)    
    # (Group 1 - Num: 4) - Basic characteristics
    norm_constant = stats['Number_of_Nodes'] + stats['Number_of_Edges'] 
    features['node_percentage'] = stats['Number_of_Nodes'] / norm_constant
    features['edge_percentage'] = stats['Number_of_Edges']  / norm_constant
    features['DFF_percentage'] = stats['Number_of_DFF'] / stats['Number_of_Nodes']
    features['Combinational_Nodes_percentage'] = stats['Number_of_Combinational_Nodes'] / stats['Number_of_Nodes']

    # (Group 2 - Num: 2) - ABC Area / Level
    # as area / level may goes up during optimization , we doubled the original LUTCount/Levels as norm constant
    features['Norm_LUTCount'] = abc_features['Current_Level'] / (2*abc_features['Ori_Level'])
    features['Norm_Levels'] = abc_features['Current_LUTCount'] / (2*abc_features['Ori_LUTCount'])   

    # (Group 3 - Num: 6) - Characteristic of Delay structure
    norm_delay_structure = 2 # as we only have [0 , 1 , 2 ] three type of circuits
    features['Norm_Node_shape'] = stats['Node_shape'] / norm_delay_structure
    features['Norm_Input_shape'] = stats['Input_shape:'] / norm_delay_structure
    features['Norm_Output_shape'] = stats['Output_shape'] / norm_delay_structure
    features['Norm_Latched_shape'] = stats['Latched_shape'] / norm_delay_structure
    features['Norm_POshape'] = stats['POshape'] / norm_delay_structure
    features['Norm_Edge_length_distribution'] = stats['Edge_length_distribution'] / norm_delay_structure

    # (Group 4 - Num: 8) -Characteristic of Fanout from Nodes (Feature Count: 8 + 2(optional) + 5？)
    # Try Avg_fanout with clipping
    features['Norm_Avg_fanout'] = stats['Avg_fanout' ]/stats['Maximum_fanout']
    features['Norm_Std_fanout'] = stats['Std_fanout' ]/stats['Maximum_fanout']
    features['Norm_Avg_fanout_comb'] = stats['Avg_fanout_comb' ] /stats['Maximum_fanout']
    features['Norm_Std_fanout_comb'] = stats['Std_fanout_comb' ]/stats['Maximum_fanout']
    features['Norm_Avg_fanout_pi'] = stats['Avg_fanout_pi' ] /stats['Maximum_fanout']
    features['Norm_Std_fanout_pi'] = stats['Std_fanout_pi' ]/stats['Maximum_fanout']
    features['Norm_Avg_fanout_dff'] = stats['Avg_fanout_dff' ] /stats['Maximum_fanout']
    features['Norm_Std_fanout_dff'] = stats['Std_fanout_dff' ]/stats['Maximum_fanout']
    #Enabled only for mapped circuit, normalization constant = 6 as we are mapped to LUT 6 FPGA
    # features['Norm_Avg_fanin'] = stats['Avg_fanin' ] / 6 
    # features['Norm_Avg_fanin'] = stats['Std_fanin' ] / 6

    # (Group 5 - Num: 1) Reconvergence value
    # R is guranteedto be bounded by [0,1] for LUT 2 , direct input
    features['Norm_R'] = stats['Reconvergence']

    #-----------To be decided-------------------
    # stats['Number_of_high_degree_comb'] 
    # stats['Number_of_high_degree_pi'] 
    # stats['Number_of_high_degree_dff'] 
    # stats['Number_of_10plus_degree_comb'] 
    # stats['Number_of_10plus_degree_pi'] 

    # From Yosys - gate types percentages --- maybe only for standard cell
    # features['percentage_of_ands'] = stats['ands'] / stats['number_of_cells']
    # features['percentage_of_ors'] = stats['ors'] / stats['number_of_cells']
    # features['percentage_of_nots'] = stats['nots'] / stats['number_of_cells']    

    return np.array([features['node_percentage'],features['edge_percentage'],features['DFF_percentage'],features['Combinational_Nodes_percentage']\
        ,features['Norm_LUTCount'],features['Norm_Levels'] \
        ,features['Norm_Node_shape'],features['Norm_Input_shape'],features['Norm_Output_shape']\
        ,features['Norm_Latched_shape'],features['Norm_POshape'],features['Norm_Edge_length_distribution']\
        ,features['Norm_Avg_fanout'],features['Norm_Std_fanout'],features['Norm_Avg_fanout_comb'],features['Norm_Std_fanout_comb']\
        ,features['Norm_Avg_fanout_pi'],features['Norm_Std_fanout_pi'],features['Norm_Avg_fanout_dff'],features['Norm_Std_fanout_dff']
        ,features['Norm_R']],dtype = np.float32)

#if __name__ == "__main__":
def test():
    read_file = ["adder","div","square","sqrt","sin","hyp","log2","bar","multiplier","max"] # Possible another options, change adder to bar/div/log2/hyp/max/multiplier/sin/sqrt/square
    for file in read_file:
        stats = defaultdict(list)
        print("Start to analyze node shape for circuit: " + file)
        Analyzefile = "../DRiLLS/benchmarks/arithmetic/" + file + ".blif"
        ccirc_binary = "Cgen/ccirc/ccirc"
        ccirc_stats(Analyzefile, ccirc_binary, stats,"Exp0")
        #print(stats)
        print("Reconvergence value : " + str(stats['Reconvergence']) + " ,max/min : "+ str(stats['Reconvergence_max' ])+ " / " + str(stats['Reconvergence_min' ]))
        
