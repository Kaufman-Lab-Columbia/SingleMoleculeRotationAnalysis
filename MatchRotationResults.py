# -*- coding: utf-8 -*-
"""
@author: Alec Meacham
"""

import numpy as np

"""Define all necessary methods"""

def create_saveto_filepath(filepath, new_end):

    """Create a new filepath for saving an image, .txt, etc."""

    if '.csv' in filepath:
        save_to_filepath = filepath.replace('.csv', new_end)

    return save_to_filepath

def log_input(log_path, print_statement):

    "Print instruction, gather user input and write program output into a log.txt file"

    input_val = input(print_statement)

    logFile = open(log_path, 'a')
    logFile.write(print_statement + input_val + '\n\n')
    logFile.close()

    return input_val

def inputs():

    """Prompt user to input filepath, TBF and movie type"""

    traj_filepath = input("Input filepath of Filtered Trajectories:\n")
    filepath_tuple = traj_filepath.rpartition('\\')
    folder = filepath_tuple[0]
    
    log_path = folder + r'\log.txt'
    logFile = open(log_path, 'w')
    logFile.write('Trajectory Filepath: ' + traj_filepath + '\n\n')
    logFile.close()
    
    LD_filepath = log_input(log_path, "Input filepath of LD Rotation Results:\n")
    Int_filepath = log_input(log_path, "Input filepath of Intensity Rotation Results:\n")
    
    coord_type = log_input(log_path, "Is the tracking (0) simulated positions, (1) from Particle Tracker or (2) from THUNDERSTORM?\n")

    # trajectories = np.loadtxt(traj_filepath, delimiter=',')   
    trajectories = []
    LD_res = np.loadtxt(LD_filepath, delimiter=',')[2: :]
    Int_res = np.loadtxt(Int_filepath, delimiter=',')[2:, :]

    return traj_filepath, LD_filepath, Int_filepath, folder, trajectories, LD_res, Int_res, coord_type

def processTraj(trajectories, coord_type):
    
    """ Compute Average Positions, Number of Tracked Positions and Trajectory Length (last - first) """
    
    # Compute Average Positions
    
    num_frames = trajectories.shape[0]
    num_features = int(trajectories.shape[1] / 2)
    
    centers = np.empty((num_frames, num_features*2))
    centers[:] = np.nan
    
    # *** TO CORRECT FOR 0.5 PIXEL SHIFT FROM PARTICLE TRACKER, IDK WHY IT DOES THIS
    coord_type = 1 #i.e.; from Particle Tracker
    
    if coord_type == 1:
        trajectories = trajectories - 0.5
    
    for i in range(num_frames):
        for j in range(num_features * 2):
            
            if np.isnan(trajectories[i, j]) == True:
                continue
            
            centers[i, j] = np.rint(trajectories[i, j])
            
    avgPos = np.zeros((num_features, 2))
    
    if coord_type == 1:
        for i in range(num_features):
            avgPos[i, 0] = np.nanmean(centers[:, i*2+1]) #x_vag
            avgPos[i, 1] = np.nanmean(centers[:, i*2]) #y_avg
            
    else: 
        for i in range(num_features):
            avgPos[i, 0] = np.nanmean(centers[:, i*2]) #x_vag
            avgPos[i, 1] = np.nanmean(centers[:, i*2+1]) #y_avg
        
    # Compute Number of Tracked Positions
    
    numTracked = np.zeros(num_features)
    for i in range(num_features):
        numTracked[i] = np.count_nonzero(~np.isnan(trajectories[:, i*2]))
    
    # Compute Trajectory Length (last - first)
    
    trajLen = np.zeros(num_features)
    for i in range(num_features):
        first_frame = np.argmax(~np.isnan(trajectories[:, i*2]))
        last_frame = np.argwhere(~np.isnan(trajectories[:, i*2]))[-1][0]
        trajLen[i] = last_frame - first_frame
    
    # avgPos_filepath = create_saveto_filepath(traj_filepath, "_avgPos.csv")
    # np.savetxt(avgPos_filepath, avgPos, delimiter=',')
    
    return avgPos, numTracked, trajLen

def match_results(folder, first_results, second_results):
    
    """ Match the coordinates of found Features in the two loaded results files """
    
    x_dim = first_results.shape[1]-3 + second_results.shape[1]+1
    y_dim = second_results.shape[0]

    matched_results = np.zeros((y_dim, x_dim))
    check = 0

    for i in range(second_results.shape[0]):
        for j in range(first_results.shape[0]):
            if np.all(second_results[i, 0:2] == first_results[j, 0:2]):
                matched_results[check, 0:first_results.shape[1]-3] = first_results[j, :-3]
                matched_results[check, first_results.shape[1]-3+1:] = second_results[i, :]
                check += 1
                break

    index = y_dim
    for i in range(y_dim):
        if np.all(matched_results[i, :] == 0):
            index = i
            break
        
    matched_results = matched_results[:index, :]
    
    return matched_results

def clean_output(matched_results, avgPos, numTracked, trajLen):
    
    """ Clean and Organize Output """
    
    numberResults = matched_results.shape[0]
    goodIndices = np.zeros(numberResults)
    
    for i in range(numberResults):
        for j in range(avgPos.shape[0]):
            if np.all(matched_results[i, 0:2] == avgPos[j, :]):
                goodIndices[i] = j
                break
            
    # print(matched_results[:, 0:2])
    # print(avgPos)
    # print(goodIndices)
    # input()
            
    addVals = np.zeros((numberResults, 2))
    for i in range(numberResults):
        index = int(goodIndices[i])
        addVals[i, 0] = numTracked[index]
        addVals[i, 1] = trajLen[index]
    
    matched_results = np.append(addVals, matched_results, axis=1)
    
    empty_add = np.zeros((2, matched_results.shape[1]))
    matched_results = np.append(empty_add, matched_results, axis=0)
    
    for i in range(matched_results.shape[1]):
        matched_results[0, i] = np.median(matched_results[2:, i])

    matched_results_filepath = folder + r'\matched_results.csv'
    np.savetxt(matched_results_filepath, matched_results, delimiter=',', 
               header='#TrackedPos, TrajLen, Yavg, Xavg, LD A, LD tauc, LD tauf, LD log tauc, LD log tauf, LD beta, LD pts/tauf, LD r^2, SPACER, Yavg, Xavg, Int. A, Int. tauc, Int. tauf, Int. log tauc, Int. log tauf, Int. beta, Int. pts/tauf, Int. r^2')  
    
    return

"""Run Analysis"""

traj_filepath, LD_filepath, Int_filepath, folder, trajectories, LD_res, Int_res, coord_type = inputs()
# avgPos, numTracked, trajLen = processTraj(trajectories, coord_type)
matched_results = match_results(folder, LD_res, Int_res)

empty_add = np.zeros((2, matched_results.shape[1]))
matched_results = np.append(empty_add, matched_results, axis=0)

for i in range(matched_results.shape[1]):
    matched_results[0, i] = np.median(matched_results[2:, i])

matched_results_filepath = folder + r'\matched_results.csv'
np.savetxt(matched_results_filepath, matched_results, delimiter=',', 
           header='Yavg, Xavg, LD A, LD tauc, LD tauf, LD log tauc, LD log tauf, LD beta, LD pts/tauf, LD r^2, SPACER, Yavg, Xavg, Int. A, Int. tauc, Int. tauf, Int. log tauc, Int. log tauf, Int. beta, Int. pts/tauf, Int. r^2, #Pos, TrajLen, Track%')  


# clean_output(matched_results, avgPos, numTracked, trajLen)