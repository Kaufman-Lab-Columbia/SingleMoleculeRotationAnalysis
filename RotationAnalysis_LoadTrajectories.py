# -*- coding: utf-8 -*-
"""
Combining both LD and Intensity derived Rotational Analysis from 
Coordinates either from Translational Tracking or Simulated Trajectories

@author: Glassy
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.offsetbox import AnchoredText
from skimage import io
import statsmodels.tsa.stattools
import os
import sys
from scipy import optimize, special
from sklearn.metrics import r2_score

""" Define Methods """

def log_input(log_path, print_statement):

    """ Print instruction, gather user input and write program output into a log.txt file """

    input_val = input(print_statement)

    logFile = open(log_path, 'a')
    logFile.write(print_statement + input_val + '\n\n')
    logFile.close()

    return input_val  

def create_saveto_filepath(filepath, new_end):

    """Create a new filepath for saving an image, .txt, etc."""

    if '.tif' in filepath:
        save_to_filepath = filepath.replace('.tif', new_end)

    else:
        save_to_filepath = filepath.replace('.bin', new_end)

    return save_to_filepath

def inputs():
    
    """ Prompt user to input trajectory filepath, movie filepath and associated parameters """
    
    movie_filepath = input('Input the movie filepath: \n')
    folder = movie_filepath.rpartition('\\')[0]
    
    log_path = folder + r'\log.txt'
    logFile = open(log_path, 'w')
    logFile.write('Movie Filepath: ' + movie_filepath + '\n\n')
    logFile.close()
    
    trajectory_filepath = log_input(log_path, 'Input the trajectory filepath: \n')
     
    tbf = float(log_input(log_path, 'What is the time between frames (s)?\n'))
    
    movie_type = 5
    while (movie_type != 1 and movie_type != 2):
        movie_type = int(log_input(log_path, "Is the movie you are loading (1) one-channel (intensity) or (2) two-channel (LD)?\n"))
    
    coord_type = 5
    while (coord_type != 0 and coord_type != 1 and coord_type != 2):
        coord_type = int(log_input(log_path, "Are your coordinates (0) simulated, (1) from Particle Tracker or (2) from THUNDERSTORM?\n"))
    
    pixel_size = 0
    if coord_type == 2:
        pixel_size = float(log_input(log_path, "What is the pixel size you input to THUNDERSTORM?\n"))
    
    return movie_filepath, trajectory_filepath, folder, tbf, movie_type, coord_type, log_path, pixel_size

def load_movie(filepath):

    """Load movie file"""

    # Case 1: Filetype .tif or .ome.tif
    if '.tif' in filepath:
        img = io.imread(filepath)

        return img, img.shape[0], img.shape[1], img.shape[2]

    # Case 2: Filetype .bin
    elif '.bin' in filepath:
        img = np.fromfile(filepath, dtype='int16', sep="")

        # .bin files do NOT have metadata like .tif and as such dimensions and length must be input manually

        x_dimension = int(log_input(log_path, 'What is the x-dimension of the movie? \n'))
        y_dimension = int(log_input(log_path, 'What is the y-dimension of the movie? \n'))
        frame_number = int(log_input(log_path, 'How many frames are in the movie? \n'))

        img = img.reshape(frame_number, y_dimension, x_dimension)
        img[img < 0] = 0

        return img, img.shape[0], img.shape[1], img.shape[2]

    # Case 3: Invalid File
    else:
        sys.exit('Invalid filename... Terminating program.')
        return

def find_centers(trajectory_filepath, num_frames, pixel_size):
    
    """ Generate Center Pixel Coordinates by taking Integer Rounding of ImageJ XY Coords """
    if coord_type == 2:
        xy_trajarr = np.loadtxt(trajectory_filepath, delimiter=',', skiprows=1)[:, :-1] / pixel_size

        ct = 0
        for i in range(int(xy_trajarr.shape[1] / 3)):
            xy_trajarr = np.delete(xy_trajarr, 3*i-ct, 1)
            ct += 1
        
    else: 
        xy_trajarr = np.loadtxt(trajectory_filepath, delimiter=',')
        
    num_features = int(xy_trajarr.shape[1] / 2)
    
    centers = np.empty((num_frames, num_features*2))
    centers[:] = np.nan
    
    # *** TO CORRECT FOR 0.5 PIXEL SHIFT FROM PARTICLE TRACKER, IDK WHY IT DOES THIS
    if coord_type == 1:
        xy_trajarr = xy_trajarr - 0.5    
    
    for i in range(num_frames):
        for j in range(num_features * 2):
            
            if np.isnan(xy_trajarr[i, j]) == True:
                continue
            
            centers[i, j] = np.rint(xy_trajarr[i, j])
            
    centers_filepath = create_saveto_filepath(movie_filepath, '_centers.csv')
    np.savetxt(centers_filepath, centers, delimiter=',')
    
    return centers, num_features, xy_trajarr

def find_avgPos(xy_traj, coord_type, num_feats):
    
    """ Compute the Average X and Y positions of all trajectories for output in Results file (and to be used for matching) """
    
    avg_Pos = np.empty((num_feats, 2))
    avg_Pos[:] = np.nan
        
    if coord_type == 1 or coord_type == 2: 
        for i in range(num_feats):
            avg_Pos[i, 1] = np.nanmean(xy_traj[:, i*2])     # X-coord
            avg_Pos[i, 0] = np.nanmean(xy_traj[:, i*2+1])   # Y-coord
            
    else:
        for i in range(num_feats):
            avg_Pos[i, 0] = np.nanmean(xy_traj[:, i*2])     # Y-coord
            avg_Pos[i, 1] = np.nanmean(xy_traj[:, i*2+1])   # X-coord
                       
    
    avgPos_filepath = create_saveto_filepath(movie_filepath, '_avg_Positions.csv')
    np.savetxt(avgPos_filepath, avg_Pos, delimiter=',')
    
    return avg_Pos

def compute_TrajInfo(xy_traj, num_feats):
    
    """ Compute the Number of Tracked Positions, Trajectory Length (final - initial frame) and Track% """
    
    trajInfo = np.empty((num_feats, 3))
    trajInfo[:] = np.nan
    for i in range(num_feats):
        
        # Compute Number of Tracked Positions
        trajInfo[i, 0] = np.count_nonzero(~np.isnan(xy_traj[:, i*2]))
        
        # Compute Trajectory Length
        if np.isnan(xy_traj[:, i*2]).all():
            
            trajInfo[i, :] = 0
            continue
        
        else:
        
            first_frame = np.argmax(~np.isnan(xy_traj[:, i*2]))
            last_frame = np.argwhere(~np.isnan(xy_traj[:, i*2]))[-1][0]
            trajInfo[i, 1] = last_frame - first_frame
            
            # Compute Track Percent
            trajInfo[i, 2] = trajInfo[i, 0] / trajInfo[i, 1] * 100
    
    return trajInfo

def generate_intensity_trajectories(movie_file, centers, num_frames, num_feats, x_dim, y_dim, coord_type, LCRC):
    
    """ Generate Intensity Arrays from a 3x3 Pixel Window around Found Feature Centers at each Frame """
    
    int_traj = np.empty((num_frames, num_feats))
    int_traj[:] = np.nan
    
    for i in range(num_frames):
        for j in range(num_feats):
            
            if np.isnan(centers[i, j*2]) == True:
                continue
            
            if coord_type == 1 or coord_type == 2:
                # For Experimental Movies
                x_coord = int(centers[i, j*2])
                y_coord = int(centers[i, j*2+1])
            
            else:
                # For Simulated Movies
                y_coord = int(centers[i, j*2])
                x_coord = int(centers[i, j*2+1])
            
            if np.abs(x_dim - x_coord) < 3 or np.abs(y_dim - y_coord) < 3:
                continue
                        
            int_traj[i, j] = np.mean(movie_file[i, y_coord - 1:y_coord + 2, x_coord - 1:x_coord + 2])
    
    
    if LCRC == 0:
        inttraj_filepath = create_saveto_filepath(movie_filepath, '_LC_intensity_trajectories.csv')
    elif LCRC == 1:
        inttraj_filepath = create_saveto_filepath(movie_filepath, '_RC_intensity_trajectories.csv')
    else:
        inttraj_filepath = create_saveto_filepath(movie_filepath, '_intensity_trajectories.csv')
        
    np.savetxt(inttraj_filepath, int_traj, delimiter=',')
    
    return int_traj

def compute_LD(LC_inttraj, RC_inttraj, num_frames, num_feats):
    
    LD_arr = np.empty((num_frames, num_feats))
    LD_arr[:] = np.nan
    
    for i in range(num_frames):
        for j in range(num_feats):
            
            LC_int = LC_inttraj[i, j]
            RC_int = RC_inttraj[i, j]
            
            diff = LC_int - RC_int
            sum = LC_int + RC_int
            
            if (sum == 0 or np.isnan(sum)):
                LD = np.nan
                
            else:
                LD = diff / sum
                
            LD_arr[i, j] = LD
            
    LD_filepath = create_saveto_filepath(movie_filepath, '_LD_values.csv')
    np.savetxt(LD_filepath, LD_arr, delimiter=',')
    
    return LD_arr

def compute_ACF(data, num_frames, num_feats):
    
    """ Compute ACF Values for All Available Offset Datapoints """
    
    ACF_arr = np.empty((num_frames-1, num_feats))
    ACFuncert_arr = np.empty((num_frames-1, num_feats))
    ACF_qstat = np.empty((num_frames-1, num_feats))
    
    ACF_arr[:] = np.nan
    ACFuncert_arr[:] = np.nan
    ACF_qstat[:] = np.nan
    
    # ACF Calculated using Scipy Statsmodels
    
    with np.errstate(divide='ignore', invalid='ignore'):
        for i in range(num_feats):
            ACF = statsmodels.tsa.stattools.acf(data[:, i], nlags=num_frames-1, qstat=True, fft=True, alpha=0.05, missing='conservative')
    
            ACF_arr[:, i] = ACF[0][1:]
            ACFuncert_arr[:, i] = ACF[0][1:] - ACF[1][1:, 0]
            ACF_qstat[:, i] = ACF[2] 
            
    ACF_filepath = create_saveto_filepath(movie_filepath, '_ACFvals.csv')
    ACFuncert_filepath = create_saveto_filepath(movie_filepath, '_ACFuncert.csv')
    ACFqstat_filepath = create_saveto_filepath(movie_filepath, '_ACFqstat.csv')
    
    np.savetxt(ACF_filepath, ACF_arr, delimiter=',')
    np.savetxt(ACFuncert_filepath, ACFuncert_arr, delimiter=',')
    np.savetxt(ACFqstat_filepath, ACF_qstat, delimiter=',')
    
    return ACF_arr, ACFuncert_arr

def compute_ACFpts(data_arr, num_frames, num_feats):
    
    """ Compute the number of data points (i.e.; the number of non-NaN timelags) contributing to each ACF data point """
    
    ACFcts_arr = np.empty((num_frames, num_feats))
    ACFcts_arr[:] = np.nan
    
    print('Computing ACF Counts...\n')
    
    for i in range(num_feats): # iterate over all feats
        data = data_arr[:, i]
        data_avg = np.nanmean(data)
    
        for j in range(num_frames): 
            data_unshifted = data[0:num_frames-j-1]
            data_shifted = data[j:num_frames-1]
            
            numer = (data_unshifted - data_avg) * (data_shifted - data_avg)
            ACFcts_arr[j, i] = np.count_nonzero(~np.isnan(numer))  
        
    ACFcts_filepath = create_saveto_filepath(movie_filepath, '_ACFcts.csv')
    np.savetxt(ACFcts_filepath, ACFcts_arr, delimiter=',')
    
    return

def KWW(x, A, tauf, beta):

    """Define the KWW equation to fit ACFs to"""

    return A * np.exp(-np.power((x/tauf), beta))

def kww_fit(ACF_arr, ACFuncert_arr, num_frames, num_feats):
    
    """Fit ACF data to KWW equation to yield A, taufit and beta"""
    
    # Initialize variables
    
    x_data = (np.arange(ACF_arr.shape[0]) + 1) * tbf
    raw_results = np.zeros((num_feats, 3))
    raw_uncert = np.zeros((num_feats, 3))
    raw_r2_arr = np.zeros(num_feats)
    index_check = np.zeros(num_feats, dtype=int)
    
    min_points = int(log_input(log_path, 'What is the minimum number of acceptable points to fit for KWW: \n'))
    ACF_init_check = float(log_input(log_path, "What is the minimum initial ACF value that is acceptable? "
                                         "(0-1, Typically 0.3)\n"))
    ACF_fin_check = float(log_input(log_path, "What is the minimum final ACF value that is acceptable? "
                                        "(0-1, Typically 0.1)\n"))
    r2_cutoff = float(log_input(log_path, "What r^2 cutoff would you like to apply? \n"))
    
    # Create a folder for saving graphs
    
    new_folder = folder + '\KWW_fit_graphs'
    
    try:
        os.mkdir(new_folder)

    except OSError:
        print('Failed to create directory for KWW fit graphs. Folder likely already exists.\n')

    else:
        print('Created directory for KWW fit graphs. Now calculating fits...\n')
        
    # Compute KWW Fits    
    for i in range(num_feats):
        for j in range(num_frames-1):

            if ACF_arr[0, i] < ACF_init_check:
                break

            if ACF_arr[j, i] < ACF_fin_check: # or ACFuncert_arr[j, i] / ACF_arr[j, i] > ACF_uncert_cutoff:
                index_check[i] = int(j)
                break
            
            if np.isnan(ACF_arr[j, i]):
                index_check[i] = int(j);
                break
            
            else:
                index_check[i] = int(num_frames-1)
        
        if (np.isnan(ACF_arr[0, i]) == True or index_check[i] < min_points):
            raw_results[i, :] = [1, 1, 1]
            raw_uncert[i, :] = [10000, 10000, 10000]
            raw_r2_arr[i] = 0.01

        #   Do the actual fitting with Least Squares and initial guesses of A=1, tauf=where(ACF<0.4), beta=1
        #   A and beta are bounded by 0.01 and 2, taufit is bounded by 0 and inf

        elif index_check[i] > min_points: 

            # ORIGINAL FITTING PARAMETERS
            
            tau_init_loc = np.argmax(ACF_arr[:, i] < 0.4)
            if tau_init_loc > 0:
                tau_init = tau_init_loc * tbf
                
            else:
                tau_init = tbf*10
            
            p0 = [1, tau_init, 1]
            params, params_covariance = optimize.curve_fit(KWW, x_data[0:index_check[i]],
                                                           ACF_arr[0:index_check[i], i], p0, max_nfev=500,
                                                           bounds=([0.01, tbf, 0.01], [2, (num_frames/2)*tbf, 2]))
            
            # Compute R^2 of Fit        
            y_pred = KWW(x_data[0:index_check[i]], *params)
            r2 = r2_score(ACF_arr[0:index_check[i], i], y_pred)
            
            raw_results[i, :] = params
            raw_uncert[i, :] = np.sqrt(np.diag(params_covariance))
            raw_r2_arr[i] = r2      
            
    # Save Raw Results
    
    rawfit_filepath = create_saveto_filepath(movie_filepath, "_raw_kwwfit_results.csv")
    rawuncert_filepath = create_saveto_filepath(movie_filepath, "_raw_kwwfit_uncert.csv")
    rawr2_filepath = create_saveto_filepath(movie_filepath, "_raw_r2_filepath.csv")
    indexcheck_filepath = create_saveto_filepath(movie_filepath, "_indexchecks.csv")
    
    np.savetxt(rawfit_filepath, raw_results, delimiter=',')
    np.savetxt(rawuncert_filepath, raw_uncert, delimiter=',')
    np.savetxt(rawr2_filepath, raw_r2_arr, delimiter=',')
    np.savetxt(indexcheck_filepath, index_check, delimiter=',')
        
    # Filter results that meet the specified fitting uncertainties
    
    num_good = 0
    for i in range(num_feats):
        
        if raw_r2_arr[i] >= r2_cutoff:
            num_good += 1
        
            if num_good == 1:
                index = i
                good_fits = [index]
                
                good_idxcheck = [index_check[index]]
                results = raw_results[i, :].reshape((1, 3))
                uncert = raw_uncert[i, :].reshape((1, 3))
                r2_arr = raw_r2_arr[i].reshape((1, 1))
                
            else: 
                index = i
                good_fits = np.append(good_fits, [index], axis=0)
                
                add_res = np.array(raw_results[i, :]).reshape((1, 3))
                add_uncert = np.array(raw_uncert[i, :]).reshape((1, 3))
                add_r2 = np.array(raw_r2_arr[i]).reshape((1, 1))
                
                good_idxcheck = np.append(good_idxcheck, [index_check[index]], axis=0)
                results = np.append(results, add_res, axis=0)
                uncert = np.append(uncert, add_uncert, axis=0)
                r2_arr = np.append(r2_arr, add_r2, axis=0)
                
    num_good_feats = results.shape[0]
        
    # Graph Fits    
    for i in range(num_good_feats):
            
        index = good_fits[i]
            
        graph_file = new_folder + '/feature' + str(good_fits[i] + 1) + '.jpg'
        title = 'Feature ' + str(good_fits[i] + 1)
        fig, ax = plt.subplots(1, 1)
        
        plt.scatter(x_data[0:good_idxcheck[i]], ACF_arr[0:good_idxcheck[i], index], label='Data')
        plt.plot(x_data[0:good_idxcheck[i]],
                 KWW(x_data[0:good_idxcheck[i]], results[i, 0], results[i, 1], results[i, 2]),
                 label='Fitted function')
        
        plt.xlabel('Time (s)')
        plt.ylabel('ACF')
        plt.title(title)
        
        txt_str = 'A =' + str(results[i, 0])[0:5] + r'$(\pm)$' + str(uncert[i, 0])[0:7] + '\n' + r'$\tau_{fit} =$' + str(results[i, 1])[0:5] + r'$(\pm)$' + str(uncert[i, 1])[0:7] + '\n' + r'$\beta =$' + str(results[i, 2])[0:5] + r'$(\pm)$' + str(uncert[i, 2])[0:7] + '\n' + r'$r^{2} =$' + str(r2_arr[i][0])[0:7]
        
        at = AnchoredText(txt_str, prop=dict(size=12), frameon=True, loc='upper right')
        at.patch.set_boxstyle("round, pad=0., rounding_size=0.2")
        ax.add_artist(at)
        
        plt.savefig(graph_file, dpi=300)
        
        # Display the first graph for reference during analysis
        if i == 0:
            print('\n Feature', good_fits[i] + 1, 'results: \n')
            plt.show()
            print('\nSaving all graphs... Please wait...\n')

        plt.close()
        
    kwwfit_filepath = create_saveto_filepath(movie_filepath, '_kwwfit_results.csv')
    r2_filepath = create_saveto_filepath(movie_filepath, '_kww_r2_vals.csv')
    
    np.savetxt(kwwfit_filepath, results, delimiter=',')
    np.savetxt(r2_filepath, r2_arr, delimiter=',')
        
    return results, r2_arr, good_fits

def tauc_equation(tauf, beta):

    """Equation to calculate tauc from taufit and beta via the gamma function"""

    tauc = (tauf / beta) * special.gamma(1 / beta)

    return tauc

def calc_tauc(results_arr):

    """Calculate tauc for all found features"""

    empty_col = np.zeros((1, 1))
    results_arr = np.insert(results_arr, 1, empty_col, axis=1)
    features = results_arr.shape[0]

    for i in range(features):
        tauf = results_arr[i, 2]
        beta = results_arr[i, 3]
        tauc = tauc_equation(tauf, beta)
        results_arr[i, 1] = tauc

    return results_arr

def clean_output(prelim_res, r2_arr, avg_Pos, trajInfo, good_fits, num_frames):
    
    """ Clean up output array with coordinate centers, r2 values, log values and pts/tauf """
    
    output = np.zeros((good_fits.shape[0], 13))
    good_trajectories = np.zeros((num_frames, good_fits.shape[0]*2))
    
    # Add A, tauc, tauf and beta values to the output file
    output[:, 2:5] = prelim_res[:, 0:3]
    output[:, 7] = prelim_res[:, 3]
    
    # Add log tauc and log tauf to the output file
    output[:, 5:7] = np.log10(output[:, 3:5])
    
    # Add r^2 values to output array
    output[:, 9] = r2_arr.reshape((good_fits.shape[0],))
    
    # Add average (Y, X) coordinates of each feature to the output file
    for i in range(good_fits.shape[0]):
        index = good_fits[i]
        output[i, 0:2] = avg_Pos[index]
        
    # Add trajectory info of each feature to the output file
    for i in range(good_fits.shape[0]):
        index = good_fits[i]
        output[i, 10:14] = trajInfo[index]
        
    # Add pts/taufit to output array
    output[:, 8] = prelim_res[:, 2] / tbf
    
    # Filter good trajectories
    for i in range(good_fits.shape[0]):
        index = good_fits[i]
        good_trajectories[:, 2*i:2*i+2] = xy_traj[:, 2*index:2*index+2]
    
    # Add median values and a line of zeros to the array to deliniate
    empty_add = np.zeros((2, 13))
    output = np.append(empty_add, output, axis=0)
    
    for i in range(output.shape[1]):
        output[0, i] = np.median(output[2:, i])
    
    # Save output files
    output_filepath = create_saveto_filepath(movie_filepath, "_final_results.csv")
    goodtraj_filepath = create_saveto_filepath(movie_filepath, "_good_trajectories.csv")
    
    np.savetxt(output_filepath, output, delimiter=',', header='Yavg, Xavg, A, tauc, tauf, log tauc, log tauf, beta, pts/tauf, r^2, #Pos, TrajLen, Track%')
    np.savetxt(goodtraj_filepath, good_trajectories, delimiter=',')
    
    # Filtered Output to remove A = 2, Beta = 2 and pts/taufit < 2
    filtered_output = output[2:, :]
    filtered_goodIndices = good_fits
    num_del = 0

    for i in range(filtered_output.shape[0]):
        if np.abs(filtered_output[i-num_del, 2] - 2) < 0.0000001:
            filtered_output = np.delete(filtered_output, i-num_del, axis=0)
            filtered_goodIndices = np.delete(filtered_goodIndices, i-num_del, axis=0)
            num_del += 1
            continue
            
        if np.abs(filtered_output[i-num_del, 7] - 2) < 0.0000001:
            filtered_output = np.delete(filtered_output, i-num_del, axis=0)
            filtered_goodIndices = np.delete(filtered_goodIndices, i-num_del, axis=0)
            num_del += 1
            continue
            
        if filtered_output[i-num_del, 8] < 2.0:
            filtered_output = np.delete(filtered_output, i-num_del, axis=0)
            filtered_goodIndices = np.delete(filtered_goodIndices, i-num_del, axis=0)
            num_del += 1
            continue
        
    empty_add = np.zeros((2, filtered_output.shape[1]))
    filtered_output = np.append(empty_add, filtered_output, axis=0)

    for i in range(filtered_output.shape[1]):
        filtered_output[0, i] = np.median(filtered_output[2:, i])    

    filteredout_filepath = create_saveto_filepath(movie_filepath, "_filtered_final_results.csv")
    np.savetxt(filteredout_filepath, filtered_output, delimiter=',', header='Yavg, Xavg, A, tauc, tauf, log tauc, log tauf, beta, pts/tauf, r^2, #Pos, TrajLen, Track%')
    
    # Print Filtered Median Values
    print("Median LogTauc: " + str(filtered_output[0, 5]) + " Median LogTauf: " + str(filtered_output[0, 6]) + " Median Beta: " + str(filtered_output[0, 7]))
    
    return filtered_output, filtered_goodIndices

def QE_calc(ACF_arr, indicesArr):
    
    """Calculate QE ACF Values and Fit to KWW to Extract QE Rotational Information"""
    
    filtered_ACF = np.empty((ACF_arr.shape[0], indicesArr.shape[0]))
    for i in range(indicesArr.shape[0]):
        idx = indicesArr[i]
        filtered_ACF[:, i] = ACF_arr[:, idx]
        
    filteredACF_filepath = create_saveto_filepath(movie_filepath, "_filtered_ACF.csv")
    np.savetxt(filteredACF_filepath, filtered_ACF, delimiter=',')
    
    # Take average of each ACF offset up until ACF < 0.1
    QE_ACF = np.empty(filtered_ACF.shape[0],)
    
    for i in range(filtered_ACF.shape[0]):
        QE_ACF[i] = np.nanmean(filtered_ACF[i, :])
        
        if QE_ACF[i] < 0.1:
            QE_fin = i
            break
        
    QEACF_filepath = create_saveto_filepath(movie_filepath, "_QE_ACFvals.csv")
    np.savetxt(QEACF_filepath, QE_ACF, delimiter=',')
    
    # Compute KWW Fit
    QE_ACF = QE_ACF[0:QE_fin]
    x_data = (np.arange(QE_ACF.shape[0]) + 1) * tbf
    
    tau_init_loc = np.argmax(QE_ACF[:] < 0.4)
    if tau_init_loc > 0:
        tau_init = tau_init_loc * tbf
        
    else:
        tau_init = tbf
    
    p0 = [1, tau_init, 1]
    params, params_covariance = optimize.curve_fit(KWW, x_data, QE_ACF, p0, max_nfev=500,
                                                   bounds=([0.01, tbf, 0.01], [2, (num_frames/2)*tbf, 2]))
    
    # Compute R^2 of Fit        
    y_pred = KWW(x_data, *params)
    r2 = r2_score(QE_ACF, y_pred)
    
    QE_results = params
    QE_uncert = np.sqrt(np.diag(params_covariance))
    QE_r2 = r2      
    
    # Plot
    QE_graphfile = create_saveto_filepath(movie_filepath, "_QE_kwwfit.jpg")
    title = 'QE KWW Fit'
    fig, ax = plt.subplots(1, 1)
    
    plt.scatter(x_data, QE_ACF, label='Data')
    plt.plot(x_data,
             KWW(x_data, QE_results[0], QE_results[1], QE_results[2]),
             label='Fitted function')
    
    plt.xlabel('Time (s)')
    plt.ylabel('ACF')
    plt.title(title)
    
    txt_str = 'A =' + str(QE_results[0])[0:5] + r'$(\pm)$' + str(QE_uncert[0])[0:7] + '\n' + r'$\tau_{fit} =$' + str(QE_results[1])[0:5] + r'$(\pm)$' + str(QE_uncert[1])[0:7] + '\n' + r'$\beta =$' + str(QE_results[2])[0:5] + r'$(\pm)$' + str(QE_uncert[2])[0:7] + '\n' + r'$r^{2} =$' + str(QE_r2)[0:7]
    
    at = AnchoredText(txt_str, prop=dict(size=12), frameon=True, loc='upper right')
    at.patch.set_boxstyle("round, pad=0., rounding_size=0.2")
    ax.add_artist(at)
    
    plt.savefig(QE_graphfile, dpi=300)
    
    return

""" Run Analysis """

# INPUTS
movie_filepath, trajectory_filepath, folder, tbf, movie_type, coord_type, log_path, pixel_size = inputs()

# LOAD MOVIE
movie_file, num_frames, y_dim, x_dim = load_movie(movie_filepath)
if movie_type == 2:
    LC = movie_file[:, :, 0:int(x_dim/2)]
    RC = movie_file[:, :, int(x_dim/2):]

# GENERATE CENTERS ARRAY
centers, num_feats,  xy_traj = find_centers(trajectory_filepath, num_frames, pixel_size)

# GENERATE AVERAGE POSITION ARRAY
avg_Pos = find_avgPos(xy_traj, coord_type, num_feats)

# COMPUTE TRAJECTORY INFORMATION
trajInfo = compute_TrajInfo(xy_traj, num_feats)

print("Computing Intensity Trajectories...\n")
# GENERATE INTENSITY TRAJECTORIES
if movie_type == 1:
    int_traj = generate_intensity_trajectories(movie_file, centers, num_frames, num_feats, x_dim, y_dim, coord_type, LCRC = 5)
else:
    LC_inttraj = generate_intensity_trajectories(LC, centers, num_frames, num_feats, x_dim/2, y_dim, coord_type, 0)
    RC_inttraj = generate_intensity_trajectories(RC, centers, num_frames, num_feats, x_dim/2, y_dim, coord_type, 1)

# COMPUTE LD (for movie_type == 2)
if movie_type == 2:
    print("Computing LD...\n")
    LD_arr = compute_LD(LC_inttraj, RC_inttraj, num_frames, num_feats)
    
# COMPUTE ACF AND NUMBER OF ACF PTS
print("Computing ACF...\n")
if movie_type == 1:
    ACF_arr, ACFuncert_arr = compute_ACF(int_traj, num_frames, num_feats)
    compute_ACFpts(int_traj, num_frames, num_feats)
else:
    ACF_arr, ACFuncert_arr = compute_ACF(LD_arr, num_frames, num_feats)
    compute_ACFpts(LD_arr, num_frames, num_feats)

# COMPUTE KWW FITS
print("Computing KWW Fits...\n")
results, r2_arr, good_fits = kww_fit(ACF_arr, ACFuncert_arr, num_frames, num_feats)

# COMPUTE TAUC
results = calc_tauc(results)
prelim_results_filepath = create_saveto_filepath(movie_filepath, '_prelim_results.csv')
np.savetxt(prelim_results_filepath, results, delimiter=',')

# CLEAN UP OUTPUT FILE
print("Cleaning Output File...\n")
filtered_output, filtered_goodIndices = clean_output(results, r2_arr, avg_Pos, trajInfo, good_fits, num_frames)

# COMPUTE QE RESULT
print("Computing QE Results...\n")
QE_calc(ACF_arr, filtered_goodIndices)

print("Analysis Complete!")