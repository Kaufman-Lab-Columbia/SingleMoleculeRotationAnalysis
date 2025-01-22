# -*- coding: utf-8 -*-
"""
This code includes an analysis pipeline for Rotational Analysis analogous to 
that previously implemented in IDL. 

This analysis is only suitable for two-channel, linear-dichroism data and does
not accurately extract rotational timescales from single-channel, intensity
measurements. For that code reference "RotationalAnalysis_LoadTrajectories.py" 

@author: Alec Meacham
"""

"""

Imports

"""

import numpy as np
from matplotlib import pyplot as plt
from skimage import io, img_as_ubyte, morphology
from skimage.color import rgb2gray
from skimage.feature import blob_log
import statsmodels.tsa.stattools
import os
import sys
from scipy import optimize, special
from sklearn.metrics import r2_score

"""

Define Methods

"""

def log_input(log_path, print_statement):

    """ 
    Print instruction, gather user input and write program output into a 
    log.txt file. 
    
    Args: 
        log_path: (str) Filepath where the log.txt file is located
        print_statement: (str) Prompt displayed to the user
        
    Returns:
        input_val: (str) User input value
    """

    input_val = input(print_statement)

    logFile = open(log_path, 'a')
    logFile.write(print_statement + input_val + '\n\n')
    logFile.close()

    return input_val

def fileinputs():

    """ 
    Prompt user to input trajectory filepath, movie filepath and associated parameters 
    
    Args: 
        None
    
    Returns: 
        filepath: (str) Filepath of .tif or .bin movie
        filename: (str) Name of movie file without full path
        folder: (str) Folder where all analyis outputs are being written
        tbf: (float) Time-Between-Frames of movie
        movie_type: (int) Type of Movie: 
            (1) Single-Channel for Intensity -- NOT USABLE
            (2) Two-Channel for Linear Dichroism
        log_path: (str) Path of log.txt file
    """

    filepath = input('Input filepath: \n')
    filepath_tuple = filepath.rpartition('\\')
    filename = filepath_tuple[len(filepath_tuple)-1]
    folder = filepath_tuple[0]

    log_path = folder + r'\log.txt'
    logFile = open(log_path, 'w')
    logFile.write('Filepath: ' + filepath + '\n\n')
    logFile.close()

    tbf = float(log_input(log_path, 'What is the time between frames (s)? \n'))

    movie_type = 2 #int(log_input(log_path, 'Is this a 2ch or 1ch movie? \n'))
    #while movie_type != 1 and movie_type != 2:
        #movie_type = int(log_input(log_path, 'Invalid movie type. Please try again.'))

    return filepath, filename, folder, tbf, movie_type, log_path

def load_movie(filepath):

    """
    Load movie file
    
    Args: 
        filepath: (str) Filepath of .tif or .bin movie
        
    Returns
        img: (ndarray (num_frames, y_dim, x_dim)): Movie Array
        img.shape[0]: (int) num_frames, Number of frames in movie
        img.shape[1]: (int) y_dim, y-dimension of movie
        img.shape[2]: (int) x_dim, x-dimension of movie
    """

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

def create_saveto_filepath(filepath, new_end):

    """
    Create a new filepath for saving output files
    
    Args: 
        filepath: (str) Filepath of original input as a base for appending
        new_end: (str) String to append to the end of the original filepath
    
    Returns: 
        save_to_filepath: (str) Filepath with newly appended ending
    """

    if '.tif' in filepath:\
        save_to_filepath = filepath.replace('.tif', new_end)

    else:
        save_to_filepath = filepath.replace('.bin', new_end)

    return save_to_filepath

def sum_middle_frames(img, frame_number):

    """
    Sum the middle N frames, display this summed image and save this summed 
    image.
    
    Args: 
        img: (ndarray (frame_number, y_dim, x_dim)) Array containing the pixel
            intensities for all frames of the movie
        frame_number: (int) Number of frames in movie
        
    Returns:
        summed_image: (ndarray (y_dim, x_dim)) Array containing the summed 
            pixel intensities for the middle N frames of the movie
    """

    sum_count = int(log_input(log_path, 'How many frames would you like to sum for feature finding? \n'))
    while sum_count > frame_number:
        sum_count = int(log_input(log_path, 'Frame number to sum invalid, please try again: \n'))

    middle_frame = int(frame_number / 2)
    sum_start = int(middle_frame - sum_count / 2)
    sum_end = int(middle_frame + sum_count / 2)
    #sum_start = int(input('Which frame start to sum? \n'))
    #sum_end = int(input('Which frame end to sum? \n'))

    print('Summing from', sum_start, 'to', sum_end, '\n')

    summed_image = np.sum(img[sum_start:sum_end, :, :], axis=0) / sum_count
    si_max = np.max(summed_image)
    si_min = np.min(summed_image)
    si_med = np.median(summed_image)
    summed_image_filepath = create_saveto_filepath(filepath, '_summed_middle_frames.jpg')

    title = 'Summed Middle Frames'
    fig, ax = plt.subplots(1, 1)
    ax.set_title(title)
    ax.imshow(summed_image, cmap='gray')
    ax.set_axis_off()
    plt.savefig(summed_image_filepath, dpi=300)
    plt.show()
    plt.close()

    return summed_image

def find_features(summed_middle_frames):

    """
    
    Identify features using skimage white_tophat thresholding and save 
    
    Args: 
        summed_middle_frames: (ndarray (y_dim, x_dim)) Array containing the summed 
            pixel intensities for the middle N frames of the movie
        
    Returns: 
        coord_arr: (ndarray (num_molecules, 2)) Array of coordinates for all
            molecules found
    """

    is_good = False

    # Make structural element to filter features by
    selem = morphology.disk(3)
    res = morphology.white_tophat(summed_middle_frames, selem)

    title = 'Isolated Features'
    fig, ax = plt.subplots(1, 1)
    ax.set_title(title)
    ax.imshow(res, cmap='gray')
    ax.set_axis_off()
    plt.show()
    plt.close()

    while is_good == False:

        # Calculate local threshold intensity scaled by an inputted factor
        threshold_scaling = float(log_input(log_path, 'What scaling factor should be applied to intensity threshold? '
                                        '(Start with 100; ideally you want ~1000 features) \n'))
        if threshold_scaling <= 0:
            threshold_scaling = float(log_input(log_path, 'Please input a positive scaling factor: \n'))

        with np.errstate(divide='ignore', invalid='ignore'):

            # Blob detection - Determine what IS and ISNT a feature
            blobs_log = blob_log(res, min_sigma=1, max_sigma=5, num_sigma=3, threshold=threshold_scaling)
            blobs_log[:, 2] = blobs_log[:, 2] * np.sqrt(2)

            # Remove features with radii greater than 3x the standard radius
            to_delete = blobs_log[:, 2] > 3 * np.sqrt(2)
            blobs_log = np.delete(blobs_log, np.where(to_delete), 0)
            number_molecules_found = blobs_log.shape[0]

        # Display found features
        print('\n Found', number_molecules_found, 'features:')

        found_features_filepath = create_saveto_filepath(filepath, '_found_features.jpg')

        title = 'LoG Found Features'
        color = 'lime'
        fig, ax = plt.subplots(1, 1)
        ax.set_title(title)
        ax.imshow(res, cmap='gray')
        for blob in blobs_log:
            y, x, r = blob
            c = plt.Circle((x, y), r, color=color, linewidth=2, fill=False)
            ax.add_patch(c)
        ax.set_axis_off()
        plt.savefig(found_features_filepath, dpi=300)
        plt.show()
        plt.close()

        goodness_check = str(log_input(log_path, 'Is this acceptable (Y/N)? \n'))

        if goodness_check == 'y' or goodness_check == 'Y':
            is_good = True

    # Return coordinate array without radii values and sort by X coordinate; [Ys, Xs]
    coord_arr = blobs_log[:, 0:2]
    coord_arr = coord_arr[coord_arr[:,1].argsort()]
    coord_arr_filepath = create_saveto_filepath(filepath, '_coord_arr.csv')
    np.savetxt(coord_arr_filepath, coord_arr, delimiter=',')

    return coord_arr

def generate_intensity_arrays(coordinate_array, x_dim, y_dim, frame_number, movie_file):

    """
    Generate an array of ints and of the form (Ys, Xs, Int) for all features 
    and frames by integrating intensities in a 3x3 pixel area
    
    Args: 
        coordinate_array: (ndarray (num_molecules, 2)) Array of coordinates for 
        all molecules found
        x_dim: (int) x-dimension of movie
        y_dim: (int) y-dimension of movie
        frame_number: (int) Number of frames in movie
        movie_file: (ndarray (frame_number, y_dim, x_dim)) Array containing the 
            pixel intensities for all frames of the movie
        
    Returns: 
        good_coordinate_int_array: (ndarray (frame_number, good_particle, 3)) 
            Array with each molecule's (that is not too close to an edge to 
            integrate) y-coordinate, x-coordinate and integrated intensity values
        Int_values: (ndarray (frame_number, good_particle)) Array with all
            integrated intensities for each molecule at every frame
    """

    number_particles = coordinate_array.shape[0]

    # Remove particles on the very edge of the FOV
    good_particle = 0
    last_good = 0
    bad_particle = 0
    first_particle = 0

    for i in range(number_particles):
        y_coord = int(coordinate_array[i, 0])
        x_coord = int(coordinate_array[i, 1])
        if x_coord > 2 and x_coord < x_dim-3 and y_coord > 2 and y_coord < y_dim-3:
            first_particle += 1
            good_particle += 1

            if first_particle == 1:
                good_indices = np.zeros(1, dtype=int)
                good_indices[0] = i
                last_good = good_particle
                first_particle +=1

            elif good_particle > 1 and good_particle > last_good:
                add = np.zeros(1, dtype=int)
                add[0] = i
                good_indices = np.append(good_indices, add)
                last_good = good_particle

        else:
            good_particle += 1
            last_good = good_particle
            bad_particle += 1

    good_particle = good_particle - bad_particle
    raw_Int_values = np.zeros((frame_number, good_particle), dtype='float32')
    Int_values = np.zeros((frame_number, good_particle), dtype = 'float32')
    bkg_values = np.zeros((frame_number, good_particle), dtype = 'float32')
    good_coordinate_int_array = np.zeros((frame_number, good_particle, 3), dtype = 'float32')
    int_coords = np.zeros((good_particle, 2))

    for k in range(good_particle):
        i = good_indices[k]
        y_coord = int(coordinate_array[i, 0])
        x_coord = int(coordinate_array[i, 1])

        for j in range(frame_number):
            raw_Int_values[j, k] = np.mean(movie_file[j, y_coord - 1:y_coord + 2, x_coord - 1:x_coord + 2]) # ORIGINAL 3x3 FEATURE WINDOW

            # raw_Int_values[j, k] = np.mean(movie_file[j, y_coord - 2:y_coord + 3, x_coord - 2:x_coord + 3]) # TRYING A LARGER 5X5 FEATURE WINDOW

            Int_values[j, k] = raw_Int_values[j, k]
            # (Int_values[j, k], bkg_values[j, k]) = bkg_subtr(raw_Int_values[j, k], x_coord, y_coord, movie_file, j)

            good_coordinate_int_array[j, k, 0] = y_coord
            good_coordinate_int_array[j, k, 1] = x_coord
            good_coordinate_int_array[j, k, 2] = Int_values[j, k]

    int_coords = good_coordinate_int_array[0, :, 0:2]

    raw_int_filepath = create_saveto_filepath(filepath, '_no_bkgsub_Int_data.csv')
    int_filepath = create_saveto_filepath(filepath, '_Int_data.csv')
    int_coords_filepath = create_saveto_filepath(filepath, '_Int_coords.csv')
    np.savetxt(int_filepath, Int_values, delimiter=',')
    np.savetxt(raw_int_filepath, raw_Int_values, delimiter=',')
    np.savetxt(int_coords_filepath, int_coords, delimiter=',')

    return good_coordinate_int_array, Int_values

def Set_Sep(coordinate_int_arr, filepath, log_path):

    """
    Set channel seperation using a generated heatmap
    
    Args: 
        coordinate_int_array: (ndarray (frame_number, good_particle, 3)) 
            Array with each molecule's (that is not too close to an edge to 
            integrate) y-coordinate, x-coordinate and integrated intensity values
        filepath: (str) Filepath of .tif or .bin movie
        log_path: (str) Path of log.txt file
    
    Returns: 
        coords_int_2ch: (ndarray (num_frames, num_pairs, 4)) Array of 
            all integrated feature intensities paired for left and right channels
    """

    is_good = False
    num_features = coordinate_int_arr.shape[1]
    num_frames = coordinate_int_arr.shape[0]

    while is_good == False:
        x_sep = int(log_input(log_path, 'What is the x-offset: \n'))
        y_sep = int(log_input(log_path, 'What is the y-offset: \n'))
        uncert = int(log_input(log_path, 'What is the offset uncertainty: \n'))
        print('X-offset:', x_sep, 'Y-offset:', y_sep, 'Uncertainty:', uncert, '\n')
        print('A total of', num_features, 'good features were identified.\n') # good meaning not on the edge

        heatmap = np.zeros((5, 5))
        lower = -2
        upper = 3

        print('Calculating offset heatmap...\n')

        for a in range(lower, upper):
            for b in range(lower, upper):
                num_matches = 0
                last_match = 100000 #arbitrary high value

                for i in range(num_features-1):
                    for j in range(i + 1, num_features):

                        if np.abs(np.abs(coordinate_int_arr[0, i, 0] - coordinate_int_arr[0, j, 0]) - (y_sep + a)) \
                                <= uncert and np.abs(np.abs(coordinate_int_arr[0, i, 1] - coordinate_int_arr[0, j, 1])
                                - (x_sep + b)) <= uncert:
                            num_matches += 1

                            if num_matches == 1 and a == 0 and b == 0:
                                coords_int_2ch = np.zeros((num_frames, 1, 6))
                                coords_int_2ch[:, 0, 0] = coordinate_int_arr[:, i, 0]
                                coords_int_2ch[:, 0, 1] = coordinate_int_arr[:, i, 1]
                                coords_int_2ch[:, 0, 2] = coordinate_int_arr[:, i, 2]
                                coords_int_2ch[:, 0, 3] = coordinate_int_arr[:, j, 0]
                                coords_int_2ch[:, 0, 4] = coordinate_int_arr[:, j, 1]
                                coords_int_2ch[:, 0, 5] = coordinate_int_arr[:, j, 2]
                                num_matches += 1
                                last_match = 2

                            elif num_matches > last_match and a == 0 and b == 0:
                                append_row = np.zeros((num_frames, 1, 6))
                                append_row[:, 0, 0] = coordinate_int_arr[:, i, 0]
                                append_row[:, 0, 1] = coordinate_int_arr[:, i, 1]
                                append_row[:, 0, 2] = coordinate_int_arr[:, i, 2]
                                append_row[:, 0, 3] = coordinate_int_arr[:, j, 0]
                                append_row[:, 0, 4] = coordinate_int_arr[:, j, 1]
                                append_row[:, 0, 5] = coordinate_int_arr[:, j, 2]
                                coords_int_2ch = np.append(coords_int_2ch, append_row, axis=1)
                                last_match += 1

                num_matches -= 1
                heatmap[np.abs(a - 2), b + 2] = num_matches

        fig, ax = plt.subplots()
        im = ax.imshow(heatmap)
        ax.set_xticks(np.arange(5))
        ax.set_yticks(np.arange(5))
        ax.set_xticklabels((x_sep-2, x_sep-1, x_sep, x_sep+1, x_sep+2))
        ax.set_yticklabels((y_sep+2, y_sep+1, y_sep, y_sep-1, y_sep-2))

        for i in range(5):
            for j in range(5):
                text = ax.text(j, i, heatmap[i, j],
                               ha="center", va="center", color="w")

        ax.set_title("Heatmap of matches found based on offset:")
        fig.tight_layout()
        heatmap_filepath = create_saveto_filepath(filepath, '_heatmap.jpg')
        plt.savefig(heatmap_filepath, dpi=300)
        plt.show()
        plt.close()

        goodness_check = log_input(log_path, 'Found ' + str(heatmap[2, 2]) + ' pairs. ' 
                                                                           'Is this acceptable (Y/N)? \n').capitalize()
        while goodness_check != 'Y' and goodness_check != 'N':
            goodness_check = log_input(log_path, 'Invalid response please try again:\n').capitalize()
        if goodness_check == 'Y':
            is_good = True

    return coords_int_2ch

def calc_LD(coords_int_2ch, filepath):

    """
    Calculate Reduced Linear Dichroism and return an array of LDs
    
    Args: 
        coords_int_2ch: (ndarray (num_frames, num_pairs, 4)) Array of 
            all integrated feature intensities paired for left and right channels
        filepath: (str) Filepath of .tif or .bin movie
        
    Returns: 
        LD_arr: (ndarray (num_frames, num_pairs)) Array of LD values for all
            pairings and all frames
        LD_coords: (ndarray (num_pairs, 4)) Array of all paired coordinates
            from left and right channel
    """

    print('Generating LD Array \n')

    num_frames = coords_int_2ch.shape[0]
    num_pairs = coords_int_2ch.shape[1]
    LD_arr = np.zeros((num_frames, num_pairs))
    LD_coords = np.zeros((num_pairs, 4))
    summed_int_pairs = np.zeros((num_frames, num_pairs))

    for i in range(num_frames):
        for j in range(num_pairs):
            left_int = coords_int_2ch[i, j, 2]
            right_int = coords_int_2ch[i, j, 5]
            diff = left_int - right_int
            sum = left_int + right_int

            if sum == 0:
                LD = 'nan'

            else:
                LD = diff / sum

            LD_arr[i, j] = LD
            summed_int_pairs[i, j] = sum
            LD_coords[j, 0] = coords_int_2ch[i, j, 0]
            LD_coords[j, 1] = coords_int_2ch[i, j, 1]
            LD_coords[j, 2] = coords_int_2ch[i, j, 3]
            LD_coords[j, 3] = coords_int_2ch[i, j, 4]

    LD_filepath = create_saveto_filepath(filepath, '_LD.csv')
    LD_coords_filepath = create_saveto_filepath(filepath, '_LD_coords.csv')
    summed_int_pairs_filepath = create_saveto_filepath(filepath, '_summed_paired_ints.csv')
    np.savetxt(LD_filepath, LD_arr, delimiter=',')
    np.savetxt(LD_coords_filepath, LD_coords, delimiter=',', header='Y1, X1, Y2, X2')
    np.savetxt(summed_int_pairs_filepath, summed_int_pairs, delimiter=',')

    return LD_arr, LD_coords

def compute_ACF(data_to_fit, filepath, movie_type):

    """
    Compute ACF and return ACF array and Uncertainty Array with Statistical 
    Checks
    
    Args: 
        data_to_fit: (ndarray (num_frames, num_features)) Array of data to be 
            used for ACF computation. Only LD data should be used with this
            code. 
        filepath: (str) Filepath of .tif or .bin movie
        movie_type: (int) Type of Movie: 
            (1) Single-Channel for Intensity -- NOT USABLE
            (2) Two-Channel for Linear Dichroism
            
    Returns: 
        ACF_arr: (ndarray, (num_frames-1, num_features)) Array of ACF values
            for all molecules at all possible timelags
        ACF_uncert_arr: (ndarray (num_frames-1, num_features)) Array of ACF
            uncertainty values
        ACF_indices: (ndarray, (m,)) List of all indices of molecules who meet
            the defined checked for KWW fitting
        ACF_init_check: (float) Minimum cutoff value that the first time-lag
            ACF value must exceed
        ACF_fin_check: (float) Minimum cutoff value for fitting the long
            time-lag tail of the ACF
        ACF_uncert_cutoff: (float) Cutoff of uncertainty for fitting the ACF
            to the KWW equation
        min_points: (int) Minimum number of points which must be fit for the
            KWW fit to be considered significant
    """

    num_features = data_to_fit.shape[1]
    num_frames = data_to_fit.shape[0]

    min_points = int(log_input(log_path, 'What is the minimum number of acceptable points to fit for KWW: \n'))

    # -1 to num_frames is so we don't keep the 0-lag ACF point of 1
    raw_ACF_arr = np.zeros([num_frames-1, num_features])
    raw_ACF_uncert_arr = np.zeros([num_frames-1, num_features])

    # COMPUTE ACF USING STATSMODELS METHOD
    with np.errstate(invalid='ignore'):
        for i in range(num_features):
            data = data_to_fit[:, i]
            ACF = statsmodels.tsa.stattools.acf(data, nlags=num_frames-1, qstat=False, fft=True,
                                                alpha=0.05)  # missing = 'drop')
            ACF_data = ACF[0]
            ACF_uncert = ACF[1]
            raw_ACF_arr[:, i] = ACF_data[1:]
            # The ACF_uncert array is really the upper and lower bound so the subtraction below gives the +- value
            raw_ACF_uncert_arr[:, i] = ACF_data[1:] - ACF_uncert[1:, 0]
            # raw_ACF_uncert_arr[:, i] = 1 - (ACF_uncert[1:, 0] / ACF_data[1:])

    # COMPUTE ACF USING THE HAND-MATH FROM IDL
    # for i in range(num_features):
    #     data = data_to_fit[:, i]
    #     data_avg = np.mean(data)
    
    #     for j in range(num_frames):
    #         data_unshifted = data[0:num_frames-i-1]
    #         data_shifted = data[i:num_frames-1]
    #         ACF_numerator = (data_unshifted - data_avg) * (data_shifted - data_avg)
    #         ACF_denominator = (data_unshifted - data_avg) * (data_unshifted - data_avg)
    
    #         ACF_norm = ACF_numerator / ACF_denominator
    
    #         raw_ACF_arr[:, i] = ACF_norm[1:]

    # BACK TO NORMAL

    features = raw_ACF_arr.shape[1]
    num_ACF = raw_ACF_arr.shape[0]
    ACF_arr = np.zeros((1, num_frames))
    ACF_uncert_arr = np.zeros((1, num_frames))
    ACF_indices = np.zeros((1, 1))
    number_good = 0
    index_check = 0

    # Check for molcules with ACF[0]>0.3 and with >10pts before ACF dips below 0.1 or uncertainty is greater than 25%

    ACF_init_check = float(log_input(log_path, "What is the minimum initial ACF value that is acceptable? "
                                         "(0-1, Typically 0.3)\n"))
    ACF_fin_check = float(log_input(log_path, "What is the minimum final ACF value that is acceptable? "
                                        "(0-1, Typically 0.1)\n"))
    ACF_uncert_cutoff = float(log_input(log_path, "What is the acceptable ACF uncertainty? (0-1, Typically 0.25)\n"))

    for i in range(features):
        for j in range(num_ACF):
            if raw_ACF_arr[0, i] < ACF_init_check:
                index_check = 0
                break

            if np.isnan(raw_ACF_arr[j, i]) == True:
                index_check = 0
                break

            if raw_ACF_arr[j, i] < ACF_fin_check or raw_ACF_arr[j, i] * ACF_uncert_cutoff < raw_ACF_uncert_arr[j, i]:
                index_check = j
                break

        if index_check >= min_points:
            number_good += 1

            if number_good == 1:
                ACF_arr = raw_ACF_arr[:, i]
                ACF_uncert_arr = raw_ACF_uncert_arr[:, i]
                ACF_arr = ACF_arr.reshape((1, num_ACF))
                ACF_uncert_arr = ACF_uncert_arr.reshape((1, num_ACF))
                ACF_indices[0, 0] = i

            else:

                add = np.array(raw_ACF_arr[:, i])
                add = add.reshape((1, num_ACF))
                add_uncert = np.array(raw_ACF_uncert_arr[:, i])
                add_uncert = add_uncert.reshape((1, num_ACF))
                ACF_arr = np.append(ACF_arr, add, axis=0)
                ACF_uncert_arr = np.append(ACF_uncert_arr, add_uncert, axis=0)
                add_good_index = np.array([[i]])
                add_good_index = add_good_index.reshape((1, 1))
                ACF_indices = np.append(ACF_indices, add_good_index, axis=0)

    # Transpose such that each column is a feature
    ACF_arr = ACF_arr.T
    ACF_uncert_arr = ACF_uncert_arr.T

    ACF_filepath = create_saveto_filepath(filepath, '_ACFdata.csv')
    orig_ACF_filepath = create_saveto_filepath(filepath, '_origACFdata.csv')
    ACF_uncert_filepath = create_saveto_filepath(filepath, '_ACF_uncert.csv')
    orig_ACF_uncert_filepath = create_saveto_filepath(filepath, '_origACF_uncert.csv')
    np.savetxt(ACF_filepath, ACF_arr, delimiter=',')
    np.savetxt(orig_ACF_filepath, raw_ACF_arr, delimiter=',')
    np.savetxt(ACF_uncert_filepath, ACF_uncert_arr, delimiter=',')
    np.savetxt(orig_ACF_uncert_filepath, raw_ACF_uncert_arr, delimiter=',')

    return ACF_arr, ACF_uncert_arr, ACF_indices, ACF_init_check, ACF_fin_check, ACF_uncert_cutoff, min_points

def KWW(x, A, tauf, beta):

    """Define the KWW equation to fit ACFs to"""

    return A * np.exp(-np.power((x/tauf), beta))

def fit_KWW(ACF_arr, ACF_uncert_arr, ACF_indices, tbf, filepath, ACF_init_check, ACF_fin_check, ACF_uncert_cutoff):

    """
    Fit ACF data to KWW equation to yield A, taufit and beta
    
    Args: 
        ACF_arr: (ndarray, (num_frames-1, num_features)) Array of ACF values
            for all molecules at all possible timelags
        ACF_uncert_arr: (ndarray (num_frames-1, num_features)) Array of ACF
            uncertainty values
        ACF_indices: (ndarray, (m,)) List of all indices of molecules who meet
            the defined checked for KWW fitting
        tbf: (float) Time-Between-Frames of movie
        filepath: (str) Filepath of .tif or .bin movie
        ACF_init_check: (float) Minimum cutoff value that the first time-lag
            ACF value must exceed
        ACF_fin_check: (float) Minimum cutoff value for fitting the long
            time-lag tail of the ACF
        ACF_uncert_cutoff: (float) Cutoff of uncertainty for fitting the ACF
            to the KWW equation
    Returns: 
        results: (ndarray (good_feats, 3) Array of KWW fit paramters A, tauf,
              and beta for each molecule
        r2_arr: (ndarray (good_feats,)) List of r^2 values for each KWW fit
        good_index: (ndarray ()) Tracking indices for later
        good_indicies: (ndarray (m,)) Indices of all molecules that meet
            user requirements
    """

    num_features = ACF_arr.shape[1]
    num_frames = ACF_arr.shape[0]
    x_data = np.arange(ACF_arr.shape[0])*tbf+tbf
    raw_results = np.zeros((num_features, 3))
    raw_uncert = np.zeros((num_features, 3))
    raw_r2_arr = np.zeros(num_features)

    # Create folder for saving fit graphs

    #   For Ubuntu
    # filepath_tuple = filepath.rpartition('/')
    # filepath = filepath_tuple[0]
    # new_folder = filepath + '/KWW_fit_graphs'

    #   For Windows
    filepath_tuple = filepath.rpartition('\\')
    filepath = filepath_tuple[0]
    new_folder = filepath + '\KWW_fit_graphs'

    try:
        os.mkdir(new_folder)

    except OSError:
        print('Failed to create directory for KWW fit graphs. Folder likely already exists.\n')

    else:
        print('Created directory for KWW fit graphs. Now calculating fits...\n')

    for i in range(num_features):
        for j in range(num_frames):

            if ACF_arr[j, i] < ACF_fin_check or ACF_arr[j, i] * ACF_uncert_cutoff < ACF_uncert_arr[j, i]:
                index_check = j
                break

        #   Do the actual fitting with Least Squares and initial guesses of all 1
        #   A and beta are bounded by 0 and 2, taufit is bounded by 0 and inf

        p0 = [1, 1, 1]
        params, params_covariance = optimize.curve_fit(KWW, x_data[0:index_check],
                                                       ACF_arr[0:index_check, i], p0, max_nfev=10000,
                                                       bounds=([0.01, tbf, 0.01], [2, num_frames*tbf/2, 2]))

        # Compute R^2 of Fit
        y_pred = KWW(x_data[0:index_check], *params)
        r2 = r2_score(ACF_arr[0:index_check, i], y_pred)

        # tau_guess = np.where(ACF_arr[0:, i] < 0.4)[0][0]
        #
        # if tau_guess > np.floor(num_frames * tbf / 2.0):
        #     tau_guess = np.floor(num_frames * tbf / 2.0 - 1.0)

        # p0 = [1.0, tau_guess, 1.0]
        # params, params_covariance = optimize.curve_fit(KWW, x_data[0:index_check],
        #                                                ACF_arr[0:index_check, i], p0, max_nfev=500,
        #                                                bounds=([0.01, tbf, 0.01], [2, np.floor(num_frames * tbf / 2.0), 2]), method='lm')
        # p0 = [1.0, tau_guess, 1.0]
        # params, params_covariance = optimize.curve_fit(KWW, x_data[0:index_check],
        #                                                ACF_arr[0:index_check, i], p0, method='lm')

        raw_results[i, :] = params
        perr = np.sqrt(np.diag(params_covariance))
        raw_uncert[i, :] = perr
        raw_r2_arr[i] = r2
        

    kww_uncert_cutoff = float(log_input(log_path, "What is the acceptable KWW fit parameter uncertainty? "
                                            "(0-1, Typically 0.1)\n"))

    num_good = 0

    for i in range(num_features):
        if raw_uncert[i, 0]/raw_results[i, 0]<kww_uncert_cutoff and \
                raw_uncert[i, 1]/raw_results[i, 1]<kww_uncert_cutoff and \
                raw_uncert[i, 2]/raw_results[i, 2]<kww_uncert_cutoff:
            num_good += 1

            if num_good == 1:
                index = i
                orig_index = int(ACF_indices[i])
                good_index = [index]
                good_orig_indices = [orig_index]
                results = raw_results[i, :].reshape((1, 3))
                uncert = raw_uncert[i, :].reshape((1, 3))
                r2_arr = raw_r2_arr[i].reshape((1,1))

            else:

                index = i
                orig_index = int(ACF_indices[i])
                good_index = np.append(good_index, [index], axis=0)
                good_orig_indices = np.append(good_orig_indices, [orig_index], axis=0)\
                
                add_res = np.array(raw_results[i, :]).reshape((1, 3))
                add_uncert = np.array(raw_uncert[i, :]).reshape((1, 3))
                add_r2 = raw_r2_arr[i].reshape((1,1))
                
                results = np.append(results, add_res, axis=0)
                uncert = np.append(uncert, add_uncert, axis=0)
                r2_arr=np.append(r2_arr, add_r2, axis=0)

    num_good_features = results.shape[0]

    for i in range(num_good_features):
        index = good_index[i]
        for j in range(num_frames):
            if ACF_arr[j, index] < ACF_fin_check or ACF_uncert_arr[j, index] / ACF_arr[j, index] > ACF_uncert_cutoff:
                index_check = j
                break

        if i > 200:
            break;

        graph_file = new_folder + '/feature' + str(i + 1) + '.jpg'
        title = 'Feature ' + str(i + 1)
        x_text = 0.6 * x_data[index_check]
        y_text1 = 0.85 * ACF_arr[0, index]
        y_text2 = y_text1 - 0.05 * ACF_arr[0, index]
        y_text3 = y_text2 - 0.05 * ACF_arr[0, index]
        y_text4 = y_text3 - 0.05 * ACF_arr[0, i]

        plt.scatter(x_data[0:index_check], ACF_arr[0:index_check, index], label='Data')
        plt.plot(x_data[0:index_check],
                 KWW(x_data[0:index_check], results[i, 0], results[i, 1], results[i, 2]),
                 label='Fitted function')
        plt.plot(x_data[0:index_check], KWW(x_data[0:index_check], results[i, 0] + uncert[i, 0],
                                                results[i, 1] + uncert[i, 1], results[i, 2] + uncert[i, 2]),
                                                label='Fitted function + error')
        plt.plot(x_data[0:index_check], KWW(x_data[0:index_check], results[i, 0] - uncert[i, 0],
                                                results[i, 1] - uncert[i, 1], results[i, 2] - uncert[i, 2]),
                                                label='Fitted function - error')
        plt.xlabel('Time (s)')
        plt.ylabel('ACF')
        plt.title(title)
        plt.text(x_text, y_text1, 'A =' + str(results[i, 0])[0:5] + r'$(\pm)$' + str(uncert[i, 0])[0:7])
        plt.text(x_text, y_text2,
                 r'$\tau_{fit} =$' + str(results[i, 1])[0:5] + r'$(\pm)$' + str(uncert[i, 1])[0:7])
        plt.text(x_text, y_text3,
                 r'$\beta =$' + str(results[i, 2])[0:5] + r'$(\pm)$' + str(uncert[i, 2])[0:7])
        plt.text(x_text, y_text4, r'$r^{2} =$' + str(r2_arr[i][0])[0:7])

        if i < 200:
            plt.savefig(graph_file, dpi=300)

        # Display the first 5 graphs for reference during analysis
        if i == 0:
            print('\n Feature', i + 1, 'results: \n')
            plt.show()
            print('\nSaving all graphs... Please wait...\n')

        plt.close('all')

    return results, r2_arr, good_index, good_orig_indices

def tauc_equation(tauf, beta):

    """
    Equation to calculate tauc from taufit and beta via the gamma function
    
    Args: 
        tauf: (float) Taufit value of individual molecule
        beta: (float) Beta value of individual molecule
    
    Returns:
        tauc: (float) Tauc, average rotational timescale, of individual molecule
    """

    tauc = (tauf / beta) * special.gamma(1 / beta)

    return tauc

def calc_tauc(results_arr):

    """
    Calculate tauc for all molecules with good KWW fits
    
    Args:
        results_arr: (ndarray (num_good_feats, 3)) Array of KWW fit parameters
            A, taufit and beta for each molecule with a good fit. 
            
    Returns: 
        results_arr: (ndarray (num_good_feats, 4)) Array of KWW fit parameters
            A, taufit and beta as well as the computed tauc value for each
            molecule with a good fit. 
    """

    empty_col = np.zeros((1, 1))
    results_arr = np.insert(results_arr, 1, empty_col, axis=1)
    features = results_arr.shape[0]

    for i in range(features):
        tauf = results_arr[i, 2]
        beta = results_arr[i, 3]
        tauc = tauc_equation(tauf, beta)
        results_arr[i, 1] = tauc

    return results_arr

def quasi_ensemble_tau(ACF_arr, good_indices, min_points, tbf, folder):

    """
    Calculate Quasi-Ensemble ACF Values and Fit to KWW to Extract Quasi-Ensemble 
    Rotational Information
    
    Args: 
        ACF_arr: (ndarray (num_frames, num_Feats)) Array of all molecule
            ACFs
        good_indicies: (ndarray (m,)) Indices of all molecules that meet
            user requirements
        min_points: (int) Minimum number of points which must be fit for the
            KWW fit to be considered significant
        tbf: (float) Time-Between-Frames of movie
        folder: (str) Folder where all analyis outputs are being written
            
    Returns: 
        None
    """

    num_features = good_indices.shape[0]
    good_ACF_vals = np.zeros((min_points, num_features))

    for i in range(num_features):
        index = good_indices[i]
        good_ACF_vals[:, i] = ACF_arr[0:min_points, index]

    avg_ACF_vals = np.zeros((min_points))
    for i in range(min_points):
        avg_ACF_vals[i] = np.average(good_ACF_vals[i, :])

    x_data = np.arange(min_points)*tbf
    y_data = avg_ACF_vals

    p0 = [1, 1, 1]
    params, params_covariance = optimize.curve_fit(KWW, x_data, y_data, p0, max_nfev=10000,
                                                   bounds=([0.01, 0.01, 0.01], [2, np.inf, 2]))
    perr = np.sqrt(np.diag(params_covariance))

    graph_file = folder + r'\Quasi_Ensemble_KWW_fit.jpg'
    title = 'Quasi-Ensemble KWW Fit'
    x_text = x_data[min_points - 1] - 0.3 * x_data[min_points - 1]
    y_text1 = (y_data[min_points - 1] + 3 * y_data[0]) / 4
    y_text2 = (y_data[min_points - 1] + 2 * y_data[0]) / 3
    y_text3 = (y_data[min_points - 1] + 1.4 * y_data[0]) / 2.4

    plt.scatter(x_data, y_data, label='Data')
    plt.plot(x_data, KWW(x_data, params[0], params[1], params[2]), label='Fitted function')
    plt.plot(x_data, KWW(x_data, params[0] + perr[0], params[1] + perr[0], params[2] + perr[0]), label = 'Fitted function ' \
                                                                                                  '+ error')
    plt.plot(x_data, KWW(x_data, params[0] - perr[0], params[1] - perr[0], params[2] - perr[0]), label = 'Fitted function ' \
                                                                                                  '- error')
    plt.xlabel('Time (s)')
    plt.ylabel('ACF')
    plt.title(title)
    plt.text(x_text, y_text1, 'A = ' + str(params[0])[0:5]+ r'$(\pm)$' + str(perr[0])[0:7])
    plt.text(x_text, y_text2, r'$\tau_{fit} = $ ' + str(params[1])[0:5] + r'$(\pm)$' + str(perr[1])[0:7])
    plt.text(x_text, y_text3, r'$\beta = $' + str(params[2])[0:5] + r'$(\pm)$' + str(perr[2])[0:7])
    plt.savefig(graph_file, dpi=300)

    print('Show Quasi-Ensemble KWW Fit\n')
    plt.show()
    plt.close()

    tauf = params[1]
    beta = params[2]
    tauc = tauc_equation(tauf, beta)
    log_tauf = np.log10(tauf)
    log_tauc = np.log10(tauc)

    quasiEnsemblePath = folder + r'\Quasi_Ensemble_Values.csv'
    quasiEnsembleFile = open(quasiEnsemblePath, 'w')
    quasiEnsembleFile.write('Quasi-Ensemble:\nTauc: ' + str(tauc) + '\nTaufit: ' + str(tauf) + '\nBeta: ' + str(beta) +
                  '\nLog Tauc: ' + str(log_tauc) + '\nLog Taufit: ' + str(log_tauf))
    quasiEnsembleFile.close()

    print('Quasi-Ensemble:\nTauc: ' + str(tauc) + '\nTaufit: ' + str(tauf) + '\nBeta: ' + str(beta) +
                  '\nLog Tauc: ' + str(log_tauc) + '\nLog Taufit: ' + str(log_tauf) + '\n')

    return

def clean_coords(good_orig_indices, coordinate_int_arr, movie_type, filepath):

    """
    Generate an array of feature coordinates containing only those which 
    generated good results
    
    Args: 
        good_orig_indices: (ndarray ()) Array of all molecules which were
            note too close to the edge
        coordinate_int_arr: (ndarray (frame_number, good_particle, 3)) 
            Array with each molecule's (that is not too close to an edge to 
            integrate) y-coordinate, x-coordinate and integrated intensity values
        movie_type: (int) Type of Movie: 
            (1) Single-Channel for Intensity -- NOT USABLE
            (2) Two-Channel for Linear Dichroism
        filepath: (str) Filepath of .tif or .bin movie
        
    Returns: 
        good_coords: (ndarray ()) Array of the coordinates of all molecules
            which meet the specified KWW fit requirements
    """

    if movie_type == 2:
        num_coords = 4   # (y1, x1, y2, x2)
        coords = np.zeros((coordinate_int_arr.shape[1], 4))
        coords[:, 0:2] = coordinate_int_arr[0, :, 0:2]
        coords[:, 2:4] = coordinate_int_arr[0, :, 3:5]

    else:
        num_coords = 2   # (y, x)
        coords = np.zeros((coordinate_int_arr.shape[1], 2))
        coords = coordinate_int_arr[0, :, 0:2]

    num_features = good_indices.shape[0]
    good_coords = np.zeros((num_features, num_coords))

    for i in range(num_features):
        index = good_orig_indices[i]
        good_coords[i, :] = coords[index, :]

    coords_filepath = create_saveto_filepath(filepath, '_COORDS_CHECK.csv')
    good_coords_filepath = create_saveto_filepath(filepath, '_GOOD_COORDS_CHECK.csv')
    np.savetxt(coords_filepath, coords, delimiter=',')
    np.savetxt(good_coords_filepath, good_coords, delimiter=',')

    return good_coords

def output(results, r2_arr, coords, tbf, filepath, movie_type, summed_middle_frames):

    """
    Clean up the output and save a results .csv file
    
    Args: 
        results: (ndarray (num_good_feats, 4)) Array of KWW fit parameters
             A, taufit and beta as well as the computed tauc value for each
             molecule with a good fit.  
        r2_arr: (ndarray (good_feats,)) List of r^2 values for each KWW fit
        coords: (ndarray ()) Array of the coordinates of all molecules
            which meet the specified KWW fit requirements
        tbf: (float) Time-Between-Frames of movie
        filepath: (str) Filepath of .tif or .bin movie
        movie_type: (int) Type of Movie: 
            (1) Single-Channel for Intensity -- NOT USABLE
            (2) Two-Channel for Linear Dichroism
        summed_middle_frames: (ndarray (y_dim, x_dim)) Array containing the summed 
            pixel intensities for the middle N frames of the movie
        
    Returns: 
        None
    """

    output_arr = np.append(coords, results, axis=1)

    #   Remove duplicate pairs i.e.; situations where LC feature matches to more than 1 RC feature and vice versa

    print('Removing duplicate matches...\n')

    duplicate_indices = np.array([100000000])
    first_duplicate = 0
    for i in range(output_arr.shape[0]):
        for j in range(output_arr.shape[0] - i - 1):
            if (output_arr[i, 0] == output_arr[i + j + 1, 0] and output_arr[i, 1] == output_arr[i + j + 1, 1]) \
                    or (output_arr[i, 3] == output_arr[i + j + 1, 3] and output_arr[i, 4] == output_arr[i + j + 1, 4]):
                if first_duplicate == 0:
                    duplicate_indices = np.array([i, i + j + 1])
                    first_duplicate += 1

                else:
                    duplicate_indices = np.append(duplicate_indices, [i, i + j + 1], axis=0)

    if np.all((duplicate_indices != 100000000)):
        for i in range(duplicate_indices.shape[0]):
            index = duplicate_indices[i] - i
            output_arr = np.delete(output_arr, index, axis=0)

    # Add a plot of molecules giving good results. Han 9/24/2021

    selem = morphology.disk(3)
    res = morphology.white_tophat(summed_middle_frames, selem)

    title = 'Molecules with Good Results'
    color = 'red'
    fig, ax = plt.subplots(1, 1)
    ax.set_title(title)
    ax.imshow(res, cmap='gray')

    if movie_type == 2:
        for i in range(output_arr.shape[0]):
            y1, x1 = output_arr[i, 0], output_arr[i, 1]
            y2, x2 = output_arr[i, 2], output_arr[i, 3]
            c1 = plt.Circle((x1, y1), 2, color=color, linewidth=2, fill=False)
            c2 = plt.Circle((x2, y2), 2, color=color, linewidth=2, fill=False)
            ax.add_patch(c1)
            ax.add_patch(c2)

    else:
        for i in range(output_arr.shape[0]):
            y1, x1 = output_arr[i, 0], output_arr[i, 1]
            c1 = plt.Circle((x1, y1), 2, color=color, linewidth=2, fill=False)
            ax.add_patch(c1)

    ax.set_axis_off()
    final_features_filepath = create_saveto_filepath(filepath, '_final_features.jpg')
    plt.savefig(final_features_filepath, dpi=300)
    plt.show()
    plt.close()

    new_entries = np.zeros((output_arr.shape[0], 4))
    for i in range(output_arr.shape[0]):

        tauc = output_arr[i, -3]
        taufit = output_arr[i, -2]

        pts_per_taufit = taufit / tbf
        log_tauc = np.log10(tauc)
        log_taufit = np.log10(taufit)

        new_entries[i, 0] = pts_per_taufit
        new_entries[i, 1] = log_tauc
        new_entries[i, 2] = log_taufit
        new_entries[i, 3] = r2_arr[i]

    output_arr = np.append(output_arr, new_entries, axis=1)
    num_values = output_arr.shape[1]
    output_arr = np.insert(np.insert(output_arr, 0, 0, axis=0), 0, 0, axis=0)

    for i in range(num_values):
        output_arr[0, i] = np.median(output_arr[2:, i])

    # print('Median Values...\nA:', output_arr[0, -7], 'tauc:', output_arr[0, -6], 'taufit:', output_arr[0, -5],
    #       'beta:', output_arr[0, -4], 'pts/taufit:', output_arr[0, -3], 'log tauc:', output_arr[0, -2],
    #       'log taufit:', output_arr[0, -1])

    output_filepath = create_saveto_filepath(filepath, '_results.csv')
    if movie_type == 2:
        np.savetxt(output_filepath, output_arr, delimiter=',',
                   header='Y1, X1, Y2, X2, A, tauc, taufit, beta, pts/taufit, log tauc, log taufit, r^2')

    else:
        np.savetxt(output_filepath, output_arr, delimiter=',',
                   header='Y, X, A, tauc, taufit, beta, pts/taufit, log tauc, log taufit, r^2')

    return

"""

Run Analysis

"""

# Read Inputs and Load Movie
(filepath, filename, folder, tbf, movie_type, log_path) = fileinputs()
(movie_file, frame_number, y_dim, x_dim) = load_movie(filepath)

# Perform Baseline Subtraction
# movie_file = baseline_subtr(movie_file, frame_number, filepath)
#
# Find Features from Summed Middle Frames
summed_middle_frames = sum_middle_frames(movie_file, frame_number)
coord_arr = find_features(summed_middle_frames)

# Generate Intensity Arrays
(coordinate_int_arr, Int_values) = generate_intensity_arrays(coord_arr, x_dim, y_dim, frame_number, movie_file)

if movie_type == 2:
    coord_int_2ch = Set_Sep(coordinate_int_arr, filepath, log_path)
    LD_arr, LD_coords = calc_LD(coord_int_2ch, filepath)

# Compute ACF, KWW fit and Tauc
if movie_type == 1:
    data_to_fit = Int_values

else:
    data_to_fit = LD_arr

(ACF_arr, ACF_uncert_arr, ACF_indices, ACF_init_check, ACF_fin_check, ACF_uncert_cutoff, min_points) = \
    compute_ACF(data_to_fit, filepath, movie_type)

(kww_fit_arr, r2_arr, good_indices, good_orig_indices) = \
    fit_KWW(ACF_arr, ACF_uncert_arr, ACF_indices, tbf, filepath, ACF_init_check, ACF_fin_check, ACF_uncert_cutoff)
results_arr = calc_tauc(kww_fit_arr)

# Calculate Quasi-Ensemble Taus

quasi_ensemble_tau(ACF_arr, good_indices, min_points, tbf, folder)

# Clean up output
if movie_type == 2:
    good_coords = clean_coords(good_orig_indices, coord_int_2ch, movie_type, filepath)
else:
    good_coords = clean_coords(good_orig_indices, coordinate_int_arr, movie_type, filepath)

output(results_arr, r2_arr, good_coords, tbf, filepath, movie_type, summed_middle_frames)

print('Calculation complete!')