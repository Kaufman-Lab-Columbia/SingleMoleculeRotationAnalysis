# SingleMoleculeRotationAnalysis
This repository contains Python scripts used to analysis single molecule rotational dynamics from either linear dichroism or single channel fluorescence intensity fluctuation measurements.

# Getting Started

## Dependencies
- NumPy
- Pandas
- Matplotlib
- scikit-image
- scikit-learn
- statsmodels
- SciPy

# Basic Usage
## Rotational Analysis with Tracked Particles
This method of rotational analysis is suitable for extracting rotational timescales from either two-channel linear dichroism measurements or single-channel intensity measurements. In the case of two-channel measurements it is necessary to, prior to analysis, "book-match" the left and right channel of pixel coordinates of the movie by cropping them in ImageJ. This is done to achieve the following: 
- The y-offset of a molecule in the left-channel to that in the right-channel is 0-pixels i.e.; they are at the same y-pixel value.
- The x-offset of a molecule in the left-channel to that in the right-channel is half the width of the two channels laid next to one another.
These two bookmatched channels should also be summed into an effective single-channel movie for the translational tracking and linking of trajectories. 

Analysis of rotational dynamics using *RotationalAnalysis_LoadTrajectories.py* requires a *.csv* file containing the xy-coordinates of each molecules for all frames of the movie with NaNs indicating frames in which the molecule could not be localized. These can be generated either from simulations or from tracking experimental data using the ImageJ ParticleTracker plugin. The ImageJ THUNDERSTORM plugin may also be used. 

In the case of trajectories tracked with ParticleTracker or THUNDERSTORM the user must first link these trajectories using the *TranslationalAnalysis_HierarchicalClustering.py* script provided in this repository. This script will output a file with the ending *_filtered_trajectories.csv* which will be used as an input for the *RotationalAnalysis_LoadTrajectories.py* script. 

The *RotationalAnalysis_LoadTrajectories.py* script takes the *.tif* (or *.bin*) movie file as well as the *.csv* file containing the xy-trajectories of each molecule. The user will also be prompted with several other parameters during the analysis based on the conditions of the experiment and the checks on KWW functional fitting to the autocorrelations computed. 

## Rotational Analysis with Static Window
This method of rotational analysis is suitable **only** for two-channel linear dichroism measurments. It should **NOT** be used for any single-channel intensity measurements. This method is not recommended and, if possible, the above method for *Rotational Analysis with Tracked Particles* is suggested. 

With this method no translational tracking is done and instead molecules are localized by summing some number of frames from the middle of the movie and identifying particle locations from this summed image. This mimics the general analysis procedure previously implemented in IDL as used in other publications from the Kaufman Lab at Columbia University. 

The *Rotational_Analysis_StaticWindow.py* script takes only the *.tif* (or *.bin*) as a file input along with numerous use prompts regarding conditions of the experiment, feature finding and the checks on KWW functional fitting to the autocorrelations computed. 

# License
SingleMoleculeRotationAnalysis is licensed with an MIT license. See LICENSE file for more information.

# Referencing
If you use SingleMoleculeRotationAnalysis for your work, cite it with the following:
```
Alec R. Meacham, Jaladhar Mahato, Han Yang, and Laura J. Kaufman
The Journal of Physical Chemistry B 2024 128 (38), 9233-9243
DOI: 10.1021/acs.jpcb.4c02097
```

# Contact
No Current Contact
