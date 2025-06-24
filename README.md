# ASL_MRlinac_simulation

## Overview

Python implementation of point spread function model for flow alternating inversion recovery (FAIR) arterial spin labeling (ASL) MRI sequence with a balanced Steady State Free Precession (bSSFP) readout. A schematic of the pulse sequence is shown in the following figure.

| ![sequence_illustration_V003](https://github.com/user-attachments/assets/c0441c78-1e36-484a-9a53-f12686d3739f) |
|:--:| 
| A visualization of the pulse sequence consisting of FAIR perfusion preparation and bSSFP readout. α = flip angle, TI = inversion time, TR = repetition time, TM = measurement repetition time, t_acq = signal readout time period. The first signal acquisition only starts after a set of dummy pulses |

For more details on the implementation, check out the linked publication in the Credit section of this readme.
The original implementation the sequence is described by Martirosian, et al. (2004) in DOI: 10.1002/mrm.10709

_Dependencies:_

numpy,
Matplotlib

## Structure

fair_bssfp_simulation.py contains functions that simulate the magneitzation difference and PSF for the FAIR bSSFP sequence.

util.py contains functions for estimating arterial blood T1 and T2 at different field strengths and to calculate the FWHM of a function.

visualize.py contains a function to plot the signal curves.

simulations_for_publication.py is a script that runs fair bssfp simulations with different settings and determines the ideal parameters for a given input situation.

## Example output

The simulated PSFs for different flip angles may look a bit like this:

![PSF_varyFA_withidealsinc](https://github.com/user-attachments/assets/0fef0cd5-9db9-41d9-bd74-c95e98e46776)

## Credit

Created by Philip Wild and Philipp Wallimann, Department of Radiation Oncology, University Hospital Zürich.

__Manuscript in submission. A link will be provided here as soon as it is available.__

_Contact: philipp.wallimann@usz.ch_
