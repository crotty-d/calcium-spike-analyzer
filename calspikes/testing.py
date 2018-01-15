# -*- coding: utf-8 -*-
#  -*- coding: utf-8 -*-
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Functions for analyzing fluorescence intensity signals to identify and describe spikes in intraneuronal calcium.
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Import class containing default directories for input and output data
# Note: Can modify this file to change default directories
from directories import default_dir
# Import spikeAnalyzer module for accessing and processing data
import data_processing

# Import required external modules
import sys
import os.path
import numpy as np
import matplotlib.pylab as plt
from bokeh.plotting import figure, output_file, show
from bokeh.palettes import Spectral11


# ------------------------------------------------------------------------------
# Define module functions
# ------------------------------------------------------------------------------

def detection_tester(peakTimes, refFilename=None, error=0.35): # TODO: Move to testing module
    """
    Calculates percentage of true (verified) spikes that were detected along with the false spike rate (extra spikes per second of data).

    The detected spike times are compared to the actual spike times (determined by visual inspection)
    """

    if refFilename == None:
        print("True spike times were not provided, so the spike detection perfromance cannot be evaluated")
        percentTrueSpikes, falseSpikeRate = "N/A", "N/A"

        return {'percent_true_spikes': percentTrueSpikes, 'false_spike_rate': falseSpikeRate}

    else:
        # Create numpy array of the values in the reference csv file
        trueTimes = np.genfromtxt(refFilename, delimiter=',')

        # First match the two arrays of spike times. Anything within the given error is a match.

        # Ensure times are in sequntial order
        peakTimes = np.sort(peakTimes)
        trueTimes = np.sort(trueTimes)

        # Remove spikes with the same times (false spikes)
        detected = np.append(peakTimes, -1)
        uniqueDetected = peakTimes[plt.find(plt.diff(detected) != 0)]

        # Find matching spikes and mark as true detections
        trueDetected = [];
        # Find indices of dedected spikes that are within the margin of error around each true spike
        for spike in trueTimes:
            detectedWithinError = plt.find((uniqueDetected >= spike - error) & (uniqueDetected <= spike + error))
            # If detected spikes found...
            if len(detectedWithinError) > 0:
                # ...for each one, check if already present in our list of true dectections, ...
                for i in detectedWithinError:
                    alreadyMarked = plt.find(trueDetected == uniqueDetected[i])
                    # ...and if not, append it to to that list
                    if len(alreadyMarked) == 0:
                        trueDetected = np.append(trueDetected, uniqueDetected[i])

        percentTrueSpikes = 100.0 * len(trueDetected) / len(trueTimes)

        # Everything else is a false spike
        totalTime = (trueTimes[len(trueTimes) - 1] - trueTimes[0])
        falseSpikeRate = (len(peakTimes) - len(trueTimes)) / totalTime

        print("\nAction potential detector performance:")
        print("     Number of true spikes =", len(trueTimes))
        print("     Percentage of true spikes detected =", percentTrueSpikes)
        print("     False spike rate = ", falseSpikeRate, "spikes/s")

        return {'percent_true_spikes': percentTrueSpikes, 'false_spike_rate': falseSpikeRate}
    
def analysis_tester:
    pass

