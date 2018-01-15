# -*- coding: utf-8 -*-
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Functions for teting calcium spike analysis
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Import class containing default directories for input and output data
# Note: Can modify this file to change default directories
from directories import default_dir
# Import module containing functions to convert and analyze file
import spikes_analysis
import data_processing

# Import required external modules
import sys
import os.path
import numpy as np
import matplotlib.pylab as plt


# ------------------------------------------------------------------------------
# Define module functions
# ------------------------------------------------------------------------------

def detection_tester(peakTimes, refFilepath=None, delimiter=None, tolerance=0.35):
    """
    Compares spike times from the spike_detector function to the actual spike times.
    
    The actual (validated) times should be listed in a reference text file located at
    refFilepath; set delimiter via the corresponding paramter, e.g. ',' for CSV.
    
    The function ouutputs and returns the percentage of true (verified) spikes that
    were detected along with the false spike  rate (extra spikes per second of data).
    """

    if refFilepath == None:
        print("True spike times were not provided, so the spike detection perfromance cannot be evaluated")
        percentTrueSpikes, falseSpikeRate = "N/A", "N/A"

        return {'percent_true_spikes': percentTrueSpikes, 'false_spike_rate': falseSpikeRate}

    else:
        # Create numpy array of the values in the reference csv file
        trueTimes = np.genfromtxt(refFilepath, delimiter=delimiter)

        # First match the two arrays of spike times. Anything within the given tolerance is a match.

        # Ensure times are in sequntial order
        peakTimes = np.sort(peakTimes)
        trueTimes = np.sort(trueTimes)

        # Remove spikes with the same times (false spikes)
        detected = np.append(peakTimes, -1)
        uniqueDetected = peakTimes[plt.find(plt.diff(detected) != 0)]

        # Find matching spikes and mark as true detections
        trueDetected = [];
        # Find indices of dedected spikes that are within the margin of tolerance around each true spike
        for spike in trueTimes:
            detectedWithinTol = plt.find((uniqueDetected >= spike - tolerance) & (uniqueDetected <= spike + tolerance))
            # If detected spikes found...
            if len(detectedWithinTol) > 0:
                # ...for each one, check if already present in our list of true dectections, ...
                for i in detectedWithinTol:
                    alreadyMarked = plt.find(trueDetected == uniqueDetected[i])
                    # ...and if not, append it to to that list
                    if len(alreadyMarked) == 0:
                        trueDetected = np.append(trueDetected, uniqueDetected[i])

        percentTrueSpikes = 100.0 * len(trueDetected) / len(trueTimes)

        # Everything else is a false spike
        totalTime = (trueTimes[len(trueTimes) - 1] - trueTimes[0])
        falseSpikeRate = (len(peakTimes) - len(trueTimes)) / totalTime

        print("\nSpike detector performance:")
        print("\n     Number of spikes detected in test analysis =", len(peakTimes))
        print("     Number of true spikes =", len(trueTimes))
        print("     Percentage of true spikes detected =", percentTrueSpikes)
        print("     False spike rate = ", falseSpikeRate, "spikes/s")

        return {'percent_true_spikes': percentTrueSpikes, 'false_spike_rate': falseSpikeRate}


def analysis_tester(testFilepath, refFilepath):
    """
    Runs an analysis of a file and compares the results to those from a validated refernce file. 
    """
    
    # Load file data into two numpy arrays
    t, sig = data_processing.xydata_to_vectors(testFilepath)
    
    # Call analysis function
    peakTimes = spikes_analysis.spike_detect(t, sig)[0]
    
    # Call tester function
    detection_tester(peakTimes, refFilepath, delimiter=',')


# ------------------------------------------------------------------------------
# Main code
# ------------------------------------------------------------------------------

def main():
    """Main code calling above module functions"""
    
    # Locate testing data
    testFiledir = default_dir.data
    refFiledir = default_dir.reference
    testFilename = '599region_A.txt'
    refFilename = '599regionA_ref_times.csv'
    testFilepath = os.path.join(testFiledir, testFilename)
    refFilepath = os.path.join(refFiledir, refFilename)
    
    # Run test
    analysis_tester(testFilepath, refFilepath)

    # Exit program (try/except to avoid exception message iPython console)
    try:
        sys.exit(0)
    except SystemExit:
        print("\n++++++++ TESTING COMPLETE +++++++++\n")


if __name__ == '__main__':

    main()
    

