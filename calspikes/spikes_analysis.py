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

def spikeDetect(time, sig):
    """
    Detects spikes (peaks) in a signal representing calcium-dependent fluorescence intensitiy versus time.

    Parameters:
    time - 1D numpy array (vector) where each number is a time in seconds.
    sig - 1D numpy array (vector) where each number is the signal value at a different time.

    At a given index of either vector, the time and signal values are assumed to
    be in correspondence so the vectors must be the same size. The peak times, amplitudes
    and rise times are returned as 3 numpy arrays.

    The function will not work well if the baseline of the signal is not close to flat.
    """

    # Initialize return variables
    peakTimes = []
    amps = []
    riseTs = []

    # Check that the input arrays representing the signal and time data are 1D Numpy arrays of appropriate lengths (equal and > 2).
    if type(time) != np.ndarray or type(sig) != np.ndarray or len(np.shape(time)) != 1 or len(np.shape(sig)) != 1:
        print("Error:", __name__, "takes 1D Numpy arrays representing the signal and time data as arguments.")
    elif len(sig) != len(time):
        print("Error: The lengths of the vectors (1D Numpy arrays) representing the signal and time data must be the same.")
    elif len(sig) < 2 or len(time) < 2:
        print("Error: The vectors (1D Numpy arrays) representing the signal and time data must have at least two data points (length > 1).")
    else:
        # Get sample rate of data (per s); total number of samples divided total time (final value of time array)
        SR = int(round(len(sig) / time[len(time) - 1]))

        # Normalize signal using max and min:
        # 1. Subtract baseline (subtract min value from all)
        sig = sig - min(sig)
        # 2. Divide by max of new sig array (will equal 1)
        sig = sig / max(sig)

        # Rough approximation of noise in signal based on first 9 samples (prior to stimulation of neurons).
        # Simple range (max -  min) was more reliable estimator than standard deviation or similar owing
        # to small number of samples. Main purpose of noise value is to exclude peaks in data files where
        # noise peaks and the max peak are similar in magnitude, i.e. signal is all noise with no detectable spikes.
        noise = max(sig[0:10]) -  min(sig[0:10])

        # Tolerance of small dips in the signal during rise
        tolerance = 0.03

        # Then get spike (peak) times and properties like the spike amplitude and rise time.
        i = 0
        while i < len(time) - 2: # Scan along signal trace...
            if sig[i+1] > sig[i]:
                # If find potential spike record the index (i) where it begins (onset)
                onsetIdx = i
                # While the signal is rising, increment the index, but tolerate small dips in the signal along the way
                while (sig[i+1] > sig[i] or sig[i] - sig[i+1] < tolerance) and i < len(time) - 2:
                    i += 1 # move on 1 time step

                # Find the index of the actual peak by getting the max signal value between spike onset and i.
                # The tolerance of small dips means that a number of indices prior to the final value of i
                # may correspond to the actual peak signal value.
                potentialPeaks = sig[onsetIdx:i+1][::-1] # reversed to facilitate peakIdx calculation below (sig[0] = i, sig[1] = i-1, ...)
                potentialPeaksIdx = potentialPeaks.argmax() # index of highest potetial peak
                peakIdx = i - (potentialPeaksIdx)

                # Calculate amplitude and rise time of spike
                amp = sig[peakIdx] - sig[onsetIdx]
                riseT = time[peakIdx] - time[onsetIdx]

                # Threshold values of properties that indicate a spike.
                localWindow = sig[max(int(i - 10*SR), 0):int(min(i + 10*SR, len(sig)))] # max/min functions to handle windows close to array ends
                localMax = max(localWindow)
                if amp > 1.7 * noise and amp > 0.34 * localMax and riseT < 2.0 and (len(peakTimes) == 0 or time[peakIdx] - peakTimes[-1] > 2):
                    amps = np.append(amps, amp)
                    riseTs = np.append(riseTs, riseT) # TODO: Check if most efficient method as np.append creates copy of the array with the new values appended (not in place)
                    peakTimes = np.append(peakTimes, time[peakIdx])
            else:
                i += 1

    return peakTimes, amps, riseTs

def threshFind(peakTimes, initDelay = 1.4, stimPeriod = 12, initStim = 200, endStim = 800, stimIncr = 100):
    """
    Get time when first spike detected and check which 12 s (3s x 4) stimulation period it occured in.

    Starting at 200 V/m, E rises 100 V/m each interval up to 800 V/m.
    """
    # E pulse parameters
    totalTime = initDelay + stimPeriod
    stim = initStim

    # Determine which stimulation period (level) contains the first spike.
    if len(peakTimes) > 0:
        while stim <= endStim:
            if np.any(peakTimes < totalTime):
                return stim
            else:
                stim = stim + stimIncr
                totalTime = totalTime + stimPeriod
    else:
        return None


def results(t, sig, singleSpike_values=False):
    """
    Calculates the stimulation threshold and various statistics for the spike pattern.

    The previously defined spikeDetect and stimThresh functions are used to get
    the results which are returned as a dictionary.
    """
    peakTimes, amps, riseTs = spikeDetect(t, sig)

    stats = {'spike_count': len(peakTimes), 'stimulation_threshold': threshFind(peakTimes), 'mean_amplitude': np.mean(amps), 'amplitude_sd': np.std(amps), 'mean_rise_time': np.mean(riseTs), 'rise_time_sd': np.std(riseTs)}

    if singleSpike_values:
        return stats, peakTimes, amps, riseTs
    else:
        return stats


def detector_tester(peakTimes, refFilename=None, error=0.35): # TODO: Move to testing module
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


def plot_spikes(time, sig, peakTimes, title="Plot of signal versus time", xLabel="Time (s)", yLabel="Signal", outdir=default_dir.results, outHTML=None, plotter='all'):
    """
    The function creates a labeled plot showing the normalized sig signal
    and indicating the location of detected spikes with a marker above the spike.

    It takes four arguments - the array of recording times, the signal array,
    the time of the detected action potentials, and the title of your plot.
    """

    # First, zero signal baseline (subtract min value from all)
    sig = sig - min(sig)

    if plotter == 'matplotlib' or plotter == 'all':
        # Create figure to contain plot
        fig = plt.figure()

        # Create plot
        plt.plot(time, sig, 'b')
        plt.xlabel(xLabel)
        plt.ylabel(yLabel)
        plt.xticks(np.arange(min(time), max(time) + 1, 2.0))
        plt.title(title)

        # Spike markers

        # Define vertical tick y-position and length relative to max peak height.
        # x and y values in the plot() function below must be of the same dimension;
        # hence, np.ones() is used to create a 1D array equal in length to spikeTime
        # (x values) and containing only the desired y value (a little above max sig) repeated over and over.
        marker_y = np.ones(len(peakTimes)) * (max(sig) * 1.1)
        marker_x = peakTimes
        # Add markers at each spike
        plt.plot(marker_x,marker_y,'r|', markeredgewidth=1, markersize=100)

        # Display Matplotlib figure
        fig.show()

    # Optional plot in web browser using Bokeh (also saves a HTML file)
    if outHTML != None:
        # Create HTML file
        output_file(os.path.join(outdir, outHTML))

        p = figure(plot_width=1000, plot_height=400, title=title, x_axis_label=xLabel, y_axis_label=yLabel)

        # Render signal in plot
        p.line(time, sig, line_width=2)

        # Add markers to mark detected spikes
        marker_y = [sig[np.where(time == i)[0][0]] + (max(sig) * 0.04) for i in peakTimes]
        p.inverted_triangle(marker_x, marker_y, color='red', size=7)

        if plotter == 'bokeh' or plotter == 'all':
            show(p)


def plot_waveforms(time, sig, peakTimes, title="Waveforms of signal spikes", xLabel="Time (s)", yLabel="Signal", outdir=default_dir.results, outHTML=None, plotter='all'):
    """
    The function creates a labeled plot showing the waveforms for each
    signal spike.

    The times of the spikes must be provided as a list using the 'peakTimes' parameter,
    since the function does not detect spikes; it just displays them.
    """

    # Get sample rate of data (per s); total number of samples divided total time (final value of time array)
    SR = int(round(len(sig) / time[len(time) - 1]))

    # Index for time of spike (= index of signal data [signal/time, i.e. y/x] at that time).
    peakIndices = []
    for peakT in peakTimes:
        peakIndices = np.append(peakIndices, plt.find(time==peakT))

    # Make array of appropriate x values (time window for a waveform)
    waveformTime = np.linspace(-1, 2, num=3*SR)

    # Option: Create figure to contain plot
    if plotter == 'matplotlib' or plotter == 'all':
        fig = plt.figure()

        # Plot each spike waveform: y from spikeindex minus number of indices equivalent 1 s to spikeindex plus number of indices equivalent 2 s
        plt.xlabel(xLabel)
        plt.ylabel(yLabel)
        plt.title(title)

        for i in peakIndices:
            startIdx = int(i - 1*SR)
            endIdx = int(i + 2*SR)
            waveformSig = sig[startIdx:endIdx]
            # Zero signal baseline (subtract min value from all)
            waveformSig = waveformSig - min(waveformSig)
            # Create plot
            plt.plot(waveformTime, waveformSig, '-')

            #Display Matplotlib figure
            fig.show()

    # Option: plot in web browser using Bokeh (also saves a HTML file)
    if outHTML != None:
        # Create HTML file
        output_file(os.path.join(outdir, outHTML))

        p = figure(plot_width=900, plot_height=600, title=title, x_axis_label=xLabel, y_axis_label=yLabel)

        # Initialize lists of waveforms for Bokeh browser plot
        times = sigs = []

        # Create waveform input arrays for plt
        numLines = len(peakTimes)
        times = [waveformTime] * numLines
        sigs = [sig[int(i - 1*SR):int(i + 2*SR)] for i in peakIndices]
        # Zero signal baseline (subtract min value from all)
        sigs = [s - min(s) for s in sigs]

        # Render signal trace in plot
        palette = Spectral11[0:numLines]
        p.multi_line(times, sigs, line_width=2, line_color=palette)

        if plotter == 'bokeh' or plotter == 'all':
            show(p)


def spikes_analyze(filename, indir=default_dir.data, outdir=default_dir.results, refdir=default_dir.reference, refFile=None, plotter='all'):
    """
    Analysis and plot (complete signal and spike waveforms) for a single data file

    To automatically save plots to files (HTML) in directory, plotter must equal 'bokeh' or 'all'.
    plotter='matplotlib' to display using matplotlib only.

    If you have a csv file of validated times for the signal being analyzed, assign it
    as the value of the refFile parameter. Otherwise perfromance testing will be skipped.
    """

    # Input filepath
    inFilePath = os.path.join(indir, filename)

    # Load file data into two numpy arrays
    t, sig = data_processing.xydata_to_vectors(inFilePath)

    print("\nFile analyzed:", filename)

    # Analyze signal and store results in variables
    stats, peakTimes, amps, riseTs = results(t, sig, singleSpike_values=True)

    # Plot complete signal and individual spikes
    if plotter == 'bokeh' or plotter == 'all':
        spikesfile = filename[0:filename.index('.')] + '_wholeplot.html'
    else:
        spikesfile = None

    plot_spikes(t, sig, peakTimes, title="Intracellular calcium spikes", yLabel="Relative intracellular calcium (RFU)", outdir=outdir, outHTML=spikesfile, plotter=plotter)

    # Plots plots overlaid close-up views of all spike waveforms with their peaks centred on 0.
    # Use specific indices or slices of the peakTimes argument to show only a selection waveforms.
    # e.g. peakTimes=peakTimes[0:10] or peakTimes=peakTimes[0::4] for first 10 or every 4th spike, respectively.
    if plotter == 'bokeh' or plotter == 'all':
        wavesfile = filename[0:filename.index('.')] + '_waveforms.html' # TODO: Currently html file is always overwritten -- provide option and/or add timestamp to filename
    else:
        wavesfile = None

    plot_waveforms(t, sig, peakTimes=peakTimes, title="Waveforms of each spike in intracellular calcium", yLabel="Relative intracellular calcium (RFU)", outdir=outdir, outHTML=wavesfile, plotter=plotter)

    # Output spike times aquired above as csv
    outFilePath = os.path.join(outdir, filename[0:filename.index('.')] + '_times.csv')
    np.savetxt(outFilePath, peakTimes, delimiter=',')

    print("\nSpike times (s):", peakTimes, "\n\nSpike amplitudes (RFU):", amps, "\n\nSpike rise times (RFU/s):", riseTs)
    print("\nNumber of spikes:", stats['spike_count'])

    # Optional: Test spike detection performance against validated times for the same signal.
    if refFile:
        refFilePath = os.path.join(refdir, refFile)
        detector_tester(peakTimes, refFilePath)

    # Output analysis results
    print("\nStimulation threshold: %s V/m." % stats['stimulation_threshold'])
    print("\nAverage spike amplitude:", stats['mean_amplitude'], "+/-", stats['amplitude_sd'], "RFU.")
    print("Average spike rise time:", stats['mean_rise_time'], "+/-", stats['rise_time_sd'], "s.")
    print()


# ------------------------------------------------------------------------------
# Main code
# ------------------------------------------------------------------------------

def main():
    """Main code to call above module functions"""
    spikes_analyze('599region_A.txt', refFile='599regionA_ref_times.csv', plotter='all')

    # Exit program (try/except to avoid exception message iPython console)
    try:
        sys.exit(0)
    except SystemExit:
        print("\n++++++++ ANALYSIS COMPLETE +++++++++\n")


if __name__ == '__main__':

    main()