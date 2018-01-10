#  -*- coding: utf-8 -*-
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Functions for batch anlaysis of sets (directories) of data files
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Import class containing default directories for input and output data
# Note: Can modify this file to change default directories
from directories import default_dir
# Import module containing functions to convert and analyze file
import spikes_analysis
import data_processing

# Import external modules
import sys
import os
import numpy as np
import pandas as pd
import time
import inspect
import importlib# TODO: Remove from final version
# Reload module to ensure latest changes are included
importlib.reload(spikes_analysis)# TODO: Remove from final version


# ------------------------------------------------------------------------------
# Define module functions
# ------------------------------------------------------------------------------

def batch_analysis(analysis, dataDirectory=default_dir.data, filetype='.txt', delimiter=None, createCSV=True): # TODO: Add batch plotting to files
    '''
    Walks through the directory and analyses all files of a certain type using a specified function (analysis).

    Fileype argument must be string representing chosen file extension including '.' (e.g. '.txt', '.csv', etc.).
    'directory' argument is also a string.
    '''

    print("Batch analysis called")
    np.warnings.filterwarnings('ignore')

    # Initialize dictionary to hold the analysis results for each file
    # (key is filename base, value is output of analysis)
    resultDict = {}

    # Walk through given (above) directory containing the data files and if they
    # are of the given file type, apply the given analysis (function)
    for root, dirs, files in os.walk(dataDirectory):
        for filename in files:
            base, ext = os.path.splitext(filename)
            if ext == filetype:
                filepath = os.path.join(root, filename)
                x, y = data_processing.xydata_to_vectors(filepath, delimiter=delimiter)
                resultDict[base] = analysis(x, y)

    # Check that at least one appropriate file has been found and analyzed; if not, print error.
    if len(resultDict) < 1:
        print('Error: No', filetype[1:], 'files present in the', dataDirectory, 'directory.')
    else:
        # Create Pandas DataFrame from results dictionary above
        results = pd.DataFrame.from_dict(resultDict, orient='index')
        # Create csv file from this DataFrame unless 'createCSV' argument is False'
        if createCSV:
            dirResults = default_dir.results
            outFilename = 'batch_' + inspect.getmodule(analysis).__name__ + '(' + time.strftime("%Y-%m-%d_%H.%M.%S", time.localtime()) + ').csv'
            outFilepath = os.path.join(dirResults, outFilename)
            results.to_csv(outFilepath)

        print("\nBATCH ANALYSIS COMPLETE")
        print("\nOutput files saved to: ", outFilepath)
        return results


# ------------------------------------------------------------------------------
# Main code
# ------------------------------------------------------------------------------

def main():
    """ Main code to use module functions as a standalone program"""

    # Run batch analysis of directory
    batch_analysis(spikes_analysis.results)

    # Exit program (try/except to avoid exception message iPython console)
    try:
        sys.exit(0)
    except SystemExit:
        print("\n++++++++ PROGRAM FINISHED SUCCESSFULLY ++++++++\n")

if __name__ == '__main__':

    main()