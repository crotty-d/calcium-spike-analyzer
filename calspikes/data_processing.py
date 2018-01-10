# -*- coding: utf-8 -*-
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Functions for accessing and processing data
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Import external modules
import numpy as np


# ------------------------------------------------------------------------------
# Define module functions
# ------------------------------------------------------------------------------

def xydata_to_vectors(filepath, delimiter=None):
    """
    Produces two numpy arrays from a text file (txt, csv, etc.) of two-dimensional data (y versus x).
    
    Returns two 1D numpy arrays (vectors), the first containing the times
    when the signal values were recorded (units assumed to be seconds),
    and the second containing the corresponding signal values recorded.
    """
    # Convert text file to m x 2 numpy ndarray (m rows, 2 columns)
    data = np.genfromtxt(filepath, skip_header=1, usecols=(0,1), names=True, delimiter=delimiter)
    # Names of x (0) and y (1) columns
    colNames = data.dtype.names # tuple
    
    return np.array(data[colNames[0]]), np.array(data[colNames[1]])