# -*- coding: utf-8 -*-
import os.path

"""
Default directories for different types of data used in the analysis package.
"""

# Module directory
module = os.path.dirname(os.path.abspath(__file__))

# Default data directory
data = os.path.join(module, 'data')

# Default results directory
results = os.path.join(module, 'results')

# Default directory for reference files used for testing
reference = os.path.join(module, 'reference_files')