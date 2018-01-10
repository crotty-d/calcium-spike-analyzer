# -*- coding: utf-8 -*-
import os.path

class default_dir:
    """
    Default directories for different data types.
    
    The directories (class attributes) have been protected from modification by code outside the class via the
    __setattr__ method; however, you can change them by directly modifying the string arguments below.
    """
    
    # Module directory
    module = os.path.dirname(os.path.abspath(__file__))
    
    # Default data directory
    data = os.path.join(module, 'data')
    
    # Default results directory
    results = os.path.join(module, 'results')
    
    # Default directory for reference files used for testing
    reference = os.path.join(module, 'reference_files')
    
    # Protect attributes from modification by code outside the class
    def __setattr__(self, *_):
        pass