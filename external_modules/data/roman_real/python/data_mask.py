"""
The elements of data vector used in the actual analysis are 
usually a subset of the full data vector that are contained in 
the data file.

The discarded data vectors are specified in the config and
the element selection is done in other modules (e.g. 2pt_like.py 
for two point likelihood).

This module is used to keep track of the mask of the data
vector elements that are or are not used in the actual analysis.


NOTE: This information had better be stored by the likelihood modules, 
        but they are not designed to do so. Henceforce, this module is
        a temporary solution to keep track of the used data vector 
        indices until the likelihood modules are updated.
"""
from cosmosis.datablock import option_section, names
import numpy as np
import os
from astropy.io import fits

def get_mask(signal_full, signal_used):
    out = []
    i = 0

    for name, data_vector in signal_full.items():
        _ = np.zeros(data_vector.size)
        for j in range(data_vector.size):
            if i>=len(signal_used):
                break
            if data_vector[j] == signal_used[i]:
                _[j] = True
                i += 1
        out.append(_)
    out = np.hstack(out)

    return out

def setup(options):
    """
    Necessary config parameters:
        data_file (str) : path to the data file
        data_sets (str) : names of the data sets in the data file
        like_name (str) : name of the likelihood with which this mask is associated
    """
    # read the config parameters
    data_file = options.get_string(option_section, 'data_file')
    data_sets = options.get_string(option_section, 'data_sets').split()
    like_name = options.get_string(option_section, 'like_name')

    # read the data file
    data = fits.open(data_file)
    signal_full = {}
    for data_set in data_sets:
        signal_full[data_set] = data[data_set].data['VALUE']

    # config
    config = {
        'signal_full': signal_full,
        'like_name': like_name
    }

    return config

def execute(block, config):
    # retrieve the full data vector
    signal_full = config['signal_full']
    like_name   = config['like_name']

    # get the data vector
    signal_used = block[names.data_vector, f'{like_name}_data']

    # get the indices of the used data vector elements
    indices = get_mask(signal_full, signal_used)

    # save the indices to the block
    block.put_int_array_1d(names.data_vector, f'{like_name}_mask', indices)

    return 0