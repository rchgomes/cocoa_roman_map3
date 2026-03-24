"""
This module convert natural-component (nc) of shear 3pcf to map3.
"""
from cosmosis.datablock import option_section, names
import numpy as np
from fast_map3 import calculateMap3

def setup(options):
    config = dict()
    return config

def execute(block, config):
    # Get bins for natural component predictions:
    section_nc = 'natural_components'
    phi = block[section_nc, 'phi']
    t1  = block[section_nc, 't1'] * 180*60/np.pi # in arcmin
    t2  = block[section_nc, 't2'] * 180*60/np.pi # in arcmin
    logr_bin_size = np.log(t2[1])-np.log(t2[0])
    phi_bin_size = phi[1]-phi[0]
    t1, t2, phi = np.meshgrid(t1, t2, phi, indexing='ij')

    # convert natural component to map3:
    for scomb in block["map3", "sample_combinations"]:
        if np.isscalar(scomb):
            name = str(scomb)
        else:
            name = '_'.join([str(s) for s in scomb])
        filters = block['map3', 'filters_'+name]
        gamma = block[section_nc, f'real-bin_{name}'] + 1j*block[section_nc, f'imag-bin_{name}']
        #map3 = calculateMap3(gamma, t2, t1, phi, logr_bin_size, phi_bin_size, filters)
        map3 = calculateMap3(gamma, t1, t2, phi, logr_bin_size, phi_bin_size, filters)
        block['map3', f'map3-bin_{name}'] = map3

    return 0

def cleanup(config):
    pass
