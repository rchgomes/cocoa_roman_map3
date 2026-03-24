"""
Prepare the binning for the map3 signals.

Becasue the third order shear statistics is very computational expensive,
we first fix the combinations of redshift bins and angular bins for which
we want to calculate the third order shear statistics.

This is to output the map3 without line-of-sight integration, zslice.
This is used to generate the emulator withouy los integration, which can
be taken into account easily after emulation.
"""
from cosmosis.datablock import option_section, names
import numpy as np
import os
from threepoint import ThreePointDataClass

def setup(options):
    config = {}
    scombs = options.get_double_array_1d(option_section, "sample_combinations")
    
    filter_1 = options.get_double_array_1d(option_section, "theta_filter_1")
    filter_2 = options.get_double_array_1d(option_section, "theta_filter_2")
    filter_3 = options.get_double_array_1d(option_section, "theta_filter_3")

    # We assume we use the same filters for all thee z-bin combinations
    filters = {}
    for scomb in scombs:
        name = str(scomb)
        filters[name] = np.array([filter_1, filter_2, filter_3])

    # put them on config
    config['sample_combinations'] = scombs
    config['filters'] = filters

    return config

def execute(block, config):
    # map3:
    block['map3', 'sample_combinations'] = config['sample_combinations']
    for scomb in config['sample_combinations']:
        name= str(scomb)
        block['map3', 'filters_{}'.format(name)] = config['filters'][name]
    # Because the natural components are also needed for map3
    # we need to inform the z-bin combination to the natural component
    block['natural_components', 'sample_combinations'] = config['sample_combinations']
    return 0

def cleanup(config):
    pass
