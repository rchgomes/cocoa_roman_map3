"""
Prepare the binning for the map3 signals.

Becasue the third order shear statistics is very computational expensive,
we first fix the combinations of redshift bins and angular bins for which
we want to calculate the third order shear statistics.
"""
from cosmosis.datablock import option_section, names
import numpy as np
import os
from threepoint import ThreePointDataClass

def get_string_array_1d(options, section, name):
    # I was not able to utilize
    # CosmoSIS python API 
    # options.get_string_array_1d.
    # This is tentative...
    o = options.get_string(section, name).split()
    o = [x for x in o if x]  # Remove empty strings
    return o

def selection_on_sample_combination(options, thpt):
    """
    sample_combinations must be list of followings:
    - all
    - auto
    - cross
    - #,#,#  (e.g. 1.1.1   or 1.2.3)
    """
    scombs = get_string_array_1d(options, option_section, 'sample_combinations')
    if len(scombs) == 0:
        scombs = ['all']
    
    n = np.max(thpt.get_z_bin(unique=True)) # number of zbins
    scombs2 = []
    for scomb in scombs:
        if scomb == 'all':
            for i in range(1,n+1):
                for j in range(i, n+1):
                    for k in range(j, n+1):
                        scombs2.append([i,j,k])
        if scomb == 'auto':
            for i in range(1,n+1):
                scombs2.append([i,i,i])
        if scomb == 'cross':
            for i in range(1,n+1):
                for j in range(i, n+1):
                    for k in range(j, n+1):
                        if not (i==j==k):
                            scombs2.append([i,j,k])
        if ',' in scomb:
            scombs2.append([int(i) for i in scomb.split(',')])
    scombs = np.array(scombs2)
    del scombs2

    print(scombs)

    print('Preselection on sample_combination', thpt.size)
    sel = np.zeros(thpt.size, dtype=bool)
    for scomb in scombs:
        sel |= thpt.selection_z_bin(scomb, 'z123', condition='==')
    thpt.replace(sel)
    print('Postselection on sample_combination', thpt.size)
    assert thpt.size >0 
    
    return thpt

def selection_on_theta(options, thpt):
    print('Preselection on theta', thpt.size)
    sel = np.ones(thpt.size, dtype=bool)
    if options.has_value(option_section, 'theta_filter_1_range'):
        tmin, tmax = options.get_double_array_1d(option_section, 'theta_filter_1_range')
        sel &= thpt.selection_SSS_bin([tmin, tmax], 'theta1', condition=['<=', '>='])
    if options.has_value(option_section, 'theta_filter_2_range'):
        tmin, tmax = options.get_double_array_1d(option_section, 'theta_filter_2_range')
        sel &= thpt.selection_SSS_bin([tmin, tmax], 'theta2', condition=['<=', '>='])
    if options.has_value(option_section, 'theta_filter_3_range'):
        tmin, tmax = options.get_double_array_1d(option_section, 'theta_filter_3_range')
        sel &= thpt.selection_SSS_bin([tmin, tmax], 'theta3', condition=['<=', '>='])
    thpt.replace(sel)
    print('Postselection on theta', thpt.size)
    assert thpt.size >0 
    return thpt

def setup(options):
    config = {}
    fname = options.get_string(option_section, "data_file")
    thpt = ThreePointDataClass.from_fits(fname)
    # Apply selection here
    thpt = selection_on_sample_combination(options, thpt)
    thpt = selection_on_theta(options, thpt)
    # get binning
    scombs  = thpt.get_z_bin(unique=True).T
    filters = {}
    for scomb in scombs:
        sel = thpt.selection_z_bin(scomb, 'z123', condition='==')
        name= '_'.join([str(s) for s in scomb])
        filters[name] = thpt.get_t_bin(sel=sel)
    # put them on config
    config['sample_combinations'] = scombs
    config['filters'] = filters

    return config

def execute(block, config):
    # map3:
    block['map3', 'sample_combinations'] = config['sample_combinations']
    for scomb in config['sample_combinations']:
        name= '_'.join([str(s) for s in scomb])
        block['map3', 'filters_{}'.format(name)] = config['filters'][name]
    # Because the natural components are also needed for map3
    # we need to inform the z-bin combination to the natural component
    block['natural_components', 'sample_combinations'] = config['sample_combinations']
    return 0

def cleanup(config):
    pass
