from cosmosis.datablock import option_section, names
import numpy as np
import os

def setup(options):
    config = {}
    return config

def execute(block, config):
    for scomb in block["map3", "sample_combinations"]:
        mbias = np.array([block['shear_calibration_parameters', 'm{}'.format(s)] for s in scomb])
        factor = np.prod(1+mbias)
        name = '_'.join([str(s) for s in scomb])
        block['map3', f'map3-bin_{name}'] *= factor
    return 0

def cleanup(config):
    pass