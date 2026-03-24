"""
Returns zero for prior-only chain
"""
from cosmosis.datablock import option_section, names
import numpy as np
import os
from astropy.io import fits
import scipy.linalg
from time import time


def setup(options):

    name = options.get_string(option_section, "like_name", 'moped')
    likelihoods = options.get_string(option_section, "likelihoods").split()

    config = {'name':name, 'likelihoods': likelihoods}

    return config

def execute(block, config):
    block[names.likelihoods, f'{config["name"]}_like'] = 0.0
    return 0

def cleanup(config):
    pass
