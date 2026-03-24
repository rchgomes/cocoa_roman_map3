from cosmosis.datablock import option_section, names
import numpy as np
from threepoint import ThreePointDataClass

def setup(options):
    config = {}
    config['fname_out'] = options.get_string(option_section, 'filename')
    config['fname_cov']= options.get_string(option_section, 'copy_covariance', default=None)
    return config

def execute(block, config):
    this_is_new = config['fname_cov'] is None

    # data class
    if this_is_new:
        data = ThreePointDataClass()
    else:
        data = ThreePointDataClass.from_fits(config['fname_cov'])

    # set values
    for scomb in block['map3', 'sample_combinations']:
        name = '_'.join([str(s) for s in scomb])
        # get values
        z1, z2, z3 = scomb
        t1, t2, t3 = block['map3', 'filters_'+name]
        map3 = block['map3', 'map3-bin_'+name]
        # determine where to set the map3
        if this_is_new:
            where = None
        else:
            where = data.where_to_set(z1, z2, z3, t1, t2, t3)
        # assign
        data.set_value(z1, z2, z3, t1, t2, t3, map3, where=where)

    print(f"Saving to {config['fname_out']}")
    data.to_fits(filename=config['fname_out'])
    
    return 0

def cleanup(config):
    pass
