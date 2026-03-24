from cosmosis.datablock import option_section, names
import numpy as np
from threepoint import ThreePointDataClass

def setup(options):
    config = dict()
    config['data'] = ThreePointDataClass.from_fits(options.get_string(option_section, "data_file"))
    config['nsim'] = options.get_int(option_section, "covariance_realizations", -1)
    config['npar'] = options.get_int(option_section, "free_parameters", -1)

    Percival = options.get_bool(option_section, "Percival", False)

    config['Percival'] = Percival

    return config

def execute(block, config):
    data = config['data'].copy()
    model= config['data'].copy()

    used = np.zeros(data.size, dtype=bool)
    for scomb in block['map3', 'sample_combinations']:
        name = '_'.join([str(s) for s in scomb])
        # get values
        z1, z2, z3 = scomb
        t1, t2, t3 = block['map3', 'filters_'+name]
        map3 = block['map3', 'map3-bin_'+name]
        # determine where to set the map3
        where = model.where_to_set(z1, z2, z3, t1, t2, t3)
        # assign
        model.set_value(z1, z2, z3, t1, t2, t3, map3, where=where)
        # mark as used
        used[where] = True

    # restrict oursefves to the elements that were used
    model.replace(used)
    data.replace(used)
    
    # compute chi2
    map3_data = data.get_signal()
    map3_model= model.get_signal()
    diff = map3_data - map3_model
    icov = data.get_inverse_covariance(Hartlap=False) # turn off Hartlap internal function

    Percival = config['Percival']

    if not Percival:
        # Anderson-Hartlap factor
        if config['nsim'] > 0:
            nsim = config['nsim']
            n = icov.shape[0]
            f = (nsim-n-2)/(nsim-1)
            icov*= f

        # Dodelson-Schneider factor
        if config['npar'] > 0 and config['nsim'] > 0:
            npar = config['npar']
            n = icov.shape[0]
            f2 = 1/(1 + (n-npar)*(nsim-n-2)/((nsim-n-1)*(nsim-n-4)))
            print(f'Dodelson-Schneider {nsim} {n} {npar} {f2}')
            icov *= f2

        chi2 = np.matmul(diff, np.matmul(icov, diff))
        block[names.likelihoods, 'map3_like'] = -0.5 * chi2
        block[names.data_vector, 'map3_chi2'] = chi2

    else:
        print('Using Percival likelihood')
        n = icov.shape[0]
        nsim = config['nsim']
        npar = config['npar']
        factor = (n-npar)*(nsim-n-2)/((nsim-n-1)*(nsim-n-4))
        m_power = npar + 2 + (nsim-1+factor)/(1+factor)
        chi2 = np.matmul(diff, np.matmul(icov, diff))
        like = -m_power/2*np.log(1+(chi2/(nsim-1)))
        block[names.likelihoods, 'map3_like'] = like
        block[names.data_vector, 'map3_chi2'] = chi2

    block[names.data_vector, 'map3_n'] = diff.size

    # append data vector
    block[names.data_vector, 'map3_data'] = map3_data
    block[names.data_vector, 'map3_theory'] = map3_model
    block[names.data_vector, 'map3_inverse_covariance'] = icov

    return 0

def cleanup(config):
    pass
