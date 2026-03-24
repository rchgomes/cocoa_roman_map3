from cosmosis.datablock import option_section, names
import numpy as np
from threepoint import ThreePointDataClass
import scipy.linalg

def setup(options):
    # name of this likelihood
    name = options.get_string(option_section, "like_name", 'moped')

    # List of likelihood names
    likelihoods = options.get_string(option_section, "likelihoods").split()

    # Load transformation matrix
    # Assumes matrix shape is (data_dim, moped_dim), which is the output of moped_sampler.
    data_file = options.get_string(option_section, "data_file")
    transform_matrix = np.loadtxt(data_file).T

    # number of simulations
    nsim = options.get_int(option_section, 'covariance_realizations', -1)
    npar = options.get_int(option_section, "free_parameters", -1)

    # Possible option to add:
    # max_n = maximum number of MOPED mode to use
    # moped_index = index of the MOPED mode to use

    Percival = options.get_bool(option_section, "Percival", False)

    config = {"name":name, "likelihoods": likelihoods, "transform_matrix": transform_matrix, 'nsim':nsim, 'npar':npar, 'Percival':Percival}

    #If you want to directly input a compressed 2pt data file
    #(for example, a noisy realization generated with the compressed covariance)
    #Don't use the following if you want to use the full data vector as an input
    #and perform compression when running the pipeline

    compressed_2pt_data = None
    if options.has_value(option_section, "use_input_compressed_data") and options.get_bool(option_section,
                                                                                           "use_input_compressed_data"):
        compressed_2pt_data_file = options.get_string(option_section, "compressed_2pt_data_file")
        compressed_2pt_data = np.loadtxt(compressed_2pt_data_file)

    config['compressed_2pt_data'] = compressed_2pt_data


    return config

def execute(block, config):

    # Loop over likelihoods
    full_theo = []
    full_data = []
    full_cov  = []
    for likelihood in config["likelihoods"]:
        # Load data
        data = block[names.data_vector, likelihood+"_data"]
        theo = block[names.data_vector, likelihood+"_theory"]
        cov  = np.linalg.inv(block[names.data_vector, likelihood+'_inverse_covariance'])

        # append to full data/theory
        full_data.append(data)
        full_theo.append(theo)
        full_cov.append(cov)
    full_data = np.hstack(full_data)
    full_theo = np.hstack(full_theo)
    full_cov = scipy.linalg.block_diag(*full_cov)

    # Transform data and theory
    if config['compressed_2pt_data'] is None:
        transformed_data = np.dot(config["transform_matrix"], full_data)
    else:
        transformed_data = config["compressed_2pt_data"]
    transformed_theo = np.dot(config["transform_matrix"], full_theo)
    transformed_cov  = np.dot(config["transform_matrix"], np.dot(full_cov, config["transform_matrix"].T))
    transformed_icov = np.linalg.inv(transformed_cov)
    diff = transformed_data - transformed_theo

    Percival = config['Percival']
    if not Percival:
        # hartlap factor
        if config['nsim'] > 0:
            nsim = config['nsim']
            n = transformed_data.size
            f = (nsim-n-2)/(nsim-1)
            print(f'Hartlap {nsim} {n} {f} in moped like')
            transformed_icov *= f

        # MOPED modes are uncorrelated to each other and normalized
        # by construction, so the covariance matrix is unity.
        #chi2 = np.sum((transformed_data - transformed_theo)**2)*f
        chi2 = np.dot(diff, np.dot(transformed_icov, diff))

        # set values
        block[names.likelihoods, f'{config["name"]}_like'] = -0.5*chi2
        block[names.data_vector, f'{config["name"]}_chi2'] = chi2

    else:
        print('Using Percival likelihood.')
        n = transformed_data.size
        nsim = config['nsim']
        npar = config['npar']
        factor = (n-npar)*(nsim-n-2)/((nsim-n-1)*(nsim-n-4))
        m_power = npar + 2 + (nsim-1+factor)/(1+factor)
        chi2 = np.dot(diff, np.dot(transformed_icov, diff))
        like = -m_power/2*np.log(1+(chi2/(nsim-1)))
        block[names.likelihoods, f'{config["name"]}_like'] = like
        block[names.data_vector, f'{config["name"]}_chi2'] = chi2


    block[names.data_vector, f'{config["name"]}_data'] = transformed_data
    block[names.data_vector, f'{config["name"]}_theory'] = transformed_theo
    block[names.data_vector, f'{config["name"]}_inverse_covariance'] = transformed_icov
    block[names.data_vector, f'{config["name"]}_transform_matrix'] = config["transform_matrix"]

    return 0

def cleanup(config):
    pass
