"""
Compute the joint likelihood using the existing data entries 
in the data block of cosmosis.

If the data vectors are uncorrelated, we do not have to use
the joint likelihood, and we can simply sum up the individual
log likelihoods. So this joint likelihood module is especially
useful when the data vectors are correlated.
"""
from cosmosis.datablock import option_section, names
import numpy as np
import os
from astropy.io import fits
import scipy.linalg
from time import time

def read_covariance(filename, covmat_name, data_sets=None):
    # Read the covariance matrix from the data file
    extension = fits.open(filename)[covmat_name]

    # Get the covariance matrix and the header
    cov_name = extension.name
    covmat = extension.data
    header = extension.header
    i = 0
    measurement_names = []
    start_indices = []
    while True:
        name_card = 'NAME_{}'.format(i)
        if name_card not in header:
            break
        measurement_names.append(header[name_card])
        start_index_card = 'STRT_{}'.format(i)
        start_indices.append(header[start_index_card])
        i += 1
    lengths = []
    current_length = 0
    # this only works if more than one spectrum
    if len(start_indices) > 1:
        for start, end in zip(start_indices[:-1], start_indices[1:]):
            lengths.append(end - start)
        if start_indices:
            lengths.append(covmat.shape[0] - start_indices[-1])
    else:
        lengths.append(covmat.shape[0])

    # indices for each measurement
    indices = {}
    for i, measurement_name in enumerate(measurement_names):
        indices[measurement_name] = np.arange(start_indices[i], start_indices[i] + lengths[i]).astype(int)

    return covmat, indices

def setup(options):
    """
    Parameters:
        data_file (str) : the path to the data file that contains 
                        the full covariance matrix
        like_names(list): the list of likelihood names whose data vectors 
                        are combined.
        {like_name}_data_sets (list): the list of data sets for each likelihood

    Optional Parameters:
        like_name (str) : the name of this likelihood, default is 'joint'
        {like_name}_moped (str) : the name of the corresponding MOPED
        covmat_name (str): the name of the covariance matrix in the data file
    """

    # Read parameters from the ini file
    data_file   = options.get_string(option_section, "data_file")
    covmat_name = options.get_string(option_section, "covmat_name", 'COVMAT')
    like_names  = options.get_string(option_section, "like_names").split()
    data_sets   = {}
    for like_name in like_names:
        param = "%s_data_sets"%like_name
        data_sets[like_name] = options.get_string(option_section, param).split()

    # Get full covariance
    covmat, indices = read_covariance(data_file, covmat_name)

    # Get the indices for each likelihood
    like_indices = {}
    for like_name in like_names:
        param = "%s_data_sets"%like_name
        data_sets = options.get_string(option_section, param).split()
        _ = []
        for data_set in data_sets:
            _.append(indices[data_set])
        like_indices[like_name] = np.hstack(_)

    # Select the covariance
    covariance = {}
    for like_name1 in like_names:
        for like_name2 in like_names:
            name = "%s--%s"%(like_name1, like_name2)
            covariance[name] = covmat[np.ix_(like_indices[like_name1], like_indices[like_name2])]

    # name of this likelihood
    name = options.get_string(option_section, "like_name", 'joint')

    # names of moped
    moped_names = []
    for like_name in like_names:
        param = "%s_moped"%like_name
        if options.has_value(option_section, param):
            moped_names.append(options.get_string(option_section, param))
        else:
            moped_names.append(None)

    # option to exclude cross-covariance (for sanity check)
    exclude_cross_cov = options.get_bool(option_section, 'exclude_cross_cov', False)

    # Number of simulation realizations used for covariance estimation
    nsim = options.get_int(option_section, 'covariance_realizations', -1)

    # Number of free parameters -- used to compute the Dodelson-Schneider factor
    npar = options.get_int(option_section, 'free_parameters', -1)

    Percival = options.get_bool(option_section, "Percival", False)

    # set config
    config = {"name":name, "like_names": like_names, "covariance": covariance, "moped_names": moped_names, 'exclude_cross_cov':exclude_cross_cov, 'nsim':nsim, 'npar':npar, 'Percival': Percival}

    #If you want to directly input a compressed 2pt data file
    #(for example, a noisy realization generated with the compressed covariance)
    #Don't use the following if you want to use the full 2pt data vector as an input
    #and perform compression when running the pipeline

    compressed_2pt_data = None
    if options.has_value(option_section, "use_input_compressed_data") and options.get_bool(option_section,
                                                                                           "use_input_compressed_data"):
        compressed_2pt_data_file = options.get_string(option_section, "compressed_2pt_data_file")
        compressed_2pt_data = np.loadtxt(compressed_2pt_data_file)

    config['compressed_2pt_data'] = compressed_2pt_data

    return config

def execute(block, config):
    # Get the likelihood names
    like_names = config["like_names"]
    # Get the covariance matrices
    covariance = config["covariance"]

    # read masks for each likelihood
    masks = {}
    for like_name in like_names:
        masks[like_name] = block[names.data_vector, "%s_mask"%like_name].astype(bool)

    # Build the masked covariance matrix based on the masks
    covariance_masked = []
    for like_name1 in like_names:
        mask1 = masks[like_name1].astype(bool)
        _ = []
        for like_name2 in like_names:
            mask2 = masks[like_name2].astype(bool)
            # get cov
            name = "%s--%s"%(like_name1, like_name2)
            cov = covariance[name]
            # mask
            cov_masked = cov[np.ix_(mask1, mask2)]
            if config['exclude_cross_cov'] and like_name1!=like_name2:
                print('Excluding {}-{} cross cov'.format(like_name1, like_name2))
                cov_masked *= 0.0
            # append
            _.append(cov_masked)
        covariance_masked.append(_)
    covariance_masked = np.block(covariance_masked)

    # Get the data vectors
    data_vector = []
    theory_vector = []
    for like_name in like_names:

        if like_name == '2pt' and config['compressed_2pt_data'] is not None:
            data_vector.append(config['compressed_2pt_data'])
        else:
            data_vector.append(block[names.data_vector, f'{like_name}_data'])
        theory_vector.append(block[names.data_vector, f'{like_name}_theory'])

    data_vector = np.hstack(data_vector)
    theory_vector = np.hstack(theory_vector)

    # Here MOPED compression happens.
    moped_names = config["moped_names"]
    transformation_matrix = []
    for like_name, moped_name in zip(like_names, moped_names):
        # transformation matrix
        if moped_name is not None:
            mat = block[names.data_vector, f'{moped_name}_transform_matrix']
            print('Compressed {} to {}: dims={}'.format(like_name, moped_name, mat.shape))
        else:
            _ = block[names.data_vector, f'{like_name}_data']
            mat = np.eye(_.size)
            print('No compression for {}: dim={}'.format(like_name, mat.shape[0]))
        # append 
        transformation_matrix.append(mat)
    transformation_matrix = scipy.linalg.block_diag(*transformation_matrix)
    print('Dimensions of after and before MOPED compression = {}'.format(transformation_matrix.shape))

    # Transform the data vector
    if config['compressed_2pt_data'] is None:
        data_vector = np.dot(transformation_matrix, data_vector)
    theory_vector = np.dot(transformation_matrix, theory_vector)
    covariance_masked = np.dot(np.dot(transformation_matrix, covariance_masked), transformation_matrix.T)

    # Compute the joint likelihood
    diff = data_vector - theory_vector
    inv_cov = np.linalg.inv(covariance_masked)

    Percival = config['Percival']

    if not Percival:
        # Anderson-Hartlap factor
        if config['nsim'] > 0:
            nsim = config['nsim']
            n = inv_cov.shape[0]
            f = (nsim-n-2)/(nsim-1)
            print(f'Hartlap {nsim} {n} {f}')
            inv_cov *= f
        # Dodelson-Schneider factor
        if config['npar'] > 0 and config['nsim'] > 0:
            npar = config['npar']
            n = inv_cov.shape[0]
            f2 = 1/(1 + (n-npar)*(nsim-n-2)/((nsim-n-1)*(nsim-n-4)))
            print(f'Dodelson-Schneider {nsim} {n} {npar} {f2}')
            inv_cov *= f2

        chi2 = np.dot(diff, np.dot(inv_cov, diff))

        # Set the result to the block
        name = config["name"]
        block[names.likelihoods, f"{name}_like"] = -0.5 * chi2
        block[names.data_vector, f"{name}_chi2"] = chi2

    else:
        print('Using Percival likelihood')
        n = inv_cov.shape[0]
        nsim = config['nsim']
        npar = config['npar']
        factor = (n-npar)*(nsim-n-2)/((nsim-n-1)*(nsim-n-4))
        m_power = npar + 2 + (nsim-1+factor)/(1+factor)
        chi2 = np.dot(diff, np.dot(inv_cov, diff))
        like = -m_power/2*np.log(1+(chi2/(nsim-1)))
        block[names.likelihoods, f'{config["name"]}_like'] = like
        block[names.data_vector, f'{config["name"]}_chi2'] = chi2

    block[names.data_vector, f"{name}_data"] = data_vector
    block[names.data_vector, f"{name}_theory"] = theory_vector
    block[names.data_vector, f"{name}_inverse_covariance"] = inv_cov
    block[names.data_vector, f'{name}_n'] = diff.size
    
    return 0

def cleanup(config):
    pass
