
import numpy as np
import matplotlib.pyplot as plt
from getdist import MCSamples
import os
here = os.path.dirname(__file__)

##################################
# parameter name and label mapping
def get_preset_mapping(names):
    # mcmc related output
    mcmc = {'prior':['prior', 'prior'], \
            'like':['like', 'like'], \
            'post':['post', 'post'], \
            'weight':['weight', 'weight']}
    # DES related model params
    des  = {'cosmological_parameters--omega_m':['om', r'\Omega_{\rm m}'], \
            'cosmological_parameters--s_8': ['s8', r'S_8'], \
            'COSMOLOGICAL_PARAMETERS--SIGMA_8': ['sig8', r'\sigma_8'], \
            'cosmological_parameters--h0': ['h0', 'h_0'], \
            'cosmological_parameters--w': ['w0', r'w_0'], \
            'cosmological_parameters--omega_b': ['ob', r'\Omega_{\rm b}'], \
            'cosmological_parameters--n_s': ['ns', r'n_s'], \
            'cosmological_parameters--mnu': ['mnu', r'\Sigma m_{\nu}'], \
            'wl_photoz_errors--bias_1': ['dz1', r'\Delta z_1'], \
            'wl_photoz_errors--bias_2': ['dz2', r'\Delta z_2'],\
            'wl_photoz_errors--bias_3': ['dz3', r'\Delta z_3'], \
            'wl_photoz_errors--bias_4': ['dz4', r'\Delta z_4'],\
            'shear_calibration_parameters--m1': ['m1', r'm_1'], \
            'shear_calibration_parameters--m2': ['m2', r'm_2'], \
            'shear_calibration_parameters--m3': ['m3', r'm_3'], \
            'shear_calibration_parameters--m4': ['m4', r'm_4'], \
            'intrinsic_alignment_parameters--a1':['a1', r'A_1'], \
            'intrinsic_alignment_parameters--a2':['a2', r'A_2'], \
            'intrinsic_alignment_parameters--alpha1': ['alpha1', r'\alpha_1'], \
            'intrinsic_alignment_parameters--alpha2': ['alpha2', r'\alpha_2'], \
            'intrinsic_alignment_parameters--bias_ta': ['bias_ta', r'bias_ta'], \
            'DATA_VECTOR--2PT_CHI2': ['2pt_chi2', r'\chi^2_{\rm 2pt}'], \
            'DATA_VECTOR--MAP3_CHI2': ['map3_chi2', r'\chi^2_{\rm map3}']}
    # make output
    if isinstance(names, str):
        names = [names]
    mapping = {}
    for name in names:
        if name == 'des':
            mapping |= des
        if name == 'mcmc':
            mapping |= mcmc
    return mapping

##################################
# Utilities of post analysis
def read_cosmosis_param_header(fname, mapping=None):
    # get params from header
    with open(fname, 'r') as f:
        params = f.readline().replace('#','').strip().split()
        labels = [None for p in params]
    # mapping of params and labels
    if mapping is not None:
        labels = [mapping[param][1] if param in mapping else param for param in params]
        params = [mapping[param][0] if param in mapping else param for param in params]
    return params, labels

def select_name(params, take=None):
    if take is not None:
        if isinstance(take, str):
            take = [take]
        if 'prior' not in take: take.append('prior')
        if 'like' not in take: take.append('like')
        if 'post' not in take: take.append('post')
        if 'weight' not in take: take.append('weight')
        index = [i for i, p in enumerate(params) if p in take]
    else:
        index = np.arange(len(params))
    return index

def read_cosmosis_value(filename, mapping=None):
    with open(filename, 'r') as f:
        s_mark = 'START_OF_VALUES_INI'
        e_mark = 'END_OF_VALUES_INI'

        lines = f.readlines()

        for i in range(len(lines)):
            if s_mark in lines[i]:
                i_s = i
            if e_mark in lines[i]:
                i_e = i
                break
        
        lines = lines[i_s+1:i_e]
        
        params = {}
        for line in lines:
            if '[' in line and ']' in line:
                section = line[line.find('[')+1:line.find(']')]
            if '=' in line:
                name, values = line.split('=')
                name = name.replace('##', '').replace(' ', '')
                name = '%s--%s'%(section, name)
                if mapping is not None:
                    name = mapping.get(name, [name, None])[0]
                values = [float(_) for _ in values.split()]
                params[name] = values
    return params

def convert_cosmosis_value_to_range(params):
    ranges = {}
    for name, values in params.items():
        if len(values) == 3:
            ranges[name] = [min(values), max(values)]
    return ranges

def get_cosmological_parameter_mean(fname):
    params, _ = read_cosmosis_param_header(fname, None)
    chain = np.loadtxt(fname)
    weight = chain[:,params.index('weight')]
    means = []
    index = []
    for i, param in enumerate(params):
        if 'cosmological_parameters' in param.lower():
            mean = np.sum(chain[:,i]*weight)/np.sum(weight)
            means.append(mean)
            index.append(i)
    means = np.array(means)
    index = np.array(index)
    return index, means

# weight plot
def plot_weight(chain, params, nlive=500):
    w = chain[:,params.index('weight')]
    plt.figure(figsize=(4,2))
    plt.plot(w[:-nlive]) # remove live points for ease of viewing
    plt.show()

##################################
# mcmc chain reader
def read_cosmosis_mcmc_chain(fname, mapping=None, take=None, blind=True, to_mcsamples=False, fname_mean=None, wplot=False, f_icov=None, add_s8=True):
    params, labels = read_cosmosis_param_header(fname, mapping)
    ranges = convert_cosmosis_value_to_range(read_cosmosis_value(fname, mapping))
    chain = np.loadtxt(fname)
    # blind
    if blind:
        fname_mean = fname_mean or fname
        index, means = get_cosmological_parameter_mean(fname_mean)
        print('Blinding ', [params[i] for i in index])
        chain[:, index] -= means[None,:]
        for i in index:
            if params[i] in ranges:
                ranges[params[i]] -= means[i]
            labels[i] = r'\Delta '+labels[i]
    # apply selection
    index = select_name(params, take)
    chain = chain[:, index]
    params= list(np.array(params)[index])
    labels= list(np.array(labels)[index])
    # apply rescaling of samples by the rescaling factor for inverse covariance
    if f_icov is not None:
        reweight_samples_by_icov_rescale_factor(chain, params, f_icov)

    # Add s8 if missing, from sigma8 and Omegam
    if ('s8' not in params) and ('om' in params) and ('sig8' in params) and add_s8:
        s8 = chain[:, params.index('sig8')] * (chain[:, params.index('om')]/0.3)**0.5
        params.append('s8')
        labels.append(r'S_8')
        chain = np.column_stack((chain, s8))
    # wplot
    if wplot:
        plot_weight(chain, params)
    if to_mcsamples:
        samples = chain_to_mcsamples(chain, params, labels, ranges=ranges)
        return samples
    else:
        return chain, params, labels, ranges

def read_cosmosis_mcmc_des_blind_chains(fnames, mapping=None, take=None, to_mcsamples=False, fname_mean=None, wplot=False):
    if mapping is None:
        mapping = get_preset_mapping('des')
    else:
        mapping |= get_preset_mapping('des')
    return read_cosmosis_mcmc_blind_chains(fnames, mapping, take, to_mcsamples, fname_mean, wplot)

def read_cosmosis_mcmc_blind_chains(fnames, mapping=None, take=None, to_mcsamples=False, fname_mean=None, wplot=False):

    fname = fnames[0]
    print(fname)
    params, labels = read_cosmosis_param_header(fname, mapping)
    ranges = convert_cosmosis_value_to_range(read_cosmosis_value(fname, mapping))
    chain = np.loadtxt(fname)

    fname_mean = fname_mean or fname
    index, means = get_cosmological_parameter_mean(fname_mean)
    print(len(index))
    print('Blinding ', [params[i] for i in index])
    chain[:, index] -= means[None,:]
    for i in index:
        if params[i] in ranges:
            ranges[params[i]] -= means[i]
        labels[i] = r'\Delta '+labels[i]
    # apply selection
    index = select_name(params, take)
    chain = chain[:, index]
    params= list(np.array(params)[index])
    labels= list(np.array(labels)[index])
    # wplot
    if wplot:
        plot_weight(chain, params)

    if to_mcsamples:
        samples = []
        samples.append(chain_to_mcsamples(chain, params, labels, ranges=ranges))

    for chain_nums in range(1,len(fnames)):

        fname = fnames[chain_nums]
        print(fname)
        params, labels = read_cosmosis_param_header(fname, mapping)
        ranges = convert_cosmosis_value_to_range(read_cosmosis_value(fname, mapping))
        chain = np.loadtxt(fname)

        fname_mean = fname_mean or fname
        index, means0 = get_cosmological_parameter_mean(fname_mean)
        print('Blinding ', [params[i] for i in index])
        chain[:, index] -= means[None, :]
        for i in index:
            if params[i] in ranges:
                ranges[params[i]] -= means[i]
            labels[i] = r'\Delta ' + labels[i]
        # apply selection
        index = select_name(params, take)
        chain = chain[:, index]
        params = list(np.array(params)[index])
        labels = list(np.array(labels)[index])
        # wplot
        if wplot:
            plot_weight(chain, params)

        if to_mcsamples:
            samples.append(chain_to_mcsamples(chain, params, labels, ranges=ranges))

    return samples

def read_cosmosis_mcmc_des_chain(fname, mapping=None, take=None, blind=True, to_mcsamples=False, fname_mean=None, wplot=False, f_icov=None):

    if mapping is None:
        mapping = get_preset_mapping('des')
    else:
        mapping |= get_preset_mapping('des')
    return read_cosmosis_mcmc_chain(fname, mapping, take, blind, to_mcsamples, fname_mean, wplot, f_icov)

##################################
# fisher output
def _read_cosmosis_fisher_mu(fname, ndim):
    mu = np.zeros(ndim)
    i = 0
    with open(fname, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if '#mu_{}='.format(i) in line:
                mu[i] = float(line.split('=')[1])
                i += 1
            else:
                continue
            if i>ndim:
                break
    # check
    assert i==ndim, 'i={} while ndim={}'.format(i, ndim)
    
    return mu

def read_cosmosis_fisher(fname, mapping=None, take=None, to_mcsamples=False):
    # read param, label, matrix
    params, labels = read_cosmosis_param_header(fname, mapping)
    ranges = convert_cosmosis_value_to_range(read_cosmosis_value(fname, mapping))
    F = np.loadtxt(fname)
    index = select_name(params, take)
    # read reference model parameter:
    mu = _read_cosmosis_fisher_mu(fname, F.shape[0])
    # apply selection
    mu = mu[index]
    F = F[np.ix_(index, index)]
    params= list(np.array(params)[index])
    labels= list(np.array(labels)[index])
    if to_mcsamples:
        samples = fisher_to_mcsamples(mu, F, params, labels, ranges=ranges)
        return samples
    else:
        return mu, F, params, labels, ranges

def read_cosmosis_fisher_des(fname, mapping=None, take=None, to_mcsamples=False):
    if mapping is None:
        mapping = get_preset_mapping('des')
    else:
        mapping |= get_preset_mapping('des')
    return read_cosmosis_fisher(fname, mapping=mapping, take=take, to_mcsamples=to_mcsamples)

def approximate_range_by_Gauss_in_F(F, params, ranges, scale=1):
    """
    Fisher matrix does not care about the prior range.
    Here, we approximate prior range by Gaussian distribution
    whose width corresponds to range length: sigma=(max-min)/2
    """
    for i, param in enumerate(params):
        if param not in ranges:
            continue
        sigma = scale*(ranges[param][1]-ranges[param][0])/2
        F[i,i]+= 1/sigma**2
    return F

##################################
# getdist interface
def chain_to_mcsamples(chain, params, labels, **kwargs):
    w = chain[:,params.index('weight')]
    samples = MCSamples(samples=chain, names=params, labels=labels, weights=w, **kwargs)
    return samples

def fisher_to_mcsamples(mu, F, params, labels, seed=0, size=5000, **kwargs):
    rng = np.random.default_rng(seed)
    _ = rng.multivariate_normal(mu, np.linalg.inv(F), size=size)
    samples = MCSamples(samples=_, names=params, labels=labels, **kwargs)
    return samples

# utils
def wplot(chain, params, labels):
    w = chain[:,params.index('weight')]
    plt.figure(figsize=(4,2))
    plt.plot(w[:-500]) # remove live points for ease of viewing
    plt.show()

def cov_from_samples_names(samples, names):
    pars = [samples.getParamNames().list().index(name) for name in names]
    cov = samples.getCov(pars=pars)
    return cov

def FoM_from_samples_names(samples, names):
    cov = cov_from_samples_names(samples, names)
    fom = np.linalg.det(cov)**-0.5
    return fom

def reweight_samples_by_icov_rescale_factor(chain, params, f, minw=1e-3):
    """
    Rescale the weights of samples by a correction factor applied 
    to the inverse covariance.

    The factor can be either of Hartlap or Dodelson-Schneider factor,
    or product of them. See e.g. Eq (23) and (24) of 
    https://arxiv.org/pdf/2110.10141

    This function assumes Gaussian likelihood so that the rescaling on
    icov can be equivalent to the rescaling of loglike.
    """
    # Get loglikelihood and weight
    like = chain[:,params.index('like')]
    w    = chain[:,params.index('weight')]

    # We discard the samples that has relative weight smaller than
    # a certain threshold, for which the rewieghting can fail because of the 
    # too much of upweighting. Intuitively, we are removing the samples
    # at the posterior tails.
    # We use 1e-3 as a default choice, but the final result should not 
    # strongly depends on this choice, that one can always check by changing this value.
    sel  = w > w.max()*minw

    # Compute the rescaling factor
    resc = (f-1.0) * like

    # Rescale the posterior
    # here the second term is intended to avoid overflow due to the large value.
    w[sel] *= np.exp(resc[sel] - resc[sel].max())


## Post unblinding analysis
def load_TTTEEElowE(to_mcsamples=True):
    """
    Planck TT, TE, EE, lowE
    """
    # Load parameter names
    fname = '../data/planck2018/base_plikHM_TTTEEE_lowl_lowE.paramnames'
    pnames, labels = np.loadtxt(os.path.join(here,fname), dtype=str, usecols=(0,1)).T
    pnames = ['lnlike'] + list(pnames)
    labels = ['lnlike'] + list(labels)

    # Rename the param for consistency
    pnames[pnames.index('S8*')] = 's8'
    pnames[pnames.index('omegam*')] = 'om'
    pnames[pnames.index('sigma8*')] = 'sig8'
    pnames[pnames.index('omegabh2')] = 'ob' # Note we will scale later appropriately
    pnames[pnames.index('H0*')] = 'h0' # Note we will scale later appropriately
    pnames[pnames.index('omeganuh2*')] = 'mnu' # Note we will convert later appropriately
    
    # Stack chains
    chains = []
    weights= []
    for i in [1,2,3,4]:
        fname = '../data/planck2018/base_plikHM_TTTEEE_lowl_lowE_%d.txt'%i
        chain = np.loadtxt(os.path.join(here,fname))
        # We first separate out the weight
        w     = chain[:,0]
        chain = chain[:,1:]
        # Flip the sign of the likelihood column: -Log(like) -> Log(like)
        chain[:,0]*= -1
        # H0 -> h0
        idx = pnames.index('h0')
        chain[:,idx] = chain[:,idx]/100.0
        # omega_nu h2 -> mnu
        idx = pnames.index('mnu')
        chain[:,idx] = 0.06*(chain[:,idx]/0.00064)
        # omega_b h^2 -> Omega_b
        idx = pnames.index('ob')
        idx_h0 = pnames.index('h0')
        chain[:,idx] = chain[:,idx] / chain[:,idx_h0]**2
        # append
        chains.append(chain)
        weights.append(w)
    samples = np.vstack(chains)
    weights = np.hstack(weights)

    # Check
    assert chains[0].shape[1] == len(pnames), \
        'chain shape does not match with parameter names array.'
    
    # Make MCSamples
    mc = MCSamples(samples=samples, names=pnames, labels=labels, weights=weights)

    if to_mcsamples:
        return mc
    else:
        return samples, weights, pnames, labels
    