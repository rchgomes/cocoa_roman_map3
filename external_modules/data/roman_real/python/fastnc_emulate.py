from cosmosis.datablock import option_section, names
import numpy as np
from scipy.interpolate import interp1d
import os
import pickle
from astropy.cosmology import wCDM
from cosmopower import cosmopower_NN
import fastnc

def get_healpix_window_function(nside):
    import healpy as hp
    from scipy.interpolate import interp1d
    w = hp.sphtfunc.pixwin(nside)
    l = np.arange(3*nside)
    fnc = interp1d(l, w, kind='linear', bounds_error=False, fill_value=(1,0))
    window = lambda l1, l2, l3: fnc(l1) * fnc(l2) * fnc(l3)
    return window

def rescale_params(params, scale):
    params_rescaled = np.zeros_like(params)
    params_rescaled[0] = (params[0] - scale['Omega_m']['small']) / (
                scale['Omega_m']['large'] - scale['Omega_m']['small'])
    params_rescaled[1] = (params[1] - scale['s8']['small']) / (scale['s8']['large'] - scale['s8']['small'])
    params_rescaled[2] = (params[2] - scale['h0']['small']) / (scale['h0']['large'] - scale['h0']['small'])
    params_rescaled[3] = (params[3] - scale['Omega_b']['small']) / (scale['Omega_b']['large'] - scale['Omega_b']['small'])
    params_rescaled[4] = (params[4] - scale['ns']['small']) / (scale['ns']['large'] - scale['ns']['small'])

    if len(params_rescaled) == 6:
        params_rescaled[5] = (params[5] - scale['w']['small']) / (scale['w']['large'] - scale['w']['small'])

    return (params_rescaled)

def post_process(array, scale):
    out_array = np.zeros_like(array)
    maxv = scale['large']
    minv = scale['small']

    for i in range(len(array[0])):
        tmp = array[:, i] * (maxv[i] - minv[i]) + minv[i]
        out_array[:, i] = 10 ** (tmp)

    return (out_array)

def upsampling(z, pred, n):
    if n <= len(z):
        return z, pred
    z_new = np.linspace(z.min(), z.max(), n)
    pred_new = np.zeros((n, pred.shape[1]))
    for i in range(pred.shape[1]):
        f = interp1d(z, pred[:, i])
        pred_new[:, i] = f(z_new)
    return z_new, pred_new

def setup(options):
    """
    Necessary keys in the ini file:
    - Lmax: maximum multipole
    - Mmax: maximum multipole
    - l12bin: number of bins for l12
    - projection: projection of shear, x, cent, ortho (default: x)
    """
    # config
    config = {}

    # bispectrum
    config['bispectrum'] = fastnc.bispectrum.BispectrumHalofit()

    # filters
    filter_1 = options.get_double_array_1d(option_section, "theta_filter_1")
    filter_2 = options.get_double_array_1d(option_section, "theta_filter_2")
    filter_3 = options.get_double_array_1d(option_section, "theta_filter_3")
    filter_num = len(filter_1)

    model_cosmo = options.get_string(option_section, "cosmo_model", default="LCDM")

    zarray = options.get_double_array_1d(option_section, "z_values")

    modes = np.arange(filter_num*len(zarray))

    if model_cosmo == "wCDM":

        cp_nn = cosmopower_NN(parameters=['Omega_m', 's8', 'h0', 'Omega_b', 'ns', 'w'],
                              modes=modes,
                              n_hidden=[64, 256, 1024, 1024, 256, 192],
                              )
    else:

        cp_nn = cosmopower_NN(parameters=['Omega_m', 's8', 'h0', 'Omega_b', 'ns'],
                          modes=modes,
                          n_hidden=[64, 256, 1024, 1024, 384, 192],
                          )

    model_filename = options.get_string(option_section, "model_filename")
    cp_nn.restore(model_filename)

    rescaling_filename = options.get_string(option_section, "rescaling_filename", default="rescaling_for_map3_network.pkl")
    with open(rescaling_filename, 'rb') as file:
        rescaling_values = pickle.load(file)

    config['rescaling_params'] = rescaling_values['params']
    config['rescaling_features'] = rescaling_values['features']
    config["filter_num"] = filter_num
    config['zarray'] = zarray
    config['network'] = cp_nn
    config['nz_upsampling'] = options.get_int(option_section, "nz_upsampling", default=100)
    config['model_cosmo'] = model_cosmo

    # IA (NLA) modeling choice
    # If you want to use A1 parameter of IA per bin,
    # you use this flag=True.
    # You need to accordingly define the A1_1, A1_2, ...
    # Also you should fix the A1 = 1.0 and alpha1 = 1.0.
    config['perbin'] = options.get_bool(option_section, 'perbin', default=False)

    return config

def execute(block, config):

    name_likelihood = 'emu_map3_like'

    filter_num = config["filter_num"]
    zarray = config['zarray']
    cp_nn = config['network']

    if config['model_cosmo'] == 'wCDM':

        parameters = np.zeros(6)
        parameters[0] = block[names.cosmological_parameters, 'omega_m']
        parameters[1] = block[names.cosmological_parameters, 'S_8']
        parameters[2] = block[names.cosmological_parameters, 'h0']
        parameters[3] = block[names.cosmological_parameters, 'omega_b']
        parameters[4] = block[names.cosmological_parameters, 'n_s']
        parameters[5] = block[names.cosmological_parameters, 'w']

        params_for_network = rescale_params(parameters, config['rescaling_params'])
        test_params_dict = {'Omega_m': [params_for_network[0]],
                        's8': [params_for_network[1]],
                        'h0': [params_for_network[2]],
                        'Omega_b': [params_for_network[3]],
                        'ns': [params_for_network[4]],
                        'w': [params_for_network[5]]}

    else:

        parameters = np.zeros(5)
        parameters[0] = block[names.cosmological_parameters, 'omega_m']
        parameters[1] = block[names.cosmological_parameters, 'S_8']
        parameters[2] = block[names.cosmological_parameters, 'h0']
        parameters[3] = block[names.cosmological_parameters, 'omega_b']
        parameters[4] = block[names.cosmological_parameters, 'n_s']

        params_for_network = rescale_params(parameters, config['rescaling_params'])
        test_params_dict = {'Omega_m': [params_for_network[0]],
                            's8': [params_for_network[1]],
                            'h0': [params_for_network[2]],
                            'Omega_b': [params_for_network[3]],
                            'ns': [params_for_network[4]]}

    predictions = cp_nn.predictions_np(test_params_dict)
    predictions_rescaled = post_process(predictions, config['rescaling_features'])

    predictions_newshape = np.zeros(shape=(len(zarray),filter_num))
    for i in range(filter_num):
        predictions_newshape[:,i] = predictions_rescaled[:,i::filter_num]

    # BISPECTRUM ######################################
    # update bispectrum with inputs:
    bs = config['bispectrum']
    # cosmological parameters (fastnc accepts cosmo as astropy format)

    if config['model_cosmo'] == 'wCDM':
        cosmo = wCDM(
                H0=100*block[names.cosmological_parameters, 'h0'], \
                Om0=block[names.cosmological_parameters, 'omega_m'], \
                Ode0=1.0-block[names.cosmological_parameters, 'omega_m'], \
                w0 = block[names.cosmological_parameters, 'w'], \
                meta = {'sigma8':block[names.cosmological_parameters, 'sigma_8'], \
                        'n':block[names.cosmological_parameters, 'n_s']}
        )

    else:
        cosmo = wCDM(
                H0=100*block[names.cosmological_parameters, 'h0'], \
                Om0=block[names.cosmological_parameters, 'omega_m'], \
                Ode0=1.0-block[names.cosmological_parameters, 'omega_m'], \
                meta = {'sigma8':block[names.cosmological_parameters, 'sigma_8'], \
                        'n':block[names.cosmological_parameters, 'n_s']}
        )

    bs.set_cosmology(cosmo)
    # set source distribution
    nzbin = block['nz_source', "nbin"]
    bs.set_source_distribution(
        [block['nz_source', "z"] for _ in range(nzbin)],
        [block['nz_source', "bin_%d" % (i+1)] for i in range(nzbin)],
        [(i+1) for i in range(nzbin)]
    )
    # Intrinsic alignment parameter
    if config['perbin']:
        NLA_param = {'alphaIA':block['intrinsic_alignment_parameters', 'alpha1'], 
                     'z0':block['intrinsic_alignment_parameters', 'z_piv'], 
                     'perbin':True}
        nzbin = block['nz_source', "nbin"]
        for i in range(nzbin):
            # Note that fastnc calls the amplitude param AIA but cosmosis calls it A1.
            NLA_param[f'AIA_{i+1}'] = block['intrinsic_alignment_parameters', f'A1_{i+1}']
        print('>>> We are using perbin option for IA NLA!!!!')
        print('>>> inputs are {}'.format(NLA_param))
        bs.set_NLA_param(NLA_param)
        bs.config_IA['NLA'] = True
    else:
        bs.set_NLA_param({'AIA':block['intrinsic_alignment_parameters', 'a1'], \
                'alphaIA':block['intrinsic_alignment_parameters', 'alpha1'] , \
                'z0':block['intrinsic_alignment_parameters', 'z_piv'], 
                'perbin':False})
        bs.config_IA['NLA'] = True
    # set linear matter power spectrum
    bs.set_pklin(
        block[names.matter_power_lin, 'k_h'],
        block[names.matter_power_lin, 'p_k'][0,:]
    )
    # set linear growth rate
    bs.set_lgr(
        block[names.growth_parameters, "z"],
        block[names.growth_parameters, "d_z"]
    )
    # set baryon paramter
    if block.has_value('baryon_parameters', 'fb'):
        fb = block['baryon_parameters', 'fb']
        bs.set_baryon_param({'fb': fb})
    # compute kernel
    bs.compute_kernel()

    sctname = "map3"

    #zarray, predictions_newshape = upsampling(zarray, predictions_newshape, 100)

    for scomb in block['natural_components', 'sample_combinations']:
        name = '_'.join([str(s) for s in scomb])

        chi = bs.z2chi(zarray)
        z2g0 = bs.z2g_dict[scomb[0]]
        z2g1 = bs.z2g_dict[scomb[1]]
        z2g2 = bs.z2g_dict[scomb[2]]
        weight = z2g0(zarray) * z2g1(zarray) * z2g2(zarray) / chi * (1+zarray)**3
        tmp = np.einsum('ij,i->ij',predictions_newshape,weight)
        map3 = np.trapz(tmp,chi, axis=0)
        block[sctname, f'map3-bin_{name}'] = map3

    return 0

def cleanup(config):
    pass
