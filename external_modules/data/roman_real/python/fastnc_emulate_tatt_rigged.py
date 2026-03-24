'''rigged version of the TATT emulator interface,
to be used only for developing and debugging purposes'''

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

    # Common setups
    Lmax   = options.get_int(option_section, "Lmax", default = 50)
    multipole_type = options.get_string(option_section, "multipole_type", default = "legendre")

    ########################################################################################
    # bisppectrum model (halofit)
    config_halofit = {'zmin':1e-4, 'zmid':0.1, 'nzbin_log':70, 'nzbin_lin':100,
                        'Lmax':Lmax,
                      'multipole_type':multipole_type,
                      'NLA':options.get_bool(option_section, 'NLA', default=True)}

    config['TATT'] = False
    # select model
    if options.get_string(option_section, "bispectrum_model", default = "bihalofit") == "bihalofit":
        print('Bispectrum model = halofit')
        bs = fastnc.bispectrum.BispectrumHalofit(config_halofit)
    elif options.get_string(option_section, "bispectrum_model") == 'gilmarin':
        print('Bispectrum model = gilmarin')
        bs = fastnc.bispectrum.BispectrumGilMarin(config_halofit)
    elif options.get_string(option_section, "bispectrum_model") == 'E_modes_TATT':
        #Here we will only compute the bispectrum of IA. We force NLA = False and use the TATT model
        print('Bispectrum model = E_modes_TATT')
        config_halofit['NLA'] = False
        bs = fastnc.bispectrum.BispectrumTATT(config_halofit)
        config['TATT'] = True
    else:
        raise ValueError('Invalid bispectrum model')
    if options.has_value(option_section, "use-pixwin") and options.get_bool(option_section, "use-pixwin"):
        bs.set_window_function(get_healpix_window_function(options.get_int(option_section, "nside")))
    config['bispectrum'] = bs

    # filters
    filter_1 = options.get_double_array_1d(option_section, "theta_filter_1")
    filter_2 = options.get_double_array_1d(option_section, "theta_filter_2")
    filter_3 = options.get_double_array_1d(option_section, "theta_filter_3")
    filter_num = len(filter_1)

    model_cosmo = options.get_string(option_section, "cosmo_model", default="LCDM")

    zarray = options.get_double_array_1d(option_section, "z_values")

    modes = np.arange(filter_num*len(zarray))

    #if model_cosmo == "wCDM":

    #    cp_nn = cosmopower_NN(parameters=['Omega_m', 's8', 'h0', 'Omega_b', 'ns', 'w'],
    #                          modes=modes,
    #                          n_hidden=[64, 256, 1024, 1024, 256, 192],
    #                          )
    #else:

    #    cp_nn = cosmopower_NN(parameters=['Omega_m', 's8', 'h0', 'Omega_b', 'ns'],
    #                      modes=modes,
    #                      n_hidden=[64, 256, 1024, 1024, 384, 192],
    #                      )

    #model_filename = options.get_string(option_section, "model_filename")
    #cp_nn.restore(model_filename)

    #rescaling_filename = options.get_string(option_section, "rescaling_filename", default="rescaling_for_map3_network.pkl")
    #with open(rescaling_filename, 'rb') as file:
    #    rescaling_values = pickle.load(file)

    #config['rescaling_params'] = rescaling_values['params']
    #config['rescaling_features'] = rescaling_values['features']
    config["filter_num"] = filter_num
    config['zarray'] = zarray
    #config['network'] = cp_nn
    config['nz_upsampling'] = options.get_int(option_section, "nz_upsampling", default=100)
    config['model_cosmo'] = model_cosmo

    return config

def execute(block, config):

    name_likelihood = 'emu_map3_like'

    filter_num = config["filter_num"]
    zarray = config['zarray']
    #cp_nn = config['network']

    #if config['model_cosmo'] == 'wCDM':

    #    parameters = np.zeros(6)
    #    parameters[0] = block[names.cosmological_parameters, 'omega_m']
    #    parameters[1] = block[names.cosmological_parameters, 'S_8']
    #    parameters[2] = block[names.cosmological_parameters, 'h0']
    #    parameters[3] = block[names.cosmological_parameters, 'omega_b']
    #    parameters[4] = block[names.cosmological_parameters, 'n_s']
    #    parameters[5] = block[names.cosmological_parameters, 'w']

    #    params_for_network = rescale_params(parameters, config['rescaling_params'])
    #    test_params_dict = {'Omega_m': [params_for_network[0]],
    #                    's8': [params_for_network[1]],
    #                    'h0': [params_for_network[2]],
    #                    'Omega_b': [params_for_network[3]],
    #                    'ns': [params_for_network[4]],
    #                    'w': [params_for_network[5]]}

    #else:

    #    parameters = np.zeros(5)
    #    parameters[0] = block[names.cosmological_parameters, 'omega_m']
    #    parameters[1] = block[names.cosmological_parameters, 'S_8']
    #    parameters[2] = block[names.cosmological_parameters, 'h0']
    #    parameters[3] = block[names.cosmological_parameters, 'omega_b']
    #    parameters[4] = block[names.cosmological_parameters, 'n_s']

    #    params_for_network = rescale_params(parameters, config['rescaling_params'])
    #    test_params_dict = {'Omega_m': [params_for_network[0]],
    #                        's8': [params_for_network[1]],
    #                        'h0': [params_for_network[2]],
    #                        'Omega_b': [params_for_network[3]],
    #                        'ns': [params_for_network[4]]}

    #predictions = cp_nn.predictions_np(test_params_dict)
    #predictions_rescaled = post_process(predictions, config['rescaling_features'])

    #predictions_rescaled_ddE = np.ndarray(shape=(1,144))
    #predictions_rescaled_dEd = np.ndarray(shape=(1, 144))
    #predictions_rescaled_Edd = np.ndarray(shape=(1, 144))
    #predictions_rescaled_EEd = np.ndarray(shape=(1, 144))
    #predictions_rescaled_EdE = np.ndarray(shape=(1, 144))
    #predictions_rescaled_dEE = np.ndarray(shape=(1, 144))
    #predictions_rescaled_EEE = np.ndarray(shape=(1, 144))

    predictions_rescaled_ddE = np.ndarray(shape=(1, 216))
    predictions_rescaled_dEd = np.ndarray(shape=(1, 216))
    predictions_rescaled_Edd = np.ndarray(shape=(1, 216))
    predictions_rescaled_EEd = np.ndarray(shape=(1, 216))
    predictions_rescaled_EdE = np.ndarray(shape=(1, 216))
    predictions_rescaled_dEE = np.ndarray(shape=(1, 216))
    predictions_rescaled_EEE = np.ndarray(shape=(1, 216))

    load_data_file_ddE = np.loadtxt("/Users/gchgomes/3pcf_integrator/output/emulator-train-tatt-4-ddE-sept-216.txt")
    load_data_file_dEd = np.loadtxt("/Users/gchgomes/3pcf_integrator/output/emulator-train-tatt-4-dEd-sept-216.txt")
    load_data_file_Edd = np.loadtxt("/Users/gchgomes/3pcf_integrator/output/emulator-train-tatt-4-Edd-sept-216.txt")
    load_data_file_EEd = np.loadtxt("/Users/gchgomes/3pcf_integrator/output/emulator-train-tatt-4-EEd-sept-216.txt")
    load_data_file_EdE = np.loadtxt("/Users/gchgomes/3pcf_integrator/output/emulator-train-tatt-4-EdE-sept-216.txt")
    load_data_file_dEE = np.loadtxt("/Users/gchgomes/3pcf_integrator/output/emulator-train-tatt-4-dEE-sept-216.txt")
    load_data_file_EEE = np.loadtxt("/Users/gchgomes/3pcf_integrator/output/emulator-train-tatt-4-EEE-sept-216.txt")
    print(np.shape(load_data_file_EEE))

    # Put all prediction arrays in a dictionary
    predictions_rescaled_dict = {
        'ddE': predictions_rescaled_ddE,
        'dEd': predictions_rescaled_dEd,
        'Edd': predictions_rescaled_Edd,
        'dEE': predictions_rescaled_dEE,
        'EdE': predictions_rescaled_EdE,
        'EEd': predictions_rescaled_EEd,
        'EEE': predictions_rescaled_EEE
    }

    # Put all loaded data arrays in a dictionary
    load_data_file_dict = {
        'ddE': load_data_file_ddE,
        'dEd': load_data_file_dEd,
        'Edd': load_data_file_Edd,
        'dEE': load_data_file_dEE,
        'EdE': load_data_file_EdE,
        'EEd': load_data_file_EEd,
        'EEE': load_data_file_EEE
    }

    predictions_newshape = dict()
    # Loop over sections
    for sec in predictions_rescaled_dict.keys():
        predictions_rescaled = predictions_rescaled_dict[sec]
        load_data_file = load_data_file_dict[sec]

        #predictions_rescaled[:, 0::4] = load_data_file[0][26:170:4]
        #predictions_rescaled[:, 1::4] = load_data_file[0][27:170:4]
        #predictions_rescaled[:, 2::4] = load_data_file[0][28:170:4]
        #predictions_rescaled[:, 3::4] = load_data_file[0][29:170:4]

        predictions_rescaled[:, 0::4] = load_data_file[0][26:242:4]
        predictions_rescaled[:, 1::4] = load_data_file[0][27:242:4]
        predictions_rescaled[:, 2::4] = load_data_file[0][28:242:4]
        predictions_rescaled[:, 3::4] = load_data_file[0][29:242:4]

        predictions_newshape[sec] = np.zeros(shape=(len(zarray),filter_num))
        for i in range(filter_num):
            predictions_newshape[sec][:,i] = predictions_rescaled[:,i::filter_num]

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
    # Intrinsic parameter
    bs.set_NLA_param({'AIA': block['intrinsic_alignment_parameters', 'a1'], \
                      'alphaIA': block['intrinsic_alignment_parameters', 'alpha1'], \
                      'z0': block['intrinsic_alignment_parameters', 'z_piv']})

    # Set TATT parameters
    if config['TATT']:
        bs.set_IA_param({'a1': block['intrinsic_alignment_parameters', 'a1'], \
                         'alphaIA': block['intrinsic_alignment_parameters', 'alpha1'], \
                         'a2': block['intrinsic_alignment_parameters', 'a2'], \
                         'alphaIA_2': block['intrinsic_alignment_parameters', 'a2'], \
                         'bias_ta': block['intrinsic_alignment_parameters', 'bias_ta'], \
                         'z0': block['intrinsic_alignment_parameters', 'z_piv']})
        bs.set_pknl(
            block[names.matter_power_nl, 'k_h'],
            block[names.matter_power_nl, 'p_k'][0, :])

    # set source distribution
    nzbin = block['nz_source', "nbin"]
    bs.set_source_distribution(
        [block['nz_source', "z"] for _ in range(nzbin)],
        [block['nz_source', "bin_%d" % (i + 1)] for i in range(nzbin)],
        [(i + 1) for i in range(nzbin)]
    )
    # set linear matter power spectrum
    bs.set_pklin(
        block[names.matter_power_lin, 'k_h'],
        block[names.matter_power_lin, 'p_k'][0, :]
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

    # update the interpolation.
    bs.compute_kernel()

    sctname = "map3"

    #zarray2, predictions_newshape['ddE'] = upsampling(zarray, predictions_newshape['ddE'], 300)
    #zarray2, predictions_newshape['dEd'] = upsampling(zarray, predictions_newshape['dEd'], 300)
    #zarray2, predictions_newshape['Edd'] = upsampling(zarray, predictions_newshape['Edd'], 300)
    #array2, predictions_newshape['EEd'] = upsampling(zarray, predictions_newshape['EEd'], 300)
    #array2, predictions_newshape['EdE'] = upsampling(zarray, predictions_newshape['EdE'], 300)
    #zarray2, predictions_newshape['dEE'] = upsampling(zarray, predictions_newshape['dEE'], 300)
    #zarray2, predictions_newshape['EEE'] = upsampling(zarray, predictions_newshape['EEE'], 300)
    #zarray = zarray2

    for scomb in block['natural_components', 'sample_combinations']:
        name = '_'.join([str(s) for s in scomb])

        print('scombs', scomb[0],scomb[1], scomb[2])
        chi = bs.z2chi(zarray)
        z2g0 = bs.z2g_dict[scomb[0]]
        z2g1 = bs.z2g_dict[scomb[1]]
        z2g2 = bs.z2g_dict[scomb[2]]
        z2W0 = bs.z2W_dict[scomb[0]]
        z2W1 = bs.z2W_dict[scomb[1]]
        z2W2 = bs.z2W_dict[scomb[2]]
        chi2g0 = bs.chi2g_dict[scomb[0]]
        chi2g1 = bs.chi2g_dict[scomb[1]]
        chi2g2 = bs.chi2g_dict[scomb[2]]
        chi2W0 = bs.chi2W_dict[scomb[0]]
        chi2W1 = bs.chi2W_dict[scomb[1]]
        chi2W2 = bs.chi2W_dict[scomb[2]]
        #weight_ddE = z2g0(zarray) * z2g1(zarray) * z2W2(zarray) / chi * (1 + zarray) ** 3
        #weight_dEd = z2g0(zarray) * z2W1(zarray) * z2g2(zarray) / chi * (1 + zarray) ** 3
        #weight_Edd = z2W0(zarray) * z2g1(zarray) * z2g2(zarray) / chi * (1 + zarray) ** 3
        #weight_dEE = z2g0(zarray) * z2W1(zarray) * z2W2(zarray) / chi * (1 + zarray) ** 3
        #weight_EEd = z2W0(zarray) * z2W1(zarray) * z2g2(zarray) / chi * (1 + zarray) ** 3
        #weight_EdE = z2W0(zarray) * z2g1(zarray) * z2W2(zarray) / chi * (1 + zarray) ** 3
        #weight_EEE = z2W0(zarray) * z2W1(zarray) * z2W2(zarray) / chi * (1 + zarray) ** 3
        weight_ddE = chi2g0(chi) * chi2g1(chi) * chi2W2(chi) / chi * (1 + zarray) ** 3
        weight_dEd = chi2g0(chi) * chi2W1(chi) * chi2g2(chi) / chi * (1 + zarray) ** 3
        weight_Edd = chi2W0(chi) * chi2g1(chi) * chi2g2(chi) / chi * (1 + zarray) ** 3
        weight_dEE = chi2g0(chi) * chi2W1(chi) * chi2W2(chi) / chi * (1 + zarray) ** 3
        weight_EEd = chi2W0(chi) * chi2W1(chi) * chi2g2(chi) / chi * (1 + zarray) ** 3
        weight_EdE = chi2W0(chi) * chi2g1(chi) * chi2W2(chi) / chi * (1 + zarray) ** 3
        weight_EEE = chi2W0(chi) * chi2W1(chi) * chi2W2(chi) / chi * (1 + zarray) ** 3

        if scomb[0] == 1 and scomb[1] == 1 and scomb[2] == 1:

            print("kernel1", chi2g0(chi))
            print("W1", chi2W0(chi))

            print("weight ddE:", weight_ddE)
            print("weight dEd:", weight_dEd)
            print("weight Edd:", weight_Edd)
            print("weight dEE:", weight_dEE)
            print("weight EdE:", weight_EdE)
            print("weight EEd:", weight_EEd)
            print("weight EEE:", weight_EEE)
        #print("z dependent ddE", predictions_newshape['ddE'])
        #print("z dependent dEd", predictions_newshape['dEd'])
        #print("z dependent Edd", predictions_newshape['Edd'])
        #print("z dependent dEE", predictions_newshape['dEE'])
        #print("z dependent EdE", predictions_newshape['EdE'])
        #print("z dependent EEd", predictions_newshape['EEd'])
        #print("z dependent EEE", predictions_newshape['EEE'])

        #print('weight shape', np.shape(weight_ddE))
        #print('pred shape', np.shape(predictions_newshape['ddE']))
        #print('weight shape', np.shape(weight_dEd))
        #print('pred shape', np.shape(predictions_newshape['dEd']))
        #print('weight shape', np.shape(weight_Edd))
        #print('pred shape', np.shape(predictions_newshape['Edd']))

        tmp_ddE = np.einsum('ij,i->ij',predictions_newshape['ddE'], weight_ddE)
        tmp_dEd = np.einsum('ij,i->ij', predictions_newshape['dEd'], weight_dEd)
        tmp_Edd = np.einsum('ij,i->ij', predictions_newshape['Edd'], weight_Edd)
        tmp_dEE = np.einsum('ij,i->ij', predictions_newshape['dEE'], weight_dEE)
        tmp_EEd = np.einsum('ij,i->ij', predictions_newshape['EEd'], weight_EEd)
        tmp_EdE = np.einsum('ij,i->ij', predictions_newshape['EdE'], weight_EdE)
        tmp_EEE = np.einsum('ij,i->ij', predictions_newshape['EEE'], weight_EEE)
        tmp = tmp_ddE + tmp_dEd + tmp_Edd + tmp_dEE + tmp_EEd + tmp_EdE + tmp_EEE
        #tmp = tmp_dEd

        map3 = np.trapz(tmp,chi, axis=0)
        block[sctname, f'map3-bin_{name}'] = map3

    return 0

def cleanup(config):
    pass
