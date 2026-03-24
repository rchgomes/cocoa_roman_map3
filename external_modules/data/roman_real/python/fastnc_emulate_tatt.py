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
    l = np.arange(3 * nside)
    fnc = interp1d(l, w, kind='linear', bounds_error=False, fill_value=(1, 0))
    window = lambda l1, l2, l3: fnc(l1) * fnc(l2) * fnc(l3)
    return window


def rescale_params(params, scale):
    params_rescaled = np.zeros_like(params)
    params_rescaled[0] = (params[0] - scale['Omega_m']['small']) / (
            scale['Omega_m']['large'] - scale['Omega_m']['small'])
    params_rescaled[1] = (params[1] - scale['s8']['small']) / (scale['s8']['large'] - scale['s8']['small'])
    params_rescaled[2] = (params[2] - scale['h0']['small']) / (scale['h0']['large'] - scale['h0']['small'])
    params_rescaled[3] = (params[3] - scale['Omega_b']['small']) / (
                scale['Omega_b']['large'] - scale['Omega_b']['small'])
    params_rescaled[4] = (params[4] - scale['ns']['small']) / (scale['ns']['large'] - scale['ns']['small'])

    if len(params_rescaled) == 6:
        params_rescaled[5] = (params[5] - scale['w']['small']) / (scale['w']['large'] - scale['w']['small'])

    # For the 11-parameter TATT emulators
    if len(params_rescaled) == 11:
        # The first 6 are cosmological, handled above (or 5 if LCDM)
        # The next 5 are TATT parameters
        params_rescaled[6] = (params[6] - scale['a1']['small']) / (scale['a1']['large'] - scale['a1']['small'])
        params_rescaled[7] = (params[7] - scale['alpha1']['small']) / (
                    scale['alpha1']['large'] - scale['alpha1']['small'])
        params_rescaled[8] = (params[8] - scale['a2']['small']) / (scale['a2']['large'] - scale['a2']['small'])
        params_rescaled[9] = (params[9] - scale['alpha2']['small']) / (
                    scale['alpha2']['large'] - scale['alpha2']['small'])
        params_rescaled[10] = (params[10] - scale['bias_ta']['small']) / (
                    scale['bias_ta']['large'] - scale['bias_ta']['small'])

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


def interpolate_cosmo(tmp_cosmo, chi_original, chi_tatt):

    if tmp_cosmo.shape[0] != len(chi_original):
        raise ValueError(
            f"Length of tmp_cosmo's first dimension ({tmp_cosmo.shape[0]}) must match length of chi_original ({len(chi_original)}).")

    j_dim = tmp_cosmo.shape[1]
    tmp_cosmo_mod = np.zeros((len(chi_tatt), j_dim))
    for j in range(j_dim):
        interp_func = interp1d(
            chi_original,
            tmp_cosmo[:, j],
            kind='cubic',
            bounds_error=False,
            fill_value=0.0
        )

        tmp_cosmo_mod[:, j] = interp_func(chi_tatt)
    return tmp_cosmo_mod


def setup(options):
    # config
    config = {}

    # bispectrum setup from fastnc_emulate_tatt_rigged.py
    Lmax = options.get_int(option_section, "Lmax", default=50)
    multipole_type = options.get_string(option_section, "multipole_type", default="legendre")
    config_halofit = {'zmin': 1e-4, 'zmid': 0.1, 'nzbin_log': 70, 'nzbin_lin': 100,
                      'Lmax': Lmax,
                      'multipole_type': multipole_type,
                      'NLA': True}
    config_tatt = {'zmin': 1e-4, 'zmid': 0.1, 'nzbin_log': 70, 'nzbin_lin': 100,
                      'Lmax': Lmax,
                      'multipole_type': multipole_type,
                      'NLA': False}

    if options.get_string(option_section, "bispectrum_model", default="bihalofit") == "bihalofit":
        print('Bispectrum model = halofit')
        bs = fastnc.bispectrum.BispectrumHalofit(config_halofit)
    elif options.get_string(option_section, "bispectrum_model") == 'gilmarin':
        print('Bispectrum model = gilmarin')
        bs = fastnc.bispectrum.BispectrumGilMarin(config_halofit)
    else:
        raise ValueError('Invalid bispectrum model')
    bs_tatt = fastnc.bispectrum.BispectrumTATT(config_tatt)
    if options.has_value(option_section, "use-pixwin") and options.get_bool(option_section, "use-pixwin"):
        bs.set_window_function(get_healpix_window_function(options.get_int(option_section, "nside")))
    config['bispectrum'] = bs
    config['bispectrum_tatt'] = bs_tatt

    # filters and z-array
    filter_1 = options.get_double_array_1d(option_section, "theta_filter_1")
    filter_num = len(filter_1)
    model_cosmo = options.get_string(option_section, "cosmo_model", default="wCDM")
    zarray = options.get_double_array_1d(option_section, "z_values")
    zarray_tatt = options.get_double_array_1d(option_section, "z_values_tatt")
    modes = np.arange(filter_num * len(zarray))
    modes_tatt = np.arange(filter_num * len(zarray_tatt))

    cp_nn_cosmo = cosmopower_NN(
        parameters=['Omega_m', 's8', 'h0', 'Omega_b', 'ns'] + (['w'] if model_cosmo == "wCDM" else []),
        modes=modes,
        n_hidden=[64, 256, 1024, 1024, 384, 192] if model_cosmo != "wCDM" else [64, 256, 1024, 1024, 256, 192],
    )
    model_filename_cosmo = options.get_string(option_section, "model_filename")
    cp_nn_cosmo.restore(model_filename_cosmo)
    config['network_cosmo'] = cp_nn_cosmo


    tatt_emulators = ['ddE', 'dEd', 'Edd', 'dEE', 'EdE', 'EEd', 'EEE']
    cp_nn_tatt = {}
    for name in tatt_emulators:

        tatt_params = ['Omega_m', 's8', 'h0', 'Omega_b', 'ns', 'w', 'a1', 'alpha1', 'a2', 'alpha2', 'bias_ta']
        cp_nn = cosmopower_NN(
            parameters=tatt_params,
            modes=modes_tatt,
            n_hidden=[128, 256, 1024, 1024, 384],
        )
        model_filename_tatt = options.get_string(option_section, f"model_filename_{name}")
        cp_nn.restore(model_filename_tatt)
        cp_nn_tatt[name] = cp_nn
    config['network_tatt'] = cp_nn_tatt

    rescaling_filename = options.get_string(option_section, "rescaling_filename",
                                            default="rescaling_for_map3_network.pkl")
    with open(rescaling_filename, 'rb') as file:
        rescaling_values = pickle.load(file)

    config['rescaling_params'] = rescaling_values['params']
    config['rescaling_features'] = rescaling_values['features']

    config['rescaling_params_tatt'] = {}
    config['rescaling_features_tatt'] = {}
    for name in tatt_emulators:
        rescaling_filename_tatt = options.get_string(option_section, f"rescaling_filename_{name}")
        with open(rescaling_filename_tatt, 'rb') as file:
            rescaling_values_tatt = pickle.load(file)
        config['rescaling_params_tatt'][name] = rescaling_values_tatt['params']
        config['rescaling_features_tatt'][name] = rescaling_values_tatt['features']

    config["filter_num"] = filter_num
    config['zarray'] = zarray
    config['zarray_tatt'] = zarray_tatt
    config['nz_upsampling'] = options.get_int(option_section, "nz_upsampling", default=100)
    config['model_cosmo'] = model_cosmo

    return config


def execute(block, config):
    filter_num = config["filter_num"]
    zarray = config['zarray']
    zarray_tatt = config['zarray_tatt']
    cp_nn_cosmo = config['network_cosmo']
    cp_nn_tatt = config['network_tatt']
    model_cosmo = config['model_cosmo']

    # --- 1. Get Cosmological Parameters (for Cosmo-only emulator) ---
    if model_cosmo == 'wCDM':
        cosmo_params = np.zeros(6)
        cosmo_params[0] = block[names.cosmological_parameters, 'omega_m']
        cosmo_params[1] = block[names.cosmological_parameters, 'S_8']
        cosmo_params[2] = block[names.cosmological_parameters, 'h0']
        cosmo_params[3] = block[names.cosmological_parameters, 'omega_b']
        cosmo_params[4] = block[names.cosmological_parameters, 'n_s']
        cosmo_params[5] = block[names.cosmological_parameters, 'w']
        param_names_cosmo = ['Omega_m', 's8', 'h0', 'Omega_b', 'ns', 'w']
    else:
        cosmo_params = np.zeros(5)
        cosmo_params[0] = block[names.cosmological_parameters, 'omega_m']
        cosmo_params[1] = block[names.cosmological_parameters, 'S_8']
        cosmo_params[2] = block[names.cosmological_parameters, 'h0']
        cosmo_params[3] = block[names.cosmological_parameters, 'omega_b']
        cosmo_params[4] = block[names.cosmological_parameters, 'n_s']
        param_names_cosmo = ['Omega_m', 's8', 'h0', 'Omega_b', 'ns']


    tatt_params_full = np.zeros(11)
    tatt_params_full[:6] = cosmo_params[:6] if model_cosmo == 'wCDM' else np.append(cosmo_params, 0.0)  # w=0 for LCDM

    # TATT part
    tatt_params_full[6] = block['intrinsic_alignment_parameters', 'a1']
    tatt_params_full[7] = block['intrinsic_alignment_parameters', 'alpha1']
    tatt_params_full[8] = block['intrinsic_alignment_parameters', 'a2']
    tatt_params_full[9] = block['intrinsic_alignment_parameters', 'alpha2']
    tatt_params_full[10] = block['intrinsic_alignment_parameters', 'bias_ta']

    param_names_tatt = ['Omega_m', 's8', 'h0', 'Omega_b', 'ns', 'w', 'a1', 'alpha1', 'a2', 'alpha2', 'bias_ta']

    params_for_network_cosmo = rescale_params(cosmo_params, config['rescaling_params'])
    test_params_dict_cosmo = {name: [params_for_network_cosmo[i]] for i, name in enumerate(param_names_cosmo)}

    predictions_cosmo = cp_nn_cosmo.predictions_np(test_params_dict_cosmo)
    predictions_rescaled_cosmo = post_process(predictions_cosmo, config['rescaling_features'])

    predictions_newshape_cosmo = np.zeros(shape=(len(zarray), filter_num))
    for i in range(filter_num):
        predictions_newshape_cosmo[:, i] = predictions_rescaled_cosmo[:, i::filter_num]

    predictions_newshape_tatt = {}

    tatt_emulators = ['ddE', 'dEd', 'Edd', 'dEE', 'EdE', 'EEd', 'EEE']
    for name in tatt_emulators:
        params_for_network_tatt = rescale_params(tatt_params_full, config['rescaling_params_tatt'][name])
        test_params_dict_tatt = {name: [params_for_network_tatt[i]] for i, name in enumerate(param_names_tatt)}
        cp_nn = cp_nn_tatt[name]
        predictions = cp_nn.predictions_np(test_params_dict_tatt)
        predictions_rescaled = post_process(predictions, config['rescaling_features_tatt'][name])

        predictions_newshape_tatt[name] = np.zeros(shape=(len(zarray_tatt), filter_num))
        for i in range(filter_num):
            predictions_newshape_tatt[name][:, i] = predictions_rescaled[:, i::filter_num]


    bs = config['bispectrum']
    bs_tatt = config['bispectrum_tatt']

    # Cosmological parameters
    if model_cosmo == 'wCDM':
        cosmo = wCDM(
            H0=100 * block[names.cosmological_parameters, 'h0'], \
            Om0=block[names.cosmological_parameters, 'omega_m'], \
            Ode0=1.0 - block[names.cosmological_parameters, 'omega_m'], \
            w0=block[names.cosmological_parameters, 'w'], \
            meta={'sigma8': block[names.cosmological_parameters, 'sigma_8'], \
                  'n': block[names.cosmological_parameters, 'n_s']}
        )
    else:
        cosmo = wCDM(
            H0=100 * block[names.cosmological_parameters, 'h0'], \
            Om0=block[names.cosmological_parameters, 'omega_m'], \
            Ode0=1.0 - block[names.cosmological_parameters, 'omega_m'], \
            meta={'sigma8': block[names.cosmological_parameters, 'sigma_8'], \
                  'n': block[names.cosmological_parameters, 'n_s']}
        )
    bs.set_cosmology(cosmo)
    bs_tatt.set_cosmology(cosmo)

    bs.set_NLA_param({'AIA': block['intrinsic_alignment_parameters', 'a1'], \
                      'alphaIA': block['intrinsic_alignment_parameters', 'alpha1'], \
                      'z0': block['intrinsic_alignment_parameters', 'z_piv']})

    bs_tatt.set_IA_param({'a1': block['intrinsic_alignment_parameters', 'a1'], \
                         'alphaIA': block['intrinsic_alignment_parameters', 'alpha1'], \
                         'a2': block['intrinsic_alignment_parameters', 'a2'], \
                         'alphaIA_2': block['intrinsic_alignment_parameters', 'a2'], \
                         'bias_ta': block['intrinsic_alignment_parameters', 'bias_ta'], \
                         'z0': block['intrinsic_alignment_parameters', 'z_piv']})
        # Assuming P_NL is needed for TATT bispectrum calculation
    bs_tatt.set_pknl(
            block[names.matter_power_nl, 'k_h'],
            block[names.matter_power_nl, 'p_k'][0, :])

    # set source distribution
    nzbin = block['nz_source', "nbin"]
    bs.set_source_distribution(
        [block['nz_source', "z"] for _ in range(nzbin)],
        [block['nz_source', "bin_%d" % (i + 1)] for i in range(nzbin)],
        [(i + 1) for i in range(nzbin)]
    )
    bs_tatt.set_source_distribution(
        [block['nz_source', "z"] for _ in range(nzbin)],
        [block['nz_source', "bin_%d" % (i + 1)] for i in range(nzbin)],
        [(i + 1) for i in range(nzbin)]
    )
    # set linear matter power spectrum
    bs.set_pklin(
        block[names.matter_power_lin, 'k_h'],
        block[names.matter_power_lin, 'p_k'][0, :]
    )
    bs_tatt.set_pklin(
        block[names.matter_power_lin, 'k_h'],
        block[names.matter_power_lin, 'p_k'][0, :]
    )
    # set linear growth rate
    bs.set_lgr(
        block[names.growth_parameters, "z"],
        block[names.growth_parameters, "d_z"]
    )
    bs_tatt.set_lgr(
        block[names.growth_parameters, "z"],
        block[names.growth_parameters, "d_z"]
    )
    # set baryon paramter
    if block.has_value('baryon_parameters', 'fb'):
        fb = block['baryon_parameters', 'fb']
        bs.set_baryon_param({'fb': fb})

    # compute kernel
    bs.compute_kernel()
    bs_tatt.compute_kernel()

    sctname = "map3"

    for scomb in block['natural_components', 'sample_combinations']:
        name = '_'.join([str(s) for s in scomb])

        chi = bs.z2chi(zarray)
        chi_tatt = bs_tatt.z2chi(zarray_tatt)

        z2g0 = bs.z2g_dict[scomb[0]]
        z2g1 = bs.z2g_dict[scomb[1]]
        z2g2 = bs.z2g_dict[scomb[2]]
        weight_cosmo = z2g0(zarray) * z2g1(zarray) * z2g2(zarray) / chi * (1 + zarray) ** 3
        tmp_cosmo = np.einsum('ij,i->ij', predictions_newshape_cosmo, weight_cosmo)

        # Initialize total map3 with the shear-only result
        total_tmp = interpolate_cosmo(tmp_cosmo, chi, chi_tatt)

        chi2g0 = bs_tatt.chi2g_dict[scomb[0]]
        chi2g1 = bs_tatt.chi2g_dict[scomb[1]]
        chi2g2 = bs_tatt.chi2g_dict[scomb[2]]
        chi2W0 = bs_tatt.chi2W_dict[scomb[0]]
        chi2W1 = bs_tatt.chi2W_dict[scomb[1]]
        chi2W2 = bs_tatt.chi2W_dict[scomb[2]]

            # TATT weights
        weight_ddE = chi2g0(chi_tatt) * chi2g1(chi_tatt) * chi2W2(chi_tatt) / chi_tatt * (1 + zarray_tatt) ** 3
        weight_dEd = chi2g0(chi_tatt) * chi2W1(chi_tatt) * chi2g2(chi_tatt) / chi_tatt * (1 + zarray_tatt) ** 3
        weight_Edd = chi2W0(chi_tatt) * chi2g1(chi_tatt) * chi2g2(chi_tatt) / chi_tatt * (1 + zarray_tatt) ** 3
        weight_dEE = chi2g0(chi_tatt) * chi2W1(chi_tatt) * chi2W2(chi_tatt) / chi_tatt * (1 + zarray_tatt) ** 3
        weight_EEd = chi2W0(chi_tatt) * chi2W1(chi_tatt) * chi2g2(chi_tatt) / chi_tatt * (1 + zarray_tatt) ** 3
        weight_EdE = chi2W0(chi_tatt) * chi2g1(chi_tatt) * chi2W2(chi_tatt) / chi_tatt * (1 + zarray_tatt) ** 3
        weight_EEE = chi2W0(chi_tatt) * chi2W1(chi_tatt) * chi2W2(chi_tatt) / chi_tatt * (1 + zarray_tatt) ** 3

            # TATT integration
        tmp_ddE = np.einsum('ij,i->ij', predictions_newshape_tatt['ddE'], weight_ddE)
        tmp_dEd = np.einsum('ij,i->ij', predictions_newshape_tatt['dEd'], weight_dEd)
        tmp_Edd = np.einsum('ij,i->ij', predictions_newshape_tatt['Edd'], weight_Edd)
        tmp_dEE = np.einsum('ij,i->ij', predictions_newshape_tatt['dEE'], weight_dEE)
        tmp_EEd = np.einsum('ij,i->ij', predictions_newshape_tatt['EEd'], weight_EEd)
        tmp_EdE = np.einsum('ij,i->ij', predictions_newshape_tatt['EdE'], weight_EdE)
        tmp_EEE = np.einsum('ij,i->ij', predictions_newshape_tatt['EEE'], weight_EEE)

            # Sum TATT results and add to total
        total_tmp += tmp_ddE + tmp_dEd + tmp_Edd + tmp_dEE + tmp_EEd + tmp_EdE + tmp_EEE

        # Final integration (trapz) and write to datablock
        map3 = np.trapz(total_tmp, chi_tatt, axis=0)
        block[sctname, f'map3-bin_{name}'] = map3

    return 0


def cleanup(config):
    pass
