'''
Author     : Sunao Sugiyama
Last edit  : 2024/07/10 23:18:00
'''
from cosmosis.datablock import option_section, names
import numpy as np
import os
import fastnc
from astropy.cosmology import wCDM

def get_healpix_window_function(nside):
    import healpy as hp
    from scipy.interpolate import interp1d
    w = hp.sphtfunc.pixwin(nside)
    l = np.arange(3*nside)
    fnc = interp1d(l, w, kind='linear', bounds_error=False, fill_value=(1,0))
    window = lambda l1, l2, l3: fnc(l1) * fnc(l2) * fnc(l3)
    return window

def setup(options):
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
                      'NLA':options.get_bool(option_section, 'NLA', default=True),
                      'nrbin':options.get_int(option_section, "nrbin", default = 35)}

    config['TATT'] = False
    config['puv_grid'] = False
    config['remove_alignment'] = False
    config['Ct'] = 0.0

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
        config_halofit['puv_grid'] = True
        if options.has_value(option_section, "remove_alignment") and options.get_bool(option_section, "remove_alignment"):
            config['remove_alignment'] = True
        bs = fastnc.bispectrum.BispectrumTATT(config_halofit)
        config['TATT'] = True
        config['puv_grid'] = True
        config['Ct'] = options.get_double(option_section, "Ct", default = 0.0)
    else:
        raise ValueError('Invalid bispectrum model')
    if options.has_value(option_section, "use-pixwin") and options.get_bool(option_section, "use-pixwin"):
        bs.set_window_function(get_healpix_window_function(options.get_int(option_section, "nside")))
    config['bispectrum'] = bs
    if options.has_value(option_section, "select_tatt_component"):
        #Here we can select to compute only a single component (e.g. ddE, EdE)
        config['select_tatt_component'] = options.get_string(option_section, "select_tatt_component")
    else:
        config['select_tatt_component'] = None

    ########################################################################################    
    # 3PCF model (fastnc)
    t1 = np.logspace(
        np.log10(options.get_double(option_section, "theta-min") * np.pi/(180*60.0)), # radians
        np.log10(options.get_double(option_section, "theta-max") * np.pi/(180*60.0)), # radians
        options.get_int(option_section, "n-theta-bin"))
    t2 = np.logspace(
        np.log10(options.get_double(option_section, "theta-min") * np.pi/(180*60.0)), # radians
        np.log10(options.get_double(option_section, "theta-max") * np.pi/(180*60.0)), # radians
        options.get_int(option_section, "n-theta-bin"))
    phi = np.linspace(
        options.get_double(option_section, "phi-min"), 
        options.get_double(option_section, "phi-max"),
        options.get_int(option_section, "n-phi-bin"))
    config_3pcf = { \
            'Lmax':Lmax, \
            'Mmax':options.get_int(option_section, "Mmax", default = 30), \
            'projection': options.get_string(option_section, "projection", default = "x"), \
            'nfft': options.get_int(option_section, "nfft", default = 150), \
            't1':t1, 
            't2':t2, \
            'phi':phi, \
            'dlnt':options.get_double(option_section, "dlnt", default = None), \
            'mu':options.get_int_array_1d(option_section, "mu") if options.has_value(option_section, "mu") else [0,1,2,3], \
            'multipole_type':multipole_type, \
            'cache':options.get_bool(option_section, 'use_cache', default = False)}
    config['fastnc'] = fastnc.fastnc.FastNaturalComponents(config_3pcf)
    config['save_multipoles'] = options.get_bool(option_section, 'save_multipoles', default = False)
    
    return config

def execute(block, config):
    
    # BISPECTRUM ######################################
    # update bispectrum with inputs:
    bs = config['bispectrum']
    # cosmological parameters (fastnc accepts cosmo as astropy format)
    cosmo = wCDM(
                H0=100*block[names.cosmological_parameters, 'h0'], \
                Om0=block[names.cosmological_parameters, 'omega_m'], \
                Ode0=1.0-block[names.cosmological_parameters, 'omega_m'], \
                w0 = block[names.cosmological_parameters, 'w'], \
                meta = {'sigma8':block[names.cosmological_parameters, 'sigma_8'], \
                        'n':block[names.cosmological_parameters, 'n_s']}
    )
    bs.set_cosmology(cosmo)
    # Intrinsic parameter
    bs.set_NLA_param({'AIA':block['intrinsic_alignment_parameters', 'a1'], \
            'alphaIA':block['intrinsic_alignment_parameters', 'alpha1'] , \
            'z0':block['intrinsic_alignment_parameters', 'z_piv']})

    #Set TATT parameters
    if config['TATT']:
        bs.set_IA_param({'a1': block['intrinsic_alignment_parameters', 'a1'], \
                         'alphaIA': block['intrinsic_alignment_parameters', 'alpha1'], \
                         'a2': block['intrinsic_alignment_parameters', 'a2'], \
                         'alphaIA_2': block['intrinsic_alignment_parameters', 'a2'], \
                         'bias_ta': block['intrinsic_alignment_parameters', 'bias_ta'], \
                         'z0': block['intrinsic_alignment_parameters', 'z_piv']})
        bs.set_pknl(
            block[names.matter_power_nl, 'k_h'],
            block[names.matter_power_nl, 'p_k'][0,:])
        
    # set source distribution
    nzbin = block['nz_source', "nbin"]
    bs.set_source_distribution(
        [block['nz_source', "z"] for _ in range(nzbin)],
        [block['nz_source', "bin_%d" % (i+1)] for i in range(nzbin)],
        [(i+1) for i in range(nzbin)]
    )
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
        
    # update the interpolation.
    bs.compute_kernel()
    bs.interpolate(scombs=block['natural_components', 'sample_combinations'], select_tatt_component=config['select_tatt_component'],
                   Ct = config['Ct'], remove_alignment=config['remove_alignment'], puv_grid=config['puv_grid'])
    bs.decompose(scombs=block['natural_components', 'sample_combinations'], puv_grid=config['puv_grid'])

    # 3PCF ############################################
    nc = config['fastnc']
    sctname = "natural_components"

    for scomb in block['natural_components', 'sample_combinations']:
        print('calculating sample_combination:', scomb)
        # set bispectrum
        nc.set_bispectrum(bs)
        # nc.set_grid()
        nc.compute(scomb=scomb)

        # stack the Gamma
        # Because the triangle notations are different in TreeCorr and Porth et al.
        # we convert the theoretical prediction to TreeCorr convention.
        # Gamma = np.array(  [np.conjugate(np.moveaxis(nc.Gamma0, 0,1)), \
        #                     np.conjugate(np.moveaxis(nc.Gamma1, 1,2)), \
        #                     np.conjugate(np.moveaxis(nc.Gamma2, 1,2)), \
        #                     np.conjugate(np.moveaxis(nc.Gamma3, 1,2))])

        Gamma =-np.array([nc.Gamma0, nc.Gamma1, nc.Gamma3, nc.Gamma2])

        # Note this is equivalent to have
        # Gamma0 -> Gamma0
        # Gamma1 -> Gamma1
        # Gamma2 -> Gamma3
        # Gamma3 -> Gamma2


        # write to block
        # Note that the Gamma has the shape of 
        # (mu.size, phi.size, t1.size, t2.size)
        if np.isscalar(scomb):
            name = str(scomb)
        else:
            name = '_'.join([str(s) for s in scomb])
        block[sctname, f'real-bin_{name}'] = Gamma.real
        block[sctname, f'imag-bin_{name}'] = Gamma.imag

        if config['save_multipoles']:
            # stack the Gamma Multipoles
            GammaM = np.array([nc.Gamma0M, nc.Gamma1M, nc.Gamma2M, nc.Gamma3M])
            block[sctname, f'real-bin_{name}_M'] = GammaM.real
            block[sctname, f'imag-bin_{name}_M'] = GammaM.imag
    
    # write common parameters
    block[sctname, 'mu'] = nc.mu
    block[sctname, 'phi'] = nc.phi
    # nc.t1 and nc.t2 is the lower edges of bins
    # block[sctname, 't1'] = nc.t1
    # block[sctname, 't2'] = nc.t2
    # Conversion of Gamma to map3 in measurement
    # uses mean t1 and mean t2 as bin values 
    # (meand2, meand3 in TreeCorr)
    # (The other option is to use exp(logmeand2) etc)
    dlnt = np.diff(np.log(nc.t1))[0]
    # 1. meant1 = t1min * 2/3 (exp(3dlnt)-1)/(exp(2dlnt)-1)
    factor = 2.0/3.0*(np.exp(3*dlnt)-1)/(np.exp(2*dlnt)-1) - 0.015226 #empirical factor from COSMOGRID
    block[sctname, 't1'] = nc.t1 * factor
    block[sctname, 't2'] = nc.t2 * factor
    # 2. exp(logmeant1) = t1min * exp( (exp(2dlnt)(2dlnt-1)+1)/2/(exp(2dlnt)-1) )
    # factor = (np.exp(2*dlnt)*(2*dlnt-1)+1)/2/(np.exp(2*dlnt)-1)
    # factor = np.exp(factor)
    # block[sctname, 'meant1'] = nc.t1 * factor
    # block[sctname, 'meant2'] = nc.t2 * factor

    if config['save_multipoles']:
        M = np.arange(-nc.Mmax, nc.Mmax+1)
        block[sctname, 'M'] = M

    return 0

def cleanup(config):
    pass
