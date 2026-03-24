import numpy as np
from bihalofit import bihalofit
from funcs import f_h3, f_psi3, f_psi1, f_psi2, transform_gamma
import matplotlib.pyplot as plt
from halo_model import halo_model_bispectrum
from tree_level import tree_level_bispectrum
from gil_marin_bispectrum import gil_marin
from one_halo_NFW import one_halo_NFW_bispectrum
import time

def compute_3pcf(cosmo_parameters, dndz_file, d2_vals, u_vals, v_vals, output_suffix,
                 kmin=10**(-3), kmax=50, n_kbins=10000, chimin=1, chimax=4000,
                 n_chibins=10, niter=5, neval=100000, rmax=50, model='bihalofit', baryons = False):

    my_k = np.logspace(np.log10(kmin), np.log10(kmax), num=n_kbins)  # h/Mpc^-1
    my_z_new = np.linspace(0,3,1000)
    if model == 'bihalofit':
        model = bihalofit(cosmo_parameters, my_k, my_z_new)
    if model == 'halo model':
        kless = np.logspace(-2, np.log10(20), 20)
        zless = np.linspace(0, 2.7, 10)
        model = halo_model_bispectrum(cosmo_parameters, kless, zless, "Halo_bispectrum_with_baryons_and_NO_DM_PROFILE_apr26.npy")
    if model == 'tree level':
        model = tree_level_bispectrum(cosmo_parameters, my_k, my_z_new)
    limits = [[0, 2*np.pi],[0, np.pi/2],[0,rmax]]
    model.compute_lensing_kernel(chimin, chimax, 10000, dndz_file)

    aa_vals = np.ndarray(shape=len(d2_vals), dtype=complex)
    for i in range(len(d2_vals)):
        test_integration_0_re = model.gamma0(limits, d2_vals[i], u_vals[i], v_vals[i], chimin,
                                             chimax, n_chibins, niter, neval, baryons, imag=False)
        test_integration_0_im = model.gamma0(limits, d2_vals[i], u_vals[i], v_vals[i], chimin,
                                             chimax, n_chibins, niter, neval, baryons, imag=True)
        aa_vals[i] = test_integration_0_re.mean + 1j*test_integration_0_im.mean

    result_array_0 = transform_gamma(aa_vals, 0, d2_vals, u_vals, v_vals)
    np.save("Gamma0_real_" + output_suffix, np.real(result_array_0))
    np.save("Gamma0_imag_" + output_suffix, np.imag(result_array_0))

    bb_vals = np.ndarray(shape=len(d2_vals), dtype=complex)
    for i in range(len(d2_vals)):
        test_integration_1_re = model.gamma1(limits, d2_vals[i], u_vals[i], v_vals[i], chimin,
                                             chimax, n_chibins, niter, neval, baryons, imag=False)
        test_integration_1_im = model.gamma1(limits, d2_vals[i], u_vals[i], v_vals[i], chimin,
                                             chimax, n_chibins, niter, neval, baryons, imag=True)
        bb_vals[i] = test_integration_1_re.mean + 1j*test_integration_1_im.mean

    result_array_1 = transform_gamma(bb_vals, 1, d2_vals, u_vals, v_vals)
    np.save("Gamma1_real_" + output_suffix, np.real(result_array_1))
    np.save("Gamma1_imag_" + output_suffix, np.imag(result_array_1))

    cc_vals = np.ndarray(shape=len(d2_vals), dtype=complex)
    for i in range(len(d2_vals)):
        test_integration_2_re = model.gamma2(limits, d2_vals[i], u_vals[i], v_vals[i], chimin,
                                             chimax, n_chibins, niter, neval, baryons, imag=False)
        test_integration_2_im = model.gamma2(limits, d2_vals[i], u_vals[i], v_vals[i], chimin,
                                             chimax, n_chibins, niter, neval, baryons, imag=True)
        cc_vals[i] = test_integration_2_re.mean + 1j*test_integration_2_im.mean

    result_array_2 = transform_gamma(cc_vals, 2, d2_vals, u_vals, v_vals)
    np.save("Gamma2_real_" + output_suffix, np.real(result_array_2))
    np.save("Gamma2_imag_" + output_suffix, np.imag(result_array_2))

    dd_vals = np.ndarray(shape=len(d2_vals), dtype=complex)
    for i in range(len(d2_vals)):
        test_integration_3_re = model.gamma3(limits, d2_vals[i], u_vals[i], v_vals[i], chimin,
                                             chimax, n_chibins, niter, neval, baryons, imag=False)
        test_integration_3_im = model.gamma3(limits, d2_vals[i], u_vals[i], v_vals[i], chimin,
                                             chimax, n_chibins, niter, neval, baryons, imag=True)
        dd_vals[i] = test_integration_3_re.mean + 1j*test_integration_3_im.mean

    result_array_3 = transform_gamma(dd_vals, 3, d2_vals, u_vals, v_vals)
    np.save("Gamma3_real_" + output_suffix, np.real(result_array_3))
    np.save("Gamma3_imag_" + output_suffix, np.imag(result_array_3))

    np.save("Gamma_ttt_" + output_suffix, 1/4*(np.real(result_array_0+result_array_1+result_array_2+result_array_3)))

def plot_3pcf(files, labels, xvals, gamma_func, no_log = False):

    for i in range(len(files)):
        a = np.load(files[i])
        plt.plot(xvals, a * (-1) * np.pi ** 2,
             label=labels[i])
    plt.title('Halo model' + gamma_func + '(isosceles)')
    plt.xlabel("phi")
    if no_log == False:
        plt.yscale("log")
    plt.ylabel(gamma_func)
    plt.legend(fontsize=7)
    plt.grid()
    plt.show()

    first = np.load(files[0])
    second = np.load(files[2])
    plt.title('Ratio between 3pcfs')
    plt.plot(xvals, first/second)
    plt.xlabel("phi")
    if no_log == False:
        plt.yscale("log")
    plt.ylabel(gamma_func)
    plt.legend(fontsize=7)
    plt.grid()
    plt.show()

def compute_map3(cosmo_parameters, dndz_file, theta1_vals, theta2_vals, theta3_vals, output_suffix,
                 kmin=10**(-3), kmax=50, n_kbins=10000, chimin=1, chimax=4000,
                 n_chibins=10, niter=5, neval=50000, lmax=1000, model = 'bihalofit', baryons = False):

    my_k = np.logspace(np.log10(kmin), np.log10(kmax), num=n_kbins)  # h/Mpc^-1
    my_z_new = np.linspace(0,3,1000)
    if model == 'bihalofit':
        model = bihalofit(cosmo_parameters, my_k, my_z_new)
        model.name = 'bihalofit'
    if model == 'halo model':
        kless = np.logspace(-2, np.log10(20), 20)
        zless = np.linspace(0, 2.7, 10)
        #model = halo_model_bispectrum(cosmo_parameters, kless, zless, "new_halo_model_bispectrum_with_baryons_apr14.npy")
        model = halo_model_bispectrum(cosmo_parameters, kless, zless,
                                      "new_halo_model_bispectrum_with_baryons_apr21_Mc038.npy")
        #model = halo_model_bispectrum(cosmo_parameters, kless, zless,
        #                              "new_halo_model_bispectrum_with_baryons_apr14_dmprofnorelax.npy")
        model.name = 'halo model'
    if model == 'tree level':
        model = tree_level_bispectrum(cosmo_parameters, my_k, my_z_new)
        model.name = 'tree level'
    if model == 'gil marin':
        model = gil_marin(cosmo_parameters, my_k, my_z_new)
        model.name = 'gil marin'
    if model == 'one_halo_NFW':
        model = one_halo_NFW_bispectrum(cosmo_parameters, my_k, my_z_new)
        model.name = 'one_halo_NFW'
    limits = [[0, lmax],[0, lmax],[0, np.pi]]
    timelens = time.time()
    model.compute_lensing_kernel(chimin, chimax, 10000, dndz_file)
    #print("time to kernel:", time.time()-timelens)

    aa_vals = np.ndarray(shape=len(theta1_vals), dtype=complex)
    for i in range(len(theta1_vals)):
        test_integration = model.map3_loop(limits, theta1_vals[i], theta2_vals[i], theta3_vals[i], chimin,
                                             chimax, n_chibins, niter, neval, baryons)
        aa_vals[i] = test_integration
        print("aa",i)

    np.save("Map3_" + output_suffix, aa_vals)

def compute_3pcf_ro(cosmo_parameters, dndz_file, d2_vals, u_vals, v_vals, output_suffix,
                 kmin=10**(-3), kmax=50, n_kbins=10000, chimin=1, chimax=4000,
                 n_chibins=10, niter=5, neval=50000, rmax=50, model = 'bihalofit', baryons = False):

    my_k = np.logspace(np.log10(kmin), np.log10(kmax), num=n_kbins)  # h/Mpc^-1
    my_z_new = np.linspace(0,3,1000)
    if model == 'bihalofit':
        model = bihalofit(cosmo_parameters, my_k, my_z_new)
        model.name = 'bihalofit'
    if model == 'halo model':
        kless = np.logspace(-2, np.log10(20), 20)
        zless = np.linspace(0, 2.7, 10)
        #model = halo_model_bispectrum(cosmo_parameters, kless, zless, "new_halo_model_bispectrum_with_baryons_apr14.npy")
        model = halo_model_bispectrum(cosmo_parameters, kless, zless,
                                      "new_halo_model_bispectrum_with_baryons_apr21_Mc038.npy")
        #model = halo_model_bispectrum(cosmo_parameters, kless, zless,
        #                              "new_halo_model_bispectrum_with_baryons_apr14_dmprofnorelax.npy")
        model.name = 'halo model'
    if model == 'tree level':
        model = tree_level_bispectrum(cosmo_parameters, my_k, my_z_new)
        model.name = 'tree level'
    if model == 'gil marin':
        model = gil_marin(cosmo_parameters, my_k, my_z_new)
        model.name = 'gil marin'
    if model == 'one_halo_NFW':
        model = one_halo_NFW_bispectrum(cosmo_parameters, my_k, my_z_new)
        model.name = 'one_halo_NFW'
    limits = [[0, 2*np.pi],[0, np.pi/2],[0,rmax]]
    timelens = time.time()
    model.compute_lensing_kernel(chimin, chimax, 10000, dndz_file)
    #print("time to kernel:", time.time()-timelens)

    aa_vals = np.ndarray(shape=len(d2_vals), dtype=complex)
    for i in range(len(d2_vals)):
        test_integration_0_re = model.gamma0_loop(limits, d2_vals[i], u_vals[i], v_vals[i], chimin,
                                             chimax, n_chibins, niter, neval, baryons, imag=False)
        test_integration_0_im = model.gamma0_loop(limits, d2_vals[i], u_vals[i], v_vals[i], chimin,
                                             chimax, n_chibins, niter, neval, baryons, imag=True)
        aa_vals[i] = test_integration_0_re + 1j*test_integration_0_im
        print("aa",i)

    result_array_0 = transform_gamma(aa_vals, 0, d2_vals, u_vals, v_vals)
    np.save("Gamma0_real_" + output_suffix, np.real(result_array_0))
    np.save("Gamma0_imag_" + output_suffix, np.imag(result_array_0))
    #np.save("Gamma0_real_" + output_suffix, np.real(aa_vals))
    #np.save("Gamma0_imag_" + output_suffix, np.imag(aa_vals))

    bb_vals = np.ndarray(shape=len(d2_vals), dtype=complex)
    for i in range(len(d2_vals)):
        test_integration_1_re = model.gamma1_loop(limits, d2_vals[i], u_vals[i], v_vals[i], chimin,
                                             chimax, n_chibins, niter, neval, baryons, imag=False)
        test_integration_1_im = model.gamma1_loop(limits, d2_vals[i], u_vals[i], v_vals[i], chimin,
                                             chimax, n_chibins, niter, neval, baryons, imag=True)
        bb_vals[i] = test_integration_1_re + 1j*test_integration_1_im
        print("bb", i)

    result_array_1 = transform_gamma(bb_vals, 1, d2_vals, u_vals, v_vals)
    np.save("Gamma1_real_" + output_suffix, np.real(result_array_1))
    np.save("Gamma1_imag_" + output_suffix, np.imag(result_array_1))
    #np.save("Gamma1_real_" + output_suffix, np.real(bb_vals))
    #np.save("Gamma1_imag_" + output_suffix, np.imag(bb_vals))

    cc_vals = np.ndarray(shape=len(d2_vals), dtype=complex)
    for i in range(len(d2_vals)):
        test_integration_2_re = model.gamma2_loop(limits, d2_vals[i], u_vals[i], v_vals[i], chimin,
                                             chimax, n_chibins, niter, neval, baryons, imag=False)
        test_integration_2_im = model.gamma2_loop(limits, d2_vals[i], u_vals[i], v_vals[i], chimin,
                                             chimax, n_chibins, niter, neval, baryons, imag=True)
        cc_vals[i] = test_integration_2_re + 1j*test_integration_2_im
        print("cc", i)

    result_array_2 = transform_gamma(cc_vals, 2, d2_vals, u_vals, v_vals)
    np.save("Gamma2_real_" + output_suffix, np.real(result_array_2))
    np.save("Gamma2_imag_" + output_suffix, np.imag(result_array_2))
    #np.save("Gamma2_real_" + output_suffix, np.real(cc_vals))
    #np.save("Gamma2_imag_" + output_suffix, np.imag(cc_vals))

    dd_vals = np.ndarray(shape=len(d2_vals), dtype=complex)
    for i in range(len(d2_vals)):
        test_integration_3_re = model.gamma3_loop(limits, d2_vals[i], u_vals[i], v_vals[i], chimin,
                                             chimax, n_chibins, niter, neval, baryons, imag=False)
        test_integration_3_im = model.gamma3_loop(limits, d2_vals[i], u_vals[i], v_vals[i], chimin,
                                             chimax, n_chibins, niter, neval, baryons, imag=True)
        dd_vals[i] = test_integration_3_re + 1j*test_integration_3_im
        print("dd", i)

    result_array_3 = transform_gamma(dd_vals, 3, d2_vals, u_vals, v_vals)
    np.save("Gamma3_real_" + output_suffix, np.real(result_array_3))
    np.save("Gamma3_imag_" + output_suffix, np.imag(result_array_3))
    #np.save("Gamma3_real_" + output_suffix, np.real(dd_vals))
    #np.save("Gamma3_imag_" + output_suffix, np.imag(dd_vals))

    #np.save("Gamma_ttt_" + output_suffix, 1/4*(np.real(result_array_0+result_array_1+result_array_2+result_array_3)))

def compute_3pcf_ro_gamma0only(cosmo_parameters, dndz_file, d2_vals, u_vals, v_vals, output_suffix,
                 kmin=10**(-3), kmax=50, n_kbins=10000, chimin=1, chimax=4000,
                 n_chibins=10, niter=5, neval=100000, rmax=50, model = 'bihalofit', baryons = False):

    my_k = np.logspace(np.log10(kmin), np.log10(kmax), num=n_kbins)  # h/Mpc^-1
    my_z_new = np.linspace(0,3,30)
    if model == 'bihalofit':
        model = bihalofit(cosmo_parameters, my_k, my_z_new)
    if model == 'halo model':
        kless = np.logspace(-2, np.log10(20), 20)
        zless = np.linspace(0, 2.7, 10)
        #model = halo_model_bispectrum(cosmo_parameters, kless, zless, "new_halo_model_bispectrum_with_baryons_apr14.npy")
        model = halo_model_bispectrum(cosmo_parameters, kless, zless,
                                      "new_halo_model_bispectrum_with_baryons_apr21_Mc038.npy")
        #model = halo_model_bispectrum(cosmo_parameters, kless, zless,
        #                              "new_halo_model_bispectrum_with_baryons_apr14_dmprofnorelax.npy")
    if model == 'tree level':
        model = tree_level_bispectrum(cosmo_parameters, my_k, my_z_new)
    if model == 'gil marin':
        model = gil_marin(cosmo_parameters, my_k, my_z_new)
    limits = [[0, 2*np.pi],[0, np.pi/2],[0,rmax]]
    timelens = time.time()
    model.compute_lensing_kernel(chimin, chimax, 10000, dndz_file)
    #print("time to kernel:", time.time()-timelens)

    '''new test of off-isosceles'''

    angles_0 = np.ndarray(shape=(200, 300))
    angles_1 = np.ndarray(shape=(200, 300))
    ri = np.ndarray(shape=(300))
    for i in range(200):
        angles_0[i] = i*np.pi/400*np.ones(shape=(300))
        angles_1[i] = 2*np.pi*i/200*np.ones(shape=(300))
    for i in range(300):
        ri[i] = i/3+0.0001

    integ_new = np.ndarray(shape=(200,300))
    for j in range(200):
        ress = np.ndarray(shape=(200, 300))
        for k in range(200):
            y = np.ndarray(shape=(300, 3))
            for i in range(300):
                y[i][0] = 2*np.pi*j/200
                y[i][1] = k*np.pi/400+0.001
                y[i][2] = i/3+0.0001
            res = model.gamma0_integrand_ro(5, 0.8, 0, 1902.5, bary_correction=False, imag= False, y=y)
            #print("k and shape are", k, np.shape(res))
            ress[k] = res
            #print("ress shape is:", np.shape(ress))
            #print(ress)
        integ_part = np.trapz(ress, angles_0, axis=0)
        #print(integ_part)
        integ_new[j] = integ_part
        print("j is", j)
    full = np.trapz(integ_new, angles_1, axis=0)
    print("done")

    print(ri, full)
    final = 27 * (100 / 299792) ** 6 * model.omegam ** 3 / 8*np.trapz(full, ri)
    print(final)
    #myval = model.gamma0_integrand_ro(5, 0.8, 0.05, 500, bary_correction=False, imag= False, y=y)
    plt.plot(ri, full)
    plt.show()


    print(sajfbkg)
    aa_vals = np.ndarray(shape=len(d2_vals), dtype=complex)
    for i in range(len(d2_vals)):
        test_integration_0_re = model.gamma0_loop(limits, d2_vals[i], u_vals[i], v_vals[i], chimin,
                                             chimax, n_chibins, niter, neval, baryons, imag=False)
        test_integration_0_im = model.gamma0_loop(limits, d2_vals[i], u_vals[i], v_vals[i], chimin,
                                             chimax, n_chibins, niter, neval, baryons, imag=True)
        aa_vals[i] = test_integration_0_re + 1j*test_integration_0_im

    result_array_0 = transform_gamma(aa_vals, 0, d2_vals, u_vals, v_vals)
    np.save("training_set/Gamma0_real_" + output_suffix, np.real(result_array_0))
    np.save("training_set/Gamma0_imag_" + output_suffix, np.imag(result_array_0))

def compute_3pcf_for_emulator(cosmo_parameters, d2_vals, u_vals, v_vals, z,
                 kmin=10**(-3), kmax=50, n_kbins=10000, zmin = 0, zmax = 3, n_zbins = 100, niter=5, neval=100000, rmax=50, model = 'bihalofit', baryons = False):

    my_k = np.logspace(np.log10(kmin), np.log10(kmax), num=n_kbins)  # h/Mpc^-1
    #my_z_new = np.linspace(0,3,1000)
    my_z_new = np.linspace(zmin, zmax, n_zbins)
    if model == 'bihalofit':
        model = bihalofit(cosmo_parameters, my_k, my_z_new)
    if model == 'halo model':
        kless = np.logspace(-2, np.log10(20), 20)
        zless = np.linspace(0, 2.7, 10)
        #model = halo_model_bispectrum(cosmo_parameters, kless, zless, "new_halo_model_bispectrum_with_baryons_apr14.npy")
        model = halo_model_bispectrum(cosmo_parameters, kless, zless,
                                      "new_halo_model_bispectrum_with_baryons_apr21_Mc038.npy")
        #model = halo_model_bispectrum(cosmo_parameters, kless, zless,
        #                              "new_halo_model_bispectrum_with_baryons_apr14_dmprofnorelax.npy")
    if model == 'tree level':
        model = tree_level_bispectrum(cosmo_parameters, my_k, my_z_new)
    if model == 'gil marin':
        model = gil_marin(cosmo_parameters, my_k, my_z_new)
    limits = [[0, 2*np.pi],[0, np.pi/2],[0,rmax]]
    timelens = time.time()
    #print("time to kernel:", time.time()-timelens)

    chi = model.r_from_z_func(z)

    if model.is_knl_zero(z):

        return(10e20*np.ones(shape=(8)))

    else:

        try:
            constant = 27 * (100 / 299792) ** 6 * model.omegam ** 3 / 8
            output = np.ndarray(shape=(8))
            test_integration_0_re = model.gamma0_ro(limits, d2_vals, u_vals, v_vals, chi, niter, neval, baryons, imag=False)
            test_integration_0_im = model.gamma0_ro(limits, d2_vals, u_vals, v_vals, chi, niter, neval, baryons, imag=True)
            #print(test_integration_0_re, test_integration_0_im)
            aa_vals = test_integration_0_re.mean + 1j*test_integration_0_im.mean

            result_array_0 = transform_gamma(aa_vals, 0, d2_vals, u_vals, v_vals)

            #output[0] = np.real(result_array_0)
            output[0] = test_integration_0_re.mean
            output[1] = np.imag(result_array_0)

            test_integration_1_re = model.gamma1_ro(limits, d2_vals, u_vals, v_vals, chi, niter, neval, baryons, imag=False)
            test_integration_1_im = model.gamma1_ro(limits, d2_vals, u_vals, v_vals, chi, niter, neval, baryons, imag=True)
            bb_vals = test_integration_1_re.mean + 1j*test_integration_1_im.mean

            result_array_1 = transform_gamma(bb_vals, 1, d2_vals, u_vals, v_vals)

            output[2] = np.real(result_array_1)
            output[3] = np.imag(result_array_1)

            test_integration_2_re = model.gamma2_ro(limits, d2_vals, u_vals, v_vals, chi, niter, neval, baryons, imag=False)
            test_integration_2_im = model.gamma2_ro(limits, d2_vals, u_vals, v_vals, chi, niter, neval, baryons, imag=True)
            cc_vals = test_integration_2_re.mean + 1j*test_integration_2_im.mean

            result_array_2 = transform_gamma(cc_vals, 2, d2_vals, u_vals, v_vals)

            output[4] = np.real(result_array_2)
            output[5] = np.imag(result_array_2)

            test_integration_3_re = model.gamma3_ro(limits, d2_vals, u_vals, v_vals, chi, niter, neval, baryons, imag=False)
            test_integration_3_im = model.gamma3_ro(limits, d2_vals, u_vals, v_vals, chi, niter, neval, baryons, imag=True)
            dd_vals = test_integration_3_re.mean + 1j*test_integration_3_im.mean

            result_array_3 = transform_gamma(dd_vals, 3, d2_vals, u_vals, v_vals)

            output[6] = np.real(result_array_3)
            output[7] = np.imag(result_array_3)

            return(constant*output)

        except ValueError:

            print("value_error")
            return(10e20*np.ones(shape=(8)))

def integrate_emulator_predictions(cosmo_parameters, z_array, emulator_pred, kmin=10**(-3), kmax=50, n_kbins=10000,
                                   zmin = 0, zmax = 3, n_zbins = 100, model = 'bihalofit', baryons = False):

    my_k = np.logspace(np.log10(kmin), np.log10(kmax), num=n_kbins)  # h/Mpc^-1
    #my_z_new = np.linspace(0,3,1000)
    my_z_new = np.linspace(zmin, zmax, n_zbins)
    if model == 'bihalofit':
        model = bihalofit(cosmo_parameters, my_k, my_z_new)
    if model == 'halo model':
        kless = np.logspace(-2, np.log10(20), 20)
        zless = np.linspace(0, 2.7, 10)
        #model = halo_model_bispectrum(cosmo_parameters, kless, zless, "new_halo_model_bispectrum_with_baryons_apr14.npy")
        model = halo_model_bispectrum(cosmo_parameters, kless, zless,
                                      "new_halo_model_bispectrum_with_baryons_apr21_Mc038.npy")
        #model = halo_model_bispectrum(cosmo_parameters, kless, zless,
        #                              "new_halo_model_bispectrum_with_baryons_apr14_dmprofnorelax.npy")
    if model == 'tree level':
        model = tree_level_bispectrum(cosmo_parameters, my_k, my_z_new)
    if model == 'gil marin':
        model = gil_marin(cosmo_parameters, my_k, my_z_new)
    limits = [[0, 2*np.pi],[0, np.pi/2],[0,rmax]]
    timelens = time.time()
    #print("time to kernel:", time.time()-timelens)

    chi = model.r_from_z_func(z_array)
    to_integrate = (model.lensing_kernel(chi) * (1 + model.z_from_r_func(chi))) ** 3 / chi * emulator_pred
    final_result = np.trapz(to_integrate, chi)

    return(final_result)
