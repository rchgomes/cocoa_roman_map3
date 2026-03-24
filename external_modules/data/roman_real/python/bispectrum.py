import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.interpolate import RegularGridInterpolator
from scipy.special import jv
from classy import Class
import vegas
from vegas import Integrator
import functools
from funcs import remove_zeros
from funcs import uhat
from inspect import signature

#Verify dependance of k_nl and n_eff by cosmology
class bispectrum:

    def __init__(self, cosmo_params, k, z):

        self.cosmo_params = cosmo_params
        self.my_cosmo = Class()
        self.my_cosmo.set(self.cosmo_params)
        self.my_cosmo.compute()

        self.k = k
        self.z = z

        self.omegam = self.cosmo_params['Omega_b'] + self.cosmo_params['Omega_cdm']
        h = self.cosmo_params['h']
        self.knl = self.my_cosmo.nonlinear_scale(z, len(z))

        #print(self.knl)
        self.knl_interp = interp1d(z, self.knl, bounds_error = False, fill_value = self.knl[-1])

        self.Pk_linear = np.ndarray(shape=(len(k),len(z)))
        self.Pk_nonlinear = np.ndarray(shape=(len(k), len(z)))
        for i in range(len(self.Pk_linear)):
            for j in range(len(self.Pk_linear[0])):
                self.Pk_linear[i][j] = self.my_cosmo.pk_lin(k[i]*h,z[j])
                self.Pk_nonlinear[i][j] = self.my_cosmo.pk(k[i]*h, z[j])

        self.Pk_linear *= h ** 3
        self.Pk_nonlinear *= h ** 3

        newpkinterp = []
        newpkinterp_nonlin = []
        for kii in range(len(self.k)):
            newpkinterp.append([])
            newpkinterp_nonlin.append([])
            for zi in range(len(self.z)):
                newpkinterp[kii].append(self.Pk_linear[kii][zi])
                newpkinterp_nonlin[kii].append(self.Pk_nonlinear[kii][zi])

        self.PL = RegularGridInterpolator((self.k, self.z), newpkinterp, bounds_error = False, fill_value = 0)
        self.PNL = RegularGridInterpolator((self.k, self.z), newpkinterp_nonlin, bounds_error=False, fill_value=0)
        self.r_from_z, self.dzdr_of_z = self.my_cosmo.z_of_r(z)

        self.r_from_z *= h
        self.dzdr_of_z *= h

        self.dzdr_from_r_func = interp1d(self.r_from_z, self.dzdr_of_z,  bounds_error = False, fill_value = 0)
        self.z_from_r_func = interp1d(self.r_from_z, z, bounds_error = False, fill_value = 0)
        self.r_from_z_func = interp1d(z, self.r_from_z, bounds_error=False, fill_value=0)

    def compute_kernel(self, k1, k2, k3):
        dot = (-k3 ** 2 + k1 ** 2 + k2 ** 2) / 2
        f2 = 5 / 7 + 2 * dot ** 2 / (7 * k1 ** 2 * k2 ** 2) - dot * (1 / k1 ** 2 + 1 / k2 ** 2) / 2
        return (f2)

    def b_of_l(self, l1, l2, l3, chi, bary_correction):

        if self.name == 'one_halo_NFW':
            b_of_l_vals = self.matter_bispectrum(l1,l2,l3)
        else:
            z_actual = self.z_from_r_func(chi)
            b_of_l_vals = self.matter_bispectrum(z_actual, l1/chi, l2/chi, l3/chi, bary_correction)

        return (b_of_l_vals)

    def compute_lensing_kernel(self, chimin, chimax, npoints, dndz):

        chivals = np.linspace(chimin, chimax, npoints)
        zvals = self.z_from_r_func(chivals)
        nz = interp1d(dndz[:, 0], dndz[:, 1], bounds_error = False, fill_value = 0)

        def_z = np.trapz(dndz[:,1], dndz[:,0])
        #print(def_z)

        try:
            nvals = nz(zvals)*self.dzdr_from_r_func(chivals)
            #nvals = nz(zvals)
        except ValueError:
            raise ValueError(f"There was an interpolation problem. Your requested redshifts span from z={zvals[0]} to z={zvals[-1]}, "
                             f"while your redshift distribution function spans from z={dndz[0][0]} to z={dndz[-1][0]}")

        #print(chimin, chimax)
        def_int = np.trapz(nvals, chivals)
        nvals /= def_int
        #plt.plot(chivals, nvals)
        #plt.show()
        lensing_kernel = np.zeros_like(chivals)
        for chii in range(len(chivals)-1):
            integrand = nvals[chii:]*(chivals[chii:]-chivals[chii]*np.ones_like(chivals[chii:]))/chivals[chii:]
            lensing_kernel[chii] = np.trapz(integrand, chivals[chii:])

        self.lensing_kernel  = interp1d(chivals, lensing_kernel, bounds_error = False, fill_value = 0)
        #plt.plot(chivals, lensing_kernel)
        #plt.yscale("log")
        #plt.show()

    def compute_kappa_bispectrum(self, l1, l2, l3, chimin, chimax, npoints, bary_correction):

        constant = 27*(100/299792)**6*self.omegam**3/8
        chivals_simple = np.linspace(chimin, chimax, npoints)
        if type(l1) == np.float64 or type(l1) == float or type(l1) == int:
            integrand = (self.lensing_kernel(chivals_simple) * (
                        1 + self.z_from_r_func(chivals_simple))) ** 3 / chivals_simple * self.b_of_l(
                l1, l2, l3, chivals_simple, bary_correction)
            integral = np.trapz(integrand, chivals_simple)
        else:
            chivals_extended = np.ndarray(shape=(len(l1), len(chivals_simple)))
            for l1i in range(len(l1)):
                chivals_extended[l1i] = chivals_simple
            integrand = np.ndarray(shape=(len(l1), len(chivals_simple)))
            for chii in range(len(chivals_simple)):
                integrand[:,chii] = (self.lensing_kernel(chivals_simple[chii]) * (1 + self.z_from_r_func(chivals_simple[chii]))) ** 3 / chivals_simple[chii] * self.b_of_l(
                    l1, l2, l3, chivals_simple[chii], bary_correction)
            integral = np.trapz(integrand, chivals_extended)

        return(constant*integral)

    def create_interpolated_kappa_bispectrum(self, lvec):

        self.interpolated_kappa_bispectrum = RegularGridInterpolator((lvec, lvec, lvec), self.kappa_bispectrum, bounds_error = False, fill_value = 0)

    def map3_integrand(self, theta1, theta2, theta3, chi, bary_correction, y):

        theta1_conv = theta1 * np.pi/ (60 * 180)
        theta2_conv = theta2 * np.pi / (60 * 180)
        theta3_conv = theta3 * np.pi / (60 * 180)

        l1 = y[:,0]
        l2 = y[:,1]
        phi = y[:,2]

        const = 2/(2*np.pi)**3
        factor = l1*l2
        l3 = np.sqrt(l1**2+l2**2+2*l1*l2*np.cos(phi))
        final_integrand = const*factor*uhat(theta1_conv*l1)*uhat(theta2_conv*l2)*uhat(theta3_conv*l3)*self.b_of_l(l1,l2, l3, chi, bary_correction)

        return(final_integrand)

    def map3(self, integ_limits, theta1, theta2, theta3, chi, niter, nevalu, baryons = False):

        timenow = time.time()
        int_obj = Integrator(integ_limits)

        @vegas.batchintegrand
        def my_integrand(y):
            return(functools.partial(self.map3_integrand, theta1, theta2, theta3, chi, baryons)(y))

        result = int_obj(my_integrand, nitn=niter, neval=nevalu)
        timeafter = time.time()
        print("the time to integrate is", timeafter-timenow)

        return(result)

    def map3_loop(self, integ_limits, theta1, theta2, theta3, chi_min, chi_max, n_chisteps, niter, nevalu, baryons = False):

        constant = 27 * (100 / 299792) ** 6 * self.omegam ** 3 / 8
        timenow = time.time()
        chivals = np.linspace(chi_min, chi_max, n_chisteps)
        result_of_chi = np.ndarray(shape=(len(chivals)))
        for i in range(len(chivals)):
            result_of_chi[i] = self.map3(integ_limits, theta1, theta2, theta3, chivals[i], niter, nevalu, baryons).mean
        to_integrate = (self.lensing_kernel(chivals) * (1 + self.z_from_r_func(chivals))) ** 3 / chivals * result_of_chi
        final_result = np.trapz(to_integrate, chivals)
        timeafter = time.time()
        #print("the time to integrate is", timeafter-timenow)

        return(final_result*constant)

    def gamma0_integrand(self, r, u, v, chimin, chimax, n_chibins, bary_correction, imag, y):

        x2 = r*np.pi/(60*180)
        x3 = u*x2
        x1 = v*x3+x2

        phi = y[:,0]
        psi = y[:,1]
        R = y[:,2]

        sinn = (2 * np.cos(psi) ** 2 - 1) * np.sin(phi)
        coss = np.cos(phi)+np.sin(2*psi)

        beta_bar_times2 = np.arctan2(sinn, coss)

        outside_term = 1 / (6 * 32 * np.pi ** 5) * np.sin(2 * psi) * R ** 3 * jv(6, R)

        exp_re = np.cos(beta_bar_times2)
        exp_im = np.sin(beta_bar_times2)

        '''internal angles of the triangle'''
        phi1 = np.arccos((x1**2-x2**2-x3**2)/(-2*x2*x3))
        phi2 = np.arccos((x2**2-x3**2-x1**2)/(-2*x3*x1))
        phi3 = np.arccos((x3**2-x1**2-x2**2)/(-2*x1*x2))
        print("here are the values", phi1, phi2, phi3)
        #print(phi1,phi2,phi3)
        """inside terms:"""

        A1_prime = np.sqrt((x3*np.cos(psi))**2+(x2*np.sin(psi))**2+x2*x3*np.sin(2*psi)*np.cos(phi+phi1))

        A1p_sin_alpha1 = (x3*np.cos(psi)-x2*np.sin(psi))*np.sin((phi+phi1)/2)
        A1p_cos_alpha1 = (x3 * np.cos(psi) + x2 * np.sin(psi)) * np.cos((phi + phi1) / 2)
        alpha1 = np.arctan2(A1p_sin_alpha1, A1p_cos_alpha1)
        E1_re = np.cos(phi2-phi3-6*alpha1)
        E1_im = np.sin(phi2-phi3-6*alpha1)

        A2_prime = np.sqrt((x1*np.cos(psi))**2+(x3*np.sin(psi))**2+x3*x1*np.sin(2*psi)*np.cos(phi+phi2))
        A2p_sin_alpha2 = (x1*np.cos(psi)-x3*np.sin(psi))*np.sin((phi+phi2)/2)
        A2p_cos_alpha2 = (x1 * np.cos(psi) + x3 * np.sin(psi)) * np.cos((phi + phi2) / 2)
        alpha2 = np.arctan2(A2p_sin_alpha2, A2p_cos_alpha2)
        E2_re = np.cos(phi3 - phi1 - 6*alpha2)
        E2_im = np.sin(phi3 - phi1 - 6*alpha2)

        A3_prime = np.sqrt((x2*np.cos(psi))**2+(x1*np.sin(psi))**2+x1*x2*np.sin(2*psi)*np.cos(phi+phi3))
        A3p_sin_alpha3 = (x2*np.cos(psi)-x1*np.sin(psi))*np.sin((phi+phi3)/2)
        A3p_cos_alpha3 = (x2 * np.cos(psi) + x1 * np.sin(psi)) * np.cos((phi + phi3) / 2)
        alpha3 = np.arctan2(A3p_sin_alpha3, A3p_cos_alpha3)
        E3_re = np.cos(phi1 - phi2 - 6*alpha3)
        E3_im = np.sin(phi1 - phi2 - 6*alpha3)

        if imag == False:

            E1 = exp_re * E1_re - exp_im * E1_im
            E2 = exp_re * E2_re - exp_im * E2_im
            E3 = exp_re * E3_re - exp_im * E3_im

        if imag == True:

            E1 = exp_re * E1_im + exp_im * E1_re
            E2 = exp_re * E2_im + exp_im * E2_re
            E3 = exp_re * E3_im + exp_im * E3_re

        l1_1 = R*np.cos(psi)/A1_prime
        l1_2 = R*np.cos(psi)/A2_prime
        l1_3 = R*np.cos(psi)/A3_prime

        l2_1 = R*np.sin(psi)/A1_prime
        l2_2 = R*np.sin(psi)/A2_prime
        l2_3 = R*np.sin(psi)/A3_prime

        l3_1 = np.sqrt((R/A1_prime)**2*(1 + 2*np.cos(psi)*np.sin(psi)*np.cos(phi)))
        l3_2 = np.sqrt((R / A2_prime) ** 2 * (1 + 2 * np.cos(psi) * np.sin(psi) * np.cos(phi)))
        l3_3 = np.sqrt((R / A3_prime) ** 2 * (1 + 2 * np.cos(psi) * np.sin(psi) * np.cos(phi)))

        #print(l1_1, l2_1,l3_1)
        if l3_1.any() == 0:
            l3_1 = remove_zeros(l3_1)

        if l3_2.any() == 0:
            l3_2 = remove_zeros(l3_2)

        if l3_3.any() == 0:
            l3_3 = remove_zeros(l3_3)

        if np.any(l2_1) == 0 or np.any(l2_2) == 0 or np.any(l2_3) == 0:
            print("yep")
            print(R,psi, A1_prime, phi)

        first_term = E1 / A1_prime ** 4 * self.compute_kappa_bispectrum(l1_1,l2_1, l3_1, chimin, chimax, n_chibins, bary_correction)
        second_term = E2 / A2_prime ** 4 * self.compute_kappa_bispectrum(l1_2, l2_2, l3_2, chimin, chimax, n_chibins, bary_correction)
        third_term = E3 / A3_prime ** 4 * self.compute_kappa_bispectrum(l1_3, l2_3, l3_3, chimin, chimax, n_chibins, bary_correction)

        complete_integrand = outside_term*(first_term+second_term+third_term)

        return(complete_integrand)

    def gamma0(self, integ_limits, r, u, v, chimin, chimax, n_chibins, niter, nevalu, baryons = False, imag = False):

        timenow = time.time()
        int_obj = Integrator(integ_limits)

        @vegas.batchintegrand
        def my_integrand(y):
            return(functools.partial(self.gamma0_integrand, r, u, v, chimin, chimax, n_chibins, baryons, imag)(y))

        result = int_obj(my_integrand, nitn=niter, neval=nevalu)
        timeafter = time.time()
        print("the time to integrate is", timeafter-timenow)

        return(result)

    def gamma1_integrand(self, r, u, v, chimin, chimax, n_chibins, bary_correction, imag, y):

        x2 = r*np.pi/(60*180)
        x3 = u*x2
        x1 = v*x3+x2

        phi = y[:,0]
        psi = y[:,1]
        R = y[:,2]

        outside_term = 1 / (6 * 32 * np.pi ** 5) * np.sin(2 * psi) * R ** 3 * jv(2, R)

        '''internal angles of the triangle'''
        phi1 = np.arccos((x1**2-x2**2-x3**2)/(-2*x2*x3))
        phi2 = np.arccos((x2**2-x3**2-x1**2)/(-2*x3*x1))
        phi3 = np.arccos((x3**2-x1**2-x2**2)/(-2*x1*x2))

        sinn = (2 * np.cos(psi) ** 2 - 1) * np.sin(phi)
        coss = np.cos(phi)+np.sin(2*psi)

        beta_bar_times2 = np.arctan2(sinn, coss)

        """inside terms:"""

        A1_prime = np.sqrt((x3*np.cos(psi))**2+(x2*np.sin(psi))**2+x2*x3*np.sin(2*psi)*np.cos(phi+phi1))
        A1p_sin_alpha1 = (x3*np.cos(psi)-x2*np.sin(psi))*np.sin((phi+phi1)/2)
        A1p_cos_alpha1 = (x3 * np.cos(psi) + x2 * np.sin(psi)) * np.cos((phi + phi1) / 2)
        alpha1 = np.arctan2(A1p_sin_alpha1, A1p_cos_alpha1)
        E1_re = np.cos(phi3-phi2-2*alpha1-beta_bar_times2)
        E1_im = np.sin(phi3-phi2-2*alpha1-beta_bar_times2)

        A2_prime = np.sqrt((x1*np.cos(psi))**2+(x3*np.sin(psi))**2+x3*x1*np.sin(2*psi)*np.cos(phi+phi2))
        A2p_sin_alpha2 = (x1*np.cos(psi)-x3*np.sin(psi))*np.sin((phi+phi2)/2)
        A2p_cos_alpha2 = (x1 * np.cos(psi) + x3 * np.sin(psi)) * np.cos((phi + phi2) / 2)
        alpha2 = np.arctan2(A2p_sin_alpha2, A2p_cos_alpha2)
        E2_re = np.cos(2*phi-2*alpha2+beta_bar_times2+phi3-phi1-2*phi2)
        E2_im = np.sin(2*phi-2*alpha2+beta_bar_times2+phi3-phi1-2*phi2)

        A3_prime = np.sqrt((x2*np.cos(psi))**2+(x1*np.sin(psi))**2+x1*x2*np.sin(2*psi)*np.cos(phi+phi3))
        A3p_sin_alpha3 = (x2*np.cos(psi)-x1*np.sin(psi))*np.sin((phi+phi3)/2)
        A3p_cos_alpha3 = (x2 * np.cos(psi) + x1 * np.sin(psi)) * np.cos((phi + phi3) / 2)
        alpha3 = np.arctan2(A3p_sin_alpha3, A3p_cos_alpha3)
        E3_re = np.cos(-2*phi-2*alpha3+beta_bar_times2+phi1-phi2+2*phi3)
        E3_im = np.sin(-2*phi-2*alpha3+beta_bar_times2+phi1-phi2+2*phi3)

        if imag == False:
            E1 = E1_re
            E2 = E2_re
            E3 = E3_re

        if imag == True:
            E1 = E1_im
            E2 = E2_im
            E3 = E3_im

        l1_1 = R*np.cos(psi)/A1_prime
        l1_2 = R*np.cos(psi)/A2_prime
        l1_3 = R*np.cos(psi)/A3_prime

        l2_1 = R*np.sin(psi)/A1_prime
        l2_2 = R*np.sin(psi)/A2_prime
        l2_3 = R*np.sin(psi)/A3_prime

        l3_1 = np.sqrt((R/A1_prime)**2*(1 + 2*np.cos(psi)*np.sin(psi)*np.cos(phi)))
        l3_2 = np.sqrt((R / A2_prime) ** 2 * (1 + 2 * np.cos(psi) * np.sin(psi) * np.cos(phi)))
        l3_3 = np.sqrt((R / A3_prime) ** 2 * (1 + 2 * np.cos(psi) * np.sin(psi) * np.cos(phi)))

        if l3_1.any() == 0:
            l3_1 = remove_zeros(l3_1)

        if l3_2.any() == 0:
            l3_2 = remove_zeros(l3_2)

        if l3_3.any() == 0:
            l3_3 = remove_zeros(l3_3)

        first_term = E1 / A1_prime ** 4 * self.compute_kappa_bispectrum(l1_1,l2_1, l3_1, chimin, chimax, n_chibins, bary_correction)
        second_term = E2 / A2_prime ** 4 * self.compute_kappa_bispectrum(l1_2, l2_2, l3_2, chimin, chimax, n_chibins, bary_correction)
        third_term = E3 / A3_prime ** 4 * self.compute_kappa_bispectrum(l1_3, l2_3, l3_3, chimin, chimax, n_chibins, bary_correction)

        complete_integrand = outside_term*(first_term+second_term+third_term)

        return(complete_integrand)

    def gamma1(self, integ_limits, r, u, v, chimin, chimax, n_chibins, niter, nevalu, baryons = False, imag = False):

        timenow = time.time()
        int_obj = Integrator(integ_limits)

        @vegas.batchintegrand
        def my_integrand(y):
            return(functools.partial(self.gamma1_integrand, r, u, v, chimin, chimax, n_chibins, baryons, imag)(y))

        result = int_obj(my_integrand, nitn=niter, neval=nevalu)
        timeafter = time.time()
        print("the time to integrate is", timeafter-timenow)

        return(result)

    def gamma2(self, integ_limits, r, u, v, chimin, chimax, n_chibins, niter, nevalu, baryons = False, imag = False):

        x2 = r*np.pi/(60*180)
        x3 = u*x2
        x1 = v*x3+x2

        new_r = x3*(60*180)/np.pi
        new_u = x1/x3
        new_v = (x2-x3)/x1

        timenow = time.time()
        int_obj = Integrator(integ_limits)

        @vegas.batchintegrand
        def my_integrand(y):
            return (functools.partial(self.gamma1_integrand, new_r, new_u, new_v, chimin, chimax, n_chibins, baryons, imag)(y))

        result = int_obj(my_integrand, nitn=niter, neval=nevalu)
        timeafter = time.time()
        print("the time to integrate is", timeafter-timenow)

        return(result)

    def gamma3(self, integ_limits, r, u, v, chimin, chimax, n_chibins, niter, nevalu, baryons = False, imag = False):

        x2 = r * np.pi / (60 * 180)
        x3 = u * x2
        x1 = v * x3 + x2

        new_r = x1 * (60 * 180) / np.pi
        new_u = x2 / x1
        new_v = (x3 - x1) / x2

        timenow = time.time()
        int_obj = Integrator(integ_limits)

        @vegas.batchintegrand
        def my_integrand(y):
            return (functools.partial(self.gamma1_integrand, new_r, new_u, new_v, chimin, chimax, n_chibins, baryons, imag)(y))

        result = int_obj(my_integrand, nitn=niter, neval=nevalu)
        timeafter = time.time()
        print("the time to integrate is", timeafter - timenow)

        return (result)

    def gamma0_integrand_ro(self, r, u, v, chi,bary_correction, imag, y):

        x2 = r*np.pi/(60*180)
        x3 = u*x2
        x1 = v*x3+x2

        phi = y[:,0]
        psi = y[:,1]
        R = y[:,2]

        sinn = (2 * np.cos(psi) ** 2 - 1) * np.sin(phi)
        coss = np.cos(phi)+np.sin(2*psi)

        beta_bar_times2 = np.arctan2(sinn, coss)

        outside_term = 1 / (6 * 32 * np.pi ** 5) * np.sin(2 * psi) * R ** 3 * jv(6, R)

        exp_re = np.cos(beta_bar_times2)
        exp_im = np.sin(beta_bar_times2)

        '''internal angles of the triangle'''
        phi1 = np.arccos((x1**2-x2**2-x3**2)/(-2*x2*x3))
        phi2 = np.arccos((x2**2-x3**2-x1**2)/(-2*x3*x1))
        phi3 = np.arccos((x3**2-x1**2-x2**2)/(-2*x1*x2))
        #print("here are the values", phi1, phi2, phi3)
        #print(phi1,phi2,phi3)
        """inside terms:"""

        A1_prime = np.sqrt((x3*np.cos(psi))**2+(x2*np.sin(psi))**2+x2*x3*np.sin(2*psi)*np.cos(phi+phi1))
        #print("val of A1prime", A1_prime)

        A1p_sin_alpha1 = (x3*np.cos(psi)-x2*np.sin(psi))*np.sin((phi+phi1)/2)
        A1p_cos_alpha1 = (x3 * np.cos(psi) + x2 * np.sin(psi)) * np.cos((phi + phi1) / 2)
        alpha1 = np.arctan2(A1p_sin_alpha1, A1p_cos_alpha1)
        E1_re = np.cos(phi2-phi3-6*alpha1)
        E1_im = np.sin(phi2-phi3-6*alpha1)

        A2_prime = np.sqrt((x1*np.cos(psi))**2+(x3*np.sin(psi))**2+x3*x1*np.sin(2*psi)*np.cos(phi+phi2))
        A2p_sin_alpha2 = (x1*np.cos(psi)-x3*np.sin(psi))*np.sin((phi+phi2)/2)
        A2p_cos_alpha2 = (x1 * np.cos(psi) + x3 * np.sin(psi)) * np.cos((phi + phi2) / 2)
        alpha2 = np.arctan2(A2p_sin_alpha2, A2p_cos_alpha2)
        E2_re = np.cos(phi3 - phi1 - 6*alpha2)
        E2_im = np.sin(phi3 - phi1 - 6*alpha2)

        A3_prime = np.sqrt((x2*np.cos(psi))**2+(x1*np.sin(psi))**2+x1*x2*np.sin(2*psi)*np.cos(phi+phi3))
        A3p_sin_alpha3 = (x2*np.cos(psi)-x1*np.sin(psi))*np.sin((phi+phi3)/2)
        A3p_cos_alpha3 = (x2 * np.cos(psi) + x1 * np.sin(psi)) * np.cos((phi + phi3) / 2)
        alpha3 = np.arctan2(A3p_sin_alpha3, A3p_cos_alpha3)
        E3_re = np.cos(phi1 - phi2 - 6*alpha3)
        E3_im = np.sin(phi1 - phi2 - 6*alpha3)

        if imag == False:

            E1 = exp_re * E1_re - exp_im * E1_im
            E2 = exp_re * E2_re - exp_im * E2_im
            E3 = exp_re * E3_re - exp_im * E3_im

        if imag == True:

            E1 = exp_re * E1_im + exp_im * E1_re
            E2 = exp_re * E2_im + exp_im * E2_re
            E3 = exp_re * E3_im + exp_im * E3_re

        l1_1 = R*np.cos(psi)/A1_prime
        l1_2 = R*np.cos(psi)/A2_prime
        l1_3 = R*np.cos(psi)/A3_prime

        l2_1 = R*np.sin(psi)/A1_prime
        l2_2 = R*np.sin(psi)/A2_prime
        l2_3 = R*np.sin(psi)/A3_prime

        l3_1 = np.sqrt((R/A1_prime)**2*(1 + 2*np.cos(psi)*np.sin(psi)*np.cos(phi)))
        l3_2 = np.sqrt((R / A2_prime) ** 2 * (1 + 2 * np.cos(psi) * np.sin(psi) * np.cos(phi)))
        l3_3 = np.sqrt((R / A3_prime) ** 2 * (1 + 2 * np.cos(psi) * np.sin(psi) * np.cos(phi)))

        #print(l1_1, l2_1,l3_1)
        if l3_1.any() == 0:
            l3_1 = remove_zeros(l3_1)

        if l3_2.any() == 0:
            l3_2 = remove_zeros(l3_2)

        if l3_3.any() == 0:
            l3_3 = remove_zeros(l3_3)

        #if np.any(l2_1) == 0 or np.any(l2_2) == 0 or np.any(l2_3) == 0:
            #print("yep")
            #print(R,psi, A1_prime, phi)

        first_term = E1 / A1_prime ** 4 * self.b_of_l(l1_1,l2_1, l3_1, chi, bary_correction)
        second_term = E2 / A2_prime ** 4 * self.b_of_l(l1_2, l2_2, l3_2, chi, bary_correction)
        third_term = E3 / A3_prime ** 4 * self.b_of_l(l1_3, l2_3, l3_3,chi, bary_correction)

        complete_integrand = outside_term*(first_term+second_term+third_term)

        return(complete_integrand)

    def gamma0_ro(self, integ_limits, r, u, v, chi, niter, nevalu, baryons, imag = False):

        timesv= time.time()
        int_obj = Integrator(integ_limits)

        @vegas.batchintegrand
        def my_integrand(y):
            return(functools.partial(self.gamma0_integrand_ro, r, u, v, chi, baryons, imag)(y))

        result = int_obj(my_integrand, nitn=niter, neval=nevalu)
        #print("time to single unprojected integral with niter and neval:", time.time()-timesv, niter, nevalu)

        return(result)

    def gamma0_loop(self, integ_limits, r, u, v, chi_min, chi_max, n_chisteps, niter, nevalu, baryons = False, imag = False):

        constant = 27 * (100 / 299792) ** 6 * self.omegam ** 3 / 8
        timenow = time.time()
        chivals = np.linspace(chi_min, chi_max, n_chisteps)
        result_of_chi = np.ndarray(shape=(len(chivals)))
        for i in range(len(chivals)):
            result_of_chi[i] = self.gamma0_ro(integ_limits, r, u, v, chivals[i], niter, nevalu, baryons, imag).mean
        to_integrate = (self.lensing_kernel(chivals) * (1 + self.z_from_r_func(chivals))) ** 3 / chivals * result_of_chi
        final_result = np.trapz(to_integrate, chivals)
        timeafter = time.time()
        #print("the time to integrate is", timeafter-timenow)

        return(final_result*constant)

    def gamma1_integrand_ro(self, r, u, v, chi, bary_correction, imag, y):

        x2 = r*np.pi/(60*180)
        x3 = u*x2
        x1 = v*x3+x2

        phi = y[:,0]
        psi = y[:,1]
        R = y[:,2]

        outside_term = 1 / (6 * 32 * np.pi ** 5) * np.sin(2 * psi) * R ** 3 * jv(2, R)

        '''internal angles of the triangle'''
        phi1 = np.arccos((x1**2-x2**2-x3**2)/(-2*x2*x3))
        phi2 = np.arccos((x2**2-x3**2-x1**2)/(-2*x3*x1))
        phi3 = np.arccos((x3**2-x1**2-x2**2)/(-2*x1*x2))

        sinn = (2 * np.cos(psi) ** 2 - 1) * np.sin(phi)
        coss = np.cos(phi)+np.sin(2*psi)

        beta_bar_times2 = np.arctan2(sinn, coss)

        """inside terms:"""

        A1_prime = np.sqrt((x3*np.cos(psi))**2+(x2*np.sin(psi))**2+x2*x3*np.sin(2*psi)*np.cos(phi+phi1))
        A1p_sin_alpha1 = (x3*np.cos(psi)-x2*np.sin(psi))*np.sin((phi+phi1)/2)
        A1p_cos_alpha1 = (x3 * np.cos(psi) + x2 * np.sin(psi)) * np.cos((phi + phi1) / 2)
        alpha1 = np.arctan2(A1p_sin_alpha1, A1p_cos_alpha1)
        E1_re = np.cos(phi3-phi2-2*alpha1-beta_bar_times2)
        E1_im = np.sin(phi3-phi2-2*alpha1-beta_bar_times2)

        A2_prime = np.sqrt((x1*np.cos(psi))**2+(x3*np.sin(psi))**2+x3*x1*np.sin(2*psi)*np.cos(phi+phi2))
        A2p_sin_alpha2 = (x1*np.cos(psi)-x3*np.sin(psi))*np.sin((phi+phi2)/2)
        A2p_cos_alpha2 = (x1 * np.cos(psi) + x3 * np.sin(psi)) * np.cos((phi + phi2) / 2)
        alpha2 = np.arctan2(A2p_sin_alpha2, A2p_cos_alpha2)
        E2_re = np.cos(2*phi-2*alpha2+beta_bar_times2+phi3-phi1-2*phi2)
        E2_im = np.sin(2*phi-2*alpha2+beta_bar_times2+phi3-phi1-2*phi2)

        A3_prime = np.sqrt((x2*np.cos(psi))**2+(x1*np.sin(psi))**2+x1*x2*np.sin(2*psi)*np.cos(phi+phi3))
        A3p_sin_alpha3 = (x2*np.cos(psi)-x1*np.sin(psi))*np.sin((phi+phi3)/2)
        A3p_cos_alpha3 = (x2 * np.cos(psi) + x1 * np.sin(psi)) * np.cos((phi + phi3) / 2)
        alpha3 = np.arctan2(A3p_sin_alpha3, A3p_cos_alpha3)
        E3_re = np.cos(-2*phi-2*alpha3+beta_bar_times2+phi1-phi2+2*phi3)
        E3_im = np.sin(-2*phi-2*alpha3+beta_bar_times2+phi1-phi2+2*phi3)

        if imag == False:
            E1 = E1_re
            E2 = E2_re
            E3 = E3_re

        if imag == True:
            E1 = E1_im
            E2 = E2_im
            E3 = E3_im

        l1_1 = R*np.cos(psi)/A1_prime
        l1_2 = R*np.cos(psi)/A2_prime
        l1_3 = R*np.cos(psi)/A3_prime

        l2_1 = R*np.sin(psi)/A1_prime
        l2_2 = R*np.sin(psi)/A2_prime
        l2_3 = R*np.sin(psi)/A3_prime

        l3_1 = np.sqrt((R/A1_prime)**2*(1 + 2*np.cos(psi)*np.sin(psi)*np.cos(phi)))
        l3_2 = np.sqrt((R / A2_prime) ** 2 * (1 + 2 * np.cos(psi) * np.sin(psi) * np.cos(phi)))
        l3_3 = np.sqrt((R / A3_prime) ** 2 * (1 + 2 * np.cos(psi) * np.sin(psi) * np.cos(phi)))

        if l3_1.any() == 0:
            l3_1 = remove_zeros(l3_1)

        if l3_2.any() == 0:
            l3_2 = remove_zeros(l3_2)

        if l3_3.any() == 0:
            l3_3 = remove_zeros(l3_3)

        first_term = E1 / A1_prime ** 4 * self.b_of_l(l1_1,l2_1, l3_1,chi, bary_correction)
        second_term = E2 / A2_prime ** 4 * self.b_of_l(l1_2, l2_2, l3_2, chi, bary_correction)
        third_term = E3 / A3_prime ** 4 * self.b_of_l(l1_3, l2_3, l3_3, chi, bary_correction)

        complete_integrand = outside_term*(first_term+second_term+third_term)

        return(complete_integrand)

    def gamma1_ro(self, integ_limits, r, u, v, chi, niter, nevalu, baryons, imag = False):

        timenow = time.time()
        int_obj = Integrator(integ_limits)

        @vegas.batchintegrand
        def my_integrand(y):
            return(functools.partial(self.gamma1_integrand_ro, r, u, v, chi, baryons, imag)(y))

        result = int_obj(my_integrand, nitn=niter, neval=nevalu)
        timeafter = time.time()
        #print("the time to integrate is", timeafter-timenow)

        return(result)

    def gamma1_loop(self, integ_limits, r, u, v, chi_min, chi_max, n_chisteps, niter, nevalu, baryons = False, imag = False):

        constant = 27 * (100 / 299792) ** 6 * self.omegam ** 3 / 8
        timenow = time.time()
        chivals = np.linspace(chi_min, chi_max, n_chisteps)
        result_of_chi = np.ndarray(shape=(len(chivals)))
        for i in range(len(chivals)):
            result_of_chi[i] = self.gamma1_ro(integ_limits, r, u, v, chivals[i], niter, nevalu, baryons, imag).mean
        to_integrate = (self.lensing_kernel(chivals) * (1 + self.z_from_r_func(chivals))) ** 3 / chivals * result_of_chi
        final_result = np.trapz(to_integrate, chivals)
        timeafter = time.time()
        #print("the time to integrate is", timeafter-timenow)

        return(final_result*constant)

    def gamma2_ro(self, integ_limits, r, u, v, chi, niter, nevalu, baryons, imag = False):

        x2 = r*np.pi/(60*180)
        x3 = u*x2
        x1 = v*x3+x2

        new_r = x3*(60*180)/np.pi
        new_u = x1/x3
        new_v = (x2-x3)/x1

        timenow = time.time()
        int_obj = Integrator(integ_limits)

        @vegas.batchintegrand
        def my_integrand(y):
            return (functools.partial(self.gamma1_integrand_ro, new_r, new_u, new_v,chi, baryons, imag)(y))

        result = int_obj(my_integrand, nitn=niter, neval=nevalu)
        timeafter = time.time()
        #print("the time to integrate is", timeafter-timenow)

        return(result)

    def gamma2_loop(self, integ_limits, r, u, v, chi_min, chi_max, n_chisteps, niter, nevalu, baryons = False, imag = False):

        constant = 27 * (100 / 299792) ** 6 * self.omegam ** 3 / 8
        timenow = time.time()
        chivals = np.linspace(chi_min, chi_max, n_chisteps)
        result_of_chi = np.ndarray(shape=(len(chivals)))
        for i in range(len(chivals)):
            result_of_chi[i] = self.gamma2_ro(integ_limits, r, u, v, chivals[i], niter, nevalu, baryons, imag).mean
        to_integrate = (self.lensing_kernel(chivals) * (1 + self.z_from_r_func(chivals))) ** 3 / chivals * result_of_chi
        final_result = np.trapz(to_integrate, chivals)
        timeafter = time.time()
        #print("the time to integrate is", timeafter-timenow)

        return(final_result*constant)

    def gamma3_ro(self, integ_limits, r, u, v, chi, niter, nevalu, baryons, imag = False):

        x2 = r * np.pi / (60 * 180)
        x3 = u * x2
        x1 = v * x3 + x2

        new_r = x1 * (60 * 180) / np.pi
        new_u = x2 / x1
        new_v = (x3 - x1) / x2

        timenow = time.time()
        int_obj = Integrator(integ_limits)

        @vegas.batchintegrand
        def my_integrand(y):
            return (functools.partial(self.gamma1_integrand_ro, new_r, new_u, new_v, chi, baryons, imag)(y))

        result = int_obj(my_integrand, nitn=niter, neval=nevalu)
        timeafter = time.time()
        #print("the time to integrate is", timeafter - timenow)

        return (result)

    def gamma3_loop(self, integ_limits, r, u, v, chi_min, chi_max, n_chisteps, niter, nevalu, baryons = False, imag = False):

        constant = 27 * (100 / 299792) ** 6 * self.omegam ** 3 / 8
        timenow = time.time()
        chivals = np.linspace(chi_min, chi_max, n_chisteps)
        result_of_chi = np.ndarray(shape=(len(chivals)))
        for i in range(len(chivals)):
            result_of_chi[i] = self.gamma3_ro(integ_limits, r, u, v, chivals[i], niter, nevalu, baryons, imag).mean
        to_integrate = (self.lensing_kernel(chivals) * (1 + self.z_from_r_func(chivals))) ** 3 / chivals * result_of_chi
        final_result = np.trapz(to_integrate, chivals)
        timeafter = time.time()
        #print("the time to integrate is", timeafter-timenow)

        return(final_result*constant)