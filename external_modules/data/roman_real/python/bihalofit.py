import numpy as np
import matplotlib.pyplot as plt
from bispectrum import bispectrum
from scipy.interpolate import interp1d
import time

class bihalofit(bispectrum):

    def __init__(self, cosmo_params, k, z):

        timebih = time.time()
        bispectrum.__init__(self, cosmo_params, k, z)

        self.model = 'bihalofit'

        ns = self.cosmo_params['n_s']
        omegam = self.omegam

        '''neff'''
        ratio_k_knl = np.ndarray(shape=(len(k), len(z)))
        k_ext = np.ndarray(shape=(len(k), len(z)))
        for ki in range(len(k)):
            ratio_k_knl[ki] = k[ki]/self.knl
        for zi in range(len(z)):
            k_ext[:,zi] = k
        common_prod = self.Pk_linear*np.exp(-ratio_k_knl**2)
        integrand_sigma2 = np.einsum('ij,i -> ij', common_prod, k**3) #or k**2 anf k**4, with k_ext instead of np.log(k_ext)
        integrand_deriv = np.einsum('ij,i -> ij', common_prod, k**5)
        sigma2 = np.trapz(integrand_sigma2, np.log(k_ext), axis=0)
        deriv = np.trapz(integrand_deriv, np.log(k_ext), axis=0)
        #print(np.shape(sigma2), np.shape(deriv), np.shape(self.knl))
        self.neff = -3 + 2*deriv/sigma2/self.knl**2


        self.neff_interp = interp1d(z, self.neff)
        #print(self.neff)

        #self.neff = -1.55*np.ones(shape=(len(self.k), len(self.z)))
        #self.neff = -1.55
        self.paramns = np.log10(1 - 2 * ns / 3)
        #self.logsigma8 = np.log10(self.my_cosmo.sigma(8, 0, h_units = True))
        self.logsigma8 = np.array([np.log10(self.my_cosmo.sigma(8, zi, h_units=True)) for zi in z])
        self.logsigma8_interp = interp1d(z, self.logsigma8, bounds_error = False, fill_value = self.logsigma8[-1])
        #print(10**(self.logsigma8))

        # bihalofit global params

        # one halo
        self.bn = 10 ** (-3.428 - 2.681 * self.logsigma8 + 1.624 * self.logsigma8 ** 2 - 0.095 * self.logsigma8 ** 3)
        self.bn_interp = interp1d(z, self.bn, bounds_error = False, fill_value = self.bn[-1])
        self.cn = 10 ** (0.159 - 1.107 * self.neff)
        self.cn_interp = interp1d(z, self.cn, bounds_error = False, fill_value = self.cn[-1])
        self.gamman = 10 ** (0.182 + 0.57 * self.neff)
        self.gamman_interp = interp1d(z, self.gamman, bounds_error=False, fill_value=self.gamman[-1])

        # three halo
        self.fn = 10 ** (-10.533 - 16.838 * self.neff - 9.3048 * self.neff ** 2 - 1.8263 * self.neff ** 3)
        self.fn_interp = interp1d(z, self.fn, bounds_error=False, fill_value=self.fn[-1])
        self.gn = 10 ** (2.787 + 2.405 * self.neff + 0.4577 * self.neff ** 2)
        self.gn_interp = interp1d(z, self.gn, bounds_error=False, fill_value=self.gn[-1])
        self.hn = 10 ** (-1.118 - 0.394 * self.neff)
        self.hn_interp = interp1d(z, self.hn, bounds_error=False, fill_value=self.hn[-1])
        self.mn = 10 ** (-2.605 - 2.434 * self.logsigma8 + 5.710 * self.logsigma8 ** 2)
        self.mn_interp = interp1d(z, self.mn, bounds_error = False, fill_value = self.mn[-1])
        self.nn = 10 ** (-4.468 - 3.080 * self.logsigma8 + 1.035 * self.logsigma8 ** 2)
        self.nn_interp = interp1d(z, self.nn, bounds_error=False, fill_value=self.nn[-1])
        self.mun = 10 ** (15.312 + 22.977 * self.neff + 10.9579 * self.neff ** 2 + 1.6586 * self.neff ** 3)
        self.mun_interp = interp1d(z, self.mun, bounds_error=False, fill_value=self.mun[-1])
        self.nun = 10 ** (1.347 + 1.246 * self.neff + 0.4525 * self.neff ** 2)
        self.nun_interp = interp1d(z, self.nun, bounds_error=False, fill_value=self.nun[-1])
        self.pn = 10 ** (0.071 - 0.433 * self.neff)
        self.pn_interp = interp1d(z, self.pn, bounds_error=False, fill_value=self.pn[-1])
        self.dn = 10 ** (-0.483 + 0.892 * self.logsigma8 - 0.086 * omegam)
        self.dn_interp = interp1d(z, self.dn, bounds_error = False, fill_value = self.dn[-1])
        self.en = 10 ** (-0.632 + 0.646 * self.neff)
        self.en_interp = interp1d(z, self.en, bounds_error=False, fill_value=self.en[-1])

        #print("time to init bihalofit:", time.time()-timebih)

    def compute_dependent_params_old(self, z, k1, k2, k3):

        tosort = [k1, k2, k3]
        sortedlist = np.sort(tosort)
        kmin = sortedlist[0]
        kmid = sortedlist[1]
        kmax = sortedlist[2]
        r1 = kmin / kmax
        r2 = (kmid + kmin - kmax) / kmax

        r1_full = np.ndarray(shape=(len(self.k), len(self.z)))
        r2_full = np.ndarray(shape=(len(self.k), len(self.z)))

        for i in range(len(self.z)):
            r1_full[:,i] = r1
            r2_full[:, i] = r2

        an = 10 ** (-2.167 - 2.944 * self.logsigma8_interp(z) - 1.106 * self.logsigma8_interp(z) ** 2 - 2.865 * self.logsigma8_interp(z) ** 3 - 0.310 * r1_full ** self.gamman_interp(z))
        alphan = 10 ** (
            np.minimum(-4.348 - 3.006 * self.neff - 0.5745 * self.neff ** 2 + 10 ** (-0.9 + 0.2 * self.neff) * r2_full ** 2, self.paramns))
        betan = 10 ** (-1.731 - 2.845 * self.neff - 1.4995 * self.neff ** 2 - 0.2811 * self.neff ** 3 + 0.007 * r2_full)
        return (an, alphan, betan)

    def compute_dependent_params(self, z, k1, k2, k3):

        if type(k1) == float:
            kvals = np.array([k1,k2,k3])
        else:
            kvals = np.ndarray(shape=(3, len(k1)))
            kvals[0] = k1
            kvals[1] = k2
            kvals[2] = k3

        sortedlist = np.sort(kvals, axis=0)
        kmin = sortedlist[0]
        kmid = sortedlist[1]
        kmax = sortedlist[2]
        r1 = kmin / kmax

        r2 = (kmid + kmin - kmax) / kmax

        an = 10 ** (-2.167 - 2.944 * self.logsigma8_interp(z) - 1.106 * self.logsigma8_interp(z) ** 2 - 2.865 * self.logsigma8_interp(z) ** 3 - 0.310 * r1 ** self.gamman_interp(z))
        alphan = 10 ** (
            np.minimum(-4.348 - 3.006 * self.neff_interp(z) - 0.5745 * self.neff_interp(z) ** 2 + 10 ** (-0.9 + 0.2 * self.neff_interp(z)) * r2 ** 2, self.paramns))
        betan = 10 ** (-1.731 - 2.845 * self.neff_interp(z) - 1.4995 * self.neff_interp(z) ** 2 - 0.2811 * self.neff_interp(z) ** 3 + 0.007 * r2)
        return (an, alphan, betan)

    def compute_one_halo(self, z, k1, k2, k3):

        #print("z is", z)
        knl = self.knl_interp(z)/self.cosmo_params['h'] #UPDATED ISSUE ON K_NL
        #print("knl is", knl)
        #print("k1 is", k1)
        if type(k1) == float and type(z) == float:
            qvec = np.array(k1/knl,k2/knl,k3/knl)
        elif type(k1) == float:
            qvec = np.ndarray(shape=(3,len(z)))
            qvec[0] = k1 / knl
            qvec[1] = k2 / knl
            qvec[2] = k3 / knl
        else:
            qvec = np.ndarray(shape=(3,len(k1)))
            qvec[0] = k1 / knl
            qvec[1] = k2 / knl
            qvec[2] = k3 / knl
        an, alphan, betan = self.compute_dependent_params(z, k1, k2, k3)
        valuetot = 1
        for q in qvec:
            value = (1 / (an * q ** alphan + self.bn_interp(z) * q ** betan)) * (1 / (1 + (self.cn_interp(z) * q) ** (-1)))
            valuetot = valuetot * value

        return (valuetot)

    def I_func(self, ki, z):

        knl = self.knl_interp(z)
        return (1 / (1 + self.en_interp(z) * ki / knl))

    def P_enhanced(self, ki, z):

        knl = self.knl_interp(z)
        first = (1 + self.fn_interp(z) * (ki / knl) ** 2) / (1 + self.gn_interp(z) * (ki / knl) + self.hn_interp(z) * (ki / knl) ** 2)
        second = 1 / (self.mn_interp(z) * (ki / knl) ** self.mun_interp(z) + self.nn_interp(z) * (ki / knl) ** self.nun_interp(z))
        third = 1 / (1 + (self.pn_interp(z) * ki / knl) ** (-3))

        tmp = first * self.PL((ki, z)) + second * third
        return (tmp)

    def compute_three_halo_part(self, z, k1, k2, k3):

        first_term = self.compute_kernel(k1,k2,k3) + self.dn_interp(z) * k3 / self.knl_interp(z)
        second_term = self.I_func(k1, z) * self.I_func(k2, z) * self.I_func(k3, z)
        third_term = self.P_enhanced(k1, z) * self.P_enhanced(k2, z)

        return (2 * first_term * second_term * third_term)

    def compute_three_halo(self, z, k1, k2, k3):
        self.three_halo = self.compute_three_halo_part(z, k1, k2, k3)+self.compute_three_halo_part(z, k2, k3, k1)+ self.compute_three_halo_part(z, k3, k1, k2)
        return(self.three_halo)

    def step(self, x):
        if x >= 0.0:
            return (1.0)
        else:
            return (0.0)

    def one_baryon(self, z, ki):

        scalefactor = 1 / (1 + z)
        if scalefactor >= 0.5:
            a0_baryon = 0.068 * (scalefactor - 0.5) ** 0.47
        else:
            a0_baryon = 0.0
        if scalefactor >= 0.2:
            a1_baryon = 1.052 * (scalefactor - 0.2) ** 1.41
        else:
            a1_baryon = 0.0
        alpha0_baryon = 2.346
        alpha2_baryon = 2.25
        mu0_baryon = 0.018 * scalefactor + 0.837 * scalefactor ** 2
        sigma0_baryon = 0.881 * mu0_baryon
        mu1_baryon = np.abs(0.172 + 3.048 * scalefactor - 0.675 * scalefactor ** 2)
        sigma1_baryon = (0.494 - 0.039 * scalefactor) * mu1_baryon
        kstar = 29.90 - 38.73 * scalefactor + 24.30 * scalefactor ** 2
        beta2_baryon = 0.563 / (alpha2_baryon * ((scalefactor / 0.06) ** 0.02 + 1))

        x = np.log10(ki)
        first = a0_baryon * np.exp(-np.abs((x - mu0_baryon) / sigma0_baryon) ** alpha0_baryon)
        second = a1_baryon * np.exp(-((x - mu1_baryon) / sigma1_baryon) ** 2)
        third = ((ki / kstar) ** alpha2_baryon + 1) ** beta2_baryon
        return (first - second + third)

    def baryonic_correction(self, z, k1, k2, k3):
        return (self.one_baryon(z, k1) * self.one_baryon(z, k2) * self.one_baryon(z, k3))

    def matter_bispectrum(self, z, k1, k2, k3, baryons = False):

        if baryons == False:
            return(self.compute_one_halo(z,k1,k2,k3)+self.compute_three_halo(z, k1, k2, k3))
            #return (self.compute_one_halo(z, k1, k2, k3))
            #return (self.compute_three_halo(z, k1, k2, k3))

        if baryons == True:
            return((self.compute_one_halo(z,k1,k2,k3)+self.compute_three_halo(z, k1, k2, k3))*self.baryonic_correction(z, k1, k2, k3))

    def is_knl_zero(self, z):

        return (self.knl_interp(z) == 0.0)
