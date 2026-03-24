import numpy as np
import matplotlib.pyplot as plt
from bispectrum import bispectrum
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import interp1d
from scipy.special import gamma
from scipy.optimize import root
import time

class halo_model_bispectrum(bispectrum):

    def __init__(self, cosmo_params, k, z, file_name):

        bispectrum.__init__(self, cosmo_params, k, z)

        bispec_full = np.load(file_name)
        bispec_reshaped = np.reshape(bispec_full, newshape=(len(z), len(k), len(k), len(k)))
        self.halomodel = RegularGridInterpolator((z, k, k, k), bispec_reshaped, bounds_error = False, fill_value = 0)

    def matter_bispectrum(self, zi, k1,k2,k3, baryons):
        return(np.pi*self.halomodel((zi, k1,k2,k3)))

class halo_model_bispectrum_all(bispectrum):

    def __init__(self, cosmo_params, k, z, bcm_params = None):

        bispectrum.__init__(self, cosmo_params, k, z)
        self.G_const = 4.301 * 10 ** (-9) * self.cosmo_params['h']  # units of km**2*Mpc/(h*M_sun*s**2)
        self.H_const = 100  # units of h*km/(s*Mpc)
        self.rho_crit = 3 * self.H_const ** 2 / (8 * np.pi * self.G_const)  # units of M_sun*h**3/Mpc**3
        self.delta_c = 1.686
        self.bcm_params = bcm_params

    def rho_bar(self, z):
        return (self.rho_crit * self.omegam * (z + 1) ** 3)

    def radius(self, m, z):
        return ((3 * m / (4 * np.pi * self.rho_bar(z))) ** (1 / 3))

    def window(self, x):
        return (3 / x ** 3 * (np.sin(x) - x * np.cos(x)))

    def dsigma2(self, k, m, z):
        return (k ** 2 * self.PL((k, z)) / (2 * np.pi ** 2) * np.abs(self.window(k * self.radius(m, z))) ** 2)

    def sigma2(self, m, z, kgrid):
        integral = np.trapz(np.array([self.dsigma2(kgrid, mi, z) for mi in m]), kgrid)
        return (integral)

    def nu(self, m, z, k):
        return (self.delta_c ** 2 / self.sigma2(m, z, k))

    def dnudm(self, m, z, k):
        nu_vals = self.nu(m, z, k)
        deriv = np.ndarray(shape=len(m) - 1)
        new_m = np.ndarray(shape=len(m) - 1)
        for i in range(len(deriv)):
            deriv[i] = (nu_vals[i + 1] - nu_vals[i]) / (m[i + 1] - m[i])
            new_m[i] = (m[i + 1] + m[i]) / 2
        return (new_m, deriv)

    def f_halomodel(self, nu):
        p = 0.3
        q = 0.75
        A = (1 + 2 ** (-p) * gamma(0.5 - p) / np.pi ** (1 / 2)) ** (-1)
        qnu = q * nu
        f = (A * (1 + qnu ** (-p)) * (qnu / (2 * np.pi)) ** (1 / 2) * np.exp(-qnu / 2)) / nu
        return (f)

    def halo_mass_function(self, m, z, k):

        timee = time.time()
        m_prime, derivnudm = self.dnudm(m, z, k)
        nu_vals = self.nu(m_prime, z, k)
        term = self.rho_bar(z) / m_prime
        print(time.time()-timee)
        return (m_prime, term * derivnudm * self.f_halomodel(nu_vals))

    def do_halo_mass(self, k):

        masses = np.logspace(10,16, num=100)
        m, hmf = self.halo_mass_function(masses, 0, k)
        plt.plot(m, hmf)
        plt.xscale("log")
        plt.yscale("log")
        plt.show()

    def concentration(self, m, z):
        m_star = 2 * 10 ** 13  # approx from cooray & sheth, since we are using constant delta_c, we elimnate the z dependence
        return ((9 / (1 + z)) * (m / m_star) ** (-0.13))

    def r_virial(self, m, z):
        #density = 18 * np.pi ** 2 * self.rho_bar(z)
        density = 200 * self.rho_bar(z) #r_200
        volume = m / density
        radius = (3 / (4 * np.pi) * volume) ** (1 / 3)
        return (radius)

    def nfw_profile(self, r, m, z):
        # c = concentration(m,z)
        # c = concentration(m,z)/2
        c = self.concentration(m, z) * 2
        r_vir = self.r_virial(m, z)
        r_s = r_vir / c
        tmp = np.log(1 + c) - c / (1 + c)
        rho_s = m / (4 * np.pi * r_s ** 3 * tmp)
        ratio = r / r_s
        # truncate = (r<r_vir)
        result = rho_s / (ratio * (1 + ratio) ** 2)

        return (result)

    def ejected_gas_profile(self, r, m, z, eta):

        r_esc = 1/2*np.sqrt(200)*self.r_virial(m,z) #CHECK THE 200
        r_ej = eta*0.75*r_esc
        exp = np.exp(-0.5*(r/r_ej)**2)
        profile = m*exp/(2*np.pi*r_ej**2)**(3/2)

        return(profile)

    def bound_gas_profile(self, r, m, z):

        r_vir = self.r_virial(m,z)
        c = self.concentration(m, z) * 2
        x_transition = c/np.sqrt(5)
        gamma_eff = (1+3*c/np.sqrt(5))*np.log(1+c*np.sqrt(5))/((1+c/np.sqrt(5))*np.log(1+c/np.sqrt(5))-c/np.sqrt(5))
        y0_div_y1 = x_transition**(-1)*(1+x_transition)**(-2)/(x_transition**(-1)*np.log(1+x_transition))**gamma_eff
        x = r*c/r_vir
        first_part = (x**(-1)*np.log(1+x))**gamma_eff
        second_part = x**(-1)*(1+x)**(-2)
        eval1 = np.zeros_like(r)
        eval2 = np.zeros_like(r)
        for ii in range(len(r)):
            if r[ii] < r_vir/np.sqrt(5):
                eval1[ii] = 1
            if r[ii] >= r_vir/np.sqrt(5) and r[ii] < r_vir:
                eval2[ii] = 1

        unnormalized_profile = y0_div_y1*first_part*eval1 + second_part*eval2
        y1 = m/(np.trapz(4*np.pi*r**2*unnormalized_profile, r))

        self.bound_gas_profile_at_rvir = y1*c**(-1)*(1+c)**(-2)
        prod = y1*unnormalized_profile

        return(prod)

    def central_galaxy_profile(self, r, m, z):

        r_vir = self.r_virial(m, z)
        Rh = 0.015*r_vir
        exp = np.exp(-(r/(2*Rh))**2)
        profile = m*exp/(4*np.pi**(3/2)*Rh*r**2)
        return(profile)

    def find_r_prime(self, rprime, rho, fdm, m, r_vir, c):

        rs = r_vir/c
        y = rprime*(rs+rprime)**2*rho*(np.log(1+rprime/rs)-(rprime/(rprime+rs)))+rho/3*(r_vir**3-rprime**3) - fdm*m/4*np.pi
        return(y)

    def dark_matter_profile_before_relaxing(self, r, m, z):

        r_vir = self.r_virial(m, z)
        c = self.concentration(m, z)
        prof_at_rvir = self.nfw_profile(r_vir, m, z) - self.bound_gas_profile_at_rvir
        print(m)
        print(prof_at_rvir)
        init = 0.0001
        #print("shape args:", np.shape(init), np.shape(prof_at_rvir), np.shape(r_vir), np.shape(c))
        res = root(self.find_r_prime, x0 = init, args=(prof_at_rvir, self.frac_dm, m, r_vir, c))
        r_prime = res.x
        print("ratio", r_prime / r_vir)
        #print("my r prime", np.shape(r_prime))
        print(r_vir)
        first_zeros = np.zeros_like(r)
        second_zeros = np.zeros_like(r)
        for ri in range(len(r)):
            if ri<r_prime:
                first_zeros[ri] = 1
            if ri>r_prime:
                second_zeros[ri] = 1

        if r_prime > r_vir:
            r_prime = r_vir
        norm_profile = prof_at_rvir/(r_prime*c/r_vir*(1+(r_prime*c/r_vir))**2)

        my_inner_profile = norm_profile/(r*c/r_vir*(1+(r*c/r_vir))**2)
        final_profile = my_inner_profile*first_zeros + prof_at_rvir*second_zeros

        return(final_profile)

    def compute_frac_cg(self, m, z, mi_param):

        scalefactor = 1/(z+1)
        nu = np.exp(-4*scalefactor**2)
        epsilon = 10**(np.log10(0.023)-0.006*(scalefactor-1)*nu -0.119*(scalefactor-1))
        alpha = -1.779 + 0.731*(scalefactor-1)*nu
        delta = 4.394 + (2.608*(scalefactor-1) -0.043*z)*nu
        gamma = 0.547 + (1.319*(scalefactor-1)+0.279*z)*nu
        m1 = 10**(np.log10(mi_param) + (-1.793*(scalefactor-1)-0.251*z)*nu)
        gx = -np.log10(10**(alpha*np.log10(m/m1))+1) + delta*(np.log10(1+np.exp(np.log10(m/m1)))**gamma)/(1+np.exp(m1/m))
        g0 = -np.log10(2) + delta*np.log10(2)**gamma/(1+np.exp(1))

        frac_cg = epsilon*m1/m*10**(gx-g0)

        return(frac_cg)

    def compute_frac_bg(self, frac_cg, m, mc_param, beta_param):

        num = self.cosmo_params['Omega_b']/self.omegam - frac_cg
        den = 1 + (mc_param/m)**beta_param

        frac_bg = num/den

        return(frac_bg)

    def compute_frac_dm(self):

        frac_dm = self.cosmo_params['Omega_cdm']/self.omegam
        self.frac_dm = frac_dm

        return(frac_dm)

    def compute_relaxed_dm_profile(self, r, m, z, cg_prof, eg_prof, bg_prof, nfw_prof):

        rf_init = self.r_virial(m, z)
        cg_mass_prof = np.zeros_like(r)
        eg_mass_prof = np.zeros_like(r)
        bg_mass_prof = np.zeros_like(r)
        nfw_mass_prof = np.zeros_like(r)
        i = 0
        nfw_val = 0
        cg_val = 0
        eg_val = 0
        bg_val = 0
        while i < len(r):
            cg_mass_prof[i] = cg_val
            eg_mass_prof[i] = eg_val
            bg_mass_prof[i] = bg_val
            nfw_mass_prof[i] = nfw_val
            if i < len(r)-1:
                cg_val += 4*np.pi*(cg_prof[i]*r[i]**2 + cg_prof[i + 1]*r[i+1]**2) * (r[i + 1] - r[i]) / 2
                eg_val += 4*np.pi*(eg_prof[i]*r[i]**2 + eg_prof[i + 1]*r[i+1]**2) * (r[i + 1] - r[i]) / 2
                bg_val += 4*np.pi*(bg_prof[i]*r[i]**2 + bg_prof[i + 1]*r[i+1]**2) * (r[i + 1] - r[i]) / 2
                nfw_val += 4 * np.pi * (nfw_prof[i] * r[i] ** 2 + nfw_prof[i + 1] * r[i + 1] ** 2) * (r[i + 1] - r[i]) / 2
            i+=1
        cg_mass_prof_interp = interp1d(r, cg_mass_prof, bounds_error = False, fill_value = 0)
        eg_mass_prof_interp = interp1d(r, eg_mass_prof, bounds_error = False, fill_value = 0)
        bg_mass_prof_interp = interp1d(r, bg_mass_prof, bounds_error = False, fill_value = 0)
        nfw_mass_prof_interp = interp1d(r, nfw_mass_prof, bounds_error=False, fill_value=0)

        diff = 1
        count = 0
        rf = rf_init
        while np.abs(diff) > 0.00001:

            Mf = self.frac_dm*m + cg_mass_prof_interp(rf) + eg_mass_prof_interp(rf) + bg_mass_prof_interp(rf)
            #print(rf, Mf)
            new_rf = (((m/Mf)**2 - 1)*0.3+1)*rf_init
            diff = (new_rf-rf)/rf
            rf = new_rf
            count += 1
            #print(diff)

        #print('done this many iterations:', count)
        csi = rf/rf_init
        #print('csi is', self.csi)
        relaxed_profile = nfw_mass_prof_interp(r/csi)*self.frac_dm/csi**3

        return(relaxed_profile)


    def compute_fourier_nfw(self, baryons = False):
        #times = time.time()
        #print("computing fourier nfw")
        m = np.logspace(10, 18, num=100)
        z = np.linspace(0, 3, 10)
        k = np.logspace(-3, np.log10(50), 100)
        full_grid = np.ndarray(shape=(len(z), len(k), len(m)))
        if baryons == True:

            frac_dm = self.compute_frac_dm()
            mi_param = self.bcm_params['M_i']
            mc_param = self.bcm_params['M_c']
            beta_param = self.bcm_params['beta']
            eta_param = self.bcm_params['eta']

            for zi in range(len(z)):
                timeee = time.time()
                for mi in range(len(m)):

                    frac_cg = self.compute_frac_cg(m[mi], z[zi], mi_param)
                    frac_bg = self.compute_frac_bg(frac_cg, m[mi], mc_param, beta_param)
                    frac_ej = 1 - frac_dm - frac_cg - frac_bg

                    r_vir = self.r_virial(m[mi], z[zi])
                    r = np.logspace(np.log10(r_vir) - 3, np.log10(r_vir), num=96)

                    #print(r)
                    central_galaxy = self.central_galaxy_profile(r, m[mi], z[zi])*frac_cg
                    bound_gas = self.bound_gas_profile(r, m[mi], z[zi])*frac_bg
                    ejected_gas = self.ejected_gas_profile(r, m[mi], z[zi], eta_param)*frac_ej
                    #dark_matter = self.dark_matter_profile_before_relaxing(r, m[mi], z[zi])*frac_dm
                    nfw = self.dark_matter_profile_before_relaxing(r,m[mi],z[zi])
                    #nfw = self.nfw_profile(r, m[mi], z[zi])
                    dark_matter = self.compute_relaxed_dm_profile(r, 0.01*m[mi], z[zi], central_galaxy, ejected_gas, bound_gas, nfw)
                    #integrand = central_galaxy + bound_gas + ejected_gas + dark_matter
                    integrand = central_galaxy + bound_gas + ejected_gas

                    extended_r = np.ndarray(shape=(len(k), len(r)))
                    consts_r = np.ndarray(shape=(len(k), len(r)))
                    integrand_ext = np.ndarray(shape=(len(k), len(r)))

                    for ki in range(len(k)):
                        extended_r[ki] = r
                        integrand_ext[ki] = integrand
                        consts_r[ki] = 4 * np.pi * r ** 2 * np.sin(k[ki] * r) / (k[ki] * r * m[mi])

                    full_grid[zi, :, mi] = np.trapz(consts_r*integrand_ext, extended_r)
                #print("did z", zi)
                #print(time.time()-timeee)

        if baryons == False:

            for zi in range(len(z)):
                timeee = time.time()
                for mi in range(len(m)):
                    r_vir = self.r_virial(m[mi], z[zi])
                    r = np.logspace(np.log10(r_vir) - 3, np.log10(r_vir), num=100)
                    integrand = self.nfw_profile(r, m[mi], z[zi])

                    extended_r = np.ndarray(shape=(len(k), len(r)))
                    consts_r = np.ndarray(shape=(len(k), len(r)))
                    integrand_ext = np.ndarray(shape=(len(k), len(r)))

                    for ki in range(len(k)):
                        extended_r[ki] = r
                        integrand_ext[ki] = integrand
                        consts_r[ki] = 4 * np.pi * r ** 2 * np.sin(k[ki] * r) / (k[ki] * r * m[mi])

                    full_grid[zi, :, mi] = np.trapz(consts_r*integrand_ext, extended_r)
                #print("did z", zi)
                #print(time.time()-timeee)

        self.fourier_nfw = RegularGridInterpolator((z, k, m), full_grid, bounds_error=False, fill_value=0)

    def plot_prof(self):

        k_prof = np.logspace(-1, 3, num=100)
        u11 = np.array([self.fourier_nfw((0, ki, 10 ** 11)) for ki in k_prof])
        u13 = np.array([self.fourier_nfw((0, ki, 10 ** 13)) for ki in k_prof])
        u14 = np.array([self.fourier_nfw((0, ki, 10 ** 14)) for ki in k_prof])
        u15 = np.array([self.fourier_nfw((0, ki, 10 ** 15))for ki in k_prof])
        u16 = np.array([self.fourier_nfw((0, ki, 10 ** 16)) for ki in k_prof])
        plt.plot(k_prof, u11, label="11")
        plt.plot(k_prof, u13, label="13")
        plt.plot(k_prof, u14, label="14")
        plt.plot(k_prof, u15, label="15")
        plt.plot(k_prof, u16, label="16")
        plt.legend()
        plt.xscale("log")
        plt.yscale("log")
        plt.show()

    def stack_profiles(self, m, k1, k2, k3, z):
        # print(m)
        #times = time.time()
        #res = np.array([self.fourier_nfw(k1i, m, z) * self.fourier_nfw(k2i, m, z) * self.fourier_nfw(k3i, m, z) for k1i, k2i, k3i in
        #                  zip(k1, k2, k3)])
        res = np.array([self.fourier_nfw((z, k1i, m)) * self.fourier_nfw((z, k2i, m)) * self.fourier_nfw((z,k3i, m)) for k1i, k2i, k3i in
                          zip(k1, k2, k3)])
        #print(time.time()-times)
        return (res)

    def one_halo_bispectrum(self, k1, k2, k3, z):
        m = np.logspace(10, 18, num=50)
        m_mod, n = self.halo_mass_function(m, z, self.k)
        profiles = np.array([self.stack_profiles(mi, k1, k2, k3, z) for mi in m_mod])
        # profiles = np.array([fourier_nfw(k1i,mi,z)*fourier_nfw(k2i,mi,z)*fourier_nfw(k3i,mi,z) for mi, k1i, k2i, k3i in zip(m_mod, k1,k2,k3)])
        # norm = m_mod**3/rho_bar**3
        norm = m_mod ** 3 / self.rho_bar(z) ** 3
        tmp = n * norm
        multi = profiles * tmp[:, None]
        integrate = np.trapz(multi, m_mod, axis=0)
        # print(integrate)
        return (integrate)

    def do_one_halo(self):
        shortks = np.logspace(-2, np.log10(20), 20)
        newkgrid = np.ndarray(shape=((3, len(shortks) ** 3)))
        for ii in range(len(shortks)):
            for jj in range(len(shortks)):
                for kk in range(len(shortks)):
                    index = kk + len(shortks) * jj + len(shortks) ** 2 * ii
                    newkgrid[0][index] = shortks[ii]
                    newkgrid[1][index] = shortks[jj]
                    newkgrid[2][index] = shortks[kk]

        bispecs_grid = np.ndarray(shape=(10, len(shortks)**3))
        zarray = np.linspace(0,2.7,10)
        for ll in range(10):
            timee = time.time()
            z = ll*0.3
            bispecs_grid[ll] = self.one_halo_bispectrum(newkgrid[0], newkgrid[1], newkgrid[2], z)
            print("done z=", z)
            print(time.time()-timee)

        bispec_newshape = np.reshape(bispecs_grid, newshape=(10, len(shortks), len(shortks), len(shortks)))
        np.save("Halo_bispectrum_with_baryons_DM_PROFILE_FROM_UNRELAXED_TO_RELAXED_apr26", bispecs_grid)
        self.halomodel = RegularGridInterpolator((zarray, shortks, shortks, shortks), bispec_newshape, bounds_error = False, fill_value = 0)

    def matter_bispectrum(self, zi, k1,k2,k3):
        return(self.halomodel((zi, k1,k2,k3)))