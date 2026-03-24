# Copyright (c) 2003-2024 by Mike Jarvis
#
# TreeCorr is free software: redistribution and use in source and binary forms,
# with or without modification, are permitted provided that the following
# conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions, and the disclaimer given in the accompanying LICENSE
#    file.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions, and the disclaimer given in the documentation
#    and/or other materials provided with the distribution.

import numpy as np
'''This file contains an adapted version of the treecorr functions _calculateT and
calculateMap, that are designed to function with theoretical predictions outside of the
treecorr environment'''

def _calculateT(s, t, k1, k2, k3):
    # First calculate q values:

    q1 = (s + t) / 3.
    q2 = q1 - t
    q3 = q1 - s

    # |qi|^2 shows up a lot, so save these.
    # The a stands for "absolute", and the ^2 part is implicit.
    a1 = np.abs(q1) ** 2
    a2 = np.abs(q2) ** 2
    a3 = np.abs(q3) ** 2
    a123 = a1 * a2 * a3

    # These combinations also appear multiple times.
    # The b doesn't stand for anything.  It's just the next letter after a.
    b1 = np.conjugate(q1) ** 2 * q2 * q3
    b2 = np.conjugate(q2) ** 2 * q1 * q3
    b3 = np.conjugate(q3) ** 2 * q1 * q2

    if k1 == 1 and k2.all() == 1 and k3.all() == 1:

        # Some factors we use multiple times
        expfactor = -np.exp(-(a1 + a2 + a3) / 2)

        # JBJ Equation 51
        # Note that we actually accumulate the Gammas with a different choice for
        # alpha_i.  We accumulate the shears relative to the q vectors, not relative to s.
        # cf. JBJ Equation 41 and footnote 3.  The upshot is that we multiply JBJ's formulae
        # be (q1q2q3)^2 / |q1q2q3|^2 for T0 and (q1*q2q3)^2/|q1q2q3|^2 for T1.
        # Then T0 becomes
        # T0 = -(|q1 q2 q3|^2)/24 exp(-(|q1|^2+|q2|^2+|q3|^2)/2)
        T0 = expfactor * a123 / 24

        # JBJ Equation 52
        # After the phase adjustment, T1 becomes:
        # T1 = -[(|q1 q2 q3|^2)/24
        #        - (q1*^2 q2 q3)/9
        #        + (q1*^4 q2^2 q3^2 + 2 |q2 q3|^2 q1*^2 q2 q3)/(|q1 q2 q3|^2)/27
        #       ] exp(-(|q1|^2+|q2|^2+|q3|^2)/2)
        T1 = expfactor * (a123 / 24 - b1 / 9 + (b1 ** 2 + 2 * a2 * a3 * b1) / (a123 * 27))
        T2 = expfactor * (a123 / 24 - b2 / 9 + (b2 ** 2 + 2 * a1 * a3 * b2) / (a123 * 27))
        T3 = expfactor * (a123 / 24 - b3 / 9 + (b3 ** 2 + 2 * a1 * a2 * b3) / (a123 * 27))

    else:
        # SKL Equation 63:
        k1sq = k1 * k1
        k2sq = k2 * k2
        k3sq = k3 * k3
        Theta2 = ((k1sq * k2sq + k1sq * k3sq + k2sq * k3sq) / 3.) ** 0.5
        k1sq /= Theta2  # These are now what SKL calls theta_i^2 / Theta^2
        k2sq /= Theta2
        k3sq /= Theta2
        Theta4 = Theta2 * Theta2
        Theta6 = Theta4 * Theta2
        S = k1sq * k2sq * k3sq

        # SKL Equation 64:
        Z = ((2 * k2sq + 2 * k3sq - k1sq) * a1 +
             (2 * k3sq + 2 * k1sq - k2sq) * a2 +
             (2 * k1sq + 2 * k2sq - k3sq) * a3) / (6 * Theta2)
        expfactor = -S * np.exp(-Z) / Theta4

        # SKL Equation 65:
        f1 = (k2sq + k3sq) / 2 + (k2sq - k3sq) * (q2 - q3) / (6 * q1)
        f2 = (k3sq + k1sq) / 2 + (k3sq - k1sq) * (q3 - q1) / (6 * q2)
        f3 = (k1sq + k2sq) / 2 + (k1sq - k2sq) * (q1 - q2) / (6 * q3)
        f1c = np.conjugate(f1)
        f2c = np.conjugate(f2)
        f3c = np.conjugate(f3)

        # SKL Equation 69:
        g1 = k2sq * k3sq + (k3sq - k2sq) * k1sq * (q2 - q3) / (3 * q1)
        g2 = k3sq * k1sq + (k1sq - k3sq) * k2sq * (q3 - q1) / (3 * q2)
        g3 = k1sq * k2sq + (k2sq - k1sq) * k3sq * (q1 - q2) / (3 * q3)
        g1c = np.conjugate(g1)
        g2c = np.conjugate(g2)
        g3c = np.conjugate(g3)

        # SKL Equation 62:
        T0 = expfactor * a123 * f1c ** 2 * f2c ** 2 * f3c ** 2 / (24. * Theta6)

        # SKL Equation 68:
        T1 = expfactor * (
                a123 * f1 ** 2 * f2c ** 2 * f3c ** 2 / (24 * Theta6) -
                b1 * f1 * f2c * f3c * g1c / (9 * Theta4) +
                (b1 ** 2 * g1c ** 2 + 2 * k2sq * k3sq * a2 * a3 * b1 * f2c * f3c) / (a123 * 27 * Theta2))
        T2 = expfactor * (
                a123 * f1c ** 2 * f2 ** 2 * f3c ** 2 / (24 * Theta6) -
                b2 * f1c * f2 * f3c * g2c / (9 * Theta4) +
                (b2 ** 2 * g2c ** 2 + 2 * k1sq * k3sq * a1 * a3 * b2 * f1c * f3c) / (a123 * 27 * Theta2))
        T3 = expfactor * (
                a123 * f1c ** 2 * f2c ** 2 * f3 ** 2 / (24 * Theta6) -
                b3 * f1c * f2c * f3 * g3c / (9 * Theta4) +
                (b3 ** 2 * g3c ** 2 + 2 * k1sq * k2sq * a1 * a2 * b3 * f1c * f2c) / (a123 * 27 * Theta2))

    return T0, T1, T2, T3


def calculateMap3(three_pt, d2_vals, d3_vals, phi_vals, logr_bin_size, phi_bin_size, filters):

    '''Adapted from the treecorr function calculateMap3 to yield the integration
    from a predicted set of Gamma values on the SAS binning'''

    R = filters[0]
    k2 = filters[1]/filters[0]
    k3 = filters[2]/filters[0]

    s = np.outer(1. / R, d2_vals.ravel())
    d3 = np.outer(1. / R, d3_vals.ravel())
    t = d3 * np.exp(1j * phi_vals.ravel()) #phi vals in radians

    T0, T1, T2, T3 = _calculateT(s, t, 1., k2, k3)

    d2t = d3 ** 2 * logr_bin_size * phi_bin_size / (2 * np.pi)

    sds = s * s * logr_bin_size  # Remember bin_size is dln(s)

    T0 *= sds * d2t
    T1 *= sds * d2t
    T2 *= sds * d2t
    T3 *= sds * d2t

    # Now do the integral by taking the matrix products.
    gam0 = three_pt[0].ravel()
    gam1 = three_pt[1].ravel()
    gam2 = three_pt[2].ravel()
    gam3 = three_pt[3].ravel()
    mmm = T0.dot(gam0)
    mcmm = T1.dot(gam1)
    mmcm = T2.dot(gam2)
    mmmc = T3.dot(gam3)

    # SAS binning counts each triangle with each vertex in the c1 position.
    # Just need to account for the cases where 1-2-3 are clockwise, rather than CCW.
    if k2.all() == 1 and k3.all() == 1:
        mmm *= 2
        mcmm *= 2
        mmcm += mmmc
        mmmc = mmcm

    else:
        # Repeat the above with 2,3 swapped.
        T0, T1, T2, T3 = _calculateT(s, t, 1, k3, k2)
        T0 *= sds * d2t
        T1 *= sds * d2t
        T2 *= sds * d2t
        T3 *= sds * d2t
        mmm += T0.dot(gam0)
        mcmm += T1.dot(gam1)
        mmmc += T2.dot(gam2)
        mmcm += T3.dot(gam3)

    map3 = 0.25 * np.real(mcmm + mmcm + mmmc + mmm)

    return (map3)