import numpy as np
import matplotlib.pyplot as plt
from bispectrum import bispectrum

class tree_level_bispectrum(bispectrum):

    def __init__(self, cosmo_params, k, z):

        bispectrum.__init__(self, cosmo_params, k, z)

    def compute_tree_level(self, zi, k1,k2,k3):
        return(2*(self.compute_kernel(k1,k2,k3)*self.PL((k1,zi))*self.PL((k2, zi)) +
                  self.compute_kernel(k2,k3,k1)*self.PL((k2, zi))*self.PL((k3, zi)) +
                  self.compute_kernel(k3,k1,k2)*self.PL((k3, zi))*self.PL((k1, zi))))