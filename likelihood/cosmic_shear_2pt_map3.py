from __future__ import absolute_import, division, print_function

import importlib.util
import os

import numpy as np

from cobaya.log import LoggedError

from cobaya.likelihoods.roman_real._cosmolike_prototype_base import _cosmolike_prototype_base


class cosmic_shear_2pt_map3(_cosmolike_prototype_base):
    """Cosmic shear likelihood combining xi with an external mass-aperture theory.

    The xi contribution is evaluated with the standard CosmoLike real-space cosmic
    shear pipeline. The mass-aperture (Map3) contribution is evaluated through the
    map3 emulator with a user-provided data vector and covariance.
    Cross-covariance between xi and map3 is ignored.
    """

    #_required_cosmo_params = ("As", "ns", "H0", "omegab", "omegam", "mnu", "w")
    _required_cosmo_params = ("As", "ns", "H0", "omegab", "omegam", "mnu", "w", "sigma8")

    def get_requirements(self):
        # Inherit requirements from the parent xi pipeline, then add sigma8
        # which is needed by the Map3 emulator to compute s8 = sigma8*sqrt(Om/0.3).
        reqs = super(cosmic_shear_2pt_map3, self).get_requirements()
        reqs["sigma8"] = None
        return reqs

    def initialize(self):
        if self.use_emulator:
            raise LoggedError(
                self.log,
                "cosmic_shear_2pt_map3 does not support use_emulator=True. "
                "Use the non-emulator pipeline for xi + map3.",
            )

        super(cosmic_shear_2pt_map3, self).initialize(probe="xi")

        self.map3_module_path = getattr(self, "map3_module_path", None)
        self.map3_data_vector_file = getattr(self, "map3_data_vector_file", None)
        self.map3_cov_file = getattr(self, "map3_cov_file", None)
        self.map3_mask_file = getattr(self, "map3_mask_file", None)
        self.map3_nz_files = getattr(self, "map3_nz_files", None)
        self.map3_function_name = getattr(self, "map3_function_name", "compute_map3")
        self.map3_model_filename = getattr(self, "map3_model_filename", None)
        self.map3_rescaling_filename = getattr(self, "map3_rescaling_filename", None)
        self.map3_metadata_file = getattr(self, "map3_metadata_file", None)
        self.map3_z_values = getattr(self, "map3_z_values", None)
        self.map3_cosmo_model = getattr(self, "map3_cosmo_model", "LCDM")
        self.map3_nz_upsampling = getattr(self, "map3_nz_upsampling", 100)
        self.map3_perbin = getattr(self, "map3_perbin", False)
        self.map3_ia_alpha = getattr(self, "map3_ia_alpha", 1.0)
        self.map3_ia_z0 = getattr(self, "map3_ia_z0", 0.62)
        self.map3_ia_amplitude_prefix = getattr(self, "map3_ia_amplitude_prefix", "roman_A1_")
        self.map3_ia_global_amplitude = getattr(self, "map3_ia_global_amplitude", 0.0)
        self.map3_use_pixwin = getattr(self, "map3_use_pixwin", False)
        self.map3_nside = getattr(self, "map3_nside", None)

        missing = [
            name
            for name, value in (
                ("map3_module_path", self.map3_module_path),
                ("map3_data_vector_file", self.map3_data_vector_file),
                ("map3_cov_file", self.map3_cov_file),
                ("map3_nz_files", self.map3_nz_files),
            )
            if value in (None, "")
        ]
        if missing:
            raise LoggedError(
                self.log,
                "Missing required options for cosmic_shear_2pt_map3: %s.",
                ", ".join(missing),
            )

        self.map3_data_vector = self._load_map3_data_vector(self.map3_data_vector_file)
        self.map3_cov = self._load_map3_covariance(self.map3_cov_file)

        if self.map3_cov.ndim != 2 or self.map3_cov.shape[0] != self.map3_cov.shape[1]:
            raise LoggedError(self.log, "map3_cov_file must contain a square matrix.")

        if self.map3_cov.shape[0] != self.map3_data_vector.size:
            raise LoggedError(
                self.log,
                "Map3 covariance size (%d) does not match map3 data vector size (%d).",
                self.map3_cov.shape[0],
                self.map3_data_vector.size,
            )

        if self.map3_mask_file is None:
            self.map3_mask = np.ones(self.map3_data_vector.size, dtype=bool)
        else:
            tmp_mask = np.ravel(np.loadtxt(self.map3_mask_file)).astype(int)
            if tmp_mask.size != self.map3_data_vector.size:
                raise LoggedError(
                    self.log,
                    "Map3 mask size (%d) does not match map3 data vector size (%d).",
                    tmp_mask.size,
                    self.map3_data_vector.size,
                )
            self.map3_mask = tmp_mask != 0

        self.inv_map3_cov = np.linalg.inv(self.map3_cov[np.ix_(self.map3_mask, self.map3_mask)])
        self._map3_module = self._load_map3_module()
        self._map3_callable = getattr(self._map3_module, self.map3_function_name, None)
        if self._map3_callable is None:
            raise LoggedError(
                self.log,
                "Function '%s' was not found in map3_module_path='%s'.",
                self.map3_function_name,
                self.map3_module_path,
            )

    @staticmethod
    def _is_fits_path(filename):
        return str(filename).lower().endswith((".fits", ".fit", ".fits.gz", ".fit.gz"))

    def _import_fits(self):
        try:
            from astropy.io import fits
        except ImportError as exc:
            raise LoggedError(
                self.log,
                "Reading map3 FITS inputs requires astropy, but it could not be imported: %s.",
                exc,
            )
        return fits

    def _load_map3_data_vector(self, filename):
        if self._is_fits_path(filename):
            fits = self._import_fits()
            with fits.open(filename) as hdul:
                try:
                    data = np.asarray(hdul["map3"].data["VALUE"], dtype="float64")
                except (KeyError, ValueError) as exc:
                    raise LoggedError(
                        self.log,
                        "Could not read VALUE column from 'map3' extension in '%s': %s.",
                        filename,
                        exc,
                    )
            return np.ravel(data)
        return np.ravel(np.loadtxt(filename, dtype="float64"))

    def _load_map3_covariance(self, filename):
        if self._is_fits_path(filename):
            fits = self._import_fits()
            with fits.open(filename) as hdul:
                try:
                    cov = np.asarray(hdul["COVMAT"].data, dtype="float64")
                except KeyError as exc:
                    raise LoggedError(
                        self.log,
                        "Could not read 'COVMAT' extension in '%s': %s.",
                        filename,
                        exc,
                    )
            return cov
        return np.loadtxt(filename, dtype="float64")

    def _load_map3_module(self):
        module_path = os.path.abspath(self.map3_module_path)
        module_name = os.path.splitext(os.path.basename(module_path))[0]
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise LoggedError(self.log, "Could not import map3 module from '%s'.", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _get_cosmology_parameters(self):
        cosmo_params = {}
        for name in self._required_cosmo_params:
            cosmo_params[name] = self.provider.get_param(name)
        return cosmo_params

    def _get_map3_from_external_module(self):
        map3 = self._map3_callable(
            self._get_cosmology_parameters(),
            self.map3_data_vector_file,
            self.map3_cov_file,
            self.map3_nz_files,
            provider=self.provider,
            model_filename=self.map3_model_filename,
            rescaling_filename=self.map3_rescaling_filename,
            metadata_file=self.map3_metadata_file,
            z_values=self.map3_z_values,
            cosmo_model=self.map3_cosmo_model,
            nz_upsampling=self.map3_nz_upsampling,
            perbin=self.map3_perbin,
            ia_alpha=self.map3_ia_alpha,
            ia_z0=self.map3_ia_z0,
            ia_amplitude_prefix=self.map3_ia_amplitude_prefix,
            ia_global_amplitude=self.map3_ia_global_amplitude,
            use_pixwin=self.map3_use_pixwin,
            nside=self.map3_nside,
        )
        map3 = np.asarray(map3, dtype="float64").ravel()
        if map3.size != self.map3_data_vector.size:
            raise LoggedError(
                self.log,
                "Map3 theory vector size (%d) does not match map3 data vector size (%d).",
                map3.size,
                self.map3_data_vector.size,
            )
        return map3

    def get_datavector(self, **params):
        xi_dv = np.asarray(
            super(cosmic_shear_2pt_map3, self).get_datavector(**params),
            dtype="float64",
        ).ravel()
        map3_dv = self._get_map3_from_external_module()
        return xi_dv, map3_dv

    def compute_logp(self, datavector):
        xi_dv, map3_dv = datavector
        logp_xi = super(cosmic_shear_2pt_map3, self).compute_logp(xi_dv)

        map3_residual = map3_dv - self.map3_data_vector
        map3_masked_residual = map3_residual[self.map3_mask]
        chi2_map3 = float(map3_masked_residual @ self.inv_map3_cov @ map3_masked_residual)
        return logp_xi - 0.5 * chi2_map3

    def logp(self, **params):
        return self.compute_logp(self.get_datavector(**params))
