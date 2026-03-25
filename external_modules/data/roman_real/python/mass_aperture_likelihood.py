"""Mass aperture emulator interface rebuilt without cosmopower.
So far the filter aperture scales are hard-coded. This will be changed
in a next version
"""

from __future__ import annotations

import pickle
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn as nn
from astropy.cosmology import wCDM
from scipy.interpolate import interp1d
import fastnc
from astropy.io import fits

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from threepoint import ThreePointDataClass


class _CosmopowerActivation(nn.Module):
    """
    f(x) = (β + sigmoid(α·x) · (1 − β)) · x
    α and β are learned per-neuron parameters, initialised from N(0,1).
    """

    def __init__(self, n_units: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.randn(n_units))
        self.beta  = nn.Parameter(torch.randn(n_units))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.beta + torch.sigmoid(self.alpha * x) * (1.0 - self.beta)) * x

class _Map3MLP(nn.Module):
    """
    wCDM : 6 → [64, 256, 1024, 1024, 256, 192] → N outputs
    LCDM : 5 → [64, 256, 1024, 1024, 384, 192] → N outputs
    """

    def __init__(self, n_inputs: int, n_outputs: int, hidden: Sequence[int]):
        super().__init__()

        self.linears     = nn.ModuleList()
        self.activations = nn.ModuleList()
        in_dim = n_inputs
        for h in hidden:
            self.linears.append(nn.Linear(in_dim, h))
            self.activations.append(_CosmopowerActivation(h))
            in_dim = h
        self.output_layer = nn.Linear(in_dim, n_outputs)

        # Normalisation buffers — overwritten when the checkpoint is loaded
        self.register_buffer("params_mean",   torch.zeros(n_inputs))
        self.register_buffer("params_std",    torch.ones(n_inputs))
        self.register_buffer("features_mean", torch.zeros(n_outputs))
        self.register_buffer("features_std",  torch.ones(n_outputs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Internal input z-score normalisation
        x = (x - self.params_mean) / self.params_std
        for linear, act in zip(self.linears, self.activations):
            x = act(linear(x))
        x = self.output_layer(x)
        # Internal output rescaling
        return x * self.features_std + self.features_mean

class _Map3Network:

    def __init__(self, n_inputs: int, n_outputs: int, hidden: Sequence[int]):
        self._n_inputs  = n_inputs
        self._n_outputs = n_outputs
        self._hidden    = list(hidden)
        self._model: Optional[_Map3MLP] = None
        self._device    = torch.device("cpu")

    def restore(self, path: str) -> None:

        self._model = _Map3MLP(self._n_inputs, self._n_outputs, self._hidden)
        state = torch.load(path, map_location=self._device)
        self._model.load_state_dict(state)
        self._model.eval()

    def predictions_np(self, params_arr: np.ndarray) -> np.ndarray:

        if self._model is None:
            raise RuntimeError("Call restore() before predictions_np().")
        x = torch.tensor(params_arr.astype(np.float32), device=self._device)
        with torch.no_grad():
            out = self._model(x).cpu().numpy()
        return out  # shape (1, n_outputs)

_DEFAULT_IA_ALPHA = 1.0
_DEFAULT_IA_Z0    = 0.62
_SETUP_CACHE: Dict[Tuple[Any, ...], Dict[str, Any]] = {}

def get_healpix_window_function(nside: int):
    import healpy as hp

    w   = hp.sphtfunc.pixwin(nside)
    ell = np.arange(3 * nside)
    fnc = interp1d(ell, w, kind="linear", bounds_error=False, fill_value=(1, 0))
    return lambda l1, l2, l3: fnc(l1) * fnc(l2) * fnc(l3)


def rescale_params(
    params: np.ndarray, scale: Mapping[str, Mapping[str, float]]
) -> np.ndarray:
    params_rescaled = np.zeros_like(params)
    params_rescaled[0] = (params[0] - scale["Omega_m"]["small"]) / (
        scale["Omega_m"]["large"] - scale["Omega_m"]["small"]
    )
    params_rescaled[1] = (params[1] - scale["s8"]["small"]) / (
        scale["s8"]["large"] - scale["s8"]["small"]
    )
    params_rescaled[2] = (params[2] - scale["h0"]["small"]) / (
        scale["h0"]["large"] - scale["h0"]["small"]
    )
    params_rescaled[3] = (params[3] - scale["Omega_b"]["small"]) / (
        scale["Omega_b"]["large"] - scale["Omega_b"]["small"]
    )
    params_rescaled[4] = (params[4] - scale["ns"]["small"]) / (
        scale["ns"]["large"] - scale["ns"]["small"]
    )
    if len(params_rescaled) == 6:
        params_rescaled[5] = (params[5] - scale["w"]["small"]) / (
            scale["w"]["large"] - scale["w"]["small"]
        )
    return params_rescaled


def post_process(
    array: np.ndarray, scale: Mapping[str, Sequence[float]]
) -> np.ndarray:
    out_array = np.zeros_like(array)
    maxv = scale["large"]
    minv = scale["small"]
    for i in range(len(array[0])):
        tmp = array[:, i] * (maxv[i] - minv[i]) + minv[i]
        out_array[:, i] = 10 ** tmp
    return out_array


def upsampling(
    z: np.ndarray, pred: np.ndarray, n: int
) -> Tuple[np.ndarray, np.ndarray]:
    if n <= len(z):
        return z, pred
    z_new    = np.linspace(z.min(), z.max(), n)
    pred_new = np.zeros((n, pred.shape[1]))
    for i in range(pred.shape[1]):
        pred_new[:, i] = interp1d(z, pred[:, i])(z_new)
    return z_new, pred_new


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _normalize_path(path: Optional[str]) -> Optional[str]:
    if path in (None, ""):
        return None
    return os.path.abspath(os.path.expanduser(path))


def _load_nz_file(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.loadtxt(filename)
    if arr.ndim == 1 or arr.shape[1] < 2:
        raise ValueError(
            f"n(z) file '{filename}' must contain at least two columns: z and n(z)."
        )
    return np.asarray(arr[:, 0], dtype=float), np.asarray(arr[:, 1], dtype=float)


def _load_nz_fits(
        fits_file: str,
        bins: Optional[Sequence[int]] = None,
        extname: str = "nz_source",
        z_col: str = "Z_MID",
        bin_prefix: str = "BIN",
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    with fits.open(fits_file) as hdul:
        data = hdul[extname].data
        z = np.asarray(data[z_col], dtype=float)
        if bins is None:
            # Auto-discover all BINn columns
            cols = [c.name for c in hdul[extname].columns]
            bins = sorted(
                int(c[len(bin_prefix):])
                for c in cols
                if c.startswith(bin_prefix) and c[len(bin_prefix):].isdigit()
            )
        z_arrays = [z for _ in bins]
        nz_arrays = [np.asarray(data[f"{bin_prefix}{b}"], dtype=float) for b in bins]
    return z_arrays, nz_arrays


def _is_fits_path(path: str) -> bool:
    return str(path).lower().endswith((".fits", ".fit", ".fits.gz", ".fit.gz"))


def _load_source_distributions(
        nz_files: Sequence[str],
) -> Tuple[List[np.ndarray], List[np.ndarray]]:

    nz_files = list(nz_files)

    if len(nz_files) == 1 and (
            _is_fits_path(nz_files[0].split(":")[0])
    ):
        parts = nz_files[0].split(":", 1)
        fits_path = parts[0]
        bins = (
            [int(b) for b in parts[1].split(",") if b.strip()]
            if len(parts) > 1 else None
        )
        return _load_nz_fits(fits_path, bins=bins)

    # Text-file branch: one file per bin
    z_arrays: List[np.ndarray] = []
    nz_arrays: List[np.ndarray] = []
    for filename in nz_files:
        z, nz = _load_nz_file(filename)
        z_arrays.append(z)
        nz_arrays.append(nz)
    return z_arrays, nz_arrays

#def _load_source_distributions(
#    nz_files: Sequence[str],
#) -> Tuple[List[np.ndarray], List[np.ndarray]]:
#    z_arrays:  List[np.ndarray] = []
#    nz_arrays: List[np.ndarray] = []
#    for filename in nz_files:
#        z, nz = _load_nz_file(filename)
#        z_arrays.append(z)
#        nz_arrays.append(nz)
#    return z_arrays, nz_arrays


def _load_map3_metadata(
    metadata_file: str,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], int]:
    data = ThreePointDataClass.from_fits(metadata_file)
    if data.bin_type != "SSS":
        raise ValueError(
            f"Map3 metadata file '{metadata_file}' must contain an SSS data vector, "
            f"got '{data.bin_type}'."
        )
    scombs     = data.get_z_bin(unique=True).T
    filters:   Dict[str, np.ndarray] = {}
    filter_num = None
    for scomb in scombs:
        name = "_".join(str(s) for s in scomb)
        sel  = data.selection_z_bin(scomb, "z123", condition="==")
        filters[name] = data.get_t_bin(sel=sel)
        if filter_num is None:
            filter_num = filters[name].shape[1]
    if filter_num is None:
        raise ValueError(f"No sample combinations found in '{metadata_file}'.")
    return scombs, filters, int(filter_num)

def _build_network(model_cosmo: str, n_outputs: int) -> _Map3Network:

    if model_cosmo == "wCDM":
        return _Map3Network(
            n_inputs=6,
            n_outputs=n_outputs,
            hidden=[64, 256, 1024, 1024, 256, 192],
        )
    else:  # LCDM
        return _Map3Network(
            n_inputs=5,
            n_outputs=n_outputs,
            hidden=[64, 256, 1024, 1024, 384, 192],
        )

def _cache_key(
    model_filename:     str,
    rescaling_filename: str,
    metadata_file:      str,
    z_values:           Sequence[float],
    model_cosmo:        str,
    nz_upsampling:      int,
    perbin:             bool,
    use_pixwin:         bool,
    nside:              Optional[int],
) -> Tuple[Any, ...]:
    return (
        _normalize_path(model_filename),
        _normalize_path(rescaling_filename),
        _normalize_path(metadata_file),
        tuple(np.asarray(z_values, dtype=float).tolist()),
        model_cosmo,
        int(nz_upsampling),
        bool(perbin),
        bool(use_pixwin),
        None if nside is None else int(nside),
    )

def setup_map3_module(
    *,
    model_filename:     str,
    rescaling_filename: str,
    metadata_file:      str,
    z_values:           Sequence[float],
    cosmo_model:        str = "wCDM",
    nz_upsampling:      int = 100,
    perbin:             bool = False,
    use_pixwin:         bool = False,
    nside:              Optional[int] = None,
) -> Dict[str, Any]:
    key = _cache_key(
        model_filename, rescaling_filename, metadata_file,
        z_values, cosmo_model, nz_upsampling, perbin, use_pixwin, nside,
    )
    if key in _SETUP_CACHE:
        return _SETUP_CACHE[key]

    required = {
        "model_filename":     model_filename,
        "rescaling_filename": rescaling_filename,
        "metadata_file":      metadata_file,
    }
    missing = [name for name, value in required.items() if value in (None, "")]
    if missing:
        raise ValueError(
            "Missing required mass-aperture configuration option(s): "
            + ", ".join(missing)
        )

    z_values = np.asarray(z_values, dtype=float)
    if z_values.ndim != 1 or z_values.size == 0:
        raise ValueError("map3_z_values must be a non-empty 1D array.")

    sample_combinations, filters, filter_num = _load_map3_metadata(metadata_file)
    n_outputs = filter_num * len(z_values)

    network = _build_network(cosmo_model, n_outputs)
    network.restore(_normalize_path(model_filename))

    with open(_normalize_path(rescaling_filename), "rb") as handle:
        rescaling_values = pickle.load(handle)

    bispectrum = fastnc.bispectrum.BispectrumHalofit()
    if use_pixwin:
        if nside is None:
            raise ValueError("nside must be provided when use_pixwin=True.")
        bispectrum.set_window_function(get_healpix_window_function(nside))

    config = {
        "bispectrum":          bispectrum,
        "network":             network,
        "rescaling_params":    rescaling_values["params"],
        "rescaling_features":  rescaling_values["features"],
        "sample_combinations": sample_combinations,
        "filters":             filters,
        "filter_num":          filter_num,
        "zarray":              z_values,
        "nz_upsampling":       int(nz_upsampling),
        "model_cosmo":         cosmo_model,
        "perbin":              bool(perbin),
    }
    _SETUP_CACHE[key] = config
    return config

def _get_sigma8(
    cosmology_parameters: Mapping[str, float], provider: Any
) -> float:
    if "sigma8" in cosmology_parameters:
        return cosmology_parameters["sigma8"]
    if provider is not None:
        try:
            return provider.get_param("sigma8")
        except Exception as exc:
            raise ValueError(
                "sigma8 is required but was not available from the provider."
            ) from exc
    raise ValueError(
        "sigma8 is required in cosmology_parameters or via the Cobaya provider."
    )


def _build_raw_parameter_array(
    cosmology_parameters: Mapping[str, float],
    model_cosmo: str,
    provider: Any,
) -> np.ndarray:

    sigma8 = _get_sigma8(cosmology_parameters, provider)
    s8     = sigma8 * np.sqrt(cosmology_parameters["omegam"] / 0.3)

    if model_cosmo == "wCDM":
        return np.array(
            [
                cosmology_parameters["omegam"],
                s8,
                cosmology_parameters["H0"] / 100.0,
                cosmology_parameters["omegab"],
                cosmology_parameters["ns"],
                cosmology_parameters["w"],
            ],
            dtype=np.float32,
        ).reshape(1, -1)

    return np.array(
        [
            cosmology_parameters["omegam"],
            s8,
            cosmology_parameters["H0"] / 100.0,
            cosmology_parameters["omegab"],
            cosmology_parameters["ns"],
        ],
        dtype=np.float32,
    ).reshape(1, -1)


def _prediction_dict(
    params_for_network: np.ndarray, model_cosmo: str
) -> Dict[str, List[float]]:
    if model_cosmo == "wCDM":
        keys = ["Omega_m", "s8", "h0", "Omega_b", "ns", "w"]
    else:
        keys = ["Omega_m", "s8", "h0", "Omega_b", "ns"]
    return {k: [float(params_for_network[i])] for i, k in enumerate(keys)}


def _reshape_predictions(
    predictions_rescaled: np.ndarray, zarray: np.ndarray, filter_num: int
) -> np.ndarray:
    predictions_newshape = np.zeros((len(zarray), filter_num))
    for i in range(filter_num):
        predictions_newshape[:, i] = predictions_rescaled[:, i::filter_num]
    return predictions_newshape


def _set_linear_theory(
    bs: Any,
    provider: Any,
    cosmology_parameters: Mapping[str, float],
    zarray: np.ndarray,
):
    if provider is None:
        raise ValueError(
            "The Cobaya theory provider is required to build the linear matter "
            "power and growth inputs."
        )
    h         = cosmology_parameters["H0"] / 100.0
    z_for_pk  = np.unique(np.concatenate(([0.0], zarray)))
    pk_interp = provider.get_Pk_interpolator(("delta_tot", "delta_tot"), nonlinear=False)

    k_h = np.logspace(-4.5, np.log10(7), 400)
    p_k = pk_interp.P(z_for_pk, k_h)
    if p_k.ndim == 1:
        p_k = p_k[np.newaxis, :]
    bs.set_pklin(k_h / h, p_k[0] * h**3)

    k_growth = 5.0e-4
    growth   = np.sqrt(
        pk_interp.P(z_for_pk, k_growth) / pk_interp.P(0.0, k_growth)
    ) * (1.0 + z_for_pk)
    growth  /= growth[-1]
    bs.set_lgr(z_for_pk, growth)


def _set_ia_parameters(
    bs: Any,
    provider: Any,
    perbin: bool,
    nzbin: int,
    ia_alpha: float,
    ia_z0: float,
    ia_amplitude_prefix: str,
    ia_global_amplitude: float,
):
    if perbin:
        nla_param: Dict[str, Any] = {"alphaIA": ia_alpha, "z0": ia_z0, "perbin": True}
        for i in range(nzbin):
            if provider is None:
                raise ValueError(
                    "per-bin IA requires the Cobaya provider so nuisance "
                    "parameters can be read."
                )
            nla_param[f"AIA_{i + 1}"] = provider.get_param(
                f"{ia_amplitude_prefix}{i + 1}"
            )
        bs.set_NLA_param(nla_param)
    else:
        bs.set_NLA_param(
            {"AIA": ia_global_amplitude, "alphaIA": ia_alpha, "z0": ia_z0, "perbin": False}
        )
    bs.config_IA["NLA"] = True

def compute_map3(
    cosmology_parameters: Mapping[str, float],
    data_vector_file:     Optional[str],
    covariance_file:      Optional[str],
    nz_files:             Sequence[str],
    **kwargs: Any,
) -> np.ndarray:
    """Compute a Map3 theory vector using the emulator + fastnc configs.

    Parameters
    ----------
    cosmology_parameters
        Mapping with ``H0``, ``omegam``, ``omegab``, ``ns``, ``w`` and
        either ``sigma8`` or a Cobaya provider that can supply it.
    nz_files
        Sequence of plain-text files with at least two columns: z and n(z).
    kwargs
        Runtime configuration passed by ``cosmic_shear_2pt_map3``. Required keys:
        ``model_filename``, ``rescaling_filename``, ``metadata_file``,
        ``z_values``. Recommended key: ``provider``.
    """
    del data_vector_file, covariance_file

    provider = kwargs.get("provider")
    config   = setup_map3_module(
        model_filename     = kwargs["model_filename"],
        rescaling_filename = kwargs["rescaling_filename"],
        metadata_file      = kwargs["metadata_file"],
        z_values           = kwargs["z_values"],
        cosmo_model        = kwargs.get("cosmo_model", "wCDM"),
        nz_upsampling      = kwargs.get("nz_upsampling", 100),
        perbin             = kwargs.get("perbin", False),
        use_pixwin         = kwargs.get("use_pixwin", False),
        nside              = kwargs.get("nside"),
    )

    params_arr           = _build_raw_parameter_array(
        cosmology_parameters, config["model_cosmo"], provider
    )
    predictions          = config["network"].predictions_np(params_arr)
    predictions_rescaled = post_process(predictions, config["rescaling_features"])
    predictions_newshape = _reshape_predictions(
        predictions_rescaled, config["zarray"], config["filter_num"]
    )

    bs     = config["bispectrum"]
    sigma8 = _get_sigma8(cosmology_parameters, provider)
    cosmo  = wCDM(
        H0   = cosmology_parameters["H0"],
        Om0  = cosmology_parameters["omegam"],
        Ode0 = 1.0 - cosmology_parameters["omegam"],
        w0   = cosmology_parameters["w"],
        meta = {"sigma8": sigma8, "n": cosmology_parameters["ns"]},
    )
    bs.set_cosmology(cosmo)

    nz_paths = [_normalize_path(p) for p in _as_list(nz_files)]
    if not nz_paths:
        raise ValueError("At least one n(z) file must be provided via map3_nz_files.")
    z_list, nz_list = _load_source_distributions(nz_paths)
    nzbin           = len(nz_list)
    bs.set_source_distribution(z_list, nz_list, list(range(1, nzbin + 1)))

    _set_ia_parameters(
        bs, provider, config["perbin"], nzbin,
        kwargs.get("ia_alpha", _DEFAULT_IA_ALPHA),
        kwargs.get("ia_z0", _DEFAULT_IA_Z0),
        kwargs.get("ia_amplitude_prefix", "roman_A1_"),
        kwargs.get("ia_global_amplitude", 0.0),
    )
    _set_linear_theory(bs, provider, cosmology_parameters, config["zarray"])

    if kwargs.get("baryon_fb") is not None:
        bs.set_baryon_param({"fb": kwargs["baryon_fb"]})

    bs.compute_kernel()

    zarray = config["zarray"]
    if config["nz_upsampling"] > len(zarray):
        zarray, predictions_newshape = upsampling(
            zarray, predictions_newshape, config["nz_upsampling"]
        )

    chi           = bs.z2chi(zarray)
    theory_vector: List[np.ndarray] = []
    for scomb in config["sample_combinations"]:
        scomb_tuple = tuple(int(x) for x in np.atleast_1d(scomb))
        z2g0   = bs.z2g_dict[scomb_tuple[0]]
        z2g1   = bs.z2g_dict[scomb_tuple[1]]
        z2g2   = bs.z2g_dict[scomb_tuple[2]]
        weight = (
            z2g0(zarray) * z2g1(zarray) * z2g2(zarray) / chi * (1 + zarray) ** 3
        )
        tmp = np.einsum("ij,i->ij", predictions_newshape, weight)
        theory_vector.append(np.trapz(tmp, chi, axis=0))

    return np.concatenate(theory_vector)
