#!/usr/bin/env python
"""
Fit optical parameters from PSF second moments using batoid ray tracing.

Uses LSSTBuilder with CCD height maps and AOS degrees of freedom.
"""

import numpy as np
import polars as pl
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import os
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import argparse

# batoid imports
import batoid
from batoid_rubin import LSSTBuilder


# AOS DOF structure (50 parameters total)
# Indices 0-4:   M2 rigid body (dx, dy, dz, rx, ry) in meters/radians
# Indices 5-9:   Camera rigid body (dx, dy, dz, rx, ry) in meters/radians
# Indices 10-29: M1M3 bending modes (20 modes)
# Indices 30-49: M2 bending modes (20 modes)

# AOS DOF indices and units (all in microns for translations, arcsec for rotations)
# Note: batoid_rubin applies sign flips internally, see builder.py line ~1285
AOS_DOF_NAMES = {
    'm2_dz': 0, 'm2_dx': 1, 'm2_dy': 2, 'm2_rx': 3, 'm2_ry': 4,
    'cam_dz': 5, 'cam_dx': 6, 'cam_dy': 7, 'cam_rx': 8, 'cam_ry': 9,
}
# Add M1M3 bending modes
for i in range(20):
    AOS_DOF_NAMES[f'm1m3_bend_{i}'] = 10 + i
# Add M2 bending modes
for i in range(20):
    AOS_DOF_NAMES[f'm2_bend_{i}'] = 30 + i

# LSST detector ID to batoid_rubin detector ID mapping
# These differ because of different numbering conventions
# Mapping found by matching detector center positions (x_fp, y_fp)
LSST_TO_BATOID_DET = {
    1: 30, 2: 33, 3: 28, 4: 31, 5: 34, 6: 29, 7: 32, 8: 35,
    9: 72, 10: 75, 11: 78, 12: 73, 13: 76, 14: 79, 15: 74, 16: 77, 17: 80,
    18: 117, 19: 120, 21: 118, 22: 121, 23: 124, 24: 119, 25: 122, 26: 125,
    28: 3, 29: 6, 30: 1, 31: 4, 32: 7, 33: 2, 34: 5, 35: 8,
    36: 36, 37: 39, 38: 42, 39: 37, 40: 40, 41: 43, 42: 38, 43: 41, 44: 44,
    45: 81, 46: 84, 47: 87, 48: 82, 49: 85, 50: 88, 51: 83, 52: 86, 53: 89,
    54: 126, 55: 129, 56: 132, 57: 127, 58: 130, 59: 133, 60: 128, 61: 131, 62: 134,
    63: 162, 64: 165, 66: 163, 67: 166, 68: 169, 69: 164, 70: 167, 71: 170,
    72: 9, 73: 12, 74: 15, 75: 10, 76: 13, 77: 16, 78: 11, 79: 14, 80: 17,
    81: 45, 82: 48, 83: 51, 84: 46, 85: 49, 86: 52, 87: 47, 88: 50, 89: 53,
    90: 90, 91: 93, 92: 96, 93: 91, 94: 94, 95: 97, 96: 92, 97: 95, 98: 98,
    99: 135, 100: 138, 101: 141, 102: 136, 103: 139, 104: 142, 105: 137, 106: 140, 107: 143,
    108: 171, 109: 174, 110: 177, 111: 172, 112: 175, 113: 178, 114: 173, 115: 176, 116: 179,
    117: 18, 118: 21, 119: 24, 120: 19, 121: 22, 124: 23, 125: 26,
    126: 54, 127: 57, 128: 60, 129: 55, 130: 58, 131: 61, 132: 56, 133: 59, 134: 62,
    135: 99, 136: 102, 137: 105, 138: 100, 139: 103, 140: 106, 141: 101, 142: 104, 143: 107,
    144: 144, 145: 147, 146: 150, 147: 145, 148: 148, 149: 151, 150: 146, 151: 149, 152: 152,
    153: 180, 154: 183, 155: 186, 156: 181, 157: 184, 158: 187, 159: 182, 160: 185,
    162: 63, 163: 66, 164: 69, 165: 64, 166: 67, 167: 70, 169: 68, 170: 71,
    171: 108, 172: 111, 173: 114, 174: 109, 175: 112, 176: 115, 177: 110, 178: 113, 179: 116,
    180: 153, 181: 156, 182: 159, 183: 154, 184: 157, 185: 160, 186: 155, 187: 158,
}


def get_default_fit_params() -> Dict[str, Dict[str, Any]]:
    """
    Get default fit parameters configuration.

    Returns dict where each key is a DOF name and value is:
        - 'fit': bool, whether to fit this parameter
        - 'init': float, initial value
        - 'bounds': tuple, (min, max) bounds
        - 'value': float, fixed value if fit=False
    """
    params = {}

    # M2 rigid body - units: microns for translations, arcsec for rotations
    # NOTE: m2_dz is FIXED (degenerate with cam_dz for focus)
    params['m2_dz'] = {'fit': False, 'init': 0, 'bounds': (-100, 100), 'value': 0}  # FIXED
    params['m2_dx'] = {'fit': False, 'init': 0, 'bounds': (-100, 100), 'value': 0}
    params['m2_dy'] = {'fit': False, 'init': 0, 'bounds': (-100, 100), 'value': 0}
    params['m2_rx'] = {'fit': False, 'init': 0, 'bounds': (-50, 50), 'value': 0}  # arcsec
    params['m2_ry'] = {'fit': False, 'init': 0, 'bounds': (-50, 50), 'value': 0}

    # Camera rigid body - units: microns for translations, arcsec for rotations
    params['cam_dz'] = {'fit': True, 'init': 0, 'bounds': (-100, 100), 'value': 0}
    params['cam_dx'] = {'fit': False, 'init': 0, 'bounds': (-100, 100), 'value': 0}
    params['cam_dy'] = {'fit': False, 'init': 0, 'bounds': (-100, 100), 'value': 0}
    params['cam_rx'] = {'fit': False, 'init': 0, 'bounds': (-50, 50), 'value': 0}
    params['cam_ry'] = {'fit': False, 'init': 0, 'bounds': (-50, 50), 'value': 0}

    # M1M3 bending modes - typically small
    for i in range(20):
        params[f'm1m3_bend_{i}'] = {'fit': False, 'init': 0, 'bounds': (-1, 1), 'value': 0}

    # M2 bending modes
    for i in range(20):
        params[f'm2_bend_{i}'] = {'fit': False, 'init': 0, 'bounds': (-1, 1), 'value': 0}

    return params


@dataclass
class FitConfig:
    """Configuration for optical parameter fitting."""
    wavelength: float = 480e-9  # g-band ~480nm
    n_rays_rad: int = 15  # rays in radial direction
    n_rays_az: int = 30  # rays in azimuthal direction
    pixel_scale: float = 10e-6  # 10 micron pixels
    focal_length: float = 10.312  # meters
    ccd_size_mm: float = 42.0  # CCD size in mm (for visualization)
    fit_params: Dict[str, Dict[str, Any]] = field(default_factory=get_default_fit_params)


def load_and_bin_data(parquet_file: str, config: FitConfig):
    """Load observed data and compute one mean value per detector.

    Parameters
    ----------
    parquet_file : str
        Input parquet file path
    config : FitConfig
        Configuration object

    Returns
    -------
    dict with one mean value per CCD for T, e1, e2 and raw data for visualization
    """
    df = pl.read_parquet(parquet_file)

    x_fp = df['x_fp'].to_numpy()
    y_fp = df['y_fp'].to_numpy()
    T = df['T'].to_numpy()
    e1 = df['e1'].to_numpy()
    e2 = df['e2'].to_numpy()
    detector = df['detector'].to_numpy()
    rotator_angle = df['rotator_angle_radian'][0]

    unique_detectors = np.unique(detector)

    grid_x, grid_y, grid_T, grid_e1, grid_e2, grid_det = [], [], [], [], [], []

    for det_id in unique_detectors:
        mask = detector == det_id
        if np.sum(mask) < 5:  # Need at least 5 stars
            continue

        # Compute mean position and mean moments for this detector
        grid_x.append(np.mean(x_fp[mask]))
        grid_y.append(np.mean(y_fp[mask]))
        grid_T.append(np.mean(T[mask]))
        grid_e1.append(np.mean(e1[mask]))
        grid_e2.append(np.mean(e2[mask]))
        grid_det.append(det_id)

    return {
        'x_fp': np.array(grid_x),
        'y_fp': np.array(grid_y),
        'T': np.array(grid_T),
        'e1': np.array(grid_e1),
        'e2': np.array(grid_e2),
        'detector': np.array(grid_det),
        'rotator_angle': rotator_angle,
        'raw_x': x_fp,
        'raw_y': y_fp,
        'raw_T': T,
        'raw_e1': e1,
        'raw_e2': e2,
    }


def focal_plane_to_field_angle(x_fp, y_fp, rotator_angle, focal_length=10.312):
    """Convert focal plane coordinates (mm) to field angles (radians)."""
    x_m = x_fp * 1e-3
    y_m = y_fp * 1e-3
    th_fp_x = x_m / focal_length
    th_fp_y = y_m / focal_length

    cos_r = np.cos(-rotator_angle)
    sin_r = np.sin(-rotator_angle)
    thx = cos_r * th_fp_x - sin_r * th_fp_y
    thy = sin_r * th_fp_x + cos_r * th_fp_y

    return thx, thy


def compute_psf_moments_at_point(optic, thx, thy, wavelength, config: FitConfig):
    """Compute PSF second moments at a single field point using ray tracing."""
    rays = batoid.RayVector.asPolar(
        optic,
        wavelength=wavelength,
        theta_x=thx,
        theta_y=thy,
        nrad=config.n_rays_rad,
        naz=config.n_rays_az,
    )

    optic.trace(rays)

    good = ~rays.vignetted
    if np.sum(good) < 10:
        return np.nan, np.nan, np.nan

    x = rays.x[good] - np.mean(rays.x[good])
    y = rays.y[good] - np.mean(rays.y[good])

    ixx = np.mean(x**2)
    iyy = np.mean(y**2)
    ixy = np.mean(x * y)

    ixx_pix = ixx / config.pixel_scale**2
    iyy_pix = iyy / config.pixel_scale**2
    ixy_pix = ixy / config.pixel_scale**2

    T = ixx_pix + iyy_pix
    e1 = (ixx_pix - iyy_pix) / T
    e2 = 2 * ixy_pix / T

    return T, e1, e2


def rotate_ellipticity(e1, e2, angle):
    """Rotate ellipticity by an angle (spin-2 transformation)."""
    cos2a = np.cos(2 * angle)
    sin2a = np.sin(2 * angle)
    e1_rot = e1 * cos2a + e2 * sin2a
    e2_rot = -e1 * sin2a + e2 * cos2a
    return e1_rot, e2_rot


def params_to_dof(fit_values: np.ndarray, fit_names: List[str],
                  config: FitConfig) -> np.ndarray:
    """
    Convert fit parameter values to full 50-element AOS DOF array.

    Parameters
    ----------
    fit_values : array
        Values for parameters being fit
    fit_names : list
        Names of parameters being fit
    config : FitConfig
        Configuration with fixed parameter values

    Returns
    -------
    dof : array (50,)
        Full AOS DOF array
    """
    dof = np.zeros(50)

    # Set fixed values
    for name, param in config.fit_params.items():
        if not param['fit']:
            idx = AOS_DOF_NAMES[name]
            dof[idx] = param['value']

    # Set fit values
    for val, name in zip(fit_values, fit_names):
        idx = AOS_DOF_NAMES[name]
        dof[idx] = val

    return dof


def forward_model(fit_values: np.ndarray, fit_names: List[str], data: dict,
                  config: FitConfig, fiducial_telescope=None,
                  return_full=False, verbose=False):
    """
    Compute predicted PSF moments given optical parameters.

    Uses LSSTBuilder with with_aos_dof() and CCD height maps.
    Subtracts global focal plane mean from both data and predictions
    to focus on spatial patterns (atmosphere dominates absolute values).
    """
    if fiducial_telescope is None:
        fiducial_telescope = batoid.Optic.fromYaml("LSST_r.yaml")

    # Convert fit values to full DOF array
    dof = params_to_dof(fit_values, fit_names, config)

    # Create builder with AOS DOF
    builder = LSSTBuilder(fiducial_telescope)
    builder = builder.with_aos_dof(dof)

    n_points = len(data['x_fp'])
    T_pred = np.zeros(n_points)
    e1_pred = np.zeros(n_points)
    e2_pred = np.zeros(n_points)

    # Cache built optics and field angles per detector
    optic_cache = {}
    field_angle_cache = {}

    # Build optics and compute field angles from batoid detector centers
    unique_detectors = np.unique(data['detector'])
    for det_id in unique_detectors:
        det_id = int(det_id)
        batoid_det_id = LSST_TO_BATOID_DET.get(det_id)
        if batoid_det_id is None:
            if verbose:
                print(f"Warning: no batoid mapping for LSST detector {det_id}")
            optic_cache[det_id] = None
            field_angle_cache[det_id] = (0, 0)
            continue

        try:
            optic = builder.build_det(batoid_det_id)
            optic_cache[det_id] = optic

            # Get batoid detector center from the Bicubic surface
            detector = optic.itemDict['LSST.LSSTCamera.Detector']
            surf = detector.surface
            x_center_m, y_center_m = 0, 0
            if hasattr(surf, 'surfaces'):
                for s in surf.surfaces:
                    if hasattr(s, 'xs') and hasattr(s, 'ys'):  # Bicubic
                        x_center_m = (s.xs[0] + s.xs[-1]) / 2
                        y_center_m = (s.ys[0] + s.ys[-1]) / 2
                        break

            # Convert batoid detector center (meters) to field angle
            # Note: batoid uses meters, no rotator angle needed (telescope frame)
            thx = x_center_m / config.focal_length
            thy = y_center_m / config.focal_length
            field_angle_cache[det_id] = (thx, thy)

        except Exception as e:
            if verbose:
                print(f"Warning: could not build optic for detector {det_id} (batoid {batoid_det_id}): {e}")
            optic_cache[det_id] = None
            field_angle_cache[det_id] = (0, 0)

    for i in range(n_points):
        det_id = int(data['detector'][i])

        optic = optic_cache[det_id]
        if optic is None:
            T_pred[i], e1_pred[i], e2_pred[i] = np.nan, np.nan, np.nan
            continue

        # Use detector center field angle (not individual grid point)
        thx, thy = field_angle_cache[det_id]
        T_pred[i], e1_pred[i], e2_pred[i] = compute_psf_moments_at_point(
            optic, thx, thy, config.wavelength, config
        )

    # Rotate ellipticity back to focal plane frame
    e1_pred, e2_pred = rotate_ellipticity(e1_pred, e2_pred, data['rotator_angle'])

    # Subtract global focal plane mean from predictions
    valid = np.isfinite(T_pred) & np.isfinite(e1_pred) & np.isfinite(e2_pred)
    if np.sum(valid) == 0:
        if return_full:
            return {'T': T_pred, 'e1': e1_pred, 'e2': e2_pred,
                    'dT': T_pred, 'de1': e1_pred, 'de2': e2_pred}
        return 1e10

    T_pred_mean = np.nanmean(T_pred[valid])
    e1_pred_mean = np.nanmean(e1_pred[valid])
    e2_pred_mean = np.nanmean(e2_pred[valid])

    dT_pred = T_pred - T_pred_mean
    de1_pred = e1_pred - e1_pred_mean
    de2_pred = e2_pred - e2_pred_mean

    # Subtract global focal plane mean from observations
    T_obs = data['T']
    e1_obs = data['e1']
    e2_obs = data['e2']

    T_obs_mean = np.nanmean(T_obs)
    e1_obs_mean = np.nanmean(e1_obs)
    e2_obs_mean = np.nanmean(e2_obs)

    dT_obs = T_obs - T_obs_mean
    de1_obs = e1_obs - e1_obs_mean
    de2_obs = e2_obs - e2_obs_mean

    if return_full:
        return {
            'T': T_pred, 'e1': e1_pred, 'e2': e2_pred,
            'dT': dT_pred, 'de1': de1_pred, 'de2': de2_pred,
            'T_mean': T_pred_mean, 'e1_mean': e1_pred_mean, 'e2_mean': e2_pred_mean,
        }

    # Compute chi2 on residuals (mean-subtracted)
    sigma_dT = 0.3  # pixel^2 for spatial variations
    sigma_de = 0.02  # ellipticity

    chi2_dT = np.sum(((dT_pred[valid] - dT_obs[valid]) / sigma_dT)**2)
    chi2_de1 = np.sum(((de1_pred[valid] - de1_obs[valid]) / sigma_de)**2)
    chi2_de2 = np.sum(((de2_pred[valid] - de2_obs[valid]) / sigma_de)**2)

    return chi2_dT + chi2_de1 + chi2_de2


def fit_optics(data: dict, config: FitConfig, verbose=True):
    """
    Fit optical parameters to observed PSF moments using iminuit.
    """
    fiducial = batoid.Optic.fromYaml("LSST_r.yaml")

    # Get parameters to fit
    fit_names = [name for name, p in config.fit_params.items() if p['fit']]
    n_fit = len(fit_names)

    if n_fit == 0:
        print("No parameters to fit!")
        return np.array([]), None

    print(f"Fitting {n_fit} parameters: {fit_names}")

    # Get initial values and bounds
    initial = [config.fit_params[name]['init'] for name in fit_names]
    bounds = [config.fit_params[name]['bounds'] for name in fit_names]

    try:
        from iminuit import Minuit

        def objective(*vals):
            chi2 = forward_model(np.array(vals), fit_names, data, config, fiducial)
            if verbose:
                param_str = ', '.join([f'{n}={v:.2e}' for n, v in zip(fit_names, vals)])
                print(f"  chi2={chi2:.2f} | {param_str}")
            return chi2

        m = Minuit(objective, *initial, name=fit_names)
        for i, (lo, hi) in enumerate(bounds):
            m.limits[i] = (lo, hi)

        m.errordef = Minuit.LEAST_SQUARES

        if verbose:
            print("\nStarting Minuit optimization...")

        m.migrad()

        return np.array(m.values), m

    except ImportError:
        print("iminuit not installed. Using scipy.optimize instead.")
        from scipy.optimize import minimize

        def objective(vals):
            chi2 = forward_model(vals, fit_names, data, config, fiducial)
            if verbose:
                print(f"  chi2={chi2:.2f}")
            return chi2

        result = minimize(objective, initial, method='L-BFGS-B', bounds=bounds)
        return result.x, None


def load_ccd_geometry(geometry_file: str = None):
    """Load CCD geometry (corners) from CSV file."""
    if geometry_file is None:
        geometry_file = os.path.join(os.path.dirname(__file__), 'data', 'ccd_geometry.csv')

    df = pl.read_csv(geometry_file)
    geometry = {}
    for row in df.iter_rows(named=True):
        det_id = row['detector']
        corners = np.array([
            [row['corner0_x'], row['corner0_y']],
            [row['corner1_x'], row['corner1_y']],
            [row['corner2_x'], row['corner2_y']],
            [row['corner3_x'], row['corner3_y']],
        ])
        geometry[det_id] = {
            'center': (row['x_center'], row['y_center']),
            'corners': corners,
            'name': row['name'],
        }
    return geometry


def plot_ccd_polygons(ax, detectors, values, geometry, cmap, vmin, vmax):
    """Plot CCDs as polygons with colors based on values."""
    patches = []
    colors = []

    for det_id, val in zip(detectors, values):
        if np.isfinite(val) and det_id in geometry:
            corners = geometry[det_id]['corners']
            poly = Polygon(corners, closed=True)
            patches.append(poly)
            colors.append(val)

    if len(patches) == 0:
        return None

    collection = PatchCollection(patches, cmap=cmap, edgecolor='k', linewidth=0.3)
    collection.set_array(np.array(colors))
    collection.set_clim(vmin, vmax)
    ax.add_collection(collection)
    return collection


def plot_comparison(data, pred, output_file='fit_comparison.png', geometry=None):
    """Plot observed vs predicted PSF moments (mean-subtracted) using CCD polygons."""
    if geometry is None:
        geometry = load_ccd_geometry()

    fig, axes = plt.subplots(4, 3, figsize=(15, 18))

    # Mean-subtract the observed data for plotting
    raw_dT = data['raw_T'] - np.nanmean(data['raw_T'])
    raw_de1 = data['raw_e1'] - np.nanmean(data['raw_e1'])
    raw_de2 = data['raw_e2'] - np.nanmean(data['raw_e2'])

    # Mean per CCD observed (used in chi2)
    obs_dT = data['T'] - np.nanmean(data['T'])
    obs_de1 = data['e1'] - np.nanmean(data['e1'])
    obs_de2 = data['e2'] - np.nanmean(data['e2'])

    # Residuals
    res_dT = pred['dT'] - obs_dT
    res_de1 = pred['de1'] - obs_de1
    res_de2 = pred['de2'] - obs_de2

    vmin_dT, vmax_dT = -1.5, 1.5
    vmin_de, vmax_de = -0.10, 0.10
    detectors = data['detector']

    # Row 1: Raw observed (mean-subtracted) - scatter points
    ax = axes[0, 0]
    sc = ax.scatter(data['raw_x'], data['raw_y'], c=raw_dT, s=1,
                    cmap='seismic', vmin=vmin_dT, vmax=vmax_dT)
    plt.colorbar(sc, ax=ax, label='dT (pixel²)')
    ax.set_title('Raw Observed dT = T - <T>')
    ax.set_xlabel('x_fp (mm)')
    ax.set_ylabel('y_fp (mm)')
    ax.set_aspect('equal')
    ax.set_xlim(-350, 350)
    ax.set_ylim(-350, 350)

    ax = axes[0, 1]
    sc = ax.scatter(data['raw_x'], data['raw_y'], c=raw_de1, s=1,
                    cmap='seismic', vmin=vmin_de, vmax=vmax_de)
    plt.colorbar(sc, ax=ax, label='de1')
    ax.set_title('Raw Observed de1 = e1 - <e1>')
    ax.set_xlabel('x_fp (mm)')
    ax.set_aspect('equal')
    ax.set_xlim(-350, 350)
    ax.set_ylim(-350, 350)

    ax = axes[0, 2]
    sc = ax.scatter(data['raw_x'], data['raw_y'], c=raw_de2, s=1,
                    cmap='seismic', vmin=vmin_de, vmax=vmax_de)
    plt.colorbar(sc, ax=ax, label='de2')
    ax.set_title('Raw Observed de2 = e2 - <e2>')
    ax.set_xlabel('x_fp (mm)')
    ax.set_aspect('equal')
    ax.set_xlim(-350, 350)
    ax.set_ylim(-350, 350)

    # Row 2: Mean per CCD observed - CCD polygons
    ax = axes[1, 0]
    col = plot_ccd_polygons(ax, detectors, obs_dT, geometry,
                            'seismic', vmin_dT, vmax_dT)
    if col:
        plt.colorbar(col, ax=ax, label='dT (pixel²)')
    ax.set_title('Observed <dT> per CCD')
    ax.set_xlabel('x_fp (mm)')
    ax.set_ylabel('y_fp (mm)')
    ax.set_aspect('equal')
    ax.set_xlim(-350, 350)
    ax.set_ylim(-350, 350)

    ax = axes[1, 1]
    col = plot_ccd_polygons(ax, detectors, obs_de1, geometry,
                            'seismic', vmin_de, vmax_de)
    if col:
        plt.colorbar(col, ax=ax, label='de1')
    ax.set_title('Observed <de1> per CCD')
    ax.set_xlabel('x_fp (mm)')
    ax.set_aspect('equal')
    ax.set_xlim(-350, 350)
    ax.set_ylim(-350, 350)

    ax = axes[1, 2]
    col = plot_ccd_polygons(ax, detectors, obs_de2, geometry,
                            'seismic', vmin_de, vmax_de)
    if col:
        plt.colorbar(col, ax=ax, label='de2')
    ax.set_title('Observed <de2> per CCD')
    ax.set_xlabel('x_fp (mm)')
    ax.set_aspect('equal')
    ax.set_xlim(-350, 350)
    ax.set_ylim(-350, 350)

    # Row 3: Predicted - CCD polygons
    ax = axes[2, 0]
    col = plot_ccd_polygons(ax, detectors, pred['dT'], geometry,
                            'seismic', vmin_dT, vmax_dT)
    if col:
        plt.colorbar(col, ax=ax, label='dT (pixel²)')
    ax.set_title('Predicted dT')
    ax.set_xlabel('x_fp (mm)')
    ax.set_ylabel('y_fp (mm)')
    ax.set_aspect('equal')
    ax.set_xlim(-350, 350)
    ax.set_ylim(-350, 350)

    ax = axes[2, 1]
    col = plot_ccd_polygons(ax, detectors, pred['de1'], geometry,
                            'seismic', vmin_de, vmax_de)
    if col:
        plt.colorbar(col, ax=ax, label='de1')
    ax.set_title('Predicted de1')
    ax.set_xlabel('x_fp (mm)')
    ax.set_aspect('equal')
    ax.set_xlim(-350, 350)
    ax.set_ylim(-350, 350)

    ax = axes[2, 2]
    col = plot_ccd_polygons(ax, detectors, pred['de2'], geometry,
                            'seismic', vmin_de, vmax_de)
    if col:
        plt.colorbar(col, ax=ax, label='de2')
    ax.set_title('Predicted de2')
    ax.set_xlabel('x_fp (mm)')
    ax.set_aspect('equal')
    ax.set_xlim(-350, 350)
    ax.set_ylim(-350, 350)

    # Row 4: Residuals - CCD polygons
    ax = axes[3, 0]
    col = plot_ccd_polygons(ax, detectors, res_dT, geometry,
                            'seismic', vmin_dT, vmax_dT)
    if col:
        plt.colorbar(col, ax=ax, label='dT (pixel²)')
    ax.set_title('Residual dT (pred - obs)')
    ax.set_xlabel('x_fp (mm)')
    ax.set_ylabel('y_fp (mm)')
    ax.set_aspect('equal')
    ax.set_xlim(-350, 350)
    ax.set_ylim(-350, 350)

    ax = axes[3, 1]
    col = plot_ccd_polygons(ax, detectors, res_de1, geometry,
                            'seismic', vmin_de, vmax_de)
    if col:
        plt.colorbar(col, ax=ax, label='de1')
    ax.set_title('Residual de1 (pred - obs)')
    ax.set_xlabel('x_fp (mm)')
    ax.set_aspect('equal')
    ax.set_xlim(-350, 350)
    ax.set_ylim(-350, 350)

    ax = axes[3, 2]
    col = plot_ccd_polygons(ax, detectors, res_de2, geometry,
                            'seismic', vmin_de, vmax_de)
    if col:
        plt.colorbar(col, ax=ax, label='de2')
    ax.set_title('Residual de2 (pred - obs)')
    ax.set_xlabel('x_fp (mm)')
    ax.set_aspect('equal')
    ax.set_xlim(-350, 350)
    ax.set_ylim(-350, 350)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close(fig)
    print(f'Saved comparison plot to {output_file}')


def main():
    parser = argparse.ArgumentParser(description="Fit optical parameters from PSF moments")
    parser.add_argument('--input', type=str,
                        default='data/visit_test_2025090600388_g_band.parquet',
                        help='Input parquet file')
    parser.add_argument('--wavelength', type=float, default=480e-9,
                        help='Wavelength in meters (default: 480nm for g-band)')
    parser.add_argument('--output', type=str, default='fit_comparison.png',
                        help='Output plot file')
    parser.add_argument('--no-fit', action='store_true',
                        help='Skip fitting, just compute forward model')
    parser.add_argument('--verbose', action='store_true',
                        help='Print optimization progress')
    parser.add_argument('--fit-params', type=str, nargs='+', default=['cam_dz'],
                        help='Parameters to fit (default: cam_dz). Note: m2_dz is fixed (degenerate with cam_dz)')

    args = parser.parse_args()

    config = FitConfig(wavelength=args.wavelength)

    # Set which parameters to fit based on command line
    for name in config.fit_params:
        config.fit_params[name]['fit'] = name in args.fit_params

    print(f"Loading data from {args.input}...")
    data = load_and_bin_data(args.input, config)
    n_det = len(np.unique(data['detector']))
    print(f"Data: {n_det} detectors (one mean per CCD)")
    print(f"Rotator angle: {np.degrees(data['rotator_angle']):.2f} deg")

    fit_names = [name for name, p in config.fit_params.items() if p['fit']]

    if args.no_fit:
        print("\nComputing forward model with default values...")
        fit_values = np.array([config.fit_params[n]['value'] for n in fit_names])
    else:
        print(f"\nFitting parameters: {fit_names}")
        fit_values, minuit = fit_optics(data, config, verbose=args.verbose)

        print("\nFitted values:")
        for name, val in zip(fit_names, fit_values):
            print(f"  {name}: {val:.4e}")

        if minuit is not None:
            print(f"\nFit converged: {minuit.fmin.is_valid}")

    print("\nComputing predicted moments...")
    pred = forward_model(fit_values, fit_names, data, config, return_full=True)

    print("\nPlotting comparison...")
    plot_comparison(data, pred, args.output)


if __name__ == "__main__":
    main()
