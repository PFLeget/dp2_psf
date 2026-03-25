#!/usr/bin/env python
"""
Fit optical parameters from PSF second moments using batoid ray tracing.

Based on fit_batoid.py approach:
- Fits AOS DOFs + atmospheric seeing moments (smxx, smyy, smxy)
- Uses WCS to convert focal plane (mm) to tangent plane angles (degrees)
- Residuals: seeing_moment + batoid_moment - observed_moment
"""

import numpy as np
import pandas as pd
import polars as pl
import pickle
import argparse
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

import batoid
from batoid_rubin import LSSTBuilder
from scipy.optimize import leastsq

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection


# Central wavelengths (nm)
CENTRAL_WAVELENGTH = {'u': 360, 'g': 480, 'r': 625, 'i': 760, 'z': 875, 'y': 970}

# AOS DOF indices
AOS_DOF_INDICES = {
    'm2_dz': 0, 'm2_dx': 1, 'm2_dy': 2, 'm2_rx': 3, 'm2_ry': 4,
    'cam_dz': 5, 'cam_dx': 6, 'cam_dy': 7, 'cam_rx': 8, 'cam_ry': 9,
} | {f'm1m3_bend_{i}': 10 + i for i in range(20)} | {f'm2_bend_{i}': 30 + i for i in range(20)}

MICRONS_TO_PIXELS = 0.1
METERS_TO_MM = 1e3


def get_telescope(band):
    """Load telescope model for given band."""
    return batoid.Optic.fromYaml(f'Rubin_v3.14_{band}.yaml')


def launch_rays(telescope, band, ax, ay):
    """
    Launch rays for spots at angles (degrees) ax, ay in tangent plane.
    Returns DataFrame with moments in pixels^2.
    """
    wavelength = CENTRAL_WAVELENGTH[band]

    def compute_rays(alpha, beta):
        rays = batoid.RayVector.asPolar(
            optic=telescope,
            inner=telescope.pupilSize / 2 * telescope.pupilObscuration,
            theta_x=np.deg2rad(alpha),
            theta_y=np.deg2rad(beta),
            nrad=24, naz=96,
            wavelength=wavelength * 1e-9
        )
        telescope.trace(rays)
        return rays

    spots = pd.DataFrame(columns=['theta_x', 'theta_y', 'x0', 'y0', 'mxx', 'myy', 'mxy'])

    for alpha, beta in zip(ax, ay):
        rays = compute_rays(alpha, beta)
        w0 = ~rays.vignetted
        if np.sum(w0) < 10:
            spots.loc[len(spots)] = [alpha, beta, np.nan, np.nan, np.nan, np.nan, np.nan]
            continue

        # Position in microns
        x0 = rays[w0].x.mean() * 1e6
        y0 = rays[w0].y.mean() * 1e6
        # Moments in microns^2, then convert to pixels^2
        mxx = rays[w0].x.var() * 1e12 * MICRONS_TO_PIXELS**2
        myy = rays[w0].y.var() * 1e12 * MICRONS_TO_PIXELS**2
        mxy = ((rays[w0].x * 1e6 - x0) * (rays[w0].y * 1e6 - y0)).mean() * MICRONS_TO_PIXELS**2

        spots.loc[len(spots)] = [alpha, beta, x0, y0, mxx, myy, mxy]

    return spots


class WCS:
    """Mapping from focal plane (mm) to angles in tangent plane (degrees)."""

    def __init__(self, telescope, band):
        wavelength = CENTRAL_WAVELENGTH[band]

        def compute_rays(alpha, beta):
            rays = batoid.RayVector.asPolar(
                optic=telescope,
                inner=telescope.pupilSize / 2 * telescope.pupilObscuration,
                theta_x=np.deg2rad(alpha),
                theta_y=np.deg2rad(beta),
                nrad=2, naz=8,
                wavelength=wavelength * 1e-9
            )
            telescope.trace(rays)
            w0 = ~rays.vignetted
            x0 = rays[w0].x.mean() * METERS_TO_MM
            y0 = rays[w0].y.mean() * METERS_TO_MM
            return x0, y0

        # Build polynomial mapping
        theta_x = np.linspace(-0.9, 0.9, 5)
        theta_y = theta_x
        aa, bb = np.meshgrid(theta_x, theta_y)
        aa, bb = aa.flatten(), bb.flatten()

        X, Y = [], []
        for a, b in zip(aa, bb):
            x0, y0 = compute_rays(a, b)
            X.append(x0)
            Y.append(y0)

        X = np.array(X).squeeze()
        Y = np.array(Y).squeeze()
        A = np.array([X * 0 + 1, X, Y, X**2, Y**2, X * Y]).T
        B = np.array([aa, bb]).T
        coeff, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
        self.coeff = coeff

    def __call__(self, xfp, yfp):
        """Convert xfp, yfp (mm) to tangent plane angles (degrees)."""
        A = np.array([xfp * 0 + 1, xfp, yfp, xfp**2, yfp**2, xfp * yfp])
        return self.coeff.T.dot(A)


class BatoidFitter:
    """Fitter for AOS DOFs using batoid ray tracing."""

    def __init__(self, ref_telescope, param_names):
        for param in param_names:
            if param not in AOS_DOF_INDICES:
                raise ValueError(f"{param} is not in AOS param list: {list(AOS_DOF_INDICES.keys())}")

        self.builder = LSSTBuilder(ref_telescope)
        self.param_names = param_names
        self.n_extra_params = 3  # Atmospheric seeing: smxx, smyy, smxy
        self.to_fit = None
        self.band_for_fit = None

    def move_parts(self, what, shifts):
        """Build telescope with given AOS DOF shifts."""
        aos_shifts = np.zeros(50)
        for param, how_much in zip(what, shifts):
            aos_shifts[AOS_DOF_INDICES[param]] = how_much
        return self.builder.with_aos_dof(aos_shifts).build()

    def eval_tg_plane_angles(self, data, band):
        """Add tangent plane angles to data."""
        self.to_fit = data.copy()
        self.band_for_fit = band
        wcs = WCS(self.builder.fiducial, band)
        a, b = wcs(self.to_fit.xfp.to_numpy(), self.to_fit.yfp.to_numpy())
        self.to_fit['ax'] = a
        self.to_fit['ay'] = b

    def residuals(self, parameters):
        """Compute residuals: seeing + batoid - observed."""
        offset_tel = self.move_parts(self.param_names, parameters[:-self.n_extra_params])
        smxx, smyy, smxy = parameters[-3:]

        ax = self.to_fit.ax.to_numpy()
        ay = self.to_fit.ay.to_numpy()
        spots = launch_rays(offset_tel, self.band_for_fit, ax, ay)

        momres = np.array([
            smxx + spots.mxx - self.to_fit.mxx,
            smyy + spots.myy - self.to_fit.myy,
            smxy + spots.mxy - self.to_fit.mxy
        ])
        return momres

    def chi2_func(self, parameters):
        """Chi2 function for optimizer."""
        out = self.residuals(parameters)
        chi2 = (out**2).sum()
        print(f'chi2={chi2:.2f}, params={parameters}')
        return out.flatten()

    def fit(self, data, band, seeing=None, start=None, verbose=True):
        """
        Fit AOS DOFs to data.

        Parameters
        ----------
        data : DataFrame
            Must contain columns: xfp, yfp, mxx, myy, mxy
        band : str
            Filter band
        seeing : tuple, optional
            Initial guess for (smxx, smyy, smxy)
        start : dict, optional
            Initial values for parameters

        Returns
        -------
        params : array
            Fitted parameters
        cov : array
            Covariance matrix
        """
        self.eval_tg_plane_angles(data, band)

        # Starting point
        starting_point = np.zeros(len(self.param_names) + self.n_extra_params)
        if seeing is not None:
            starting_point[-3:] = np.array(seeing)

        if start is not None:
            for name, val in start.items():
                if name in self.param_names:
                    i = self.param_names.index(name)
                    starting_point[i] = val
                elif name in ['smxx', 'smyy', 'smxy']:
                    i = ['smxx', 'smyy', 'smxy'].index(name)
                    starting_point[len(self.param_names) + i] = val

        # Fit
        if verbose:
            print(f"Fitting {len(self.param_names)} DOFs: {self.param_names}")
            print(f"Starting point: {starting_point}")

        fitted_params, cov_params, _, mesg, ierr = leastsq(
            self.chi2_func, starting_point, full_output=True
        )

        if ierr not in [1, 2, 3, 4]:
            print(f'Warning: leastsq ierr={ierr}, message: {mesg}')

        return fitted_params, cov_params


def load_ccd_geometry(geometry_file=None):
    """Load CCD geometry from CSV file."""
    if geometry_file is None:
        geometry_file = os.path.join(os.path.dirname(__file__), 'data', 'ccd_geometry.csv')

    if not os.path.exists(geometry_file):
        return None

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
    """Plot CCDs as polygons colored by values."""
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


def plot_fit_results(data, fit_params, dof_names, output_file='fit_results.png', geometry=None):
    """
    Plot observed vs fitted moments.

    Parameters
    ----------
    data : DataFrame
        Must contain: det, xfp, yfp, mxx, myy, mxy, fmxx, fmyy, fmxy
    fit_params : dict
        Fitted parameters including smxx, smyy, smxy
    dof_names : list
        Names of fitted DOFs
    """
    if geometry is None:
        geometry = load_ccd_geometry()

    fig, axes = plt.subplots(3, 4, figsize=(18, 14))

    # Extract data
    detectors = data['det'].to_numpy() if hasattr(data['det'], 'to_numpy') else data['det'].values
    xfp = data['xfp'].to_numpy() if hasattr(data['xfp'], 'to_numpy') else data['xfp'].values
    yfp = data['yfp'].to_numpy() if hasattr(data['yfp'], 'to_numpy') else data['yfp'].values

    # Observed moments (mean-subtracted for spatial pattern)
    obs_mxx = data['mxx'].to_numpy() if hasattr(data['mxx'], 'to_numpy') else data['mxx'].values
    obs_myy = data['myy'].to_numpy() if hasattr(data['myy'], 'to_numpy') else data['myy'].values
    obs_mxy = data['mxy'].to_numpy() if hasattr(data['mxy'], 'to_numpy') else data['mxy'].values

    # Fitted moments
    fit_mxx = data['fmxx'].to_numpy() if hasattr(data['fmxx'], 'to_numpy') else data['fmxx'].values
    fit_myy = data['fmyy'].to_numpy() if hasattr(data['fmyy'], 'to_numpy') else data['fmyy'].values
    fit_mxy = data['fmxy'].to_numpy() if hasattr(data['fmxy'], 'to_numpy') else data['fmxy'].values

    # Residuals
    res_mxx = obs_mxx - fit_mxx
    res_myy = obs_myy - fit_myy
    res_mxy = obs_mxy - fit_mxy

    # Compute T, e1, e2
    obs_T = obs_mxx + obs_myy
    obs_e1 = (obs_mxx - obs_myy) / obs_T
    obs_e2 = 2 * obs_mxy / obs_T

    fit_T = fit_mxx + fit_myy
    fit_e1 = (fit_mxx - fit_myy) / fit_T
    fit_e2 = 2 * fit_mxy / fit_T

    # Mean-subtract for spatial patterns
    obs_dT = obs_T - np.nanmean(obs_T)
    obs_de1 = obs_e1 - np.nanmean(obs_e1)
    obs_de2 = obs_e2 - np.nanmean(obs_e2)

    fit_dT = fit_T - np.nanmean(fit_T)
    fit_de1 = fit_e1 - np.nanmean(fit_e1)
    fit_de2 = fit_e2 - np.nanmean(fit_e2)

    res_dT = obs_dT - fit_dT
    res_de1 = obs_de1 - fit_de1
    res_de2 = obs_de2 - fit_de2

    # Color scales
    vmin_dT, vmax_dT = -0.5, 0.5
    vmin_de, vmax_de = -0.15, 0.15
    lim = 350

    # Plot settings
    plot_data = [
        # Row 0: Observed
        (0, 0, obs_dT, 'Observed dT', vmin_dT, vmax_dT, 'dT (pixel$^2$)'),
        (0, 1, obs_de1, 'Observed de1', vmin_de, vmax_de, 'de1'),
        (0, 2, obs_de2, 'Observed de2', vmin_de, vmax_de, 'de2'),
        # Row 1: Fitted (batoid)
        (1, 0, fit_dT, 'Fitted dT (batoid)', vmin_dT, vmax_dT, 'dT (pixel$^2$)'),
        (1, 1, fit_de1, 'Fitted de1 (batoid)', vmin_de, vmax_de, 'de1'),
        (1, 2, fit_de2, 'Fitted de2 (batoid)', vmin_de, vmax_de, 'de2'),
        # Row 2: Residuals
        (2, 0, res_dT, 'Residual dT', vmin_dT, vmax_dT, 'dT (pixel$^2$)'),
        (2, 1, res_de1, 'Residual de1', vmin_de, vmax_de, 'de1'),
        (2, 2, res_de2, 'Residual de2', vmin_de, vmax_de, 'de2'),
    ]

    for row, col, values, title, vmin, vmax, label in plot_data:
        ax = axes[row, col]
        if geometry is not None:
            coll = plot_ccd_polygons(ax, detectors, values, geometry, 'seismic', vmin, vmax)
            if coll:
                plt.colorbar(coll, ax=ax, label=label)
        else:
            sc = ax.scatter(xfp, yfp, c=values, s=40, cmap='seismic', vmin=vmin, vmax=vmax)
            plt.colorbar(sc, ax=ax, label=label)

        ax.set_title(title)
        ax.set_xlabel('x_fp (mm)')
        ax.set_ylabel('y_fp (mm)')
        ax.set_aspect('equal')
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)

    # Column 4: Scatter plots and stats
    # Scatter: observed vs fitted
    ax = axes[0, 3]
    ax.scatter(obs_dT, fit_dT, s=20, alpha=0.7)
    ax.plot([-0.5, 0.5], [-0.5, 0.5], 'k--', alpha=0.5)
    ax.set_xlabel('Observed dT')
    ax.set_ylabel('Fitted dT')
    rho_T = np.corrcoef(obs_dT, fit_dT)[0, 1]
    ax.set_title(f'dT: corr={rho_T:.3f}')
    ax.set_aspect('equal')

    ax = axes[1, 3]
    ax.scatter(obs_de1, fit_de1, s=20, alpha=0.7, label='de1')
    ax.scatter(obs_de2, fit_de2, s=20, alpha=0.7, label='de2')
    ax.plot([-0.15, 0.15], [-0.15, 0.15], 'k--', alpha=0.5)
    ax.set_xlabel('Observed')
    ax.set_ylabel('Fitted')
    rho_e1 = np.corrcoef(obs_de1, fit_de1)[0, 1]
    rho_e2 = np.corrcoef(obs_de2, fit_de2)[0, 1]
    ax.set_title(f'de1: corr={rho_e1:.3f}, de2: corr={rho_e2:.3f}')
    ax.legend()
    ax.set_aspect('equal')

    # Stats text
    ax = axes[2, 3]
    ax.axis('off')

    stats_text = "Fitted parameters:\n"
    stats_text += "-" * 30 + "\n"
    for name in dof_names:
        if name in fit_params:
            stats_text += f"{name}: {fit_params[name]:.4f}\n"
    stats_text += "-" * 30 + "\n"
    stats_text += f"smxx: {fit_params.get('smxx', 0):.4f}\n"
    stats_text += f"smyy: {fit_params.get('smyy', 0):.4f}\n"
    stats_text += f"smxy: {fit_params.get('smxy', 0):.4f}\n"
    stats_text += "-" * 30 + "\n"
    stats_text += f"Correlations:\n"
    stats_text += f"  dT:  {rho_T:.3f}\n"
    stats_text += f"  de1: {rho_e1:.3f}\n"
    stats_text += f"  de2: {rho_e2:.3f}\n"
    stats_text += "-" * 30 + "\n"
    stats_text += f"RMS residuals:\n"
    stats_text += f"  dT:  {np.std(res_dT):.4f}\n"
    stats_text += f"  de1: {np.std(res_de1):.4f}\n"
    stats_text += f"  de2: {np.std(res_de2):.4f}\n"

    ax.text(0.1, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace')

    plt.suptitle('Batoid Optical Fit Results', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close(fig)
    print(f'Saved {output_file}')


def main():
    parser = argparse.ArgumentParser(description="Fit optical parameters from PSF moments")
    parser.add_argument('--input', type=str, default='iq_dat.parquet',
                        help='Input parquet file with mxx, myy, mxy, xfp, yfp columns')
    parser.add_argument('--band', type=str, default='r',
                        help='Filter band (default: r)')
    parser.add_argument('--output', type=str, default='fit_results.png',
                        help='Output plot file')
    parser.add_argument('--tag', type=str, default='',
                        help='Tag for output files')
    parser.add_argument('--params', type=str, nargs='+',
                        default=['m2_dz', 'm2_rx', 'm2_ry', 'cam_dz', 'cam_rx', 'cam_ry'],
                        help='DOF parameters to fit')
    parser.add_argument('--start', type=str, default=None,
                        help='Pickle file with starting values')
    parser.add_argument('--rcut', type=float, default=300,
                        help='Radial cut for vignetted data (mm)')
    parser.add_argument('--no-fit', action='store_true',
                        help='Skip fitting, just plot existing results')
    parser.add_argument('--fit-file', type=str, default=None,
                        help='Use existing fit parquet file for plotting')

    args = parser.parse_args()

    # If just plotting existing results
    if args.fit_file is not None:
        print(f"Loading existing fit from {args.fit_file}")
        data = pd.read_parquet(args.fit_file)

        # Try to load fit params
        params_file = f'fit_params{args.tag}.pkl'
        if os.path.exists(params_file):
            with open(params_file, 'rb') as f:
                fit_params = pickle.load(f)
        else:
            fit_params = {}

        plot_fit_results(data, fit_params, args.params, args.output)
        return

    # Load data
    print(f"Loading data from {args.input}")
    data = pd.read_parquet(args.input)

    # Apply radial cut
    r = np.sqrt(data.xfp**2 + data.yfp**2)
    data = data[r < args.rcut].reset_index(drop=True)
    print(f"After r < {args.rcut} mm cut: {len(data)} points")

    if args.no_fit:
        print("Skipping fit (--no-fit specified)")
        return

    # Load telescope
    print(f"Loading telescope for band {args.band}")
    telescope = get_telescope(args.band)

    # Load starting values if provided
    start_values = None
    if args.start is not None:
        with open(args.start, 'rb') as f:
            start_values = pickle.load(f)
        print(f"Using {args.start} as starting point")

    # Fit
    print(f"Fitting DOFs: {args.params}")
    fitter = BatoidFitter(telescope, args.params)
    params, cov_params = fitter.fit(data, args.band, seeing=[1.3, 1.3, 0], start=start_values)

    # Extract results
    result = {key: val for key, val in zip(args.params, params[:-3])}
    result['smxx'] = params[-3]
    result['smyy'] = params[-2]
    result['smxy'] = params[-1]

    print("\nFitted parameters:")
    for k, v in result.items():
        print(f"  {k}: {v:.4f}")

    # Save results
    tag = args.tag
    with open(f'fit_params{tag}.pkl', 'wb') as f:
        pickle.dump(result, f)
    print(f"Saved fit_params{tag}.pkl")

    with open(f'cov_params{tag}.pkl', 'wb') as f:
        pickle.dump(cov_params, f)
    print(f"Saved cov_params{tag}.pkl")

    # Compute fitted moments
    residuals = fitter.residuals(params)
    data['fmxx'] = data['mxx'] - residuals[0]
    data['fmyy'] = data['myy'] - residuals[1]
    data['fmxy'] = data['mxy'] - residuals[2]

    # Save fit data
    data.to_parquet(f'iq_fit{tag}.parquet')
    print(f"Saved iq_fit{tag}.parquet")

    # Plot
    plot_fit_results(data, result, args.params, args.output)


if __name__ == "__main__":
    main()
