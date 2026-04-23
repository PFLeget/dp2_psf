
#!/usr/bin/env python
"""
Fit optical parameters from star second moments using batoid ray tracing.

Based on fit_batoid.py approach:
- Fits AOS DOFs + atmospheric seeing moments (smxx, smyy, smxy)
- Uses WCS to convert focal plane (mm) to tangent plane angles (degrees)
- Residuals: seeing_moment + batoid_moment - observed_moment

Input: visitMappingFile (same format as FoVPlot_vs_secondMoment.py)
"""

import numpy as np
import pandas as pd
import polars as pl
import pickle
import argparse
import os
from tqdm import tqdm

import lsst.afw.cameraGeom as cameraGeom
from lsst.obs.lsst import LsstCam

import batoid
from batoid_rubin import LSSTBuilder
from scipy.optimize import leastsq

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection


# Camera for coordinate transforms
camera = LsstCam.getCamera()

# Central wavelengths (nm)
CENTRAL_WAVELENGTH = {'u': 360, 'g': 480, 'r': 625, 'i': 760, 'z': 875, 'y': 970}

# AOS DOF indices
AOS_DOF_INDICES = {
    'm2_dz': 0, 'm2_dx': 1, 'm2_dy': 2, 'm2_rx': 3, 'm2_ry': 4,
    'cam_dz': 5, 'cam_dx': 6, 'cam_dy': 7, 'cam_rx': 8, 'cam_ry': 9,
} | {f'm1m3_bend_{i}': 10 + i for i in range(20)} | {f'm2_bend_{i}': 30 + i for i in range(20)}

MICRONS_TO_PIXELS = 0.1
METERS_TO_PIXELS = 1e5
METERS_TO_MM = 1e3

# Columns to read from parquet files (star moments, not PSF model)
PARQUET_COLUMNS = [
    'slot_Shape_xx', 'slot_Shape_yy', 'slot_Shape_xy',
    'slot_Centroid_x', 'slot_Centroid_y',
    'detector', 'calib_psf_reserved',
]


def pixel_to_focal(x, y, det):
    """Convert pixel coordinates to focal plane (mm)."""
    tx = det.getTransform(cameraGeom.PIXELS, cameraGeom.FOCAL_PLANE)
    fpx, fpy = tx.getMapping().applyForward(np.vstack((x, y)))
    return fpx.ravel(), fpy.ravel()


def load_visit_data(parquet_path):
    """Load visit data from parquet file."""
    table = pl.scan_parquet(parquet_path).select(PARQUET_COLUMNS).collect()

    return {
        'mxx': table['slot_Shape_xx'].to_numpy(),
        'myy': table['slot_Shape_yy'].to_numpy(),
        'mxy': table['slot_Shape_xy'].to_numpy(),
        'xCCD': table['slot_Centroid_x'].to_numpy(),
        'yCCD': table['slot_Centroid_y'].to_numpy(),
        'detector': table['detector'].to_numpy(),
        'calib_psf_reserved': table['calib_psf_reserved'].to_numpy(),
    }


def load_single_visit_data(parquet_path, rcut=300):
    """
    Load and average star moments per CCD for a single visit.

    Parameters
    ----------
    parquet_path : str
        Path to parquet file
    rcut : float
        Radial cut in mm

    Returns
    -------
    DataFrame with columns: det, xfp, yfp, mxx, myy, mxy
    """
    data = load_visit_data(parquet_path)
    ccdIds = set(data['detector'])

    rows = []
    for ccd in ccdIds:
        mask = (data['detector'] == ccd) & np.isfinite(data['mxx'])
        if np.sum(mask) < 5:
            continue

        # Get mean position in focal plane
        xCCD = np.mean(data['xCCD'][mask])
        yCCD = np.mean(data['yCCD'][mask])
        fpx, fpy = pixel_to_focal(np.array([xCCD]), np.array([yCCD]), camera[ccd])

        # Apply radial cut
        r = np.sqrt(fpx[0]**2 + fpy[0]**2)
        if r > rcut:
            continue

        # Get mean moments
        rows.append({
            'det': ccd,
            'xfp': fpx[0],
            'yfp': fpy[0],
            'mxx': np.mean(data['mxx'][mask]),
            'myy': np.mean(data['myy'][mask]),
            'mxy': np.mean(data['mxy'][mask]),
        })

    return pd.DataFrame(rows)


def get_telescope(band):
    """Load telescope model for given band."""
    return batoid.Optic.fromYaml(f'Rubin_v3.14_{band}.yaml')


def launch_rays_simple(telescope, band, ax, ay):
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

def launch_rays(telescope, band, ax, ay, atmos, with_wcs=False, output_file=None):
    """
    launch rays for a few spots at angles (degrees) ax ay
    in tangent plane . 
    atmos is a triplet of arrays that sample x,y,value of the atmopheric seeing
    Returns a panda DataFrame.
    moments in pixels^2
    """
    pixel_size = 10 # microns
    # input_angles (degrees)
    
    wavelength = CENTRAL_WAVELENGTH[band]
    def compute_rays(alpha,beta) :
        rays = batoid.RayVector.asPolar(
            optic=telescope,
            inner=telescope.pupilSize/2*telescope.pupilObscuration,
            theta_x=np.deg2rad(alpha), theta_y=np.deg2rad(beta),
            nrad=12, naz=48, wavelength=wavelength*1e-9)
        # the speed depends a lot on nrad and naz.
        # On an example, those were 48 and 192
        telescope.trace(rays)
        return rays

    out_columns = ['theta_x','theta_y']
    spots = None
    if output_file is not None:
        all_rays = pd.DataFrame(columns=['theta_x','theta_y','x','y'])
    if with_wcs : spots['wcs'] = []
    atx,aty,atw = atmos
    for alpha,beta in zip(ax,ay):
        rays = compute_rays(alpha,beta)
        w0 = ~rays.vignetted
        # position in pixelss and moments in pix**2        
        x0, y0 = rays[w0].x.mean()*METERS_TO_PIXELS, rays[w0].y.mean()*METERS_TO_PIXELS
        xx,yy,ww = convolve_with_seeing(rays[w0].x*METERS_TO_PIXELS-x0,
                                        rays[w0].y*METERS_TO_PIXELS-y0,
                                        atx,aty,atw)
        moments_dict = gauss_moments(xx,yy,ww)
        data = [alpha,beta]+list(moments_dict.values())
        if spots is None:
            out_columns += list(moments_dict.keys())
            spots = pd.DataFrame(columns=out_columns)
            if with_wcs : spots['wcs'] = []
        if with_wcs:
            eps=1e-5
            rays_x = compute_rays(alpha+eps, beta)
            rays_y = compute_rays(alpha, beta+eps)
            # use the intersection of vignetting, or the result is biased
            w = w0 & (~rays_x.vignetted)&(~rays_y.vignetted)
            x0 = rays[w].x.mean()*METERS_TO_PIXELS
            y0 = rays[w].y.mean()*METERS_TO_PIXELS
            m2p = METERS_TO_PIXELS
            cxx, cxy = rays_x[w].x.mean()*m2p, rays_x[w].y.mean()*m2p
            cyx, cyy = rays_y[w].x.mean()*m2p,rays_y[w].y.mean()*m2p
            a11,a12 = cxx-x0, cxy - y0
            a21,a22 = cyx-x0, cyy - y0
            wcs = np.linalg.inv(np.array([[a11,a12],[a21,a22]])/eps)
            # wcs /= pixel_size # aij unit is degrees pper pixel 
            data.append(wcs)
        spots.loc[len(spots)] = data
        if output_file is not None:
            for r in rays[w0] :  # for some reason r.x is a 1-item list
                all_rays.loc[len(all_rays)] = [alpha,beta,r.x[0],r.y[0]]
    if output_file is not None: all_rays.to_parquet(output_file)
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

def gauss_moments(x,y, win=None):
    """
    Computes Gauss moments à la HSM, for discrete samples given as lists of x,y,win.
    returns a dictionnary of moments (mxx, myy, mxy), 3rd order moments, and position (x0,Y0)
    """
    mxx = x.var()
    myy = y.var()
    x0 = x.mean()
    y0 = y.mean()
    mxy = ((x-x0)*(y-y0)).mean()
    det = mxx*myy-mxy**2
    wxx = myy/det
    wyy = mxx/det
    wxy = -mxy/det
    for iter in range(60):
        dx = x-x0
        dy = y-y0
        w = np.exp(-0.5*(wxx*dx**2+wyy*dy**2+2*wxy*dx*dy))
        if win is not None : 
            w*= win
        wsum = w.sum()
        ddx = (dx*w).sum()/wsum
        ddy = (dy*w).sum()/wsum
        mxx = (w*dx**2).sum()/wsum
        myy = (w*dy**2).sum()/wsum
        mxy = (w*dx*dy).sum()/wsum
        mxx -= ddx**2
        myy -= ddy**2
        mxy -= ddx*ddy
        mxx *= 2
        myy *= 2
        mxy *= 2
        det = mxx*myy-mxy**2
        nwxx = myy/det
        nwyy = mxx/det
        nwxy = -mxy/det
        if (np.abs(wxx-nwxx)+np.abs(wyy-nwyy)+np.abs(wxy-nwxy))/wxx < 1e-6 :
            break
        x0 += ddx
        y0 += ddy
        wxx = nwxx
        wyy = nwyy
        wxy = nwxy
        #print(iter,x0,y0,wxx,wyy,wxy)
    mx3 = ((x-x0)**3*w).sum()/wsum
    mx2y = ((x-x0)**2*(y-y0)*w).sum()/wsum
    mxy2 = ((x-x0)*(y-y0)**2*w).sum()/wsum
    my3 = ((y-y0)**3*w).sum()/wsum
    return {"mxx":mxx,"myy":myy,"mxy":mxy,"x0":x0,"y0":y0, "mx2y":mx2y,
            "mx3":mx3, "mx2y":mx2y,"mxy2":mxy2, "my3":my3}


def convolve_with_seeing(x,y, xat, yat, w):
    """
    x,y array of coordinates on the focal plane (spot diagram)
    xat, yat, w : spot diagram of the atmosphere, weighted by w
    
    return 2 array of coordinates resulting froim the "convolution" of both inputs
    with len = product of the 2 inputs.
    """
    Xout = x + xat[:, np.newaxis]
    Yout = y + yat[:, np.newaxis]
    Wout = np.ones_like(x)*w[:, np.newaxis]
    return Xout.flatten(),Yout.flatten(), Wout.flatten()

def atmosphere_spot_diagram(mxx,myy,mxy, type = "Gauss"):
    """
    sample the seeing disk (mxx,myy,nmxy) in focal plane
    """
    assert type=="Gauss", "alternatives to Gauss not implemented (yet)"
    x = np.linspace(-3,3,7) # sampling : 1 pixel
    y = x
    det = mxx*myy-mxy**2
    assert det>0, "convolve_with_seeing: non pos-def PSF !"
    scale = np.pow(det,0.25)
    xx,yy = np.meshgrid(x*scale,y*scale)
    xx,yy = xx.flatten(), yy.flatten()
    wxx = myy/det
    wyy = mxx/det
    wxy = -mxy/det
    return xx, yy, np.exp(-0.5*(wxx*xx**2 + wyy*yy**2 + 2*wxy*xx*yy))


class BatoidFitter:
    """Fitter for AOS DOFs using batoid ray tracing."""

    def __init__(self, ref_telescope, param_names, use_gauss_moments= False):
        for param in param_names:
            if param not in AOS_DOF_INDICES:
                raise ValueError(f"{param} is not in AOS param list: {list(AOS_DOF_INDICES.keys())}")

        self.builder = LSSTBuilder(ref_telescope)
        self.param_names = param_names
        self.n_extra_params = 3  # Atmospheric seeing: smxx, smyy, smxy
        self.to_fit = None
        self.use_gauss_moments = use_gauss_moments
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

        if self.use_gauss_moments:
            atmos = atmosphere_spot_diagram(smxx,smyy,smxy)
            spots = launch_rays(offset_tel, self.band_for_fit, ax, ay,
                                atmos, with_wcs=False, output_file=None)
            # the atmosphere is included in the spot moments, set it to zero in the residuals calculation
            smxx = 0
            smyy = 0
            smxy = 0
        else:
            spots = launch_rays_simple(offset_tel, self.band_for_fit, ax, ay)
        
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
        """Fit AOS DOFs to data."""
        self.eval_tg_plane_angles(data, band)

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


def plot_fit_results(data, fit_params, dof_names, output_file='fit_results.png', geometry=None, visit=None, band=None):
    """Plot observed vs fitted moments."""
    if geometry is None:
        geometry = load_ccd_geometry()

    fig, axes = plt.subplots(3, 4, figsize=(18, 14))

    detectors = data['det'].to_numpy() if hasattr(data['det'], 'to_numpy') else data['det'].values
    xfp = data['xfp'].to_numpy() if hasattr(data['xfp'], 'to_numpy') else data['xfp'].values
    yfp = data['yfp'].to_numpy() if hasattr(data['yfp'], 'to_numpy') else data['yfp'].values

    obs_mxx = data['mxx'].to_numpy() if hasattr(data['mxx'], 'to_numpy') else data['mxx'].values
    obs_myy = data['myy'].to_numpy() if hasattr(data['myy'], 'to_numpy') else data['myy'].values
    obs_mxy = data['mxy'].to_numpy() if hasattr(data['mxy'], 'to_numpy') else data['mxy'].values

    fit_mxx = data['fmxx'].to_numpy() if hasattr(data['fmxx'], 'to_numpy') else data['fmxx'].values
    fit_myy = data['fmyy'].to_numpy() if hasattr(data['fmyy'], 'to_numpy') else data['fmyy'].values
    fit_mxy = data['fmxy'].to_numpy() if hasattr(data['fmxy'], 'to_numpy') else data['fmxy'].values

    obs_T = obs_mxx + obs_myy
    obs_e1 = (obs_mxx - obs_myy) / obs_T
    obs_e2 = 2 * obs_mxy / obs_T

    fit_T = fit_mxx + fit_myy
    fit_e1 = (fit_mxx - fit_myy) / fit_T
    fit_e2 = 2 * fit_mxy / fit_T

    obs_dT = obs_T - np.nanmean(obs_T)
    obs_de1 = obs_e1 - np.nanmean(obs_e1)
    obs_de2 = obs_e2 - np.nanmean(obs_e2)

    fit_dT = fit_T - np.nanmean(fit_T)
    fit_de1 = fit_e1 - np.nanmean(fit_e1)
    fit_de2 = fit_e2 - np.nanmean(fit_e2)

    res_dT = obs_dT - fit_dT
    res_de1 = obs_de1 - fit_de1
    res_de2 = obs_de2 - fit_de2

    vmin_dT, vmax_dT = -0.5, 0.5
    vmin_de, vmax_de = -0.15, 0.15
    lim = 350

    plot_data = [
        (0, 0, obs_dT, 'Observed dT', vmin_dT, vmax_dT, 'dT (pixel$^2$)'),
        (0, 1, obs_de1, 'Observed de1', vmin_de, vmax_de, 'de1'),
        (0, 2, obs_de2, 'Observed de2', vmin_de, vmax_de, 'de2'),
        (1, 0, fit_dT, 'Fitted dT (batoid)', vmin_dT, vmax_dT, 'dT (pixel$^2$)'),
        (1, 1, fit_de1, 'Fitted de1 (batoid)', vmin_de, vmax_de, 'de1'),
        (1, 2, fit_de2, 'Fitted de2 (batoid)', vmin_de, vmax_de, 'de2'),
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

    ax = axes[2, 3]
    ax.axis('off')

    stats_text = "Fitted parameters:\n" + "-" * 30 + "\n"
    for name in dof_names:
        if name in fit_params:
            stats_text += f"{name}: {fit_params[name]:.4f}\n"
    stats_text += "-" * 30 + "\n"
    stats_text += f"smxx: {fit_params.get('smxx', 0):.4f}\n"
    stats_text += f"smyy: {fit_params.get('smyy', 0):.4f}\n"
    stats_text += f"smxy: {fit_params.get('smxy', 0):.4f}\n"
    stats_text += "-" * 30 + "\n"
    stats_text += f"Correlations:\n  dT: {rho_T:.3f}\n  de1: {rho_e1:.3f}\n  de2: {rho_e2:.3f}\n"
    stats_text += "-" * 30 + "\n"
    stats_text += f"RMS residuals:\n  dT: {np.std(res_dT):.4f}\n  de1: {np.std(res_de1):.4f}\n  de2: {np.std(res_de2):.4f}\n"

    ax.text(0.1, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace')

    title = 'Batoid Optical Fit Results'
    if visit is not None:
        title += f' - Visit {visit}'
    if band is not None:
        title += f' ({band}-band)'
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close(fig)
    print(f'Saved {output_file}')


def main():
    parser = argparse.ArgumentParser(description="Fit optical parameters from star moments for a single visit")
    parser.add_argument('--visitMappingFile', type=str, required=True,
                        help='Path to visit_parquet_mapping.pkl')
    parser.add_argument('--visitID', type=int, required=True,
                        help='Visit ID to process')
    parser.add_argument('--repOut', type=str, default='./',
                        help='Output directory for all files')
    parser.add_argument('--params', type=str, nargs='+',
                        default=['m2_dz', 'm2_rx', 'm2_ry', 'cam_dz', 'cam_rx', 'cam_ry'],
                        help='DOF parameters to fit')
    parser.add_argument('--start', type=str, default=None,
                        help='Pickle file with starting values')
    parser.add_argument('--rcut', type=float, default=300,
                        help='Radial cut for vignetted data (mm)')
    parser.add_argument('--gm', action = "store_true", #default is false
                        dest="use_gauss_moments",
                        help='use HSM gauss moments ')

    args = parser.parse_args()

    os.makedirs(args.repOut, exist_ok=True)

    # Load visit mapping
    with open(args.visitMappingFile, 'rb') as f:
        visit_mapping = pickle.load(f)

    if args.visitID not in visit_mapping:
        raise ValueError(f"Visit {args.visitID} not found in mapping file")

    info = visit_mapping[args.visitID]
    band = info['band']
    parquet_path = info['parquet_path']

    print(f"Processing visit {args.visitID}, band={band}")
    print(f"  Input: {parquet_path}")

    # Load telescope for this band
    telescope = get_telescope(band)

    # Load starting values if provided
    start_values = None
    if args.start is not None:
        with open(args.start, 'rb') as f:
            start_values = pickle.load(f)
        print(f"Using {args.start} as starting point")

    # Load data
    data = load_single_visit_data(parquet_path, args.rcut)
    print(f"  Loaded {len(data)} CCDs")

    # Fit
    fitter = BatoidFitter(telescope, args.params, args.use_gauss_moments)
    params, cov_params = fitter.fit(data, band, seeing=[1.3, 1.3, 0],
                                     start=start_values, verbose=True)

    # Extract results
    result = {'visit': args.visitID, 'band': band}
    for key, val in zip(args.params, params[:-3]):
        result[key] = val
    result['smxx'] = params[-3]
    result['smyy'] = params[-2]
    result['smxy'] = params[-1]

    # Compute residuals
    residuals = fitter.residuals(params)
    result['chi2'] = (residuals**2).sum()
    result['n_ccd'] = len(data)

    print("\nFitted parameters:")
    for k, v in result.items():
        print(f"  {k}: {v}")

    # Save data with fitted moments
    data['fmxx'] = residuals[0] + data['mxx']
    data['fmyy'] = residuals[1] + data['myy'] 
    data['fmxy'] = residuals[2] + data['mxy']
    data.to_parquet(os.path.join(args.repOut, f'iq_fit_visit{args.visitID}.parquet'))
    print(f"Saved iq_fit_visit{args.visitID}.parquet")

    # Save fit params
    result_df = pd.DataFrame([result])
    result_df.to_parquet(os.path.join(args.repOut, f'fit_params_visit{args.visitID}.parquet'))
    print(f"Saved fit_params_visit{args.visitID}.parquet")

    # Save plot
    geometry = load_ccd_geometry()
    plot_fit_results(data, result, args.params,
                     os.path.join(args.repOut, f'fit_results_visit{args.visitID}.png'),
                     geometry=geometry, visit=args.visitID, band=band)


if __name__ == "__main__":
    main()
