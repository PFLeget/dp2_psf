from lsst.daf.butler import Butler
import numpy as np
from astropy.table import Table
import astropy.units as units
import treegp
print(treegp.__version__)
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import healpy as hp

import pickle
import pandas as pd
import argparse
import glob
import os


def galactic_to_equatorial(l_deg, b_deg):
    """
    Convert galactic coordinates to equatorial (RA, Dec).

    Parameters
    ----------
    l_deg : array
        Galactic longitude in degrees
    b_deg : array
        Galactic latitude in degrees

    Returns
    -------
    ra_deg, dec_deg : arrays
        Equatorial coordinates in degrees
    """
    # Galactic pole in equatorial coordinates
    ra_gp = np.radians(192.85948)  # RA of galactic north pole
    dec_gp = np.radians(27.12825)  # Dec of galactic north pole
    l_ncp = np.radians(122.93192)  # Galactic longitude of north celestial pole

    l_rad = np.radians(l_deg)
    b_rad = np.radians(b_deg)

    dec_eq = np.arcsin(np.sin(dec_gp) * np.sin(b_rad) +
                       np.cos(dec_gp) * np.cos(b_rad) * np.cos(l_rad - l_ncp))
    ra_eq = ra_gp + np.arctan2(
        np.cos(b_rad) * np.sin(l_rad - l_ncp),
        np.cos(dec_gp) * np.sin(b_rad) - np.sin(dec_gp) * np.cos(b_rad) * np.cos(l_rad - l_ncp)
    )

    return np.degrees(ra_eq) % 360, np.degrees(dec_eq)


def plot_milky_way(ax, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='Milky Way'):
    """
    Plot the Milky Way plane (galactic equator) on a healpy Mollweide projection.

    Parameters
    ----------
    ax : matplotlib axis
        Healpy Mollweide projection axis
    """
    # Galactic equator: b = 0
    gal_lon = np.linspace(0, 360, 361)
    gal_lat = np.zeros_like(gal_lon)

    ra_eq, dec_eq = galactic_to_equatorial(gal_lon, gal_lat)

    # Convert to theta, phi for healpy (colatitude, longitude in radians)
    theta = np.radians(90 - dec_eq)  # colatitude
    phi = np.radians(ra_eq)          # longitude

    # Sort by phi for continuous line
    order = np.argsort(phi)
    theta_sorted = theta[order]
    phi_sorted = phi[order]

    # Split at discontinuities to avoid lines across the plot
    diff = np.abs(np.diff(phi_sorted))
    breaks = np.where(diff > np.pi)[0]
    segments = np.split(np.arange(len(phi_sorted)), breaks + 1)

    for i, seg in enumerate(segments):
        if len(seg) > 1:
            hp.projplot(theta_sorted[seg], phi_sorted[seg], color=color, linestyle=linestyle,
                        linewidth=linewidth, alpha=alpha, label=label if i == 0 else '')


def plot_des_footprint(ax, color='cyan', linestyle='-', linewidth=2, alpha=0.8, label='DES'):
    """
    Plot the approximate DES footprint on a healpy Mollweide projection.

    The DES footprint covers ~5000 sq deg in the southern sky.
    This is an approximate boundary.

    Parameters
    ----------
    ax : matplotlib axis
        Healpy Mollweide projection axis
    """
    # DES approximate footprint boundaries
    # DES wide survey approximate corners
    des_ra = np.array([0, 90, 90, 60, 60, 0, 0, 300, 300, 0])
    des_dec = np.array([-65, -65, -40, -40, 5, 5, -40, -40, -65, -65])

    # Convert to theta, phi for healpy (colatitude, longitude in radians)
    theta = np.radians(90 - des_dec)  # colatitude
    phi = np.radians(des_ra)          # longitude

    # Plot each segment separately to handle wrap-around
    first_label = True
    for i in range(len(phi) - 1):
        phi_diff = np.abs(phi[i+1] - phi[i])
        if phi_diff < np.pi:  # Only plot if not wrapping around
            hp.projplot([theta[i], theta[i+1]], [phi[i], phi[i+1]],
                        color=color, linestyle=linestyle, linewidth=linewidth, alpha=alpha,
                        label=label if first_label else '')
            first_label = False


def plot_Sky_second_Moment(bands='g', rep="data/", repOutPlot='plots/',
                           key_second_moment='dT_T', bin_spacing=120, colorScale=0.005,
                           autoColorScale=False, autoColorScaleCst=2.,
                           colorlabel=None, title=None, pklInput=None, psf_max_value=0):
    """
    Plot spatial variation of PSF second moments on the sky using HEALPix binning.

    Parameters
    ----------
    bands : str
        Band(s) to process (e.g., 'g', 'ugrizy')
    rep : str
        Path to directory containing PSF pickle files
    repOutPlot : str
        Output directory for plots
    key_second_moment : str
        Second moment key to plot (e.g., 'dT_T', 'de1', 'de2')
    bin_spacing : float
        HEALPix bin spacing in arcsec
    colorScale : float
        Color scale range [-colorScale, +colorScale]
    autoColorScale : bool
        If True, compute color scale from data
    autoColorScaleCst : float
        Number of sigma for auto color scale
    colorlabel : str
        Label for colorbar
    title : str
        Plot title
    pklInput : str
        Path to pre-computed pickle file (to redo plot only)
    psf_max_value : float
        Exclude PSFs with max pixel value below this threshold
    """

    CMAP = plt.cm.inferno

    if pklInput is None:
        pkls = []
        for b in bands:
            pkls.append(glob.glob(os.path.join(rep, b + '/*.pkl')))
        pkls = np.concatenate(pkls)

        # Use meanify_healpix for sky coordinates
        meanifyHealpix = treegp.meanify_healpix(bin_spacing=bin_spacing)

        for pkl in tqdm(pkls, desc="Loop over visits to compute spatial average on sky:"):
            dic = pd.read_pickle(pkl)
            visit = list(dic.keys())[0]

            # Filter by psf_max_value if specified
            filtering = np.ones(len(dic[visit]["ra"]), dtype=bool)
            if psf_max_value > 0:
                filtering &= (dic[visit]["psf_max_value"] > psf_max_value)

            # Sky coordinates (RA, Dec) - convert from radians to degrees
            coord = np.array([np.degrees(dic[visit]['ra']), np.degrees(dic[visit]['dec'])]).T

            meanifyHealpix.add_field(coord[filtering], dic[visit][key_second_moment][filtering])

        meanifyHealpix.meanify()

        # Store results for saving
        coords0 = meanifyHealpix.coords0  # (RA, Dec)
        params0 = meanifyHealpix.params0
        wrms0 = meanifyHealpix.wrms0
        nside = meanifyHealpix.nside
        pixel_size_arcsec = meanifyHealpix.pixel_size_arcsec
        valid_pixels = meanifyHealpix._valid_pixels  # HEALPix pixel indices

    else:
        dicInput = pd.read_pickle(pklInput)
        coords0 = dicInput['coords0']
        params0 = dicInput['params0']
        wrms0 = dicInput['wrms0']
        nside = dicInput['nside']
        pixel_size_arcsec = dicInput['pixel_size_arcsec']
        valid_pixels = dicInput.get('valid_pixels', None)

    # Compute color scale
    if autoColorScale:
        valid = np.isfinite(params0)
        MEAN = np.median(params0[valid])
        STD = np.std(params0[valid])
        MIN = MEAN - autoColorScaleCst * STD
        MAX = MEAN + autoColorScaleCst * STD
    else:
        MIN = -colorScale
        MAX = colorScale

    # Create full HEALPix map with UNSEEN for empty pixels
    npix = hp.nside2npix(nside)
    healpix_map = np.full(npix, hp.UNSEEN)

    if valid_pixels is not None:
        healpix_map[valid_pixels] = params0
    else:
        # Fallback: convert RA/Dec to pixel indices
        import hpgeom as hpg
        pixels = hpg.angle_to_pixel(nside, coords0[:, 0], coords0[:, 1], nest=True, degrees=True)
        healpix_map[pixels] = params0

    # Set colorbar label
    if colorlabel is None:
        colorlabel = key_second_moment

    if title is None:
        title = f"DP2 {key_second_moment} | bands: ({bands}) | nside={nside} (~{pixel_size_arcsec:.0f} arcsec)"

    # Use healpy mollview for proper HEALPix rendering (no gaps)
    fig = plt.figure(figsize=(16, 10))
    hp.mollview(healpix_map, nest=True, cmap=CMAP, min=MIN, max=MAX,
                title=title, unit=colorlabel, fig=fig.number,
                cbar=True, hold=True)

    # Get the current axes for adding overlays
    ax = plt.gca()

    # Add Milky Way plane
    plot_milky_way(ax, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='Milky Way')

    # Add DES footprint
    plot_des_footprint(ax, color='cyan', linestyle='-', linewidth=2, alpha=0.8, label='DES')

    # Add legend
    ax.legend(loc='lower right', fontsize=10)

    plt.savefig(os.path.join(repOutPlot, f'{key_second_moment}_sky_{bands}_{int(bin_spacing)}_{int(psf_max_value)}.png'), dpi=150)
    plt.close()

    # Save results to pickle
    if pklInput is None:
        dicOutput = {
            'coords0': coords0,
            'params0': params0,
            'wrms0': wrms0,
            'nside': nside,
            'pixel_size_arcsec': pixel_size_arcsec,
            'valid_pixels': valid_pixels,
            'bands': bands,
            'key_second_moment': key_second_moment,
            'bin_spacing': bin_spacing,
        }
        pklFile = open(os.path.join(repOutPlot, f'{key_second_moment}_sky_{bands}_{int(bin_spacing)}_{int(psf_max_value)}.pkl'), 'wb')
        pickle.dump(dicOutput, pklFile)
        pklFile.close()


def main():

    parser = argparse.ArgumentParser(description="Sky map of PSF second moment residuals")
    parser.add_argument('--bands', type=str, required=True, help="The band(s) to process (e.g., y, g, r, i, z, u, ugrizy)")
    parser.add_argument('--pathPSFRep', type=str, required=True, help="Path to PSF File directory")

    parser.add_argument('--key_second_moment', type=str, default='dT_T', help='second moment key')
    parser.add_argument('--bin_spacing', type=float, default=120, help='HEALPix bin size in arcsec')
    parser.add_argument('--psf_max_value', type=float, default=0, help='exclude PSFs with max pixel value below this (e-)')
    parser.add_argument('--colorScale', type=float, default=0.005, help='Min/Max of color scale')
    parser.add_argument('--autoColorScaleCst', type=float, default=2., help='Number of sigma for auto color scale')
    parser.add_argument('--repOutPlot', type=str, default='plots/', help='Output directory for plots')
    parser.add_argument('--pklInput', type=str, default=None, help='Pre-computed pickle to redo plot only')

    parser.add_argument('--autoColorScale', action='store_true')

    args = parser.parse_args()

    plot_Sky_second_Moment(bands=args.bands, rep=args.pathPSFRep, repOutPlot=args.repOutPlot,
                           key_second_moment=args.key_second_moment, bin_spacing=args.bin_spacing,
                           colorScale=args.colorScale, autoColorScale=args.autoColorScale,
                           autoColorScaleCst=args.autoColorScaleCst,
                           colorlabel=None, title=None, pklInput=args.pklInput, psf_max_value=args.psf_max_value)


if __name__ == "__main__":
    main()
