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
import hpgeom as hpg

import pickle
import pandas as pd
import argparse
import glob
import os

# Import skyproj for sky visualization
from skyproj import McBrydeSkyproj
from skyproj.survey import _Survey


class SurveyMcBrydeSkyproj(_Survey, McBrydeSkyproj):
    """McBryde projection with survey footprint drawing capabilities."""
    pass


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
    npix = hpg.nside_to_npixel(nside)
    healpix_map = np.full(npix, hpg.UNSEEN)

    if valid_pixels is not None:
        healpix_map[valid_pixels] = params0
    else:
        # Fallback: convert RA/Dec to pixel indices
        pixels = hpg.angle_to_pixel(nside, coords0[:, 0], coords0[:, 1], nest=True, degrees=True)
        healpix_map[pixels] = params0

    # Set colorbar label
    if colorlabel is None:
        colorlabel = key_second_moment

    if title is None:
        title = f"DP2 {key_second_moment} | bands: ({bands}) | nside={nside} (~{pixel_size_arcsec:.0f} arcsec)"

    # Create figure and skyproj projection
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111)

    # Use McBryde projection with survey capabilities (full sky view)
    sp = SurveyMcBrydeSkyproj(ax=ax, lon_0=0.0)

    # Draw the HEALPix map
    im, lon_raster, lat_raster, values_raster = sp.draw_hpxmap(
        healpix_map, nest=True, zoom=False, vmin=MIN, vmax=MAX, cmap=CMAP
    )

    # Draw Milky Way plane
    sp.draw_milky_way(width=0, linewidth=2, color='black', linestyle='--', label='Milky Way')

    # Draw DES footprint
    sp.draw_des(edgecolor='cyan', lw=2, label='DES')

    # Add colorbar (pad moves it to the right)
    sp.draw_colorbar(label=colorlabel, fontsize=14, pad=0.02)

    # Set title (y parameter moves it higher)
    sp.ax.set_title(title, fontsize=16, y=1.05)

    # Add legend
    sp.ax.legend(loc='lower right', fontsize=10)

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
