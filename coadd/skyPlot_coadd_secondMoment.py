#!/usr/bin/env python
"""
Sky map of PSF second moments for coadd data.
Equivalent to SkyPlot_vs_secondMoment.py but for coadd (object_all from stage3).

Columns are band-prefixed: {band}_ixx, {band}_iyy, etc.
Supports: T, e1, e2, dT_T, de1, de2
"""

import numpy as np
import treegp
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import hpgeom as hpg

import os
os.environ["POLARS_MAX_THREADS"] = "1"
import polars

import pickle
import argparse

# Import skyproj for sky visualization
from skyproj import McBrydeSkyproj
from skyproj.survey import _Survey

# For galactic coordinate transformation
from astropy.coordinates import SkyCoord
import astropy.units as u


class SurveyMcBrydeSkyproj(_Survey, McBrydeSkyproj):
    """McBryde projection with survey footprint drawing capabilities."""
    pass


# LMC and SMC centers (J2000)
LMC_RA, LMC_DEC = 80.9, -69.8  # degrees
SMC_RA, SMC_DEC = 13.2, -72.8  # degrees


def filter_crowded_regions(ra_deg, dec_deg, galactic_b_min=20., lmc_radius=10., smc_radius=5.):
    """
    Filter out stars in crowded regions: Milky Way plane, LMC, SMC.

    Parameters
    ----------
    ra_deg : array
        Right ascension in degrees
    dec_deg : array
        Declination in degrees
    galactic_b_min : float
        Minimum absolute galactic latitude |b| to keep (degrees).
    lmc_radius : float
        Radius around LMC to exclude (degrees)
    smc_radius : float
        Radius around SMC to exclude (degrees)

    Returns
    -------
    mask : boolean array
        True for stars to KEEP (outside crowded regions)
    """
    coords = SkyCoord(ra=ra_deg * u.degree, dec=dec_deg * u.degree, frame='icrs')

    galactic_b = coords.galactic.b.degree
    mask_mw = np.abs(galactic_b) > galactic_b_min

    lmc_center = SkyCoord(ra=LMC_RA * u.degree, dec=LMC_DEC * u.degree, frame='icrs')
    sep_lmc = coords.separation(lmc_center).degree
    mask_lmc = sep_lmc > lmc_radius

    smc_center = SkyCoord(ra=SMC_RA * u.degree, dec=SMC_DEC * u.degree, frame='icrs')
    sep_smc = coords.separation(smc_center).degree
    mask_smc = sep_smc > smc_radius

    return mask_mw & mask_lmc & mask_smc


def get_parquet_columns(band):
    """Get the columns to read from parquet for a given band."""
    return [
        f"{band}_calib_psf_used", "detect_isPrimary", "refExtendedness",
        f"{band}_pixelFlags_inexact_psfCenter",
        f"{band}_ixxPSF", f"{band}_iyyPSF", f"{band}_ixyPSF",
        f"{band}_ixx", f"{band}_iyy", f"{band}_ixy",
        f"{band}_ra", f"{band}_dec",
    ]


def load_tract_data(parquet_path, band):
    """
    Load tract data from parquet file and compute derived columns.

    Parameters
    ----------
    parquet_path : str
        Path to the parquet file
    band : str
        Band to extract (u, g, r, i, z, y)

    Returns
    -------
    dict
        Dictionary with all necessary columns including derived ones
    """
    columns = get_parquet_columns(band)
    table = polars.scan_parquet(parquet_path).select(columns).collect()

    # Apply quality filters
    mask = table['detect_isPrimary'].to_numpy() == True
    mask &= table['refExtendedness'].to_numpy() == 0.0  # stars only
    mask &= table[f'{band}_calib_psf_used'].to_numpy() == True
    mask &= table[f'{band}_pixelFlags_inexact_psfCenter'].to_numpy() == False

    # Extract columns for filtered data
    ixx_src = table[f"{band}_ixx"].to_numpy()[mask]
    iyy_src = table[f"{band}_iyy"].to_numpy()[mask]
    ixy_src = table[f"{band}_ixy"].to_numpy()[mask]
    ixx_psf = table[f"{band}_ixxPSF"].to_numpy()[mask]
    iyy_psf = table[f"{band}_iyyPSF"].to_numpy()[mask]
    ixy_psf = table[f"{band}_ixyPSF"].to_numpy()[mask]
    ra = table[f"{band}_ra"].to_numpy()[mask]
    dec = table[f"{band}_dec"].to_numpy()[mask]

    # Compute derived quantities
    T_src = ixx_src + iyy_src
    e1_src = (ixx_src - iyy_src) / T_src
    e2_src = 2 * ixy_src / T_src

    T_psf = ixx_psf + iyy_psf
    e1_psf = (ixx_psf - iyy_psf) / T_psf
    e2_psf = 2 * ixy_psf / T_psf

    return {
        'T': T_src,
        'e1': e1_src,
        'e2': e2_src,
        'T_psf': T_psf,
        'e1_psf': e1_psf,
        'e2_psf': e2_psf,
        'dT_T': (T_src - T_psf) / T_src,
        'de1': e1_src - e1_psf,
        'de2': e2_src - e2_psf,
        'ra': ra,  # already in degrees
        'dec': dec,
    }


def plot_Sky_second_Moment_coadd(band='r', tractMappingFile="data/tract_parquet_mapping.pkl",
                                  repOutPlot='plots/',
                                  key_second_moment='dT_T', bin_spacing=120, colorScale=0.005,
                                  autoColorScale=False, autoColorScaleCst=2.,
                                  colorlabel=None, title=None, pklInput=None,
                                  exclude_crowded=False, galactic_b_min=20., lmc_radius=10., smc_radius=5.):
    """
    Plot spatial variation of PSF second moments on the sky using HEALPix binning.

    Parameters
    ----------
    band : str
        Band to process (u, g, r, i, z, y)
    tractMappingFile : str
        Path to the tract_parquet_mapping.pkl file
    repOutPlot : str
        Output directory for plots
    key_second_moment : str
        Second moment key to plot: 'T', 'e1', 'e2', 'dT_T', 'de1', 'de2'
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
    exclude_crowded : bool
        If True, exclude stars in MW plane, LMC, and SMC
    galactic_b_min : float
        Minimum |b| to keep when exclude_crowded=True (degrees)
    lmc_radius : float
        Radius around LMC to exclude (degrees)
    smc_radius : float
        Radius around SMC to exclude (degrees)
    """

   #CMAP = plt.cm.inferno
   CMAP = plt.cm.seismic

    if pklInput is None:
        # Load the tract mapping
        with open(tractMappingFile, 'rb') as f:
            tract_mapping = pickle.load(f)

        print(f"Processing {len(tract_mapping)} tracts for band: {band}")

        # Use meanify_healpix for sky coordinates
        meanifyHealpix = treegp.meanify_healpix(bin_spacing=bin_spacing)

        for tract, info in tqdm(tract_mapping.items(), desc="Loop over tracts"):
            try:
                data = load_tract_data(info['parquet_path'], band)
            except Exception as e:
                print(f"  Warning: failed to load tract {tract}: {e}")
                continue

            if len(data['ra']) == 0:
                continue

            ra_deg = data['ra']
            dec_deg = data['dec']

            # Filter by finite values
            filtering = np.isfinite(data[key_second_moment])

            # Filter crowded regions if requested
            if exclude_crowded:
                mask_crowded = filter_crowded_regions(
                    ra_deg, dec_deg,
                    galactic_b_min=galactic_b_min,
                    lmc_radius=lmc_radius,
                    smc_radius=smc_radius
                )
                filtering &= mask_crowded

            if np.sum(filtering) == 0:
                continue

            coord = np.array([ra_deg, dec_deg]).T
            meanifyHealpix.add_field(coord[filtering], data[key_second_moment][filtering])

        meanifyHealpix.meanify()

        # Store results for saving
        coords0 = meanifyHealpix.coords0
        params0 = meanifyHealpix.params0
        wrms0 = meanifyHealpix.wrms0
        nside = meanifyHealpix.nside
        pixel_size_arcsec = meanifyHealpix.pixel_size_arcsec
        valid_pixels = meanifyHealpix._valid_pixels

    else:
        with open(pklInput, 'rb') as f:
            dicInput = pickle.load(f)
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
        pixels = hpg.angle_to_pixel(nside, coords0[:, 0], coords0[:, 1], nest=True, degrees=True)
        healpix_map[pixels] = params0

    # Set labels
    key_labels = {
        'T': '$T$ (pixel$^2$)',
        'e1': '$e_1$',
        'e2': '$e_2$',
        'dT_T': '$\\delta T / T$',
        'de1': '$\\delta e_1$',
        'de2': '$\\delta e_2$',
    }
    ksm = key_labels.get(key_second_moment, key_second_moment)

    if colorlabel is None:
        colorlabel = ksm

    if title is None:
        title = f"DP2 Coadd {ksm} | band: {band}"

    # Create figure and skyproj projection
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111)

    sp = SurveyMcBrydeSkyproj(ax=ax, lon_0=0.0)

    im, lon_raster, lat_raster, values_raster = sp.draw_hpxmap(
        healpix_map, nest=True, zoom=False, vmin=MIN, vmax=MAX, cmap=CMAP
    )

    sp.draw_milky_way(label='Milky Way')
    sp.draw_des(edgecolor='blue', lw=2, label='DES footprint')
    sp.draw_colorbar(label=colorlabel, fontsize=14, pad=0.02)
    sp.ax.set_title(title, fontsize=16, y=1.05)
    sp.ax.legend(loc='lower right', fontsize=10)

    plt.subplots_adjust(left=0.05, right=0.98, top=0.98, bottom=0.08)

    # Build output filename suffix
    suffix = f'coadd_{key_second_moment}_sky_{band}_{int(bin_spacing)}'
    if exclude_crowded:
        suffix += '_noCrowded'

    os.makedirs(repOutPlot, exist_ok=True)
    plt.savefig(os.path.join(repOutPlot, f'{suffix}.png'), dpi=150)
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
            'band': band,
            'key_second_moment': key_second_moment,
            'bin_spacing': bin_spacing,
            'exclude_crowded': exclude_crowded,
            'galactic_b_min': galactic_b_min if exclude_crowded else None,
            'lmc_radius': lmc_radius if exclude_crowded else None,
            'smc_radius': smc_radius if exclude_crowded else None,
        }
        with open(os.path.join(repOutPlot, f'{suffix}.pkl'), 'wb') as f:
            pickle.dump(dicOutput, f)


def main():
    parser = argparse.ArgumentParser(description="Sky map of PSF second moments for coadd data")
    parser.add_argument('--band', type=str, required=True, help="Band to process (u, g, r, i, z, y)")
    parser.add_argument('--tractMappingFile', type=str, required=True, help="Path to tract_parquet_mapping.pkl")

    parser.add_argument('--key_second_moment', type=str, default='dT_T',
                        help='Second moment key: T, e1, e2, dT_T, de1, de2')
    parser.add_argument('--bin_spacing', type=float, default=120, help='HEALPix bin size in arcsec')
    parser.add_argument('--colorScale', type=float, default=0.005, help='Min/Max of color scale')
    parser.add_argument('--autoColorScaleCst', type=float, default=2., help='Number of sigma for auto color scale')
    parser.add_argument('--repOutPlot', type=str, default='plots/', help='Output directory for plots')
    parser.add_argument('--pklInput', type=str, default=None, help='Pre-computed pickle to redo plot only')

    parser.add_argument('--autoColorScale', action='store_true')

    # Crowded region filtering
    parser.add_argument('--exclude_crowded', action='store_true',
                        help='Exclude MW plane, LMC, and SMC')
    parser.add_argument('--galactic_b_min', type=float, default=20.,
                        help='Min |b| to keep when excluding MW (degrees)')
    parser.add_argument('--lmc_radius', type=float, default=10.,
                        help='Radius around LMC to exclude (degrees)')
    parser.add_argument('--smc_radius', type=float, default=5.,
                        help='Radius around SMC to exclude (degrees)')

    args = parser.parse_args()

    plot_Sky_second_Moment_coadd(
        band=args.band,
        tractMappingFile=args.tractMappingFile,
        repOutPlot=args.repOutPlot,
        key_second_moment=args.key_second_moment,
        bin_spacing=args.bin_spacing,
        colorScale=args.colorScale,
        autoColorScale=args.autoColorScale,
        autoColorScaleCst=args.autoColorScaleCst,
        pklInput=args.pklInput,
        exclude_crowded=args.exclude_crowded,
        galactic_b_min=args.galactic_b_min,
        lmc_radius=args.lmc_radius,
        smc_radius=args.smc_radius,
    )


if __name__ == "__main__":
    main()
