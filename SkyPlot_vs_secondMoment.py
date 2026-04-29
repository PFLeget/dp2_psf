import numpy as np
import treegp
print(treegp.__version__)
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
        Stars with |b| < galactic_b_min are excluded.
    lmc_radius : float
        Radius around LMC to exclude (degrees)
    smc_radius : float
        Radius around SMC to exclude (degrees)

    Returns
    -------
    mask : boolean array
        True for stars to KEEP (outside crowded regions)
    """
    # Convert to astropy SkyCoord
    coords = SkyCoord(ra=ra_deg * u.degree, dec=dec_deg * u.degree, frame='icrs')

    # Get galactic latitude
    galactic_b = coords.galactic.b.degree

    # Filter Milky Way: keep stars with |b| > galactic_b_min
    mask_mw = np.abs(galactic_b) > galactic_b_min

    # Filter LMC: angular distance from LMC center
    lmc_center = SkyCoord(ra=LMC_RA * u.degree, dec=LMC_DEC * u.degree, frame='icrs')
    sep_lmc = coords.separation(lmc_center).degree
    mask_lmc = sep_lmc > lmc_radius

    # Filter SMC: angular distance from SMC center
    smc_center = SkyCoord(ra=SMC_RA * u.degree, dec=SMC_DEC * u.degree, frame='icrs')
    sep_smc = coords.separation(smc_center).degree
    mask_smc = sep_smc > smc_radius

    # Combine: keep only stars that pass ALL filters
    mask = mask_mw & mask_lmc & mask_smc

    return mask


# Columns to read from parquet files (sky-coordinate moments in arcsec^2)
PARQUET_COLUMNS = [
    'shape_Iuu', 'shape_Ivv', 'shape_Iuv',
    'psfShape_Iuu', 'psfShape_Ivv', 'psfShape_Iuv',
    'coord_ra', 'coord_dec',
    'detector', 'psf_max_value',
]


def load_visit_data(parquet_path, coadd_detectors=None):
    """
    Load visit data from parquet file and compute derived columns.
    Uses sky-coordinate moments (Iuu, Ivv, Iuv) in arcsec^2.

    Parameters
    ----------
    parquet_path : str
        Path to the parquet file
    coadd_detectors : set or None
        If provided, only keep sources from detectors in this set.

    Returns
    -------
    dict
        Dictionary with all necessary columns including derived ones
    """
    # Read parquet file with polars (fast!)
    table = polars.scan_parquet(parquet_path).select(PARQUET_COLUMNS).collect()

    # Filter by coadd detectors if specified
    if coadd_detectors is not None:
        detector_col = table['detector'].to_numpy()
        mask = np.isin(detector_col, list(coadd_detectors))
        table = table.filter(polars.Series(mask))

    # Convert to numpy arrays (sky-coord moments in arcsec^2)
    iuu_src = table['shape_Iuu'].to_numpy()
    ivv_src = table['shape_Ivv'].to_numpy()
    iuv_src = table['shape_Iuv'].to_numpy()
    iuu_psf = table['psfShape_Iuu'].to_numpy()
    ivv_psf = table['psfShape_Ivv'].to_numpy()
    iuv_psf = table['psfShape_Iuv'].to_numpy()

    # Compute derived quantities
    T_src = iuu_src + ivv_src
    e1_src = (iuu_src - ivv_src) / T_src
    e2_src = 2 * iuv_src / T_src

    T_psf = iuu_psf + ivv_psf
    e1_psf = (iuu_psf - ivv_psf) / T_psf
    e2_psf = 2 * iuv_psf / T_psf

    return {
        'iuu_src': iuu_src,
        'ivv_src': ivv_src,
        'iuv_src': iuv_src,
        'iuu_psf': iuu_psf,
        'ivv_psf': ivv_psf,
        'iuv_psf': iuv_psf,
        'dT_T': (T_src - T_psf) / T_src,
        'de1': e1_src - e1_psf,
        'de2': e2_src - e2_psf,
        'ra': table['coord_ra'].to_numpy(),
        'dec': table['coord_dec'].to_numpy(),
        'detector': table['detector'].to_numpy(),
        'psf_max_value': table['psf_max_value'].to_numpy(),
    }


def plot_Sky_second_Moment(bands='g', visitMappingFile="data/visit_parquet_mapping_skycoord.pkl",
                           repOutPlot='plots/',
                           key_second_moment='dT_T', bin_spacing=120, colorScale=0.005,
                           autoColorScale=False, autoColorScaleCst=2.,
                           colorlabel=None, title=None, pklInput=None, psf_max_value=0,
                           exclude_crowded=False, galactic_b_min=20., lmc_radius=10., smc_radius=5.,
                           coaddDetectorFile=None):
    """
    Plot spatial variation of PSF second moments on the sky using HEALPix binning.

    Parameters
    ----------
    bands : str
        Band(s) to process (e.g., 'g', 'ugrizy')
    visitMappingFile : str
        Path to the visit_parquet_mapping.pkl file
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
    exclude_crowded : bool
        If True, exclude stars in MW plane, LMC, and SMC
    galactic_b_min : float
        Minimum |b| to keep when exclude_crowded=True (degrees)
    lmc_radius : float
        Radius around LMC to exclude (degrees)
    smc_radius : float
        Radius around SMC to exclude (degrees)
    coaddDetectorFile : str
        Path to coadd_detector_mapping.pkl. If provided, only use detectors that
        went into the coadd (for apples-to-apples comparison).
    """

    CMAP = plt.cm.inferno

    if pklInput is None:
        # Load the visit mapping
        with open(visitMappingFile, 'rb') as f:
            visit_mapping = pickle.load(f)

        # Load coadd detector mapping if specified
        coadd_detector_mapping = None
        if coaddDetectorFile is not None:
            with open(coaddDetectorFile, 'rb') as f:
                coadd_detector_mapping = pickle.load(f)
            print(f"Loaded coadd detector mapping: {len(coadd_detector_mapping)} visits")

        # Filter visits by band(s)
        selected_visits = []
        for visit, info in visit_mapping.items():
            if info['band'] in bands:
                selected_visits.append((visit, info))

        print(f"Selected {len(selected_visits)} visits for bands: {bands}")

        # Use meanify_healpix for sky coordinates
        meanifyHealpix = treegp.meanify_healpix(bin_spacing=bin_spacing)

        n_skipped = 0
        n_skipped_no_coadd = 0
        for visit, info in tqdm(selected_visits, desc="Loop over visits to compute spatial average on sky:"):
            # Get coadd detectors for this visit (if filtering enabled)
            coadd_detectors = None
            if coadd_detector_mapping is not None:
                if visit not in coadd_detector_mapping:
                    n_skipped_no_coadd += 1
                    continue
                coadd_detectors = coadd_detector_mapping[visit]

            # Load data directly from parquet
            try:
                data = load_visit_data(info['parquet_path'], coadd_detectors=coadd_detectors)
            except polars.exceptions.ColumnNotFoundError as e:
                n_skipped += 1
                continue

            # Sky coordinates (RA, Dec) - convert from radians to degrees
            ra_deg = np.degrees(data['ra'])
            dec_deg = np.degrees(data['dec'])

            # Initialize filter
            filtering = np.ones(len(data["ra"]), dtype=bool)

            # Filter by psf_max_value if specified
            if psf_max_value > 0:
                filtering &= (data["psf_max_value"] > psf_max_value)

            # Filter crowded regions (MW, LMC, SMC) if requested
            if exclude_crowded:
                mask_crowded = filter_crowded_regions(
                    ra_deg, dec_deg,
                    galactic_b_min=galactic_b_min,
                    lmc_radius=lmc_radius,
                    smc_radius=smc_radius
                )
                filtering &= mask_crowded

            coord = np.array([ra_deg, dec_deg]).T

            meanifyHealpix.add_field(coord[filtering], data[key_second_moment][filtering])

        if n_skipped > 0:
            print(f"Skipped {n_skipped} visits due to missing sky-coord columns")
        if n_skipped_no_coadd > 0:
            print(f"Skipped {n_skipped_no_coadd} visits not in coadd")

        meanifyHealpix.meanify()

        # Store results for saving
        coords0 = meanifyHealpix.coords0  # (RA, Dec)
        params0 = meanifyHealpix.params0
        wrms0 = meanifyHealpix.wrms0
        nside = meanifyHealpix.nside
        pixel_size_arcsec = meanifyHealpix.pixel_size_arcsec
        valid_pixels = meanifyHealpix._valid_pixels  # HEALPix pixel indices

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
        # Fallback: convert RA/Dec to pixel indices
        pixels = hpg.angle_to_pixel(nside, coords0[:, 0], coords0[:, 1], nest=True, degrees=True)
        healpix_map[pixels] = params0

    if key_second_moment == 'dT_T':
        ksm = '$\\delta T / T$'
    else:
        ksm = key_second_moment

    # Set colorbar label
    if colorlabel is None:
        colorlabel = ksm

    if title is None:
        title = f"DP2 {ksm} | bands: ({bands})"

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
    sp.draw_milky_way(label='Milky Way')

    # Draw DES footprint
    sp.draw_des(edgecolor='blue', lw=2, label='DES footprint')

    # Add colorbar (pad moves it to the right)
    sp.draw_colorbar(label=colorlabel, fontsize=14, pad=0.02)

    # Set title (y parameter moves it higher)
    sp.ax.set_title(title, fontsize=16, y=1.05)

    # Add legend
    sp.ax.legend(loc='lower right', fontsize=10)

    plt.subplots_adjust(left=0.05, right=0.98, top=0.98, bottom=0.08)

    # Build output filename suffix
    suffix = f'{key_second_moment}_sky_{bands}_{int(bin_spacing)}_{int(psf_max_value)}'
    if exclude_crowded:
        suffix += '_noCrowded'
    if coaddDetectorFile is not None:
        suffix += '_coaddOnly'

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
            'bands': bands,
            'key_second_moment': key_second_moment,
            'bin_spacing': bin_spacing,
            'exclude_crowded': exclude_crowded,
            'galactic_b_min': galactic_b_min if exclude_crowded else None,
            'lmc_radius': lmc_radius if exclude_crowded else None,
            'smc_radius': smc_radius if exclude_crowded else None,
        }
        pklFile = open(os.path.join(repOutPlot, f'{suffix}.pkl'), 'wb')
        pickle.dump(dicOutput, pklFile)
        pklFile.close()


def main():

    parser = argparse.ArgumentParser(description="Sky map of PSF second moment residuals")
    parser.add_argument('--bands', type=str, required=True, help="The band(s) to process (e.g., y, g, r, i, z, u, ugrizy)")
    parser.add_argument('--visitMappingFile', type=str, required=True, help="Path to visit_parquet_mapping.pkl file")

    parser.add_argument('--key_second_moment', type=str, default='dT_T', help='second moment key')
    parser.add_argument('--bin_spacing', type=float, default=120, help='HEALPix bin size in arcsec')
    parser.add_argument('--psf_max_value', type=float, default=0, help='exclude PSFs with max pixel value below this (e-)')
    parser.add_argument('--colorScale', type=float, default=0.005, help='Min/Max of color scale')
    parser.add_argument('--autoColorScaleCst', type=float, default=2., help='Number of sigma for auto color scale')
    parser.add_argument('--repOutPlot', type=str, default='plots/', help='Output directory for plots')
    parser.add_argument('--pklInput', type=str, default=None, help='Pre-computed pickle to redo plot only')

    parser.add_argument('--autoColorScale', action='store_true')

    # Crowded region filtering
    parser.add_argument('--exclude_crowded', action='store_true',
                        help='Exclude MW plane, LMC, and SMC')
    parser.add_argument('--galactic_b_min', type=float, default=20.,
                        help='Min |b| to keep when excluding MW (degrees, default: 20)')
    parser.add_argument('--lmc_radius', type=float, default=10.,
                        help='Radius around LMC to exclude (degrees, default: 10)')
    parser.add_argument('--smc_radius', type=float, default=5.,
                        help='Radius around SMC to exclude (degrees, default: 5)')
    parser.add_argument('--coaddDetectorFile', type=str, default=None,
                        help='Path to coadd_detector_mapping.pkl to filter only detectors in coadd')

    args = parser.parse_args()

    plot_Sky_second_Moment(bands=args.bands, visitMappingFile=args.visitMappingFile,
                           repOutPlot=args.repOutPlot,
                           key_second_moment=args.key_second_moment, bin_spacing=args.bin_spacing,
                           colorScale=args.colorScale, autoColorScale=args.autoColorScale,
                           autoColorScaleCst=args.autoColorScaleCst,
                           colorlabel=None, title=None, pklInput=args.pklInput, psf_max_value=args.psf_max_value,
                           exclude_crowded=args.exclude_crowded, galactic_b_min=args.galactic_b_min,
                           lmc_radius=args.lmc_radius, smc_radius=args.smc_radius,
                           coaddDetectorFile=args.coaddDetectorFile)


if __name__ == "__main__":
    main()
