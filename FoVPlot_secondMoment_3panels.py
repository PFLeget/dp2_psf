import numpy as np
import treegp
print(treegp.__version__)
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lsst.utils.plotting import publication_plots
publication_plots.set_rubin_plotstyle()

import os
os.environ["POLARS_MAX_THREADS"] = "1"
import polars

import pickle
import lsst.afw.cameraGeom as cameraGeom
from lsst.obs.lsst import LsstCam
import argparse

# For galactic coordinate transformation (crowded-region cut)
from astropy.coordinates import SkyCoord
import astropy.units as u


camera = LsstCam.getCamera()


# The three second-moment residuals drawn as panels
KEYS = ['dT_T', 'de1', 'de2']

# Nice panel titles
KEY_LABELS = {
    'dT_T': r'$\delta T / T$',
    'de1': r'$\delta e_1$',
    'de2': r'$\delta e_2$',
}


# LMC and SMC centers (J2000), from SkyPlot_vs_secondMoment.py
LMC_RA, LMC_DEC = 80.9, -69.8  # degrees
SMC_RA, SMC_DEC = 13.2, -72.8  # degrees


def filter_crowded_regions(ra_deg, dec_deg, galactic_b_min=30., lmc_radius=10., smc_radius=5.):
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


# Columns to read from parquet files
PARQUET_COLUMNS = [
    'slot_Shape_xx', 'slot_Shape_yy', 'slot_Shape_xy',
    'slot_PsfShape_xx', 'slot_PsfShape_xy', 'slot_PsfShape_yy',
    'coord_ra', 'coord_dec', 'slot_Centroid_x', 'slot_Centroid_y',
    'detector', 'psf_max_value', 'calib_psf_reserved',
    'base_GaussianFlux_instFlux', 'base_GaussianFlux_instFluxErr',
]


def load_visit_data(parquet_path):
    """
    Load visit data from parquet file and compute derived columns.

    Parameters
    ----------
    parquet_path : str
        Path to the parquet file

    Returns
    -------
    dict
        Dictionary with all necessary columns including derived ones
    """
    # Read parquet file with polars (fast!)
    table = polars.scan_parquet(parquet_path).select(PARQUET_COLUMNS).collect()

    # Convert to numpy arrays
    slot_Shape_xx = table['slot_Shape_xx'].to_numpy()
    slot_Shape_yy = table['slot_Shape_yy'].to_numpy()
    slot_Shape_xy = table['slot_Shape_xy'].to_numpy()
    slot_PsfShape_xx = table['slot_PsfShape_xx'].to_numpy()
    slot_PsfShape_yy = table['slot_PsfShape_yy'].to_numpy()
    slot_PsfShape_xy = table['slot_PsfShape_xy'].to_numpy()

    # Compute derived quantities
    T_src = slot_Shape_xx + slot_Shape_yy
    e1_src = (slot_Shape_xx - slot_Shape_yy) / T_src
    e2_src = 2 * slot_Shape_xy / T_src

    T_psf = slot_PsfShape_xx + slot_PsfShape_yy
    e1_psf = (slot_PsfShape_xx - slot_PsfShape_yy) / T_psf
    e2_psf = 2 * slot_PsfShape_xy / T_psf

    # Compute SNR
    flux = table['base_GaussianFlux_instFlux'].to_numpy()
    flux_err = table['base_GaussianFlux_instFluxErr'].to_numpy()
    snr = flux / flux_err

    return {
        'ixx_src': slot_Shape_xx,
        'iyy_src': slot_Shape_yy,
        'ixy_src': slot_Shape_xy,
        'ixx_psf': slot_PsfShape_xx,
        'iyy_psf': slot_PsfShape_yy,
        'ixy_psf': slot_PsfShape_xy,
        'dT_T': (T_src - T_psf) / T_src,
        'de1': e1_src - e1_psf,
        'de2': e2_src - e2_psf,
        'ra': table['coord_ra'].to_numpy(),
        'dec': table['coord_dec'].to_numpy(),
        'xCCD': table['slot_Centroid_x'].to_numpy(),
        'yCCD': table['slot_Centroid_y'].to_numpy(),
        'detector': table['detector'].to_numpy(),
        'psf_max_value': table['psf_max_value'].to_numpy(),
        'calib_psf_reserved': table['calib_psf_reserved'].to_numpy(),
        'snr': snr,
    }


def pixel_to_focal(x, y, det):
    """
    Parameters
    ----------
    x, y : array
        Pixel coordinates.
    det : lsst.afw.cameraGeom.Detector
        Detector of interest.

    Returns
    -------
    fpx, fpy : array
        Focal plane position in millimeters in DVCS
        See https://lse-349.lsst.io/
    """
    tx = det.getTransform(cameraGeom.PIXELS, cameraGeom.FOCAL_PLANE)
    fpx, fpy = tx.getMapping().applyForward(np.vstack((x, y)))

    return fpx.ravel(), fpy.ravel()


def _color_scale(averages, key, colorScales, autoColorScale, autoColorScaleCst):
    """
    Compute the (MIN, MAX) color scale for one panel.

    Parameters
    ----------
    averages : list of arrays
        The per-CCD ``_average`` arrays for this key.
    key : str
        Second-moment key ('dT_T', 'de1', 'de2').
    colorScales : dict
        Per-key fixed color scale (symmetric, +/- value).
    autoColorScale : bool
        If True, compute the scale from the data (median +/- N sigma).
    autoColorScaleCst : float
        Number of sigma for the auto color scale.
    """
    if autoColorScale:
        M = np.concatenate(averages)
        finite = M[np.isfinite(M)]
        MEAN = np.median(finite)
        STD = np.std(finite)
        return MEAN - autoColorScaleCst * STD, MEAN + autoColorScaleCst * STD
    scale = colorScales[key]
    return -scale, scale


def plot_FoV_3panels(bands='g', visitMappingFile="data/visit_parquet_mapping.pkl",
                     repOutPlot='plots/', bin_spacing=150,
                     colorScales=None, autoColorScale=False, autoColorScaleCst=2.,
                     statisticsMedian=False, title=None, pklInput=None,
                     psf_max_value=0, snr_min=0, maxVisits=0,
                     exclude_crowded=False, galactic_b_min=30.,
                     lmc_radius=10., smc_radius=5.):
    """
    Plot the spatial variation of the three PSF second-moment residuals
    (dT/T, de1, de2) on the focal plane as a single 3-panel figure.

    Parameters
    ----------
    bands : str
        Band(s) to process (e.g., 'g', 'griz', 'ugrizy'). Visits whose band is a
        member of this string are pooled into the same figure.
    visitMappingFile : str
        Path to the visit_parquet_mapping.pkl file
    repOutPlot : str
        Output directory for plots
    bin_spacing : float
        Bin spacing in pixels
    colorScales : dict or None
        Per-key symmetric color scale {key: value}. Defaults to 0.005 for each key.
    autoColorScale : bool
        If True, compute each panel's color scale from its data
    autoColorScaleCst : float
        Number of sigma for the auto color scale
    statisticsMedian : bool
        If True, use median instead of mean
    title : str
        Figure suptitle
    pklInput : str
        Path to a pre-computed pickle file (to redo the plot only)
    psf_max_value : float
        Exclude PSFs with max pixel value below this threshold
    snr_min : float
        Minimum SNR threshold (base_GaussianFlux_instFlux / base_GaussianFlux_instFluxErr)
    maxVisits : int
        If > 0, only process the first ``maxVisits`` selected visits (for testing).
    exclude_crowded : bool
        If True, exclude sources in the MW plane, LMC, and SMC.
    galactic_b_min : float
        Minimum |b| to keep when exclude_crowded=True (degrees)
    lmc_radius : float
        Radius around LMC to exclude (degrees)
    smc_radius : float
        Radius around SMC to exclude (degrees)
    """

    CMAP = plt.cm.inferno

    if colorScales is None:
        colorScales = {key: 0.005 for key in KEYS}

    if pklInput is None:
        # Load the visit mapping
        with open(visitMappingFile, 'rb') as f:
            visit_mapping = pickle.load(f)

        # Filter visits by band(s)
        selected_visits = []
        for visit, info in visit_mapping.items():
            if info['band'] in bands:
                selected_visits.append((visit, info))

        if maxVisits > 0:
            selected_visits = selected_visits[:maxVisits]

        print(f"Selected {len(selected_visits)} visits for bands: {bands}")

        # One independent per-CCD meanify stream per key
        meanifyStream = {key: {} for key in KEYS}

        n_kept = 0
        n_total = 0
        for visit, info in tqdm(selected_visits, desc="Loop over visits to compute spatial average:"):
            # Load data directly from parquet
            data = load_visit_data(info['parquet_path'])

            # Crowded-region mask (per source), computed once per visit
            if exclude_crowded:
                mask_crowded = filter_crowded_regions(
                    np.degrees(data['ra']), np.degrees(data['dec']),
                    galactic_b_min=galactic_b_min,
                    lmc_radius=lmc_radius, smc_radius=smc_radius,
                )
            else:
                mask_crowded = None

            ccdIds = set(data["detector"])

            coord = np.array([data['xCCD'], data['yCCD']]).T

            for ccd in ccdIds:
                # Build the per-source selection once (shared across the 3 keys)
                filtering = (data["detector"] == ccd)
                if psf_max_value > 0:
                    filtering &= (data["psf_max_value"] > psf_max_value)
                if snr_min > 0:
                    filtering &= (data["snr"] > snr_min)
                if mask_crowded is not None:
                    filtering &= mask_crowded

                n_total += int(np.sum(data["detector"] == ccd))
                n_kept += int(np.sum(filtering))

                for key in KEYS:
                    if ccd not in meanifyStream[key]:
                        if not statisticsMedian:
                            # New API: meanify with bounds enables streaming mode for statistics="mean"
                            meanifyStream[key][ccd] = treegp.meanify(bin_spacing=bin_spacing, statistics="mean", bounds=(0, 4100, 0, 4100))
                        else:
                            meanifyStream[key][ccd] = treegp.meanify(bin_spacing=bin_spacing, statistics='median')
                    meanifyStream[key][ccd].add_field(coord[filtering], data[key][filtering])

        if exclude_crowded and n_total > 0:
            print(f"Crowded-region cut: kept {n_kept}/{n_total} sources ({100. * n_kept / n_total:.1f}%)")

        for key in KEYS:
            for ccd in meanifyStream[key]:
                if not statisticsMedian:
                    meanifyStream[key][ccd].meanify()
                else:
                    meanifyStream[key][ccd].meanify(lu_min=0, lu_max=4100, lv_min=0, lv_max=4100)

        # Build the plot data structure (per key: {ccd: {x, y, _average}})
        dicMeanifyPlot = {key: {} for key in KEYS}
        for key in KEYS:
            for i in meanifyStream[key]:
                x, y = np.meshgrid(meanifyStream[key][i]._xedge, meanifyStream[key][i]._yedge)
                nBin0, nBin1 = np.shape(x)[0], np.shape(x)[1]
                x = x.reshape(nBin0 * nBin1)
                y = y.reshape(nBin0 * nBin1)
                x, y = pixel_to_focal(x, y, camera[i])
                x = x.reshape((nBin0, nBin1))
                y = y.reshape((nBin0, nBin1))
                dicMeanifyPlot[key][i] = {
                    'x': x,
                    'y': y,
                    '_average': meanifyStream[key][i]._average,
                }

    else:
        with open(pklInput, 'rb') as f:
            dicInput = pickle.load(f)
        dicMeanifyPlot = dicInput['panels']

    # Draw the 3 panels
    fig, axes = plt.subplots(1, 3, figsize=(30, 10))

    for ax, key in zip(axes, KEYS):
        averages = [dicMeanifyPlot[key][i]['_average'] for i in dicMeanifyPlot[key]]
        MIN, MAX = _color_scale(averages, key, colorScales, autoColorScale, autoColorScaleCst)

        mesh = None
        for i in dicMeanifyPlot[key]:
            mesh = ax.pcolormesh(dicMeanifyPlot[key][i]['x'], dicMeanifyPlot[key][i]['y'],
                                 dicMeanifyPlot[key][i]['_average'],
                                 vmin=MIN, vmax=MAX, cmap=CMAP)

        cb = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(KEY_LABELS.get(key, key), size=22)
        cb.ax.tick_params(labelsize=16)
        ax.set_xlabel('x (mm)', size=22)
        ax.set_ylabel('y (mm)', size=22)
        ax.set_title(KEY_LABELS.get(key, key), size=22)
        ax.set_aspect('equal')
        ax.tick_params(labelsize=16)

    if title is None:
        title = f"DP2 | bands: ({bands})"
        if exclude_crowded:
            title += rf" | $|b|>${int(galactic_b_min)}, no LMC/SMC"
    fig.suptitle(title, size=24)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if statisticsMedian:
        median_key = "median"
    else:
        median_key = ""
    output_name = f'secondMoment_3panel_{bands}_{int(bin_spacing)}_{median_key}_psfmax{int(psf_max_value)}_snr{int(snr_min)}'
    if exclude_crowded:
        output_name += f'_noCrowded_b{int(galactic_b_min)}'
    fig.savefig(os.path.join(repOutPlot, f'{output_name}.png'))
    plt.close(fig)

    if pklInput is None:
        dicOutput = {
            'panels': dicMeanifyPlot,
            'bands': bands,
            'bin_spacing': bin_spacing,
            'psf_max_value': psf_max_value,
            'snr_min': snr_min,
            'statisticsMedian': statisticsMedian,
            'exclude_crowded': exclude_crowded,
            'galactic_b_min': galactic_b_min if exclude_crowded else None,
            'lmc_radius': lmc_radius if exclude_crowded else None,
            'smc_radius': smc_radius if exclude_crowded else None,
        }
        with open(os.path.join(repOutPlot, f'{output_name}.pkl'), 'wb') as pklFile:
            pickle.dump(dicOutput, pklFile)


def main():

    parser = argparse.ArgumentParser(description="Focal plane 3-panel map of PSF second moment residuals (dT/T, de1, de2)")
    parser.add_argument('--bands', type=str, required=True, help="The band(s) to process (e.g., y, g, r, i, z, u, griz, ugrizy)")
    parser.add_argument('--visitMappingFile', type=str, required=True, help="Path to visit_parquet_mapping.pkl file")

    parser.add_argument('--bin_spacing', type=float, default=150, help='bin size in pixels')
    parser.add_argument('--psf_max_value', type=float, default=0, help='exclude PSFs with max pixel value below this (e-)')
    parser.add_argument('--snr_min', type=float, default=0, help='minimum SNR threshold (default: 0 = no cut)')
    parser.add_argument('--maxVisits', type=int, default=0, help='only process the first N selected visits (0 = all, for testing)')

    # Per-panel (per-key) color scale
    parser.add_argument('--colorScale_dT_T', type=float, default=0.005, help='Min/Max of color scale for dT/T panel')
    parser.add_argument('--colorScale_de1', type=float, default=0.005, help='Min/Max of color scale for de1 panel')
    parser.add_argument('--colorScale_de2', type=float, default=0.005, help='Min/Max of color scale for de2 panel')
    parser.add_argument('--autoColorScale', action='store_true', help='Auto color scale per panel (median +/- N sigma)')
    parser.add_argument('--autoColorScaleCst', type=float, default=2., help='Number of sigma for auto color scale')

    parser.add_argument('--repOutPlot', type=str, default='plots/', help='Output directory for plots')
    parser.add_argument('--pklInput', type=str, default=None, help='Pre-computed pickle to redo plot only')
    parser.add_argument('--statisticsMedian', action='store_true')

    # Crowded-region filtering (per source, on RA/Dec)
    parser.add_argument('--exclude_crowded', action='store_true',
                        help='Exclude sources in the MW plane, LMC, and SMC')
    parser.add_argument('--galactic_b_min', type=float, default=30.,
                        help='Min |b| to keep when excluding MW (degrees, default: 30)')
    parser.add_argument('--lmc_radius', type=float, default=10.,
                        help='Radius around LMC to exclude (degrees, default: 10)')
    parser.add_argument('--smc_radius', type=float, default=5.,
                        help='Radius around SMC to exclude (degrees, default: 5)')

    args = parser.parse_args()

    colorScales = {
        'dT_T': args.colorScale_dT_T,
        'de1': args.colorScale_de1,
        'de2': args.colorScale_de2,
    }

    plot_FoV_3panels(bands=args.bands, visitMappingFile=args.visitMappingFile,
                     repOutPlot=args.repOutPlot, bin_spacing=args.bin_spacing,
                     colorScales=colorScales, autoColorScale=args.autoColorScale,
                     autoColorScaleCst=args.autoColorScaleCst, statisticsMedian=args.statisticsMedian,
                     title=None, pklInput=args.pklInput, psf_max_value=args.psf_max_value,
                     snr_min=args.snr_min, maxVisits=args.maxVisits,
                     exclude_crowded=args.exclude_crowded, galactic_b_min=args.galactic_b_min,
                     lmc_radius=args.lmc_radius, smc_radius=args.smc_radius)


if __name__ == "__main__":
    main()
