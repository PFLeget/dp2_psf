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


camera = LsstCam.getCamera()


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


def plot_FoV_second_Moment(bands='g', visitMappingFile="data/visit_parquet_mapping.pkl",
                           repOutPlot='plots/',
                           key_second_moment='dT_T', bin_spacing=150, colorScale=0.005,
                           autoColorScale=False, autoColorScaleCst=2., statisticsMedian=False,
                           colorlabel=None, title=None, pklInput=None, psf_max_value=0,
                           snr_min=0):
    """
    Plot spatial variation of PSF second moments on the focal plane.

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
        Bin spacing in pixels
    colorScale : float
        Color scale range [-colorScale, +colorScale]
    autoColorScale : bool
        If True, compute color scale from data
    autoColorScaleCst : float
        Number of sigma for auto color scale
    statisticsMedian : bool
        If True, use median instead of mean
    colorlabel : str
        Label for colorbar
    title : str
        Plot title
    pklInput : str
        Path to pre-computed pickle file (to redo plot only)
    psf_max_value : float
        Exclude PSFs with max pixel value below this threshold
    snr_min : float
        Minimum SNR threshold (base_GaussianFlux_instFlux / base_GaussianFlux_instFluxErr)
    """

    CMAP = plt.cm.inferno

    if pklInput is None:
        # Load the visit mapping
        with open(visitMappingFile, 'rb') as f:
            visit_mapping = pickle.load(f)

        # Filter visits by band(s)
        selected_visits = []
        for visit, info in visit_mapping.items():
            if info['band'] in bands:
                selected_visits.append((visit, info))

        print(f"Selected {len(selected_visits)} visits for bands: {bands}")

        meanifyStream = {}

        for visit, info in tqdm(selected_visits, desc="Loop over visits to compute spatial average:"):
            # Load data directly from parquet
            data = load_visit_data(info['parquet_path'])

            ccdIds = set(data["detector"])

            for ccd in ccdIds:
                filtering = (data["detector"] == ccd)
                if psf_max_value > 0:
                    filtering &= (data["psf_max_value"] > psf_max_value)
                if snr_min > 0:
                    filtering &= (data["snr"] > snr_min)
                coord = np.array([data['xCCD'], data['yCCD']]).T
                if ccd not in meanifyStream:
                    if not statisticsMedian:
                        # New API: meanify with bounds enables streaming mode for statistics="mean"
                        meanifyStream.update({ccd: treegp.meanify(bin_spacing=bin_spacing, statistics="mean", bounds=(0, 4100, 0, 4100))})
                    else:
                        meanifyStream.update({ccd: treegp.meanify(bin_spacing=bin_spacing, statistics='median')})
                meanifyStream[ccd].add_field(coord[filtering], data[key_second_moment][filtering])

        for ccd in meanifyStream:
            if not statisticsMedian:
                meanifyStream[ccd].meanify()
            else:
                meanifyStream[ccd].meanify(lu_min=0, lu_max=4100, lv_min=0, lv_max=4100)

        ccdIds = list(meanifyStream.keys())

    else:
        with open(pklInput, 'rb') as f:
            dicInput = pickle.load(f)
        ccdIds = list(dicInput.keys())

    if autoColorScale:
        M = []
        for i in ccdIds:
            if pklInput is None:
                M.append(meanifyStream[i]._average)
            else:
                M.append(dicInput[i]['_average'])
        M = np.concatenate(M)
        MEAN = np.median(M[np.isfinite(M)])
        STD = np.std(M[np.isfinite(M)])

        MIN = MEAN - autoColorScaleCst * STD
        MAX = MEAN + autoColorScaleCst * STD
    else:
        MIN = -colorScale
        MAX = colorScale

    dicMeanifyPlot = {}
    plt.figure(figsize=(20, 12))
    for i in ccdIds:
        if pklInput is None:
            x, y = np.meshgrid(meanifyStream[i]._xedge, meanifyStream[i]._yedge)
            nBin0, nBin1 = np.shape(x)[0], np.shape(x)[1]
            x = x.reshape(nBin0*nBin1)
            y = y.reshape(nBin0*nBin1)
            x, y = pixel_to_focal(x, y, camera[i])
            x = x.reshape((nBin0, nBin1))
            y = y.reshape((nBin0, nBin1))
            plt.pcolormesh(x, y, meanifyStream[i]._average, vmin=MIN, vmax=MAX, cmap=CMAP)
            dicMeanifyPlot.update({i: {
                'x': x,
                'y': y,
                '_average': meanifyStream[i]._average
            }})
        else:
            plt.pcolormesh(dicInput[i]['x'], dicInput[i]['y'],
                           dicInput[i]['_average'], vmin=MIN, vmax=MAX, cmap=CMAP)

    cb = plt.colorbar()
    if colorlabel is None:
        colorlabel = key_second_moment

    cb.set_label(colorlabel, size=22)
    cb.ax.tick_params(labelsize=18)
    plt.xlabel('x (mm)', size=22)
    plt.ylabel('y (mm)', size=22)
    if title is None:
        title = f"DP2 {key_second_moment} | bands: ({bands})"
    plt.title(title, size=18)
    plt.axis('equal')
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    if statisticsMedian:
        median_key = "median"
    else:
        median_key = ""
    output_name = f'{key_second_moment}_2d_{bands}_{int(bin_spacing)}_{median_key}_psfmax{int(psf_max_value)}_snr{int(snr_min)}'
    plt.savefig(os.path.join(repOutPlot, f'{output_name}.png'))

    if pklInput is None:
        pklFile = open(os.path.join(repOutPlot, f'{output_name}.pkl'), 'wb')
        pickle.dump(dicMeanifyPlot, pklFile)
        pklFile.close()


def main():

    parser = argparse.ArgumentParser(description="Focal plane map of PSF second moment residuals")
    parser.add_argument('--bands', type=str, required=True, help="The band(s) to process (e.g., y, g, r, i, z, u, ugrizy)")
    parser.add_argument('--visitMappingFile', type=str, required=True, help="Path to visit_parquet_mapping.pkl file")

    parser.add_argument('--key_second_moment', type=str, default='dT_T', help='second moment key')
    parser.add_argument('--bin_spacing', type=float, default=150, help='bin size in pixels')
    parser.add_argument('--psf_max_value', type=float, default=0, help='exclude PSFs with max pixel value below this (e-)')
    parser.add_argument('--snr_min', type=float, default=0, help='minimum SNR threshold (default: 0 = no cut)')
    parser.add_argument('--colorScale', type=float, default=0.005, help='Min/Max of color scale')
    parser.add_argument('--autoColorScaleCst', type=float, default=2., help='Number of sigma for auto color scale')
    parser.add_argument('--repOutPlot', type=str, default='plots/', help='Output directory for plots')
    parser.add_argument('--pklInput', type=str, default=None, help='Pre-computed pickle to redo plot only')

    parser.add_argument('--autoColorScale', action='store_true')
    parser.add_argument('--statisticsMedian', action='store_true')

    args = parser.parse_args()

    plot_FoV_second_Moment(bands=args.bands, visitMappingFile=args.visitMappingFile,
                           repOutPlot=args.repOutPlot,
                           key_second_moment=args.key_second_moment, bin_spacing=args.bin_spacing,
                           colorScale=args.colorScale, autoColorScale=args.autoColorScale,
                           autoColorScaleCst=args.autoColorScaleCst, statisticsMedian=args.statisticsMedian,
                           colorlabel=None, title=None, pklInput=args.pklInput, psf_max_value=args.psf_max_value,
                           snr_min=args.snr_min)


if __name__ == "__main__":
    main()
