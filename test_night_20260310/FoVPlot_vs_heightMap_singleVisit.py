#!/usr/bin/env python
"""
Plot single visit PSF second moment correlation with SLAC height map.
Based on FoVPlot_vs_heightMap.py but for individual visits.

Uses butler.get() with column selection for fast S3/embargo data access.
"""

import numpy as np
import re
from sklearn.neighbors import KNeighborsRegressor
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import treegp
from lsst.daf.butler import Butler
import lsst.afw.cameraGeom as cameraGeom
from lsst.obs.lsst import LsstCam
from astropy.io import fits
from astropy.table import Table
import os
import argparse

camera = LsstCam.getCamera()

SELECTED_COLUMNS = [
    'slot_Shape_xx', 'slot_Shape_yy', 'slot_Shape_xy',
    'slot_PsfShape_xx', 'slot_PsfShape_xy', 'slot_PsfShape_yy',
    'slot_Centroid_x', 'slot_Centroid_y',
    'psf_max_value', 'calib_psf_candidate',
]


def load_visit_data_from_butler(butler, visit, detector, collection):
    """Load visit data using butler.get() with column selection for S3/embargo data."""
    # Fast access with column selection
    ref = butler.query_datasets("single_visit_star_unstandardized",
                                data_id={"instrument": 'LSSTCam', "visit": visit, "detector": detector},
                                collections=collection)[0]
    table = butler.get(ref, parameters={"columns": SELECTED_COLUMNS}, storageClass="DataFrame")

    # Filter to PSF candidates only
    table = table[table['calib_psf_candidate']]

    slot_Shape_xx = table['slot_Shape_xx'].to_numpy()
    slot_Shape_yy = table['slot_Shape_yy'].to_numpy()
    slot_Shape_xy = table['slot_Shape_xy'].to_numpy()
    slot_PsfShape_xx = table['slot_PsfShape_xx'].to_numpy()
    slot_PsfShape_yy = table['slot_PsfShape_yy'].to_numpy()
    slot_PsfShape_xy = table['slot_PsfShape_xy'].to_numpy()

    T_src = slot_Shape_xx + slot_Shape_yy
    e1_src = (slot_Shape_xx - slot_Shape_yy) / T_src
    e2_src = 2 * slot_Shape_xy / T_src

    T_psf = slot_PsfShape_xx + slot_PsfShape_yy
    e1_psf = (slot_PsfShape_xx - slot_PsfShape_yy) / T_psf
    e2_psf = 2 * slot_PsfShape_xy / T_psf

    return {
        'ixx_src': slot_Shape_xx,
        'iyy_src': slot_Shape_yy,
        'ixy_src': slot_Shape_xy,
        'T_src': T_src,
        'e1_src': e1_src,
        'e2_src': e2_src,
        'dT_T': (T_src - T_psf) / T_src,
        'de1': e1_src - e1_psf,
        'de2': e2_src - e2_psf,
        'xCCD': table['slot_Centroid_x'].to_numpy(),
        'yCCD': table['slot_Centroid_y'].to_numpy(),
        'psf_max_value': table['psf_max_value'].to_numpy(),
    }


def pixel_to_focal(x, y, det):
    """Convert pixel coordinates to focal plane coordinates (mm)."""
    tx = det.getTransform(cameraGeom.PIXELS, cameraGeom.FOCAL_PLANE)
    fpx, fpy = tx.getMapping().applyForward(np.vstack((x, y)))
    return fpx.ravel(), fpy.ravel()


def make_metrology_table(file, rsid=None):
    """Make an astropy table of the height measurement data from SLAC."""
    rows = []
    with fits.open(file) as hdulist:
        for hdu in hdulist:
            if isinstance(hdu, fits.BinTableHDU):
                table = Table(hdu.data)
                extname = hdu.header['EXTNAME']
                if rsid is not None:
                    if extname == rsid:
                        extname = re.sub(r'(R\d\d)(S\d\d)', r'\1_\2', extname)
                        for x, y, z_mod, z_meas in zip(table['X_CCS'], table['Y_CCS'],
                                                        table['Z_CCS_MODEL'], table['Z_CCS_MEASURED']):
                            rows.append([y, x, z_mod, z_meas, extname])  # fpx=y, fpy=x
                else:
                    if re.fullmatch(r'R\d\dS\d\d', extname):
                        extname = re.sub(r'(R\d\d)(S\d\d)', r'\1_\2', extname)
                        for x, y, z_mod, z_meas in zip(table['X_CCS'], table['Y_CCS'],
                                                        table['Z_CCS_MODEL'], table['Z_CCS_MEASURED']):
                            rows.append([y, x, z_mod, z_meas, extname])

    bigtable = Table(rows=rows, names=['fpx', 'fpy', 'z_mod', 'z_meas', 'det'])
    return bigtable


def plot_single_visit_heightmap(visit, butler, collection, fitHeightMap,
                                 secondMomentKey='T', bin_spacing=150,
                                 repOutPlot='plots/', subtract_focal_plane_mean=True):
    """
    Plot correlation between height map and PSF second moments for a single visit.

    Parameters
    ----------
    visit : int
        Visit ID
    butler : Butler
        Butler instance
    collection : str
        Collection to query
    fitHeightMap : str
        Path to SLAC height map FITS file
    secondMomentKey : str
        Key to plot: 'T', 'e1', 'e2', 'dT', 'de1', 'de2'
    bin_spacing : float
        Bin spacing in pixels for treegp meanify
    repOutPlot : str
        Output directory
    subtract_focal_plane_mean : bool
        If True, subtract focal plane mean from both height and second moments
    """

    if secondMomentKey not in ['T', 'e1', 'e2', 'dT', 'de1', 'de2']:
        raise ValueError(f'Invalid secondMomentKey: {secondMomentKey}')

    # Load SLAC height map
    print("Loading SLAC height map...")
    tableSLAC = make_metrology_table(file=fitHeightMap, rsid=None)

    # Get list of available detectors for this visit
    print(f"Querying detectors for visit {visit}...")
    dsrefs = list(butler.registry.queryDatasets(
        "single_visit_star_unstandardized",
        instrument="LSSTCam", visit=visit,
        collections=collection
    ))

    ccdIds = sorted(set(dsr.dataId["detector"] for dsr in dsrefs))
    print(f"Found {len(ccdIds)} detectors")

    if len(ccdIds) == 0:
        print("No detectors found!")
        return

    # Setup plot
    plt.figure(figsize=(18, 16))
    plt.subplots_adjust(top=0.95, wspace=0.3, hspace=0.25, right=0.98, left=0.08, bottom=0.06)

    # Collect data for all CCDs
    meanify = {}
    all_data = {}

    # First pass: load all data and compute focal plane mean if needed
    print("Loading data from all detectors...")
    all_field_values = []

    for ccd in tqdm(ccdIds, desc="Loading"):
        try:
            data = load_visit_data_from_butler(butler, visit, ccd, collection)
            all_data[ccd] = data

            # Compute field values
            if secondMomentKey in ['T', 'dT']:
                field = data['T_src']
            elif secondMomentKey in ['e1', 'de1']:
                field = data['e1_src']
            elif secondMomentKey in ['e2', 'de2']:
                field = data['e2_src']

            all_field_values.extend(field)
        except Exception as e:
            print(f"  Skipping detector {ccd}: {e}")

    if len(all_data) == 0:
        print("No data loaded!")
        return

    # Compute focal plane mean
    if subtract_focal_plane_mean:
        focal_plane_mean = np.nanmean(all_field_values)
        mean_height_fp = np.mean(np.array(tableSLAC['z_meas']))
        print(f"Focal plane mean: {focal_plane_mean:.4f}, Height mean: {mean_height_fp:.6f} mm")
    else:
        focal_plane_mean = 0
        mean_height_fp = 0

    # Second pass: build meanify and collect correlation data
    TTFoV = []
    ZZFoV = []

    for ccd in tqdm(all_data.keys(), desc="Processing"):
        data = all_data[ccd]

        # Initialize meanify for this CCD
        meanify[ccd] = treegp.meanify(bin_spacing=bin_spacing, statistics="median")

        # Compute field
        if secondMomentKey in ['T', 'dT']:
            field = data['T_src']
        elif secondMomentKey in ['e1', 'de1']:
            field = data['e1_src']
        elif secondMomentKey in ['e2', 'de2']:
            field = data['e2_src']

        # Subtract mean
        if subtract_focal_plane_mean:
            field = field - focal_plane_mean
        else:
            field = field - np.mean(field)

        coord = np.array([data['xCCD'], data['yCCD']]).T
        meanify[ccd].add_field(coord, field)

    # Build plots
    print("Building plots...")

    for ccd in tqdm(meanify.keys(), desc="Plotting"):
        meanify[ccd].meanify()

        # Plot 1: Height map (top left)
        plt.subplot(2, 2, 1)
        FiltDet = np.array(tableSLAC['det']) == camera[ccd].getName()
        coordSLAC = np.array([np.array(tableSLAC['fpx'])[FiltDet],
                              np.array(tableSLAC['fpy'])[FiltDet]]).T

        if subtract_focal_plane_mean:
            heightSLAC = np.array(tableSLAC['z_meas'])[FiltDet] - mean_height_fp
        else:
            heightSLAC = np.array(tableSLAC['z_meas'])[FiltDet] - np.mean(np.array(tableSLAC['z_meas'])[FiltDet])

        plt.scatter(coordSLAC[:, 0], coordSLAC[:, 1], s=12, marker='s',
                    c=heightSLAC, cmap=plt.cm.seismic, vmin=-0.005, vmax=0.005)

        # Plot 2: Second moment field (bottom left)
        plt.subplot(2, 2, 3)
        x, y = np.meshgrid(meanify[ccd]._xedge, meanify[ccd]._yedge)
        nBin0, nBin1 = np.shape(x)[0], np.shape(x)[1]
        x_flat = x.reshape(nBin0 * nBin1)
        y_flat = y.reshape(nBin0 * nBin1)
        x_fp, y_fp = pixel_to_focal(x_flat, y_flat, camera[ccd])
        x_fp = x_fp.reshape((nBin0, nBin1))
        y_fp = y_fp.reshape((nBin0, nBin1))

        # Color scale based on key
        if secondMomentKey in ['T', 'dT']:
            MAX = 0.5
            colorlabel = "T - <T> (pixel$^2$)"
        else:
            MAX = 0.05
            colorlabel = f"{secondMomentKey} - <{secondMomentKey}>"

        plt.pcolormesh(x_fp, y_fp, meanify[ccd]._average, vmin=-MAX, vmax=MAX, cmap=plt.cm.seismic)

        # Plot 3: Correlation (bottom right)
        plt.subplot(2, 2, 4)
        try:
            CoordSubmit = meanify[ccd].coords0
            csx, csy = pixel_to_focal(CoordSubmit[:, 0], CoordSubmit[:, 1], camera[ccd])
            CoordSubmit = np.array([csx, csy]).T
            PsfSubmit = meanify[ccd].params0

            knn = KNeighborsRegressor(n_neighbors=20)
            knn.fit(coordSLAC, heightSLAC)
            predict = knn.predict(CoordSubmit)

            TTFoV.append(PsfSubmit)
            ZZFoV.append(predict)

            plt.scatter(predict, PsfSubmit, color='b', s=2, alpha=0.5)
        except Exception as e:
            print(f"  KNN failed for detector {ccd}: {e}")

    # Finalize plots
    plt.subplot(2, 2, 1)
    cb = plt.colorbar()
    plt.axis('equal')
    cb.set_label("z - <z> (mm)", size=18)
    cb.ax.tick_params(labelsize=14)
    plt.xlabel('x (mm)', size=18)
    plt.ylabel('y (mm)', size=18)
    plt.title("Height map from SLAC", size=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.subplot(2, 2, 3)
    cb = plt.colorbar()
    cb.set_label(colorlabel, size=18)
    cb.ax.tick_params(labelsize=14)
    plt.xlabel('x (mm)', size=18)
    plt.ylabel('y (mm)', size=18)
    plt.title(f"Visit {visit} | {secondMomentKey}", size=16)
    plt.axis('equal')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # Correlation plot labels
    plt.subplot(2, 2, 4)
    if len(TTFoV) > 0:
        PsfAll = np.concatenate(TTFoV)
        HeightAll = np.concatenate(ZZFoV)

        valid = np.isfinite(PsfAll) & np.isfinite(HeightAll)
        if np.sum(valid) > 10:
            rho = np.corrcoef(HeightAll[valid], PsfAll[valid])[0, 1]
            plt.title(f"Correlation: $\\rho$ = {rho:.3f}", size=16)

    plt.ylabel(colorlabel, size=18)
    plt.xlabel("z - <z> (mm)", size=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    if secondMomentKey in ['T', 'dT']:
        plt.ylim(-1.5 * MAX, 1.5 * MAX)
    else:
        plt.ylim(-1.5 * MAX, 1.5 * MAX)
    plt.xlim(-0.015, 0.015)

    ylim = plt.ylim()
    xlim = plt.xlim()
    plt.plot([0, 0], ylim, 'k--', alpha=0.5)
    plt.plot(xlim, [0, 0], 'k--', alpha=0.5)
    plt.xlim(xlim)
    plt.ylim(ylim)

    # Save
    os.makedirs(repOutPlot, exist_ok=True)
    outfile = os.path.join(repOutPlot, f'heightmap_vs_{secondMomentKey}_visit{visit}.png')
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {outfile}")


def main():
    defaultFitHeightMap = "/sdf/home/l/leget/rubin-user/lsst_dev/tickets/dp2_psf/data/LSST_FP_cold_b_measurement_4col_bysurface.fits"
    defaultCollection = "u/leget/LSSTCam/HeightMapCorrelation20260311"
    defaultRepOutPlot = "/sdf/home/l/leget/rubin-user/lsst_dev/tickets/dp2_psf/test_night_20260310/plots/"

    parser = argparse.ArgumentParser(description="Single visit height map correlation plot")
    parser.add_argument('--visit', type=int, required=True, help="Visit ID")
    parser.add_argument('--collection', type=str, default=defaultCollection,
                        help="Butler collection")
    parser.add_argument('--fitHeightMap', type=str, default=defaultFitHeightMap,
                        help="Path to SLAC height map FITS file")
    parser.add_argument('--secondMomentKey', type=str, default='T',
                        help="Second moment key: T, e1, e2, dT, de1, de2")
    parser.add_argument('--bin_spacing', type=float, default=150,
                        help="Bin spacing in pixels")
    parser.add_argument('--repOutPlot', type=str, default=defaultRepOutPlot,
                        help="Output directory")
    parser.add_argument('--subtract_focal_plane_mean', action='store_true',
                        help="Subtract focal plane mean (default: True)")
    parser.add_argument('--no_subtract_focal_plane_mean', action='store_true',
                        help="Do not subtract focal plane mean")

    args = parser.parse_args()

    subtract_fp_mean = not args.no_subtract_focal_plane_mean

    butler = Butler('/repo/embargo')

    plot_single_visit_heightmap(
        visit=args.visit,
        butler=butler,
        collection=args.collection,
        fitHeightMap=args.fitHeightMap,
        secondMomentKey=args.secondMomentKey,
        bin_spacing=args.bin_spacing,
        repOutPlot=args.repOutPlot,
        subtract_focal_plane_mean=subtract_fp_mean,
    )


if __name__ == "__main__":
    main()
