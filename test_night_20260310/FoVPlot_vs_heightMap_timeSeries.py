#!/usr/bin/env python
"""
Time series video of PSF second moment correlation with SLAC height map.
Each frame is a single visit from the night.

Produces 4 subplots per frame:
- (2,2,1): Height map from SLAC
- (2,2,2): Correlation coefficient vs visit ID (last 4 digits), updated as we go
- (2,2,3): Second moments field
- (2,2,4): Scatter plot (height vs second moment)
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
import subprocess
import shutil
import gc

camera = LsstCam.getCamera()

SELECTED_COLUMNS = [
    'slot_Shape_xx', 'slot_Shape_yy', 'slot_Shape_xy',
    'slot_PsfShape_xx', 'slot_PsfShape_xy', 'slot_PsfShape_yy',
    'slot_Centroid_x', 'slot_Centroid_y',
    'psf_max_value', 'calib_psf_candidate',
]


def load_visit_data_from_butler(butler, visit, detector, collection):
    """Load visit data using butler.get() with column selection."""
    ref = butler.query_datasets("single_visit_star_unstandardized",
                                data_id={"instrument": 'LSSTCam', "visit": visit, "detector": detector},
                                collections=collection)[0]
    table = butler.get(ref, parameters={"columns": SELECTED_COLUMNS}, storageClass="DataFrame")
    table = table[table['calib_psf_candidate']]

    slot_Shape_xx = table['slot_Shape_xx'].to_numpy()
    slot_Shape_yy = table['slot_Shape_yy'].to_numpy()
    slot_Shape_xy = table['slot_Shape_xy'].to_numpy()

    T_src = slot_Shape_xx + slot_Shape_yy
    e1_src = (slot_Shape_xx - slot_Shape_yy) / T_src
    e2_src = 2 * slot_Shape_xy / T_src

    return {
        'T_src': T_src,
        'e1_src': e1_src,
        'e2_src': e2_src,
        'xCCD': table['slot_Centroid_x'].to_numpy(),
        'yCCD': table['slot_Centroid_y'].to_numpy(),
    }


def pixel_to_focal(x, y, det):
    """Convert pixel coordinates to focal plane coordinates (mm)."""
    tx = det.getTransform(cameraGeom.PIXELS, cameraGeom.FOCAL_PLANE)
    fpx, fpy = tx.getMapping().applyForward(np.vstack((x, y)))
    return fpx.ravel(), fpy.ravel()


def make_metrology_table(file):
    """Make an astropy table of the height measurement data from SLAC."""
    rows = []
    with fits.open(file) as hdulist:
        for hdu in hdulist:
            if isinstance(hdu, fits.BinTableHDU):
                table = Table(hdu.data)
                extname = hdu.header['EXTNAME']
                if re.fullmatch(r'R\d\dS\d\d', extname):
                    extname = re.sub(r'(R\d\d)(S\d\d)', r'\1_\2', extname)
                    for x, y, z_mod, z_meas in zip(table['X_CCS'], table['Y_CCS'],
                                                    table['Z_CCS_MODEL'], table['Z_CCS_MEASURED']):
                        rows.append([y, x, z_mod, z_meas, extname])
    return Table(rows=rows, names=['fpx', 'fpy', 'z_mod', 'z_meas', 'det'])


def process_single_visit(visit, butler, collection, tableSLAC, secondMomentKey, bin_spacing):
    """Process a single visit and return correlation data."""

    # Get list of available detectors
    dsrefs = list(butler.registry.queryDatasets(
        "single_visit_star_unstandardized",
        instrument="LSSTCam", visit=visit,
        collections=collection
    ))
    ccdIds = sorted(set(dsr.dataId["detector"] for dsr in dsrefs))

    if len(ccdIds) == 0:
        return None, None, None, None

    # Load all data
    meanify = {}
    all_data = {}
    all_field_values = []

    for ccd in ccdIds:
        try:
            data = load_visit_data_from_butler(butler, visit, ccd, collection)
            all_data[ccd] = data

            if secondMomentKey in ['T', 'dT']:
                field = data['T_src']
            elif secondMomentKey in ['e1', 'de1']:
                field = data['e1_src']
            elif secondMomentKey in ['e2', 'de2']:
                field = data['e2_src']
            all_field_values.extend(field)
        except:
            continue

    if len(all_data) == 0:
        return None, None, None, None

    # Compute focal plane mean
    focal_plane_mean = np.nanmean(all_field_values)
    mean_height_fp = np.mean(np.array(tableSLAC['z_meas']))

    # Build meanify and collect correlation data
    TTFoV = []
    ZZFoV = []

    for ccd in all_data.keys():
        data = all_data[ccd]
        meanify[ccd] = treegp.meanify(bin_spacing=bin_spacing, statistics="median")

        if secondMomentKey in ['T', 'dT']:
            field = data['T_src']
        elif secondMomentKey in ['e1', 'de1']:
            field = data['e1_src']
        elif secondMomentKey in ['e2', 'de2']:
            field = data['e2_src']

        field = field - focal_plane_mean
        coord = np.array([data['xCCD'], data['yCCD']]).T
        meanify[ccd].add_field(coord, field)

    # Compute correlation
    for ccd in meanify.keys():
        meanify[ccd].meanify()

        FiltDet = np.array(tableSLAC['det']) == camera[ccd].getName()
        coordSLAC = np.array([np.array(tableSLAC['fpx'])[FiltDet],
                              np.array(tableSLAC['fpy'])[FiltDet]]).T
        heightSLAC = np.array(tableSLAC['z_meas'])[FiltDet] - mean_height_fp

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
        except:
            continue

    if len(TTFoV) == 0:
        return None, None, None, None

    PsfAll = np.concatenate(TTFoV)
    HeightAll = np.concatenate(ZZFoV)

    valid = np.isfinite(PsfAll) & np.isfinite(HeightAll)
    if np.sum(valid) < 10:
        return None, None, None, None

    rho = np.corrcoef(HeightAll[valid], PsfAll[valid])[0, 1]

    return meanify, rho, HeightAll[valid], PsfAll[valid]


def create_frame(visit, meanify, rho, height_data, psf_data, tableSLAC,
                 visit_ids_so_far, rho_values_so_far, secondMomentKey, frame_idx, frames_dir):
    """Create a single frame for the video."""

    mean_height_fp = np.mean(np.array(tableSLAC['z_meas']))

    fig = plt.figure(figsize=(18, 16))
    plt.subplots_adjust(top=0.93, wspace=0.3, hspace=0.25, right=0.98, left=0.08, bottom=0.06)

    # Color scale settings
    if secondMomentKey in ['T', 'dT']:
        MAX = 0.5
        colorlabel = "T - <T> (pixel$^2$)"
    else:
        MAX = 0.05
        colorlabel = f"{secondMomentKey} - <{secondMomentKey}>"

    # (2,2,1): Height map
    ax1 = plt.subplot(2, 2, 1)
    for ccd in meanify.keys():
        FiltDet = np.array(tableSLAC['det']) == camera[ccd].getName()
        coordSLAC = np.array([np.array(tableSLAC['fpx'])[FiltDet],
                              np.array(tableSLAC['fpy'])[FiltDet]]).T
        heightSLAC = np.array(tableSLAC['z_meas'])[FiltDet] - mean_height_fp
        ax1.scatter(coordSLAC[:, 0], coordSLAC[:, 1], s=12, marker='s',
                    c=heightSLAC, cmap=plt.cm.seismic, vmin=-0.005, vmax=0.005)

    ax1.set_aspect('equal')
    ax1.set_xlabel('x (mm)', size=18)
    ax1.set_ylabel('y (mm)', size=18)
    ax1.set_title("Height map from SLAC", size=16)
    ax1.tick_params(labelsize=14)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.seismic, norm=plt.Normalize(-0.005, 0.005))
    cb = fig.colorbar(sm, ax=ax1)
    cb.set_label("z - <z> (mm)", size=18)
    cb.ax.tick_params(labelsize=14)

    # (2,2,2): Correlation vs visit ID
    ax2 = plt.subplot(2, 2, 2)
    ax2.scatter(visit_ids_so_far, rho_values_so_far, c='b', s=20)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Visit ID (last 4 digits)', size=18)
    ax2.set_ylabel('Correlation $\\rho$', size=18)
    ax2.set_title(f'Correlation time series (N={len(visit_ids_so_far)})', size=16)
    ax2.tick_params(labelsize=14)
    ax2.set_ylim(-1, 1)
    if len(visit_ids_so_far) > 1:
        ax2.set_xlim(min(visit_ids_so_far) - 5, max(visit_ids_so_far) + 5)

    # (2,2,3): Second moments field
    ax3 = plt.subplot(2, 2, 3)
    for ccd in meanify.keys():
        x, y = np.meshgrid(meanify[ccd]._xedge, meanify[ccd]._yedge)
        nBin0, nBin1 = np.shape(x)[0], np.shape(x)[1]
        x_flat = x.reshape(nBin0 * nBin1)
        y_flat = y.reshape(nBin0 * nBin1)
        x_fp, y_fp = pixel_to_focal(x_flat, y_flat, camera[ccd])
        x_fp = x_fp.reshape((nBin0, nBin1))
        y_fp = y_fp.reshape((nBin0, nBin1))
        ax3.pcolormesh(x_fp, y_fp, meanify[ccd]._average, vmin=-MAX, vmax=MAX, cmap=plt.cm.seismic)

    ax3.set_aspect('equal')
    ax3.set_xlabel('x (mm)', size=18)
    ax3.set_ylabel('y (mm)', size=18)
    ax3.set_title(f"Visit {visit} | {secondMomentKey}", size=16)
    ax3.tick_params(labelsize=14)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.seismic, norm=plt.Normalize(-MAX, MAX))
    cb = fig.colorbar(sm, ax=ax3)
    cb.set_label(colorlabel, size=18)
    cb.ax.tick_params(labelsize=14)

    # (2,2,4): Scatter plot
    ax4 = plt.subplot(2, 2, 4)
    ax4.scatter(height_data, psf_data, c='b', s=2, alpha=0.5)
    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax4.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    ax4.set_xlabel("z - <z> (mm)", size=18)
    ax4.set_ylabel(colorlabel, size=18)
    ax4.set_title(f"Correlation: $\\rho$ = {rho:.3f}", size=16)
    ax4.tick_params(labelsize=14)
    ax4.set_xlim(-0.015, 0.015)
    ax4.set_ylim(-1.5 * MAX, 1.5 * MAX)

    plt.suptitle(f"Night 2026-03-10 | Visit {visit}", fontsize=20, fontweight='bold')

    frame_file = os.path.join(frames_dir, f'frame_{frame_idx:05d}.png')
    fig.savefig(frame_file, dpi=100)
    plt.close(fig)

    return frame_file


def create_time_series_video(visitIds_file, butler, collection, fitHeightMap,
                              secondMomentKey='T', bin_spacing=150,
                              repOutPlot='plots/', fps=5, max_visits=None):
    """Create time series video of height map correlation."""

    # Load visit IDs
    with open(visitIds_file, 'r') as f:
        visits = [int(line.strip()) for line in f if line.strip()]

    print(f"Total visits: {len(visits)}")

    if max_visits is not None:
        visits = visits[:max_visits]
        print(f"Limited to {max_visits} visits for testing")

    # Load SLAC height map once
    print("Loading SLAC height map...")
    tableSLAC = make_metrology_table(file=fitHeightMap)

    # Create frames directory
    frames_dir = os.path.join(repOutPlot, 'frames_temp')
    os.makedirs(frames_dir, exist_ok=True)

    # Track correlation over time
    visit_ids_so_far = []
    rho_values_so_far = []

    frame_idx = 0

    for visit in tqdm(visits, desc="Processing visits"):
        # Get last 4 digits for x-axis
        visit_short = visit % 10000

        # Process visit
        meanify, rho, height_data, psf_data = process_single_visit(
            visit, butler, collection, tableSLAC, secondMomentKey, bin_spacing
        )

        if rho is None:
            print(f"  Skipping visit {visit}: no data")
            continue

        # Update time series
        visit_ids_so_far.append(visit_short)
        rho_values_so_far.append(rho)

        # Create frame
        create_frame(visit, meanify, rho, height_data, psf_data, tableSLAC,
                     visit_ids_so_far, rho_values_so_far, secondMomentKey,
                     frame_idx, frames_dir)

        frame_idx += 1

        # Cleanup
        if frame_idx % 20 == 0:
            gc.collect()

    print(f"\nGenerated {frame_idx} frames")

    if frame_idx == 0:
        print("No frames generated!")
        shutil.rmtree(frames_dir)
        return

    # Create video with ffmpeg
    os.makedirs(repOutPlot, exist_ok=True)
    output_file = os.path.join(repOutPlot, f'heightmap_vs_{secondMomentKey}_timeseries.mp4')

    print(f"Creating video: {output_file}")

    ffmpeg_cmd = [
        'ffmpeg', '-y',
        '-framerate', str(fps),
        '-i', os.path.join(frames_dir, 'frame_%05d.png'),
        '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-crf', '20',
        output_file
    ]

    try:
        subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
        print(f"Video saved: {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"ffmpeg error: {e.stderr.decode()}")
    finally:
        shutil.rmtree(frames_dir)
        print("Cleaned up frames")


def main():
    defaultFitHeightMap = "/sdf/home/l/leget/rubin-user/lsst_dev/tickets/dp2_psf/data/LSST_FP_cold_b_measurement_4col_bysurface.fits"
    defaultCollection = "u/leget/LSSTCam/HeightMapCorrelation20260311"
    defaultRepOutPlot = "/sdf/home/l/leget/rubin-user/lsst_dev/tickets/dp2_psf/test_night_20260310/plots/"
    defaultVisitIdsFile = "/sdf/home/l/leget/rubin-user/lsst_dev/tickets/dp2_psf/test_night_20260310/visitIds.txt"

    parser = argparse.ArgumentParser(description="Time series video of height map correlation")
    parser.add_argument('--visitIds', type=str, default=defaultVisitIdsFile,
                        help="File with visit IDs (one per line)")
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
    parser.add_argument('--fps', type=int, default=5,
                        help="Frames per second for video")
    parser.add_argument('--max_visits', type=int, default=None,
                        help="Maximum number of visits to process (for testing)")

    args = parser.parse_args()

    butler = Butler('/repo/embargo')

    create_time_series_video(
        visitIds_file=args.visitIds,
        butler=butler,
        collection=args.collection,
        fitHeightMap=args.fitHeightMap,
        secondMomentKey=args.secondMomentKey,
        bin_spacing=args.bin_spacing,
        repOutPlot=args.repOutPlot,
        fps=args.fps,
        max_visits=args.max_visits,
    )


if __name__ == "__main__":
    main()
