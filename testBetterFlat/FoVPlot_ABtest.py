#!/usr/bin/env python
"""
A/B testing analysis for flat field comparison.
Plots PSF second moment residuals on a single detector in CCD coordinates.

Uses butler.getURI + polars for fast data access without pre-computed mapping.
"""

import numpy as np
import treegp
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
os.environ["POLARS_MAX_THREADS"] = "1"
import polars as pl
import pickle
import argparse
from lsst.daf.butler import Butler


PARQUET_COLUMNS = [
    'slot_Shape_xx', 'slot_Shape_yy', 'slot_Shape_xy',
    'slot_PsfShape_xx', 'slot_PsfShape_xy', 'slot_PsfShape_yy',
    'coord_ra', 'coord_dec', 'slot_Centroid_x', 'slot_Centroid_y',
    'psf_max_value', 'calib_psf_candidate', 'slot_PsfFlux_mag',
]


def load_visit_data(parquet_path):
    """Load visit data from parquet file and compute derived columns."""
    table = pl.scan_parquet(parquet_path).select(PARQUET_COLUMNS).collect()

    # Filter to PSF candidates only
    table = table.filter(pl.col('calib_psf_candidate'))

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
        'dT_T': (T_src - T_psf) / T_src,
        'de1': e1_src - e1_psf,
        'de2': e2_src - e2_psf,
        'xCCD': table['slot_Centroid_x'].to_numpy(),
        'yCCD': table['slot_Centroid_y'].to_numpy(),
        'psf_max_value': table['psf_max_value'].to_numpy(),
        'mag': table['slot_PsfFlux_mag'].to_numpy(),
    }


def plot_detector_second_moment(butler, collection, detector, visit_file,
                                 key_second_moment='dT_T', bin_spacing=150,
                                 colorScale=0.005, repOutPlot='plots/',
                                 label=None):
    """
    Plot spatial variation of PSF second moments on a single detector.

    Parameters
    ----------
    butler : Butler
        Butler instance (repo already set)
    collection : str
        Collection to query
    detector : int
        Detector ID
    visit_file : str
        Path to file with visit IDs (one per line)
    key_second_moment : str
        Key to plot: 'dT_T', 'de1', 'de2'
    bin_spacing : float
        Bin spacing in pixels
    colorScale : float
        Color scale range
    repOutPlot : str
        Output directory
    label : str
        Label for output files (e.g., 'default', 'newflat')
    """
    # Read visit IDs
    with open(visit_file, 'r') as f:
        visits = [int(line.strip()) for line in f if line.strip()]

    print(f"Processing {len(visits)} visits for detector {detector}, collection: {collection}")

    meanify_obj = treegp.meanify(bin_spacing=bin_spacing, statistics="mean",
                                  bounds=(0, 4100, 0, 4100))

    n_processed = 0
    for visit in tqdm(visits, desc=f"Loading visits ({label})"):
        try:
            uri = butler.getURI("single_visit_star_unstandardized",
                               instrument="LSSTCam", visit=visit, detector=detector,
                               collections=collection)
            data = load_visit_data(uri.geturl())

            coord = np.array([data['xCCD'], data['yCCD']]).T
            meanify_obj.add_field(coord, data[key_second_moment])
            n_processed += 1
        except Exception as e:
            # Visit may not exist in this collection
            pass

    print(f"Processed {n_processed}/{len(visits)} visits")

    if n_processed == 0:
        print("No data found!")
        return None

    meanify_obj.meanify()

    # Plot in CCD coordinates
    plt.figure(figsize=(10, 10))
    x, y = np.meshgrid(meanify_obj._xedge, meanify_obj._yedge)
    plt.pcolormesh(x, y, meanify_obj._average, vmin=-colorScale, vmax=colorScale,
                   cmap=plt.cm.inferno)
    cb = plt.colorbar()
    cb.set_label(key_second_moment, size=18)
    plt.xlabel('x (pixels)', size=18)
    plt.ylabel('y (pixels)', size=18)
    plt.title(f"Detector {detector} | {label} | {key_second_moment} | N={n_processed}", size=14)
    plt.axis('equal')

    outname = f'{key_second_moment}_det{detector}_{label}'
    plt.savefig(os.path.join(repOutPlot, f'{outname}.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Save data for comparison
    result = {
        'x': x, 'y': y,
        '_average': meanify_obj._average,
        'n_visits': n_processed,
        'detector': detector,
        'collection': collection,
    }
    with open(os.path.join(repOutPlot, f'{outname}.pkl'), 'wb') as f:
        pickle.dump(result, f)

    return result


def plot_comparison(result_A, result_B, key_second_moment, repOutPlot, detector):
    """Plot side-by-side comparison and difference."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    vmin, vmax = -0.005, 0.005

    # A (default)
    ax = axes[0]
    im = ax.pcolormesh(result_A['x'], result_A['y'], result_A['_average'],
                       vmin=vmin, vmax=vmax, cmap=plt.cm.inferno)
    ax.set_title(f"Default flat (N={result_A['n_visits']})", size=14)
    ax.set_xlabel('x (pixels)')
    ax.set_ylabel('y (pixels)')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax)

    # B (new flat)
    ax = axes[1]
    im = ax.pcolormesh(result_B['x'], result_B['y'], result_B['_average'],
                       vmin=vmin, vmax=vmax, cmap=plt.cm.inferno)
    ax.set_title(f"New flat (N={result_B['n_visits']})", size=14)
    ax.set_xlabel('x (pixels)')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax)

    # Difference (B - A)
    ax = axes[2]
    diff = result_B['_average'] - result_A['_average']
    diff_scale = 0.002
    im = ax.pcolormesh(result_A['x'], result_A['y'], diff,
                       vmin=-diff_scale, vmax=diff_scale, cmap=plt.cm.inferno)
    ax.set_title(f"New - Default", size=14)
    ax.set_xlabel('x (pixels)')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax)

    plt.suptitle(f"Detector {detector} | {key_second_moment}", size=16)
    plt.tight_layout()
    plt.savefig(os.path.join(repOutPlot, f'{key_second_moment}_det{detector}_comparison.png'),
                dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="A/B test analysis for flat field comparison")
    parser.add_argument('--detector', type=int, default=87, help="Detector ID")
    parser.add_argument('--visit_file', type=str, default='visitIds_flat_test.txt',
                        help="File with visit IDs")
    parser.add_argument('--collection_A', type=str, default='u/leget/LSSTCam/testFlat/ABtest_default',
                        help="Collection A (default flat)")
    parser.add_argument('--collection_B', type=str, default='u/leget/LSSTCam/testFlat/ABtest_newflat',
                        help="Collection B (new flat)")
    parser.add_argument('--key_second_moment', type=str, default='dT_T',
                        help="Second moment key: dT_T, de1, de2")
    parser.add_argument('--bin_spacing', type=float, default=150, help="Bin spacing in pixels")
    parser.add_argument('--colorScale', type=float, default=0.005, help="Color scale")
    parser.add_argument('--repOutPlot', type=str, default='plots/', help="Output directory")

    args = parser.parse_args()

    os.makedirs(args.repOutPlot, exist_ok=True)

    butler = Butler('/repo/main')

    # Process both collections
    result_A = plot_detector_second_moment(
        butler, args.collection_A, args.detector, args.visit_file,
        key_second_moment=args.key_second_moment, bin_spacing=args.bin_spacing,
        colorScale=args.colorScale, repOutPlot=args.repOutPlot, label='default'
    )

    result_B = plot_detector_second_moment(
        butler, args.collection_B, args.detector, args.visit_file,
        key_second_moment=args.key_second_moment, bin_spacing=args.bin_spacing,
        colorScale=args.colorScale, repOutPlot=args.repOutPlot, label='newflat'
    )

    # Plot comparison if both succeeded
    if result_A is not None and result_B is not None:
        plot_comparison(result_A, result_B, args.key_second_moment,
                       args.repOutPlot, args.detector)
        print(f"Comparison plot saved to {args.repOutPlot}")


if __name__ == "__main__":
    main()
