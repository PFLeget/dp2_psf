#!/usr/bin/env python
"""
A/B testing analysis: dT/T, de1, de2 as a function of SNR or psf_max_value.

Compares two collections with overlay plots.
"""

import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
os.environ["POLARS_MAX_THREADS"] = "1"
import polars as pl
import pickle
import argparse
from lsst.daf.butler import Butler


PARQUET_COLUMNS = [
    'slot_Shape_xx', 'slot_Shape_yy', 'slot_Shape_xy',
    'slot_PsfShape_xx', 'slot_PsfShape_xy', 'slot_PsfShape_yy',
    'base_GaussianFlux_instFlux', 'base_GaussianFlux_instFluxErr',
    'psf_max_value', 'calib_psf_candidate',
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

    # Compute SNR
    flux = table['base_GaussianFlux_instFlux'].to_numpy()
    flux_err = table['base_GaussianFlux_instFluxErr'].to_numpy()
    snr = flux / flux_err

    return {
        'dT_T': (T_src - T_psf) / T_src,
        'de1': e1_src - e1_psf,
        'de2': e2_src - e2_psf,
        'snr': snr,
        'psf_max_value': table['psf_max_value'].to_numpy(),
    }


class meanify1D_wrms():
    """1D binned average with weighted RMS. O(1) memory."""
    def __init__(self, bin_spacing=0.3, x_min=0, x_max=1000):
        self.bin_spacing = bin_spacing
        self.x_min = x_min
        self.x_max = x_max

        self.nbin = int((x_max - x_min) / bin_spacing) + 1
        self.binning = np.linspace(x_min, x_max, self.nbin)

        self._sum = np.zeros(self.nbin - 1)
        self._sum_sq = np.zeros(self.nbin - 1)
        self._count = np.zeros(self.nbin - 1)

        self.x0 = self.binning[:-1] + (self.binning[1] - self.binning[0]) / 2.0

    def add_data(self, coord, param):
        """Add new data - accumulates directly into bins."""
        valid = np.isfinite(coord) & np.isfinite(param)
        coord = coord[valid]
        param = param[valid]

        bin_indices = np.digitize(coord, self.binning) - 1
        valid_bins = (bin_indices >= 0) & (bin_indices < self.nbin - 1)
        bin_indices = bin_indices[valid_bins]
        param = param[valid_bins]

        np.add.at(self._sum, bin_indices, param)
        np.add.at(self._sum_sq, bin_indices, param ** 2)
        np.add.at(self._count, bin_indices, 1)

    def meanify(self):
        """Compute final statistics from accumulated sums."""
        with np.errstate(divide='ignore', invalid='ignore'):
            self.average = self._sum / self._count
            variance = (self._sum_sq / self._count) - (self.average ** 2)
            self.std = np.sqrt(variance)
            self.count = self._count


def process_collection(butler, collection, visit_file, detector, x_col, x_min, x_max, bin_spacing, label):
    """Process a single collection and return binned statistics."""
    # Read visit IDs
    with open(visit_file, 'r') as f:
        visits = [int(line.strip()) for line in f if line.strip()]

    print(f"Processing {len(visits)} visits for collection: {collection} ({label})")

    # Initialize meanify objects
    meanify_dT = meanify1D_wrms(bin_spacing=bin_spacing, x_min=x_min, x_max=x_max)
    meanify_de1 = meanify1D_wrms(bin_spacing=bin_spacing, x_min=x_min, x_max=x_max)
    meanify_de2 = meanify1D_wrms(bin_spacing=bin_spacing, x_min=x_min, x_max=x_max)

    all_x = []
    n_processed = 0

    for visit in tqdm(visits, desc=f"Loading visits ({label})"):
        try:
            uri = butler.getURI("single_visit_star_unstandardized",
                               instrument="LSSTCam", visit=visit, detector=detector,
                               collections=collection)
            data = load_visit_data(uri.geturl())

            x_data = data[x_col]
            valid = np.isfinite(x_data) & np.isfinite(data['dT_T'])
            valid &= np.isfinite(data['de1']) & np.isfinite(data['de2'])
            valid &= (x_data > x_min) & (x_data < x_max)

            if np.sum(valid) > 0:
                x_vals = x_data[valid]
                all_x.append(x_vals)
                meanify_dT.add_data(x_vals, data['dT_T'][valid])
                meanify_de1.add_data(x_vals, data['de1'][valid])
                meanify_de2.add_data(x_vals, data['de2'][valid])
                n_processed += 1
        except Exception as e:
            pass

    print(f"Processed {n_processed}/{len(visits)} visits")

    if n_processed == 0:
        return None

    meanify_dT.meanify()
    meanify_de1.meanify()
    meanify_de2.meanify()

    all_x = np.concatenate(all_x) if all_x else np.array([])

    return {
        'label': label,
        'collection': collection,
        'n_visits': n_processed,
        'n_stars': len(all_x),
        'all_x': all_x,
        'x_bins': meanify_dT.x0,
        'binning': meanify_dT.binning,
        'dT_T': {'mean': meanify_dT.average, 'std': meanify_dT.std, 'count': meanify_dT.count},
        'de1': {'mean': meanify_de1.average, 'std': meanify_de1.std, 'count': meanify_de1.count},
        'de2': {'mean': meanify_de2.average, 'std': meanify_de2.std, 'count': meanify_de2.count},
    }


def plot_comparison(result_A, result_B, y_key, ylabel, x_col, x_label, x_min, x_max,
                    repOutPlot, detector, show_requirements=False):
    """Create overlay comparison plot for a single quantity."""
    fig = plt.figure(figsize=(12, 10))
    plt.subplots_adjust(left=0.12, bottom=0.08, top=0.92, right=0.95, hspace=0)
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 2])

    xlim = (x_min, x_max)

    # Top panel: x distribution (both collections)
    ax1 = plt.subplot(gs[0])
    ax1.hist(result_A['all_x'], bins=result_A['binning'], color='blue', alpha=0.5,
             edgecolor='blue', linewidth=0.5, label=f"{result_A['label']} (N={result_A['n_stars']:,})")
    ax1.hist(result_B['all_x'], bins=result_B['binning'], color='red', alpha=0.5,
             edgecolor='red', linewidth=0.5, label=f"{result_B['label']} (N={result_B['n_stars']:,})")
    ax1.set_yscale('log')
    ax1.set_ylabel('# stars', fontsize=14)
    ax1.set_xlim(xlim)
    ax1.tick_params(labelbottom=False)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.set_title(f"Detector {detector} | A/B comparison | {y_key}", fontsize=14)

    # Bottom panel: overlay comparison
    ax2 = plt.subplot(gs[1])

    # Plot A
    data_A = result_A[y_key]
    valid_A = np.isfinite(data_A['mean'])
    ax2.errorbar(result_A['x_bins'][valid_A], data_A['mean'][valid_A],
                 yerr=data_A['std'][valid_A] / np.sqrt(data_A['count'][valid_A]),
                 fmt='o-', capsize=2, markersize=4, color='blue',
                 label=f"{result_A['label']} (N_visits={result_A['n_visits']})")

    # Plot B
    data_B = result_B[y_key]
    valid_B = np.isfinite(data_B['mean'])
    ax2.errorbar(result_B['x_bins'][valid_B], data_B['mean'][valid_B],
                 yerr=data_B['std'][valid_B] / np.sqrt(data_B['count'][valid_B]),
                 fmt='s-', capsize=2, markersize=4, color='red',
                 label=f"{result_B['label']} (N_visits={result_B['n_visits']})")

    ax2.axhline(0, color='k', linestyle='--', linewidth=1, zorder=1)

    if show_requirements:
        ax2.fill_between(xlim, -0.004, 0.004, color='g', alpha=0.2, zorder=0, label='0.4% requirement')
        ax2.fill_between(xlim, -0.001, 0.001, color='g', alpha=0.3, zorder=0, label='0.1% goal')

    ax2.set_xlim(xlim)
    ax2.set_ylim(-0.02, 0.02)
    ax2.set_xlabel(x_label, fontsize=14)
    ax2.set_ylabel(ylabel, fontsize=14)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)

    output_file = os.path.join(repOutPlot, f'{x_col}_vs_{y_key}_det{detector}_comparison.png')
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"Saved: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="A/B test: dT/T vs SNR or psf_max_value")
    parser.add_argument('--detector', type=int, default=87, help="Detector ID")
    parser.add_argument('--visit_file', type=str, default='visitIds_flat_test.txt',
                        help="File with visit IDs")
    parser.add_argument('--collection_A', type=str, default='u/leget/LSSTCam/testFlat/ABtest_default',
                        help="Collection A (default)")
    parser.add_argument('--collection_B', type=str, default='u/leget/LSSTCam/testFlat/ABtest_newflat',
                        help="Collection B (new)")
    parser.add_argument('--label_A', type=str, default='default', help="Label for collection A")
    parser.add_argument('--label_B', type=str, default='newflat', help="Label for collection B")
    parser.add_argument('--repo', type=str, default='/repo/main', help="Butler repo")
    parser.add_argument('--repOutPlot', type=str, default='plots/', help="Output directory")

    # X-axis options
    parser.add_argument('--use_psf_max', action='store_true',
                        help="Use psf_max_value instead of SNR for x-axis")
    parser.add_argument('--x_min', type=float, default=None, help="Minimum x value")
    parser.add_argument('--x_max', type=float, default=None, help="Maximum x value")
    parser.add_argument('--bin_spacing', type=float, default=None, help="Bin spacing")

    args = parser.parse_args()

    # Set defaults based on x-axis type
    if args.use_psf_max:
        x_min = args.x_min if args.x_min is not None else 0
        x_max = args.x_max if args.x_max is not None else 100000
        bin_spacing = args.bin_spacing if args.bin_spacing is not None else 2000
        x_col = 'psf_max_value'
        x_label = 'psf pixel max value ($\\text{e}^{-}$)'
    else:
        x_min = args.x_min if args.x_min is not None else 10
        x_max = args.x_max if args.x_max is not None else 1000
        bin_spacing = args.bin_spacing if args.bin_spacing is not None else 20
        x_col = 'snr'
        x_label = 'SNR'

    os.makedirs(args.repOutPlot, exist_ok=True)

    butler = Butler(args.repo)

    # Process both collections
    result_A = process_collection(butler, args.collection_A, args.visit_file, args.detector,
                                   x_col, x_min, x_max, bin_spacing, args.label_A)
    result_B = process_collection(butler, args.collection_B, args.visit_file, args.detector,
                                   x_col, x_min, x_max, bin_spacing, args.label_B)

    if result_A is None or result_B is None:
        print("One or both collections have no data!")
        return

    # Create comparison plots
    plot_comparison(result_A, result_B, 'dT_T', '$\\langle \\delta T / T \\rangle$',
                    x_col, x_label, x_min, x_max, args.repOutPlot, args.detector,
                    show_requirements=True)
    plot_comparison(result_A, result_B, 'de1', '$\\langle \\delta e_1 \\rangle$',
                    x_col, x_label, x_min, x_max, args.repOutPlot, args.detector,
                    show_requirements=False)
    plot_comparison(result_A, result_B, 'de2', '$\\langle \\delta e_2 \\rangle$',
                    x_col, x_label, x_min, x_max, args.repOutPlot, args.detector,
                    show_requirements=False)

    # Save combined results
    results = {'A': result_A, 'B': result_B, 'x_col': x_col}
    with open(os.path.join(args.repOutPlot, f'{x_col}_vs_residuals_det{args.detector}_results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    print(f"Saved results to {args.repOutPlot}")


if __name__ == "__main__":
    main()
