#!/usr/bin/env python
"""
SNR vs dT/T analysis for COSMOS DDF only.

Uses Butler visit_table to select COSMOS visits.
"""

import numpy as np
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import os
os.environ["POLARS_MAX_THREADS"] = "1"
import polars

import pickle
import argparse

from lsst.daf.butler import Butler
from astropy.coordinates import SkyCoord
import astropy.units as u


COSMOS_RA = 150.12  # degrees
COSMOS_DEC = 2.21   # degrees

# Detector types
ITL_DETECTORS = np.concatenate((np.arange(0, 36), np.arange(72, 81), np.arange(162, 189)))
E2V_DETECTORS = np.concatenate((np.arange(36, 72), np.arange(81, 162)))


# Columns to read from parquet files
PARQUET_COLUMNS = [
    'slot_Shape_xx', 'slot_Shape_yy', 'slot_Shape_xy',
    'slot_PsfShape_xx', 'slot_PsfShape_xy', 'slot_PsfShape_yy',
    'base_GaussianFlux_instFlux', 'base_GaussianFlux_instFluxErr',
    'coord_ra', 'coord_dec',
    'detector', 'calib_psf_reserved', 'psf_max_value',
]


def angular_distance(ra, dec, ra_center=0, dec_center=0, unit='arcmin'):
    """Compute angular distance from a center point."""
    coords = SkyCoord(ra=ra, dec=dec, unit='deg')
    center = SkyCoord(ra=ra_center, dec=dec_center, unit='deg')
    sep = coords.separation(center)

    if unit == 'deg':
        return sep.deg
    elif unit == 'arcmin':
        return sep.arcmin
    elif unit == 'arcsec':
        return sep.arcsec
    else:
        return sep


def get_cosmos_visits(repo, collection, band, max_sep_deg=0.5):
    """Get visit IDs that overlap COSMOS using Butler visit_table."""
    butler = Butler(repo, collections=collection)
    visitTable = butler.get('visit_table', instrument="LSSTCam")

    angular_sep = angular_distance(
        visitTable['ra'], visitTable['dec'],
        ra_center=COSMOS_RA, dec_center=COSMOS_DEC, unit='deg')

    mask = angular_sep < max_sep_deg
    mask &= visitTable['band'] == band
    visitId_cosmos = visitTable['visitId'][mask]

    return list(visitId_cosmos)


def load_visit_data(parquet_path, detector_filter=None):
    """Load visit data from parquet file.

    Parameters
    ----------
    parquet_path : str
        Path to parquet file
    detector_filter : str or None
        'e2v', 'itl', or None (all detectors)
    """
    table = polars.scan_parquet(parquet_path).select(PARQUET_COLUMNS).collect()

    # Filter by detector type
    if detector_filter is not None:
        detector_col = table['detector'].to_numpy()
        if detector_filter == 'e2v':
            mask = np.isin(detector_col, E2V_DETECTORS)
        elif detector_filter == 'itl':
            mask = np.isin(detector_col, ITL_DETECTORS)
        else:
            raise ValueError(f"Unknown detector_filter: {detector_filter}")
        table = table.filter(polars.Series(mask))

    slot_Shape_xx = table['slot_Shape_xx'].to_numpy()
    slot_Shape_yy = table['slot_Shape_yy'].to_numpy()
    slot_PsfShape_xx = table['slot_PsfShape_xx'].to_numpy()
    slot_PsfShape_yy = table['slot_PsfShape_yy'].to_numpy()

    # Compute derived quantities
    T_src = slot_Shape_xx + slot_Shape_yy
    T_psf = slot_PsfShape_xx + slot_PsfShape_yy
    dT_T = (T_src - T_psf) / T_src

    # Compute SNR
    flux = table['base_GaussianFlux_instFlux'].to_numpy()
    flux_err = table['base_GaussianFlux_instFluxErr'].to_numpy()
    snr = flux / flux_err

    return {
        'dT_T': dT_T,
        'snr': snr,
        'psf_max_value': table['psf_max_value'].to_numpy(),
        'ra': table['coord_ra'].to_numpy(),
        'dec': table['coord_dec'].to_numpy(),
        'detector': table['detector'].to_numpy(),
        'calib_psf_reserved': table['calib_psf_reserved'].to_numpy(),
    }


class meanify1D_wrms():
    """
    Take data, build a 1D average with weighted RMS.
    O(1) memory implementation - keeps running sum/count per bin.
    """
    def __init__(self, bin_spacing=0.3, x_min=0, x_max=1000):
        self.bin_spacing = bin_spacing
        self.x_min = x_min
        self.x_max = x_max

        # Pre-allocate bins
        self.nbin = int((x_max - x_min) / bin_spacing) + 1
        self.binning = np.linspace(x_min, x_max, self.nbin)

        # Running statistics per bin (O(1) memory)
        self._sum = np.zeros(self.nbin - 1)
        self._sum_sq = np.zeros(self.nbin - 1)
        self._count = np.zeros(self.nbin - 1)

        # Bin centers
        self.x0 = self.binning[:-1] + (self.binning[1] - self.binning[0]) / 2.0

    def add_data(self, coord, param):
        """Add new data - accumulates directly into bins (O(1) memory)."""
        # Filter valid data
        valid = np.isfinite(coord) & np.isfinite(param)
        coord = coord[valid]
        param = param[valid]

        # Find bin indices for each data point
        bin_indices = np.digitize(coord, self.binning) - 1

        # Clip to valid bin range
        valid_bins = (bin_indices >= 0) & (bin_indices < self.nbin - 1)
        bin_indices = bin_indices[valid_bins]
        param = param[valid_bins]

        # Accumulate into bins
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


def plot_snr_vs_dT_cosmos(band='r', repo='/repo/embargo',
                          collection='LSSTCam/runs/DRP/DP2/v29_0_0/DM-50219',
                          visitMappingFile=None, repOutPlot='plots/',
                          x_min=None, x_max=None, bin_spacing=None,
                          max_visits=None, use_psf_max=False,
                          detector_filter=None):
    """
    Create a plot showing:
    - Top panel: distribution of SNR (or psf_max_value)
    - Bottom panel: dT/T as a function of SNR (or psf_max_value)

    For COSMOS DDF only.
    """
    # Set defaults based on x-axis type
    if use_psf_max:
        x_min = x_min if x_min is not None else 0
        x_max = x_max if x_max is not None else 100000
        bin_spacing = bin_spacing if bin_spacing is not None else 2000
        x_col = 'psf_max_value'
        x_label = 'psf pixel max value ($\\text{e}^{-}$)'
        x_low, x_high = 0, 100000
        file_suffix = f'psfmax_vs_dT_T_cosmos_{band}'
    else:
        x_min = x_min if x_min is not None else 10
        x_max = x_max if x_max is not None else 1000
        bin_spacing = bin_spacing if bin_spacing is not None else 20
        x_col = 'snr'
        x_label = 'SNR (base_GaussianFlux_instFlux / base_GaussianFlux_instFluxErr)'
        x_low, x_high = 50, 1000
        file_suffix = f'SNR_vs_dT_T_cosmos_{band}'

    if detector_filter:
        file_suffix += f'_{detector_filter}'

    # Get COSMOS visits from Butler
    print(f"Querying COSMOS visits from Butler...")
    cosmos_visit_ids = get_cosmos_visits(repo, collection, band)
    print(f"Found {len(cosmos_visit_ids)} COSMOS visits in {band}-band")

    if max_visits is not None and len(cosmos_visit_ids) > max_visits:
        cosmos_visit_ids = cosmos_visit_ids[:max_visits]
        print(f"Limited to {max_visits} visits")

    # Load visit mapping
    with open(visitMappingFile, 'rb') as f:
        visit_mapping = pickle.load(f)

    # Initialize meanify with x bounds for O(1) memory binning
    meanify = meanify1D_wrms(bin_spacing=bin_spacing, x_min=x_min, x_max=x_max)

    # Load all data
    all_x = []
    all_dT_T = []
    n_loaded = 0

    for visit_id in tqdm(cosmos_visit_ids, desc="Loading visits"):
        if visit_id not in visit_mapping:
            print(f"Warning: visit {visit_id} not in mapping")
            continue

        info = visit_mapping[visit_id]
        try:
            data = load_visit_data(info['parquet_path'], detector_filter=detector_filter)

            # Filter valid data
            x_data = data[x_col]
            valid = np.isfinite(x_data) & np.isfinite(data['dT_T'])
            valid &= (x_data > x_min) & (x_data < x_max)

            if np.sum(valid) > 0:
                x_vals = x_data[valid]
                dT_T = data['dT_T'][valid]

                all_x.append(x_vals)
                all_dT_T.append(dT_T)
                meanify.add_data(x_vals, dT_T)
                n_loaded += 1

        except Exception as e:
            print(f"Failed to load visit {visit_id}: {e}")

    # Compute binned statistics
    meanify.meanify()

    # Concatenate all data for histogram
    all_x = np.concatenate(all_x)
    all_dT_T = np.concatenate(all_dT_T)

    print(f"Loaded {n_loaded} visits")
    print(f"Total stars: {len(all_x):,}")

    # Create plot
    fig = plt.figure(figsize=(12, 10))
    plt.subplots_adjust(left=0.12, bottom=0.08, top=0.95, right=0.95, hspace=0)
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 2])

    # Top panel: x distribution
    ax1 = plt.subplot(gs[0])
    ax1.hist(all_x, bins=meanify.binning, color='b', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax1.set_yscale('log')
    ax1.set_ylabel('# stars', fontsize=14)
    ax1.set_xlim(x_min, x_max)
    ax1.tick_params(labelbottom=False)

    det_str = f" ({detector_filter})" if detector_filter else ""
    ax1.set_title(f"COSMOS DDF | Band: {band}{det_str} | N_visits: {n_loaded} | N_stars: {len(all_x):,}", fontsize=14)

    # Add reference lines to top panel
    ax1.axvline(x_low, color='r', linestyle='--', linewidth=2, label=f'{x_col}={x_low}')
    ax1.axvline(x_high, color='r', linestyle='--', linewidth=2, label=f'{x_col}={x_high}')
    ax1.legend(loc='upper right', fontsize=10)

    # Bottom panel: dT/T vs x
    ax2 = plt.subplot(gs[1])

    # Plot binned average
    valid_bins = np.isfinite(meanify.average)
    ax2.scatter(meanify.x0[valid_bins], meanify.average[valid_bins], s=50, c='b', zorder=3, label='Binned mean')
    ax2.errorbar(meanify.x0[valid_bins], meanify.average[valid_bins],
                 yerr=meanify.std[valid_bins] / np.sqrt(meanify.count[valid_bins]),
                 fmt='none', c='b', capsize=3, zorder=2)

    # Reference lines
    xlim = (x_min, x_max)
    ax2.axhline(0, color='k', linestyle='--', linewidth=1, zorder=1)
    ax2.fill_between(xlim, -0.004, 0.004, color='g', alpha=0.2, zorder=0, label='0.4% requirement')
    ax2.fill_between(xlim, -0.001, 0.001, color='g', alpha=0.3, zorder=0, label='0.1% goal')

    # Add reference lines to bottom panel
    ax2.axvline(x_low, color='r', linestyle='--', linewidth=2, label=f'{x_col}={x_low}')
    ax2.axvline(x_high, color='r', linestyle='--', linewidth=2, label=f'{x_col}={x_high}')

    ax2.set_xlim(xlim)
    ax2.set_ylim(-0.02, 0.02)
    ax2.set_xlabel(x_label, fontsize=14)
    ax2.set_ylabel('$\\langle \\delta T / T \\rangle$', fontsize=14)
    ax2.legend(loc='upper right', fontsize=10)

    # Save plot
    os.makedirs(repOutPlot, exist_ok=True)
    output_file = os.path.join(repOutPlot, f'{file_suffix}.png')
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"Saved: {output_file}")

    # Save results to pickle
    results = {
        'band': band,
        'n_visits': n_loaded,
        'n_stars': len(all_x),
        'x_type': x_col,
        'x_bins': meanify.x0,
        'dT_T_mean': meanify.average,
        'dT_T_std': meanify.std,
        'dT_T_count': meanify.count,
        'detector_filter': detector_filter,
    }
    results_file = os.path.join(repOutPlot, f'{file_suffix}.pkl')
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"Saved results: {results_file}")


def main():
    parser = argparse.ArgumentParser(description="SNR vs dT/T analysis for COSMOS DDF")
    parser.add_argument('--band', type=str, default='r', help='Band to process')
    parser.add_argument('--repo', type=str, default='/repo/embargo')
    parser.add_argument('--collection', type=str, default='LSSTCam/runs/DRP/DP2/v29_0_0/DM-50219')
    parser.add_argument('--visitMappingFile', type=str, required=True,
                        help='Path to visit_parquet_mapping.pkl')
    parser.add_argument('--repOutPlot', type=str, default='plots/',
                        help='Output directory for plots')
    parser.add_argument('--x_min', type=float, default=None,
                        help='Minimum x value')
    parser.add_argument('--x_max', type=float, default=None,
                        help='Maximum x value')
    parser.add_argument('--bin_spacing', type=float, default=None,
                        help='Bin spacing')
    parser.add_argument('--max_visits', type=int, default=None,
                        help='Limit to N visits for testing')
    parser.add_argument('--use_psf_max', action='store_true',
                        help='Use psf_max_value instead of SNR for x-axis')
    parser.add_argument('--detector_filter', type=str, default=None, choices=['e2v', 'itl'],
                        help='Filter by detector type: e2v or itl')

    args = parser.parse_args()

    plot_snr_vs_dT_cosmos(
        band=args.band,
        repo=args.repo,
        collection=args.collection,
        visitMappingFile=args.visitMappingFile,
        repOutPlot=args.repOutPlot,
        x_min=args.x_min,
        x_max=args.x_max,
        bin_spacing=args.bin_spacing,
        max_visits=args.max_visits,
        use_psf_max=args.use_psf_max,
        detector_filter=args.detector_filter,
    )


if __name__ == "__main__":
    main()
