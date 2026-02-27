import numpy as np
import treegp
print(treegp.__version__)
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
from scipy.stats import binned_statistic


# Columns to read from parquet files
PARQUET_COLUMNS = [
    'slot_Shape_xx', 'slot_Shape_yy', 'slot_Shape_xy',
    'slot_PsfShape_xx', 'slot_PsfShape_xy', 'slot_PsfShape_yy',
    'base_GaussianFlux_instFlux', 'base_GaussianFlux_instFluxErr',
    'detector', 'calib_psf_reserved',
]


def load_visit_data(parquet_path):
    """
    Load visit data from parquet file and compute derived columns.
    """
    table = polars.scan_parquet(parquet_path).select(PARQUET_COLUMNS).collect()

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


def plot_snr_vs_dT(bands='g', visitMappingFile="data/visit_parquet_mapping.pkl",
                   repOutPlot='plots/', snr_min=10, snr_max=1000, bin_spacing=20,
                   n_visits=None):
    """
    Create a plot showing:
    - Top panel: distribution of SNR
    - Bottom panel: dT/T as a function of SNR
    """

    # Load the visit mapping
    with open(visitMappingFile, 'rb') as f:
        visit_mapping = pickle.load(f)

    # Filter visits by band(s)
    selected_visits = []
    for visit, info in visit_mapping.items():
        if info['band'] in bands:
            selected_visits.append((visit, info))

    # Limit number of visits if specified (for testing)
    if n_visits is not None:
        selected_visits = selected_visits[:n_visits]

    print(f"Selected {len(selected_visits)} visits for bands: {bands}")

    # Initialize meanify with SNR bounds for O(1) memory binning
    meanify = meanify1D_wrms(bin_spacing=bin_spacing, x_min=snr_min, x_max=snr_max)

    # Load all data
    all_snr = []
    all_dT_T = []

    for visit, info in tqdm(selected_visits, desc="Loading visits"):
        try:
            data = load_visit_data(info['parquet_path'])

            # Filter valid data
            valid = np.isfinite(data['snr']) & np.isfinite(data['dT_T'])
            valid &= (data['snr'] > snr_min) & (data['snr'] < snr_max)

            if np.sum(valid) > 0:
                snr = data['snr'][valid]
                dT_T = data['dT_T'][valid]

                all_snr.append(snr)
                all_dT_T.append(dT_T)
                meanify.add_data(snr, dT_T)

        except Exception as e:
            print(f"Failed to load visit {visit}: {e}")

    # Compute binned statistics
    meanify.meanify()

    # Concatenate all data for histogram
    all_snr = np.concatenate(all_snr)
    all_dT_T = np.concatenate(all_dT_T)

    print(f"Total stars: {len(all_snr)}")

    # Create plot
    fig = plt.figure(figsize=(12, 10))
    plt.subplots_adjust(left=0.12, bottom=0.08, top=0.95, right=0.95, hspace=0)
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 2])

    # SNR limits of interest
    SNR_LOW = 50
    SNR_HIGH = 1000

    # Top panel: SNR distribution
    ax1 = plt.subplot(gs[0])
    ax1.hist(all_snr, bins=meanify.binning, color='b', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax1.set_yscale('log')
    ax1.set_ylabel('# stars', fontsize=14)
    ax1.set_xlim(snr_min, snr_max)
    ax1.tick_params(labelbottom=False)
    ax1.set_title(f"DP2 | Band: {bands} | N_visits: {len(selected_visits)} | N_stars: {len(all_snr):,}", fontsize=14)

    # Add SNR limit lines to top panel
    ax1.axvline(SNR_LOW, color='r', linestyle='--', linewidth=2, label=f'SNR={SNR_LOW}')
    ax1.axvline(SNR_HIGH, color='r', linestyle='--', linewidth=2, label=f'SNR={SNR_HIGH}')
    ax1.legend(loc='upper right', fontsize=10)

    # Bottom panel: dT/T vs SNR
    ax2 = plt.subplot(gs[1])

    # Plot binned average
    valid_bins = np.isfinite(meanify.average)
    ax2.scatter(meanify.x0[valid_bins], meanify.average[valid_bins], s=50, c='b', zorder=3, label='Binned mean')
    ax2.errorbar(meanify.x0[valid_bins], meanify.average[valid_bins],
                 yerr=meanify.std[valid_bins] / np.sqrt(meanify.count[valid_bins]),
                 fmt='none', c='b', capsize=3, zorder=2)

    # Reference lines
    xlim = (snr_min, snr_max)
    ax2.axhline(0, color='k', linestyle='--', linewidth=1, zorder=1)
    ax2.fill_between(xlim, -0.004, 0.004, color='g', alpha=0.2, zorder=0, label='0.4% requirement')
    ax2.fill_between(xlim, -0.001, 0.001, color='g', alpha=0.3, zorder=0, label='0.1% goal')

    # Add SNR limit lines to bottom panel
    ax2.axvline(SNR_LOW, color='r', linestyle='--', linewidth=2, label=f'SNR={SNR_LOW}')
    ax2.axvline(SNR_HIGH, color='r', linestyle='--', linewidth=2, label=f'SNR={SNR_HIGH}')

    ax2.set_xlim(xlim)
    ax2.set_ylim(-0.02, 0.02)
    ax2.set_xlabel('SNR (base_GaussianFlux_instFlux / base_GaussianFlux_instFluxErr)', fontsize=14)
    ax2.set_ylabel('$\\langle \\delta T / T \\rangle$', fontsize=14)
    ax2.legend(loc='upper right', fontsize=10)

    # Save plot
    output_file = os.path.join(repOutPlot, f'SNR_vs_dT_T_{bands}.png')
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"Saved: {output_file}")

    # Save results to pickle
    results = {
        'bands': bands,
        'n_visits': len(selected_visits),
        'n_stars': len(all_snr),
        'snr_bins': meanify.x0,
        'dT_T_mean': meanify.average,
        'dT_T_std': meanify.std,
        'dT_T_count': meanify.count,
    }
    results_file = os.path.join(repOutPlot, f'SNR_vs_dT_T_{bands}.pkl')
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"Saved results: {results_file}")


def main():
    defaultVisitMappingFile = "/sdf/home/l/leget/rubin-user/lsst_dev/tickets/dp2_psf/data/visit_parquet_mapping.pkl"
    defaultRepOutPlot = "/sdf/home/l/leget/rubin-user/lsst_dev/tickets/dp2_psf/plots/"

    parser = argparse.ArgumentParser(description="SNR vs dT/T analysis")
    parser.add_argument('--bands', type=str, required=True,
                        help="The band(s) to process (e.g., g, r, ugrizy)")
    parser.add_argument('--visitMappingFile', type=str, default=defaultVisitMappingFile,
                        help="Path to visit_parquet_mapping.pkl file")
    parser.add_argument('--repOutPlot', type=str, default=defaultRepOutPlot,
                        help="Output directory for plots")
    parser.add_argument('--snr_min', type=float, default=10,
                        help="Minimum SNR (default: 10)")
    parser.add_argument('--snr_max', type=float, default=1000,
                        help="Maximum SNR (default: 1000)")
    parser.add_argument('--bin_spacing', type=float, default=20,
                        help="SNR bin spacing (default: 20)")
    parser.add_argument('--n_visits', type=int, default=None,
                        help="Limit to N visits for testing (default: all)")

    args = parser.parse_args()

    plot_snr_vs_dT(
        bands=args.bands,
        visitMappingFile=args.visitMappingFile,
        repOutPlot=args.repOutPlot,
        snr_min=args.snr_min,
        snr_max=args.snr_max,
        bin_spacing=args.bin_spacing,
        n_visits=args.n_visits,
    )


if __name__ == "__main__":
    main()
