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

# For galactic coordinate transformation
from astropy.coordinates import SkyCoord
import astropy.units as u


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


# Columns to read from parquet files
PARQUET_COLUMNS = [
    'slot_Shape_xx', 'slot_Shape_yy', 'slot_Shape_xy',
    'slot_PsfShape_xx', 'slot_PsfShape_xy', 'slot_PsfShape_yy',
    'base_GaussianFlux_instFlux', 'base_GaussianFlux_instFluxErr',
    'coord_ra', 'coord_dec',
    'detector', 'calib_psf_reserved', 'psf_max_value',
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


def plot_snr_vs_dT(bands='g', visitMappingFile="data/visit_parquet_mapping.pkl",
                   repOutPlot='plots/', x_min=None, x_max=None, bin_spacing=None,
                   n_visits=None, use_psf_max=False,
                   exclude_crowded=False, galactic_b_min=20., lmc_radius=10., smc_radius=5.):
    """
    Create a plot showing:
    - Top panel: distribution of SNR (or psf_max_value)
    - Bottom panel: dT/T as a function of SNR (or psf_max_value)

    Parameters
    ----------
    use_psf_max : bool
        If True, use psf_max_value instead of SNR for the x-axis.
    exclude_crowded : bool
        If True, exclude stars in MW plane, LMC, and SMC
    galactic_b_min : float
        Minimum |b| to keep when exclude_crowded=True (degrees)
    lmc_radius : float
        Radius around LMC to exclude (degrees)
    smc_radius : float
        Radius around SMC to exclude (degrees)
    """
    # Set defaults based on x-axis type
    if use_psf_max:
        x_min = x_min if x_min is not None else 0
        x_max = x_max if x_max is not None else 100000
        bin_spacing = bin_spacing if bin_spacing is not None else 2000
        x_col = 'psf_max_value'
        x_label = 'psf pixel max value ($\\text{e}^{-}$)'
        x_low, x_high = 0, 100000  # Reference lines for psf_max
        file_suffix = f'psfmax_vs_dT_T_{bands}'
    else:
        x_min = x_min if x_min is not None else 10
        x_max = x_max if x_max is not None else 1000
        bin_spacing = bin_spacing if bin_spacing is not None else 20
        x_col = 'snr'
        x_label = 'SNR (base_GaussianFlux_instFlux / base_GaussianFlux_instFluxErr)'
        x_low, x_high = 50, 1000  # Reference lines for SNR
        file_suffix = f'SNR_vs_dT_T_{bands}'

    # Add suffix for crowded region exclusion
    if exclude_crowded:
        file_suffix += '_noCrowded'

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

    # Initialize meanify with x bounds for O(1) memory binning
    meanify = meanify1D_wrms(bin_spacing=bin_spacing, x_min=x_min, x_max=x_max)

    # Load all data
    all_x = []
    all_dT_T = []

    for visit, info in tqdm(selected_visits, desc="Loading visits"):
        try:
            data = load_visit_data(info['parquet_path'])

            # Filter valid data
            x_data = data[x_col]
            valid = np.isfinite(x_data) & np.isfinite(data['dT_T'])
            valid &= (x_data > x_min) & (x_data < x_max)

            # Filter crowded regions (MW, LMC, SMC) if requested
            if exclude_crowded:
                ra_deg = np.degrees(data['ra'])
                dec_deg = np.degrees(data['dec'])
                mask_crowded = filter_crowded_regions(
                    ra_deg, dec_deg,
                    galactic_b_min=galactic_b_min,
                    lmc_radius=lmc_radius,
                    smc_radius=smc_radius
                )
                valid &= mask_crowded

            if np.sum(valid) > 0:
                x_vals = x_data[valid]
                dT_T = data['dT_T'][valid]

                all_x.append(x_vals)
                all_dT_T.append(dT_T)
                meanify.add_data(x_vals, dT_T)

        except Exception as e:
            print(f"Failed to load visit {visit}: {e}")

    # Compute binned statistics
    meanify.meanify()

    # Concatenate all data for histogram
    all_x = np.concatenate(all_x)
    all_dT_T = np.concatenate(all_dT_T)

    print(f"Total stars: {len(all_x)}")

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
    ax1.set_title(f"DP2 | Band: {bands} | N_visits: {len(selected_visits)} | N_stars: {len(all_x):,}", fontsize=14)

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
    output_file = os.path.join(repOutPlot, f'{file_suffix}.png')
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"Saved: {output_file}")

    # Save results to pickle
    results = {
        'bands': bands,
        'n_visits': len(selected_visits),
        'n_stars': len(all_x),
        'x_type': x_col,
        'x_bins': meanify.x0,
        'dT_T_mean': meanify.average,
        'dT_T_std': meanify.std,
        'dT_T_count': meanify.count,
        'exclude_crowded': exclude_crowded,
        'galactic_b_min': galactic_b_min if exclude_crowded else None,
        'lmc_radius': lmc_radius if exclude_crowded else None,
        'smc_radius': smc_radius if exclude_crowded else None,
    }
    results_file = os.path.join(repOutPlot, f'{file_suffix}.pkl')
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
    parser.add_argument('--x_min', type=float, default=None,
                        help="Minimum x value (default: 10 for SNR, 0 for psf_max)")
    parser.add_argument('--x_max', type=float, default=None,
                        help="Maximum x value (default: 1000 for SNR, 100000 for psf_max)")
    parser.add_argument('--bin_spacing', type=float, default=None,
                        help="Bin spacing (default: 20 for SNR, 2000 for psf_max)")
    parser.add_argument('--n_visits', type=int, default=None,
                        help="Limit to N visits for testing (default: all)")
    parser.add_argument('--use_psf_max', action='store_true',
                        help="Use psf_max_value instead of SNR for x-axis")

    # Crowded region filtering
    parser.add_argument('--exclude_crowded', action='store_true',
                        help='Exclude MW plane, LMC, and SMC')
    parser.add_argument('--galactic_b_min', type=float, default=20.,
                        help='Min |b| to keep when excluding MW (degrees, default: 20)')
    parser.add_argument('--lmc_radius', type=float, default=10.,
                        help='Radius around LMC to exclude (degrees, default: 10)')
    parser.add_argument('--smc_radius', type=float, default=5.,
                        help='Radius around SMC to exclude (degrees, default: 5)')

    args = parser.parse_args()

    plot_snr_vs_dT(
        bands=args.bands,
        visitMappingFile=args.visitMappingFile,
        repOutPlot=args.repOutPlot,
        x_min=args.x_min,
        x_max=args.x_max,
        bin_spacing=args.bin_spacing,
        n_visits=args.n_visits,
        use_psf_max=args.use_psf_max,
        exclude_crowded=args.exclude_crowded,
        galactic_b_min=args.galactic_b_min,
        lmc_radius=args.lmc_radius,
        smc_radius=args.smc_radius,
    )


if __name__ == "__main__":
    main()
