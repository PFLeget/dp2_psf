#!/usr/bin/env python
"""
Compute rho1 for each polynomial order collection.
One plot per polynomial order with fixed axis limits.
"""

import numpy as np
import treecorr
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
os.environ["POLARS_MAX_THREADS"] = "1"
import polars

import argparse

from lsst.daf.butler import Butler


CCD_SCALE = 13.3  # arcmin
FOCAL_PLANE_SCALE = 210.0  # arcmin

# Detector types
ITL_DETECTORS = np.concatenate((np.arange(0, 36), np.arange(72, 81), np.arange(162, 189)))
E2V_DETECTORS = np.concatenate((np.arange(36, 72), np.arange(81, 162)))

PARQUET_COLUMNS = [
    'coord_ra', 'coord_dec',
    'shape_Iuu', 'shape_Ivv', 'shape_Iuv',
    'psfShape_Iuu', 'psfShape_Ivv', 'psfShape_Iuv',
    'calib_psf_used', 'calib_psf_reserved',
    'base_GaussianFlux_instFlux', 'base_GaussianFlux_instFluxErr',
    'detector',
]


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

    iuu_src = table['shape_Iuu'].to_numpy()
    ivv_src = table['shape_Ivv'].to_numpy()
    iuv_src = table['shape_Iuv'].to_numpy()
    iuu_psf = table['psfShape_Iuu'].to_numpy()
    ivv_psf = table['psfShape_Ivv'].to_numpy()
    iuv_psf = table['psfShape_Iuv'].to_numpy()

    T_src = iuu_src + ivv_src
    T_psf = iuu_psf + ivv_psf

    e1_src = (iuu_src - ivv_src) / T_src
    e2_src = 2 * iuv_src / T_src
    e1_psf = (iuu_psf - ivv_psf) / T_psf
    e2_psf = 2 * iuv_psf / T_psf

    flux = table['base_GaussianFlux_instFlux'].to_numpy()
    flux_err = table['base_GaussianFlux_instFluxErr'].to_numpy()
    snr = flux / flux_err

    return {
        'ra': np.degrees(table['coord_ra'].to_numpy()),
        'dec': np.degrees(table['coord_dec'].to_numpy()),
        'de1': e1_src - e1_psf,
        'de2': e2_src - e2_psf,
        'calib_psf_used': table['calib_psf_used'].to_numpy(),
        'calib_psf_reserved': table['calib_psf_reserved'].to_numpy(),
        'snr': snr,
    }


def compute_rho1(data, treecorr_config):
    """Compute rho1 = <de, de>."""
    ra, dec = data['ra'], data['dec']
    de1, de2 = data['de1'], data['de2']

    cat = treecorr.Catalog(ra=ra, dec=dec, g1=de1, g2=de2, ra_units='deg', dec_units='deg')

    rho1 = treecorr.GGCorrelation(config=treecorr_config)
    rho1.process(cat)

    return rho1


def plot_snr_distribution(snr_used, snr_reserved, output_file, title, snr_bins):
    """Plot SNR distribution for used vs reserved stars."""
    fig, ax = plt.subplots(figsize=(10, 7))

    ax.hist(snr_used, bins=snr_bins, color='b', alpha=0.5, edgecolor='b',
            linewidth=0.5, label=f'Used ({len(snr_used):,})')
    ax.hist(snr_reserved, bins=snr_bins, color='r', alpha=0.5, edgecolor='r',
            linewidth=0.5, label=f'Reserved ({len(snr_reserved):,})')

    ax.set_yscale('log')
    ax.set_xlabel('SNR', fontsize=12)
    ax.set_ylabel('# stars', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"Saved: {output_file}")


def plot_rho1(rho1_all, rho1_used, rho1_reserved, output_file, title, ylim,
               n_all, n_used, n_reserved):
    """Plot rho1 xip for all, used, and reserved on same plot."""
    fig, ax = plt.subplots(figsize=(10, 7))

    theta = rho1_all.meanr

    ax.errorbar(theta, rho1_all.xip, yerr=np.sqrt(rho1_all.varxip),
                fmt='o-', capsize=2, markersize=4, color='k',
                label=f'All')
    ax.errorbar(theta, rho1_used.xip, yerr=np.sqrt(rho1_used.varxip),
                fmt='s-', capsize=2, markersize=4, color='b',
                label=f'Used')
    ax.errorbar(theta, rho1_reserved.xip, yerr=np.sqrt(rho1_reserved.varxip),
                fmt='^-', capsize=2, markersize=4, color='r',
                label=f'Reserved')

    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(CCD_SCALE, color='k', linestyle='--', alpha=0.7, label='CCD scale')
    ax.axvline(FOCAL_PLANE_SCALE, color='k', linestyle=':', alpha=0.7, label='FoV scale')

    ax.set_xscale('log')
    ax.set_yscale('symlog', linthresh=1e-8)
    ax.set_ylim(ylim)
    ax.set_xlabel('Separation [arcmin]', fontsize=12)
    ax.set_ylabel(r'$\rho_1(\theta) = \langle \delta e, \delta e \rangle$', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc=4, fontsize=10)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"Saved: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Compute rho1 for polynomial order test")
    parser.add_argument('--repo', type=str, default='dp2_prep')
    parser.add_argument('--base_collection', type=str,
                        default='u/leget/LSSTCam/DM-54194/DP2/COSMOS_r_band')
    parser.add_argument('--visit_file', type=str, default='visits_cosmos_r_dp2.txt')
    parser.add_argument('--repOut', type=str, default='plots/')
    parser.add_argument('--min_sep', type=float, default=0.1, help='Min separation in arcmin')
    parser.add_argument('--max_sep', type=float, default=300.0, help='Max separation in arcmin')
    parser.add_argument('--nbins', type=int, default=20, help='Number of separation bins')
    parser.add_argument('--ylim_min', type=float, default=-1e-4)
    parser.add_argument('--ylim_max', type=float, default=1e-4)
    parser.add_argument('--detector_filter', type=str, default=None, choices=['e2v', 'itl'],
                        help='Filter by detector type: e2v or itl')

    args = parser.parse_args()

    det_str = f" ({args.detector_filter.upper()})" if args.detector_filter else ""
    det_suffix = f"_{args.detector_filter}" if args.detector_filter else ""

    os.makedirs(args.repOut, exist_ok=True)

    # Read visit IDs
    with open(args.visit_file, 'r') as f:
        visits = [int(line.strip()) for line in f if line.strip()]
    print(f"Processing {len(visits)} visits")
    print(f"Detector filter: {args.detector_filter if args.detector_filter else 'all'}")

    treecorr_config = {
        'sep_units': 'arcmin',
        'min_sep': args.min_sep,
        'max_sep': args.max_sep,
        'nbins': args.nbins,
    }

    ylim = (args.ylim_min, args.ylim_max)

    # Process each polynomial order
    for poly_order in range(5):
        collection = f"{args.base_collection}/Polynomial_{poly_order}"
        print(f"\n{'='*50}")
        print(f"Processing Polynomial_{poly_order}")
        print(f"Collection: {collection}")

        butler = Butler(args.repo, collections=collection)

        all_data = {k: [] for k in ['ra', 'dec', 'de1', 'de2', 'calib_psf_used', 'calib_psf_reserved', 'snr']}
        n_loaded = 0

        for visit in tqdm(visits, desc=f"Loading visits (poly={poly_order})"):
            try:
                uri = butler.getURI("refit_psf_star", instrument="LSSTCam", visit=visit)
                data = load_visit_data(uri.geturl(), detector_filter=args.detector_filter)

                valid = np.isfinite(data['de1']) & np.isfinite(data['de2'])
                for k in all_data:
                    all_data[k].append(data[k][valid])
                n_loaded += 1
            except Exception as e:
                pass

        if n_loaded == 0:
            print(f"No data found for Polynomial_{poly_order}")
            continue

        # Concatenate
        for k in all_data:
            all_data[k] = np.concatenate(all_data[k])

        print(f"Loaded {n_loaded} visits, {len(all_data['ra']):,} sources")

        # Split by used/reserved
        used_mask = all_data['calib_psf_used']
        reserved_mask = all_data['calib_psf_reserved']

        data_all = {k: all_data[k] for k in ['ra', 'dec', 'de1', 'de2']}
        data_used = {k: all_data[k][used_mask] for k in ['ra', 'dec', 'de1', 'de2']}
        data_reserved = {k: all_data[k][reserved_mask] for k in ['ra', 'dec', 'de1', 'de2']}

        n_all = len(data_all['ra'])
        n_used = len(data_used['ra'])
        n_reserved = len(data_reserved['ra'])

        print(f"  All: {n_all:,}, Used: {n_used:,}, Reserved: {n_reserved:,}")

        # Compute rho1 for each subset
        print("Computing rho1...")
        rho1_all = compute_rho1(data_all, treecorr_config)
        rho1_used = compute_rho1(data_used, treecorr_config)
        rho1_reserved = compute_rho1(data_reserved, treecorr_config)

        # Plot rho1
        output_file = os.path.join(args.repOut, f'rho1_Polynomial_{poly_order}{det_suffix}.png')
        title = f"Polynomial order {poly_order}{det_str} | {n_loaded} visits"
        plot_rho1(rho1_all, rho1_used, rho1_reserved, output_file, title, ylim,
                  n_all, n_used, n_reserved)

        # Plot SNR distribution
        snr_used = all_data['snr'][used_mask]
        snr_reserved = all_data['snr'][reserved_mask]
        snr_bins = np.linspace(0, 1000, 51)
        output_file_snr = os.path.join(args.repOut, f'snr_dist_Polynomial_{poly_order}{det_suffix}.png')
        title_snr = f"SNR Distribution | Polynomial order {poly_order}{det_str} | {n_loaded} visits"
        plot_snr_distribution(snr_used, snr_reserved, output_file_snr, title_snr, snr_bins)


if __name__ == "__main__":
    main()
