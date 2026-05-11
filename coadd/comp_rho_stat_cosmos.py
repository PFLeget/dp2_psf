#!/usr/bin/env python
"""
Compute Rho statistics for COSMOS DDF single visits only.

Debug script to investigate negative rho1 at small scales.
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

import pickle
import argparse

from lsst.daf.butler import Butler
from astropy.coordinates import SkyCoord
import astropy.units as u


COSMOS_RA = 150.12  # degrees
COSMOS_DEC = 2.21   # degrees

# LSSTCam scales
CCD_SCALE = 13.3  # arcmin
FOCAL_PLANE_SCALE = 210.0  # arcmin

# Detector types
ITL_DETECTORS = np.concatenate((np.arange(0, 36), np.arange(72, 81), np.arange(162, 189)))
E2V_DETECTORS = np.concatenate((np.arange(36, 72), np.arange(81, 162)))


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


def load_single_visit_data(parquet_path, detector_filter=None, snr_min=None, snr_max=None):
    """Load single visit data with sky coordinate moments.

    Parameters
    ----------
    parquet_path : str
        Path to parquet file
    detector_filter : str or None
        'e2v', 'itl', or None (all detectors)
    snr_min : float or None
        Minimum SNR cut
    snr_max : float or None
        Maximum SNR cut
    """
    columns = [
        'coord_ra', 'coord_dec', 'detector',
        'shape_Iuu', 'shape_Ivv', 'shape_Iuv',
        'psfShape_Iuu', 'psfShape_Ivv', 'psfShape_Iuv',
        'base_GaussianFlux_instFlux', 'base_GaussianFlux_instFluxErr',
    ]

    table = polars.scan_parquet(parquet_path).select(columns).collect()

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

    # Filter by SNR
    if snr_min is not None or snr_max is not None:
        flux = table['base_GaussianFlux_instFlux'].to_numpy()
        flux_err = table['base_GaussianFlux_instFluxErr'].to_numpy()
        snr = flux / flux_err
        mask = np.ones(len(snr), dtype=bool)
        if snr_min is not None:
            mask &= snr >= snr_min
        if snr_max is not None:
            mask &= snr <= snr_max
        table = table.filter(polars.Series(mask))

    return {
        'ixx': table['shape_Iuu'].to_numpy(),
        'iyy': table['shape_Ivv'].to_numpy(),
        'ixy': table['shape_Iuv'].to_numpy(),
        'ixx_psf': table['psfShape_Iuu'].to_numpy(),
        'iyy_psf': table['psfShape_Ivv'].to_numpy(),
        'ixy_psf': table['psfShape_Iuv'].to_numpy(),
        'ra': np.degrees(table['coord_ra'].to_numpy()),
        'dec': np.degrees(table['coord_dec'].to_numpy()),
    }


def compute_ellipticity(ixx, iyy, ixy, ellipticity_type='distortion'):
    """Compute ellipticity from second moments."""
    T = ixx + iyy
    if ellipticity_type == 'distortion':
        e1 = (ixx - iyy) / T
        e2 = 2 * ixy / T
    else:  # shear
        denom = T + 2 * np.sqrt(ixx * iyy - ixy**2)
        e1 = (ixx - iyy) / denom
        e2 = 2 * ixy / denom
    return e1, e2


def compute_rho_inputs(data, ellipticity_type='distortion'):
    """Compute the inputs needed for rho statistics."""
    e1, e2 = compute_ellipticity(data['ixx'], data['iyy'], data['ixy'], ellipticity_type)
    e1_psf, e2_psf = compute_ellipticity(data['ixx_psf'], data['iyy_psf'], data['ixy_psf'], ellipticity_type)

    T = data['ixx'] + data['iyy']
    T_psf = data['ixx_psf'] + data['iyy_psf']

    e1_res = e1 - e1_psf
    e2_res = e2 - e2_psf
    size_res = (T_psf - T) / T

    responsivity = 2.0 if ellipticity_type == 'distortion' else 1.0
    e1 /= responsivity
    e2 /= responsivity
    e1_res /= responsivity
    e2_res /= responsivity

    e1_size_res = e1 * size_res
    e2_size_res = e2 * size_res

    return {
        'ra': data['ra'],
        'dec': data['dec'],
        'e1': e1,
        'e2': e2,
        'e1_res': e1_res,
        'e2_res': e2_res,
        'size_res': size_res,
        'e1_size_res': e1_size_res,
        'e2_size_res': e2_size_res,
    }


def compute_rho_statistics(inputs, treecorr_config):
    """Compute all rho statistics."""
    ra, dec = inputs['ra'], inputs['dec']
    e1, e2 = inputs['e1'], inputs['e2']
    e1_res, e2_res = inputs['e1_res'], inputs['e2_res']
    size_res = inputs['size_res']
    e1_size_res, e2_size_res = inputs['e1_size_res'], inputs['e2_size_res']

    # Build catalogs
    cat_e = treecorr.Catalog(ra=ra, dec=dec, g1=e1, g2=e2, ra_units='deg', dec_units='deg')
    cat_de = treecorr.Catalog(ra=ra, dec=dec, g1=e1_res, g2=e2_res, ra_units='deg', dec_units='deg')
    cat_eT = treecorr.Catalog(ra=ra, dec=dec, g1=e1_size_res, g2=e2_size_res, ra_units='deg', dec_units='deg')
    cat_T = treecorr.Catalog(ra=ra, dec=dec, k=size_res, ra_units='deg', dec_units='deg')

    rho_stats = {}

    print("  Computing rho1: <de*, de>")
    rho1 = treecorr.GGCorrelation(config=treecorr_config)
    rho1.process(cat_de)
    rho_stats['rho1'] = rho1

    print("  Computing rho2: <e*, de>")
    rho2 = treecorr.GGCorrelation(config=treecorr_config)
    rho2.process(cat_e, cat_de)
    rho_stats['rho2'] = rho2

    print("  Computing rho3: <e*dT/T, e*dT/T>")
    rho3 = treecorr.GGCorrelation(config=treecorr_config)
    rho3.process(cat_eT)
    rho_stats['rho3'] = rho3

    print("  Computing rho4: <de*, e*dT/T>")
    rho4 = treecorr.GGCorrelation(config=treecorr_config)
    rho4.process(cat_de, cat_eT)
    rho_stats['rho4'] = rho4

    print("  Computing rho5: <e*, e*dT/T>")
    rho5 = treecorr.GGCorrelation(config=treecorr_config)
    rho5.process(cat_e, cat_eT)
    rho_stats['rho5'] = rho5

    print("  Computing rho3alt: <dT/T, dT/T>")
    rho3alt = treecorr.KKCorrelation(config=treecorr_config)
    rho3alt.process(cat_T)
    rho_stats['rho3alt'] = rho3alt

    return rho_stats


def plot_rho_statistics(rho_stats, output_file, title=None, ylims=None):
    """Plot all rho statistics with xip and xim."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    rho_labels = {
        'rho1': r"$\rho_{1}(\theta) = \langle \delta e, \delta e \rangle$",
        'rho2': r"$\rho_{2}(\theta) = \langle e, \delta e \rangle$",
        'rho3': r"$\rho_{3}(\theta) = \langle e\frac{\delta T}{T} , e\frac{\delta T}{T} \rangle$",
        'rho4': r"$\rho_{4}(\theta) = \langle \delta e, e\frac{\delta T}{T} \rangle$",
        'rho5': r"$\rho_{5}(\theta) = \langle e, e\frac{\delta T}{T} \rangle$",
        'rho3alt': r"$\rho'_{3}(\theta) = \langle \frac{\delta T}{T}, \frac{\delta T}{T}\rangle$",
    }

    if ylims is None:
        ylims = {}

    plot_order = ['rho1', 'rho2', 'rho3', 'rho4', 'rho5', 'rho3alt']

    for idx, rho_name in enumerate(plot_order):
        ax = axes.flat[idx]
        rho = rho_stats[rho_name]

        theta = rho.meanr

        if rho_name == 'rho3alt':
            ax.errorbar(theta, rho.xi, yerr=np.sqrt(rho.varxi), fmt='o-', capsize=2, markersize=4, color='blue', label='xi')
        else:
            ax.errorbar(theta, rho.xip, yerr=np.sqrt(rho.varxip), fmt='o-', capsize=2, markersize=4, color='blue', label='xip')
            ax.errorbar(theta, rho.xim, yerr=np.sqrt(rho.varxim), fmt='s--', capsize=2, markersize=4, color='red', alpha=0.7, label='xim')

        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(CCD_SCALE, color='k', linestyle='--', alpha=1, label='CCD scale' if idx == 0 else None)
        ax.axvline(FOCAL_PLANE_SCALE, color='k', linestyle=':', alpha=1, label='FoV scale' if idx == 0 else None)

        ax.set_xscale('log')
        if rho_name != 'rho3alt':
            ax.set_yscale('symlog', linthresh=1e-8)
            if rho_name in ylims:
                ax.set_ylim(-ylims[rho_name], ylims[rho_name])
        else:
            if rho_name in ylims:
                ax.set_ylim(ylims[rho_name])

        ax.set_xlabel('Separation [arcmin]')
        ax.set_ylabel(rho_labels[rho_name])
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=8)

    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"Saved plot: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Compute Rho statistics for COSMOS DDF")
    parser.add_argument('--band', type=str, default='r', help='Band to process')
    parser.add_argument('--repo', type=str, default='dp2_prep')
    parser.add_argument('--collection', type=str, default='LSSTCam/runs/DRP/DP2/v30_0_0/DM-53881/stage2')
    parser.add_argument('--visitMappingFile', type=str, required=True,
                        help='Path to visit_parquet_mapping_skycoord.pkl')
    parser.add_argument('--repOut', type=str, default='rho_stats_cosmos/', help='Output directory')
    parser.add_argument('--ellipticityType', type=str, default='distortion', choices=['distortion', 'shear'])
    parser.add_argument('--min_sep', type=float, default=0.1, help='Min separation in arcmin')
    parser.add_argument('--max_sep', type=float, default=300.0, help='Max separation in arcmin')
    parser.add_argument('--nbins', type=int, default=30, help='Number of separation bins')
    parser.add_argument('--max_visits', type=int, default=None, help='Max COSMOS visits to process')
    parser.add_argument('--detector_filter', type=str, default=None, choices=['e2v', 'itl'],
                        help='Filter by detector type: e2v or itl')
    parser.add_argument('--snr_min', type=float, default=None, help='Minimum SNR cut')
    parser.add_argument('--snr_max', type=float, default=None, help='Maximum SNR cut')

    args = parser.parse_args()

    print(f"COSMOS Rho Statistics")
    print(f"  Band: {args.band}")
    print(f"  Ellipticity type: {args.ellipticityType}")
    print(f"  Detector filter: {args.detector_filter if args.detector_filter else 'all'}")
    print(f"  SNR range: [{args.snr_min}, {args.snr_max}]")
    print(f"  Angular bins: {args.nbins} bins from {args.min_sep} to {args.max_sep} arcmin")

    # Get COSMOS visits from Butler
    print(f"\nQuerying COSMOS visits from Butler...")
    cosmos_visit_ids = get_cosmos_visits(args.repo, args.collection, args.band)
    print(f"Found {len(cosmos_visit_ids)} COSMOS visits in {args.band}-band")

    if args.max_visits is not None and len(cosmos_visit_ids) > args.max_visits:
        cosmos_visit_ids = cosmos_visit_ids[:args.max_visits]
        print(f"Limited to {args.max_visits} visits")

    # Load visit mapping
    with open(args.visitMappingFile, 'rb') as f:
        visit_mapping = pickle.load(f)

    # Load all data
    all_data = {k: [] for k in ['ixx', 'iyy', 'ixy', 'ixx_psf', 'iyy_psf', 'ixy_psf', 'ra', 'dec']}

    for visit_id in tqdm(cosmos_visit_ids, desc="Loading visits"):
        if visit_id not in visit_mapping:
            print(f"Warning: visit {visit_id} not in mapping")
            continue

        info = visit_mapping[visit_id]
        try:
            data = load_single_visit_data(info['parquet_path'], detector_filter=args.detector_filter,
                                          snr_min=args.snr_min, snr_max=args.snr_max)
            for k in all_data:
                all_data[k].append(data[k])
        except Exception as e:
            print(f"Warning: failed visit {visit_id}: {e}")

    # Concatenate
    for k in all_data:
        all_data[k] = np.concatenate(all_data[k])

    print(f"\nTotal sources: {len(all_data['ra']):,}")

    # Filter NaN
    valid = np.isfinite(all_data['ixx']) & np.isfinite(all_data['iyy']) & np.isfinite(all_data['ixy'])
    valid &= np.isfinite(all_data['ixx_psf']) & np.isfinite(all_data['iyy_psf']) & np.isfinite(all_data['ixy_psf'])
    for k in all_data:
        all_data[k] = all_data[k][valid]
    print(f"After NaN filter: {len(all_data['ra']):,}")

    # Compute rho inputs
    inputs = compute_rho_inputs(all_data, ellipticity_type=args.ellipticityType)

    # TreeCorr config
    treecorr_config = {
        'sep_units': 'arcmin',
        'min_sep': args.min_sep,
        'max_sep': args.max_sep,
        'nbins': args.nbins,
    }

    # Compute rho stats
    print("\nComputing rho statistics...")
    rho_stats = compute_rho_statistics(inputs, treecorr_config)

    # Create output directory
    os.makedirs(args.repOut, exist_ok=True)

    # Save results
    suffix = f'cosmos_{args.band}_{args.ellipticityType}'
    if args.detector_filter:
        suffix += f'_{args.detector_filter}'
    if args.snr_min is not None:
        suffix += f'_snrmin{int(args.snr_min)}'
    if args.snr_max is not None:
        suffix += f'_snrmax{int(args.snr_max)}'
    output_pkl = os.path.join(args.repOut, f'rho_stats_{suffix}.pkl')
    with open(output_pkl, 'wb') as f:
        pickle.dump({
            'rho_stats': {k: {'meanr': v.meanr,
                              'xip': v.xip if hasattr(v, 'xip') else v.xi,
                              'xim': v.xim if hasattr(v, 'xim') else None,
                              'varxip': v.varxip if hasattr(v, 'varxip') else v.varxi,
                              'varxim': v.varxim if hasattr(v, 'varxim') else None,
                              'npairs': v.npairs}
                         for k, v in rho_stats.items()},
            'band': args.band,
            'n_sources': len(inputs['ra']),
            'n_visits': len(cosmos_visit_ids),
            'treecorr_config': treecorr_config,
            'ellipticity_type': args.ellipticityType,
        }, f)
    print(f"Saved: {output_pkl}")

    # Plot
    output_plot = os.path.join(args.repOut, f'rho_stats_{suffix}.png')
    title = f"COSMOS Rho Statistics - {args.band}-band ({args.ellipticityType})\n{len(cosmos_visit_ids)} visits, {len(inputs['ra']):,} sources"
    plot_rho_statistics(rho_stats, output_plot, title=title)


if __name__ == "__main__":
    main()
