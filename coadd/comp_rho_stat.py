#!/usr/bin/env python
"""
Compute Rho statistics for coadd and single visit PSF data.

Rho statistics quantify PSF modeling errors through correlation functions
of ellipticity and size residuals.

References:
- Jarvis et al. (2016), MNRAS 460, 2245
- Rowe (2010), MNRAS 404, 350
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

from astropy.coordinates import SkyCoord
import astropy.units as u


# LMC and SMC centers (J2000)
LMC_RA, LMC_DEC = 80.9, -69.8
SMC_RA, SMC_DEC = 13.2, -72.8


def filter_crowded_regions(ra_deg, dec_deg, galactic_b_min=25., lmc_radius=10., smc_radius=5.):
    """
    Filter out stars in crowded regions: Milky Way plane, LMC, SMC.

    Returns mask of stars to KEEP.
    """
    coords = SkyCoord(ra=ra_deg * u.degree, dec=dec_deg * u.degree, frame='icrs')

    galactic_b = coords.galactic.b.degree
    mask_mw = np.abs(galactic_b) > galactic_b_min

    lmc_center = SkyCoord(ra=LMC_RA * u.degree, dec=LMC_DEC * u.degree, frame='icrs')
    mask_lmc = coords.separation(lmc_center).degree > lmc_radius

    smc_center = SkyCoord(ra=SMC_RA * u.degree, dec=SMC_DEC * u.degree, frame='icrs')
    mask_smc = coords.separation(smc_center).degree > smc_radius

    return mask_mw & mask_lmc & mask_smc


def compute_ellipticity(ixx, iyy, ixy, ellipticity_type='distortion'):
    """
    Compute ellipticity from second moments.

    Parameters
    ----------
    ixx, iyy, ixy : arrays
        Second moments
    ellipticity_type : str
        'distortion' or 'shear'

    Returns
    -------
    e1, e2 : arrays
        Ellipticity components
    """
    T = ixx + iyy

    if ellipticity_type == 'distortion':
        e1 = (ixx - iyy) / T
        e2 = 2 * ixy / T
    else:  # shear
        denom = T + 2 * np.sqrt(ixx * iyy - ixy**2)
        e1 = (ixx - iyy) / denom
        e2 = 2 * ixy / denom

    return e1, e2


def compute_size(ixx, iyy, ixy, size_type='trace'):
    """Compute size from second moments."""
    if size_type == 'trace':
        return ixx + iyy
    else:  # determinant
        return np.sqrt(ixx * iyy - ixy**2)


def load_coadd_data(parquet_path, band):
    """Load coadd data for a single tract."""
    columns = [
        f"{band}_calib_psf_used", "detect_isPrimary", "refExtendedness",
        f"{band}_pixelFlags_inexact_psfCenter",
        f"{band}_ixxPSF", f"{band}_iyyPSF", f"{band}_ixyPSF",
        f"{band}_ixx", f"{band}_iyy", f"{band}_ixy",
        f"{band}_ra", f"{band}_dec",
    ]

    table = polars.scan_parquet(parquet_path).select(columns).collect()

    # Quality filters
    mask = table['detect_isPrimary'].to_numpy() == True
    mask &= table['refExtendedness'].to_numpy() == 0.0
    mask &= table[f'{band}_calib_psf_used'].to_numpy() == True
    mask &= table[f'{band}_pixelFlags_inexact_psfCenter'].to_numpy() == False

    return {
        'ixx': table[f"{band}_ixx"].to_numpy()[mask],
        'iyy': table[f"{band}_iyy"].to_numpy()[mask],
        'ixy': table[f"{band}_ixy"].to_numpy()[mask],
        'ixx_psf': table[f"{band}_ixxPSF"].to_numpy()[mask],
        'iyy_psf': table[f"{band}_iyyPSF"].to_numpy()[mask],
        'ixy_psf': table[f"{band}_ixyPSF"].to_numpy()[mask],
        'ra': table[f"{band}_ra"].to_numpy()[mask],
        'dec': table[f"{band}_dec"].to_numpy()[mask],
    }


def load_single_visit_data(parquet_path):
    """Load single visit data with sky coordinate moments."""
    columns = [
        'coord_ra', 'coord_dec',
        'shape_Iuu', 'shape_Ivv', 'shape_Iuv',
        'psfShape_Iuu', 'psfShape_Ivv', 'psfShape_Iuv',
        'calib_psf_used',
    ]

    table = polars.scan_parquet(parquet_path).select(columns).collect()

    # Filter to PSF stars
    mask = table['calib_psf_used'].to_numpy() == True

    return {
        'ixx': table['shape_Iuu'].to_numpy()[mask],  # arcsec^2
        'iyy': table['shape_Ivv'].to_numpy()[mask],
        'ixy': table['shape_Iuv'].to_numpy()[mask],
        'ixx_psf': table['psfShape_Iuu'].to_numpy()[mask],
        'iyy_psf': table['psfShape_Ivv'].to_numpy()[mask],
        'ixy_psf': table['psfShape_Iuv'].to_numpy()[mask],
        'ra': np.degrees(table['coord_ra'].to_numpy()[mask]),
        'dec': np.degrees(table['coord_dec'].to_numpy()[mask]),
    }


def compute_rho_inputs(data, ellipticity_type='distortion', size_type='trace'):
    """
    Compute the inputs needed for rho statistics.

    Returns
    -------
    dict with: ra, dec, e1, e2, e1_res, e2_res, size_res, e1_size_res, e2_size_res
    """
    # Ellipticities
    e1, e2 = compute_ellipticity(data['ixx'], data['iyy'], data['ixy'], ellipticity_type)
    e1_psf, e2_psf = compute_ellipticity(data['ixx_psf'], data['iyy_psf'], data['ixy_psf'], ellipticity_type)

    # Sizes
    T = compute_size(data['ixx'], data['iyy'], data['ixy'], size_type)
    T_psf = compute_size(data['ixx_psf'], data['iyy_psf'], data['ixy_psf'], size_type)

    # Residuals
    e1_res = e1 - e1_psf
    e2_res = e2 - e2_psf
    size_res = (T_psf - T) / T  # Note: sign convention from analysis_tools

    # Responsivity correction for distortion
    responsivity = 2.0 if ellipticity_type == 'distortion' else 1.0
    e1 /= responsivity
    e2 /= responsivity
    e1_res /= responsivity
    e2_res /= responsivity

    # Size-weighted ellipticity
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


def corr_spin2(ra, dec, g1a, g2a, g1b=None, g2b=None, config=None):
    """Compute spin-2 (shear-like) correlation."""
    xy = treecorr.GGCorrelation(config=config)
    catA = treecorr.Catalog(ra=ra, dec=dec, g1=g1a, g2=g2a,
                            ra_units='deg', dec_units='deg', config=config)

    if g1b is None or g2b is None:
        xy.process(catA)
    else:
        catB = treecorr.Catalog(ra=ra, dec=dec, g1=g1b, g2=g2b,
                                ra_units='deg', dec_units='deg', config=config,
                                patch_centers=catA.patch_centers)
        xy.process(catA, catB)
    return xy


def corr_spin0(ra, dec, k1, k2=None, config=None):
    """Compute spin-0 (scalar) correlation."""
    xy = treecorr.KKCorrelation(config=config)
    catA = treecorr.Catalog(ra=ra, dec=dec, k=k1,
                            ra_units='deg', dec_units='deg', config=config)

    if k2 is None:
        xy.process(catA)
    else:
        catB = treecorr.Catalog(ra=ra, dec=dec, k=k2,
                                ra_units='deg', dec_units='deg', config=config,
                                patch_centers=catA.patch_centers)
        xy.process(catA, catB)
    return xy


def compute_rho_statistics(inputs, treecorr_config):
    """
    Compute all rho statistics.

    Returns dict with rho1, rho2, rho3, rho4, rho5, rho3alt
    """
    ra = inputs['ra']
    dec = inputs['dec']
    e1, e2 = inputs['e1'], inputs['e2']
    e1_res, e2_res = inputs['e1_res'], inputs['e2_res']
    size_res = inputs['size_res']
    e1_size_res, e2_size_res = inputs['e1_size_res'], inputs['e2_size_res']

    rho_stats = {}

    print("  Computing rho1: <de*, de>")
    rho_stats['rho1'] = corr_spin2(ra, dec, e1_res, e2_res, config=treecorr_config)

    print("  Computing rho2: <e*, de>")
    rho_stats['rho2'] = corr_spin2(ra, dec, e1, e2, e1_res, e2_res, config=treecorr_config)

    print("  Computing rho3: <e*dT/T, e*dT/T>")
    rho_stats['rho3'] = corr_spin2(ra, dec, e1_size_res, e2_size_res, config=treecorr_config)

    print("  Computing rho4: <de*, e*dT/T>")
    rho_stats['rho4'] = corr_spin2(ra, dec, e1_res, e2_res, e1_size_res, e2_size_res, config=treecorr_config)

    print("  Computing rho5: <e*, e*dT/T>")
    rho_stats['rho5'] = corr_spin2(ra, dec, e1, e2, e1_size_res, e2_size_res, config=treecorr_config)

    print("  Computing rho3alt: <dT/T, dT/T>")
    rho_stats['rho3alt'] = corr_spin0(ra, dec, size_res, config=treecorr_config)

    return rho_stats


def plot_rho_statistics(rho_stats, output_file, title=None):
    """Plot all rho statistics."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    rho_labels = {
        'rho1': r"$\rho_{1}(\theta) = \langle \delta e, \delta e \rangle$",
        'rho2': r"$\rho_{2}(\theta) = \langle e, \delta e \rangle$",
        'rho3': r"$\rho_{3}(\theta) = \langle e\frac{\delta T}{T} , e\frac{\delta T}{T} \rangle$",
        'rho4': r"$\rho_{4}(\theta) = \langle \delta e, e\frac{\delta T}{T} \rangle$",
        'rho5': r"$\rho_{5}(\theta) = \langle e, e\frac{\delta T}{T} \rangle$",
        'rho3alt': r"$\rho'_{3}(\theta) = \langle \frac{\delta T}{T}, \frac{\delta T}{T}\rangle$",
    }

    plot_order = ['rho1', 'rho2', 'rho3', 'rho4', 'rho5', 'rho3alt']

    for idx, rho_name in enumerate(plot_order):
        ax = axes.flat[idx]
        rho = rho_stats[rho_name]

        theta = rho.meanr  # arcmin
        if rho_name == 'rho3alt':
            y = rho.xi
            yerr = np.sqrt(rho.varxi)
        else:
            y = rho.xip
            yerr = np.sqrt(rho.varxip)

        ax.errorbar(theta, y, yerr=yerr, fmt='o-', capsize=2, markersize=4)
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xscale('log')
        if rho_name != 'rho3alt':
            ax.set_yscale('symlog', linthresh=1e-8)
        ax.set_xlabel('Separation [arcmin]')
        ax.set_ylabel(rho_labels[rho_name])
        ax.grid(True, alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"Saved plot: {output_file}")


def run_coadd(band, tractMappingFile, repOut, exclude_crowded=True, galactic_b_min=25.):
    """Run rho statistics on coadd data."""
    print(f"Computing rho statistics for COADD, band={band}")

    with open(tractMappingFile, 'rb') as f:
        tract_mapping = pickle.load(f)

    print(f"  Loading {len(tract_mapping)} tracts...")

    all_data = {k: [] for k in ['ixx', 'iyy', 'ixy', 'ixx_psf', 'iyy_psf', 'ixy_psf', 'ra', 'dec']}

    for tract, info in tqdm(tract_mapping.items(), desc="Loading tracts"):
        try:
            data = load_coadd_data(info['parquet_path'], band)
            for k in all_data:
                all_data[k].append(data[k])
        except Exception as e:
            print(f"  Warning: failed tract {tract}: {e}")

    # Concatenate
    for k in all_data:
        all_data[k] = np.concatenate(all_data[k])

    print(f"  Total sources: {len(all_data['ra'])}")

    # Filter crowded regions
    if exclude_crowded:
        mask = filter_crowded_regions(all_data['ra'], all_data['dec'], galactic_b_min=galactic_b_min)
        for k in all_data:
            all_data[k] = all_data[k][mask]
        print(f"  After crowded filter: {len(all_data['ra'])}")

    # Filter NaN
    valid = np.isfinite(all_data['ixx']) & np.isfinite(all_data['iyy']) & np.isfinite(all_data['ixy'])
    valid &= np.isfinite(all_data['ixx_psf']) & np.isfinite(all_data['iyy_psf']) & np.isfinite(all_data['ixy_psf'])
    for k in all_data:
        all_data[k] = all_data[k][valid]
    print(f"  After NaN filter: {len(all_data['ra'])}")

    # Compute rho inputs
    inputs = compute_rho_inputs(all_data)

    # Treecorr config
    treecorr_config = {
        'sep_units': 'arcmin',
        'min_sep': 0.5,
        'max_sep': 100.0,
        'nbins': 20,
    }

    # Compute rho stats
    rho_stats = compute_rho_statistics(inputs, treecorr_config)

    # Save results
    os.makedirs(repOut, exist_ok=True)
    output_pkl = os.path.join(repOut, f'rho_stats_coadd_{band}.pkl')
    with open(output_pkl, 'wb') as f:
        pickle.dump({
            'rho_stats': {k: {'meanr': v.meanr, 'xip': v.xip if hasattr(v, 'xip') else v.xi,
                              'varxip': v.varxip if hasattr(v, 'varxip') else v.varxi,
                              'npairs': v.npairs}
                         for k, v in rho_stats.items()},
            'band': band,
            'n_sources': len(inputs['ra']),
            'treecorr_config': treecorr_config,
        }, f)
    print(f"Saved: {output_pkl}")

    # Plot
    output_plot = os.path.join(repOut, f'rho_stats_coadd_{band}.png')
    plot_rho_statistics(rho_stats, output_plot, title=f"Rho Statistics - Coadd {band}-band")


def run_single_visit(band, visitMappingFile, repOut, exclude_crowded=True, galactic_b_min=25.):
    """Run rho statistics on single visit data."""
    print(f"Computing rho statistics for SINGLE VISIT, band={band}")

    with open(visitMappingFile, 'rb') as f:
        visit_mapping = pickle.load(f)

    # Filter by band
    selected_visits = [(v, info) for v, info in visit_mapping.items() if info['band'] == band]
    print(f"  Loading {len(selected_visits)} visits for band {band}...")

    all_data = {k: [] for k in ['ixx', 'iyy', 'ixy', 'ixx_psf', 'iyy_psf', 'ixy_psf', 'ra', 'dec']}

    for visit, info in tqdm(selected_visits, desc="Loading visits"):
        try:
            data = load_single_visit_data(info['parquet_path'])
            for k in all_data:
                all_data[k].append(data[k])
        except Exception as e:
            print(f"  Warning: failed visit {visit}: {e}")

    # Concatenate
    for k in all_data:
        all_data[k] = np.concatenate(all_data[k])

    print(f"  Total sources: {len(all_data['ra'])}")

    # Filter crowded regions
    if exclude_crowded:
        mask = filter_crowded_regions(all_data['ra'], all_data['dec'], galactic_b_min=galactic_b_min)
        for k in all_data:
            all_data[k] = all_data[k][mask]
        print(f"  After crowded filter: {len(all_data['ra'])}")

    # Filter NaN
    valid = np.isfinite(all_data['ixx']) & np.isfinite(all_data['iyy']) & np.isfinite(all_data['ixy'])
    valid &= np.isfinite(all_data['ixx_psf']) & np.isfinite(all_data['iyy_psf']) & np.isfinite(all_data['ixy_psf'])
    for k in all_data:
        all_data[k] = all_data[k][valid]
    print(f"  After NaN filter: {len(all_data['ra'])}")

    # Compute rho inputs
    inputs = compute_rho_inputs(all_data)

    # Treecorr config
    treecorr_config = {
        'sep_units': 'arcmin',
        'min_sep': 0.5,
        'max_sep': 100.0,
        'nbins': 20,
    }

    # Compute rho stats
    rho_stats = compute_rho_statistics(inputs, treecorr_config)

    # Save results
    os.makedirs(repOut, exist_ok=True)
    output_pkl = os.path.join(repOut, f'rho_stats_single_visit_{band}.pkl')
    with open(output_pkl, 'wb') as f:
        pickle.dump({
            'rho_stats': {k: {'meanr': v.meanr, 'xip': v.xip if hasattr(v, 'xip') else v.xi,
                              'varxip': v.varxip if hasattr(v, 'varxip') else v.varxi,
                              'npairs': v.npairs}
                         for k, v in rho_stats.items()},
            'band': band,
            'n_sources': len(inputs['ra']),
            'n_visits': len(selected_visits),
            'treecorr_config': treecorr_config,
        }, f)
    print(f"Saved: {output_pkl}")

    # Plot
    output_plot = os.path.join(repOut, f'rho_stats_single_visit_{band}.png')
    plot_rho_statistics(rho_stats, output_plot, title=f"Rho Statistics - Single Visit {band}-band")


def main():
    parser = argparse.ArgumentParser(description="Compute Rho statistics for PSF modeling")
    parser.add_argument('--mode', type=str, required=True, choices=['coadd', 'single_visit'],
                        help='Data mode: coadd or single_visit')
    parser.add_argument('--band', type=str, required=True, help='Band to process (u, g, r, i, z, y)')
    parser.add_argument('--tractMappingFile', type=str, default=None,
                        help='Path to tract_parquet_mapping.pkl (for coadd)')
    parser.add_argument('--visitMappingFile', type=str, default=None,
                        help='Path to visit_parquet_mapping_skycoord.pkl (for single_visit)')
    parser.add_argument('--repOut', type=str, default='rho_stats/', help='Output directory')
    parser.add_argument('--galactic_b_min', type=float, default=25.,
                        help='Minimum |b| for MW exclusion (degrees)')
    parser.add_argument('--no_exclude_crowded', action='store_true',
                        help='Do not exclude crowded regions')

    args = parser.parse_args()

    exclude_crowded = not args.no_exclude_crowded

    if args.mode == 'coadd':
        if args.tractMappingFile is None:
            raise ValueError("--tractMappingFile required for coadd mode")
        run_coadd(args.band, args.tractMappingFile, args.repOut,
                  exclude_crowded=exclude_crowded, galactic_b_min=args.galactic_b_min)
    else:
        if args.visitMappingFile is None:
            raise ValueError("--visitMappingFile required for single_visit mode")
        run_single_visit(args.band, args.visitMappingFile, args.repOut,
                         exclude_crowded=exclude_crowded, galactic_b_min=args.galactic_b_min)


if __name__ == "__main__":
    main()
