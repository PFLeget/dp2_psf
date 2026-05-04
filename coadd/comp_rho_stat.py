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


def load_des_y6_rho(pkl_file):
    """
    Load DES Y6 rho statistics from pkl file.

    DES Y6 index mapping (from file_order):
    - rho1 = index 14 (q2cat_q2cat)
    - rho2 = index 2 (e2cat_q2cat)
    - rho3 = index 26 (w22cat_w22cat)
    - rho4 = index 16 (q2cat_w22cat)
    - rho5 = index 4 (e2cat_w22cat)
    - rho3alt: no DES equivalent

    Returns dict with rho1, rho2, rho3, rho4, rho5 (not rho3alt)
    """
    with open(pkl_file, 'rb') as f:
        des_data = pickle.load(f)

    meanr = des_data['meanr']
    xips = des_data['xips']
    xip_errs = des_data['xip_errs']

    des_rho = {}

    des_rho['rho1'] = {
        'meanr': meanr,
        'xip': xips[14],
        'varxip': xip_errs[14]**2,
    }

    des_rho['rho2'] = {
        'meanr': meanr,
        'xip': xips[2],
        'varxip': xip_errs[2]**2,
    }

    des_rho['rho3'] = {
        'meanr': meanr,
        'xip': xips[26],
        'varxip': xip_errs[26]**2,
    }

    des_rho['rho4'] = {
        'meanr': meanr,
        'xip': xips[16],
        'varxip': xip_errs[16]**2,
    }

    des_rho['rho5'] = {
        'meanr': meanr,
        'xip': xips[4],
        'varxip': xip_errs[4]**2,
    }

    return des_rho


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


def load_single_visit_data(parquet_path, coadd_detectors=None):
    """Load single visit data with sky coordinate moments.

    Parameters
    ----------
    parquet_path : str
        Path to the parquet file
    coadd_detectors : set or None
        If provided, only keep sources from detectors in this set.
    """
    columns = [
        'coord_ra', 'coord_dec',
        'shape_Iuu', 'shape_Ivv', 'shape_Iuv',
        'psfShape_Iuu', 'psfShape_Ivv', 'psfShape_Iuv',
        'detector',
    ]

    table = polars.scan_parquet(parquet_path).select(columns).collect()

    # Filter by coadd detectors if specified
    if coadd_detectors is not None:
        detector_col = table['detector'].to_numpy()
        mask = np.isin(detector_col, list(coadd_detectors))
        table = table.filter(polars.Series(mask))

    return {
        'ixx': table['shape_Iuu'].to_numpy(),  # arcsec^2
        'iyy': table['shape_Ivv'].to_numpy(),
        'ixy': table['shape_Iuv'].to_numpy(),
        'ixx_psf': table['psfShape_Iuu'].to_numpy(),
        'iyy_psf': table['psfShape_Ivv'].to_numpy(),
        'ixy_psf': table['psfShape_Iuv'].to_numpy(),
        'ra': np.degrees(table['coord_ra'].to_numpy()),
        'dec': np.degrees(table['coord_dec'].to_numpy()),
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


def compute_rho_statistics(inputs, treecorr_config, npatch=25, patch_centers=None, patch_centers_file=None):
    """
    Compute all rho statistics with consistent patch centers for covariance estimation.

    All catalogs use the same patch centers so that treecorr.estimate_multi_cov can
    be used to combine their covariances.

    Parameters
    ----------
    inputs : dict
        Output from compute_rho_inputs
    treecorr_config : dict
        TreeCorr configuration (sep_units, min_sep, max_sep, nbins)
    npatch : int
        Number of patches for jackknife (used only if patch_centers not provided)
    patch_centers : array or str
        Pre-computed patch centers (array or path to file). If None, will be
        computed from the data and optionally saved.
    patch_centers_file : str
        If provided and patch_centers is None, save computed patch centers here.

    Returns
    -------
    rho_stats : dict
        Dict with rho1-5 and rho3alt treecorr correlation objects
    patch_centers : array
        The patch centers used (for passing to subsequent calls)
    """
    ra = inputs['ra']
    dec = inputs['dec']
    e1, e2 = inputs['e1'], inputs['e2']
    e1_res, e2_res = inputs['e1_res'], inputs['e2_res']
    size_res = inputs['size_res']
    e1_size_res, e2_size_res = inputs['e1_size_res'], inputs['e2_size_res']

    # Create the reference catalog to establish patch centers
    print(f"  Creating reference catalog with {npatch} patches...")
    if patch_centers is None:
        cat_ref = treecorr.Catalog(ra=ra, dec=dec, g1=e1_res, g2=e2_res,
                                   ra_units='deg', dec_units='deg',
                                   npatch=npatch)
        patch_centers = cat_ref.patch_centers
        if patch_centers_file is not None:
            cat_ref.write_patch_centers(patch_centers_file)
            print(f"  Saved patch centers to: {patch_centers_file}")
    else:
        cat_ref = treecorr.Catalog(ra=ra, dec=dec, g1=e1_res, g2=e2_res,
                                   ra_units='deg', dec_units='deg',
                                   patch_centers=patch_centers)

    # Build all catalogs with same patch centers
    print("  Building catalogs with consistent patch centers...")
    cat_e = treecorr.Catalog(ra=ra, dec=dec, g1=e1, g2=e2,
                             ra_units='deg', dec_units='deg',
                             patch_centers=patch_centers)
    cat_de = cat_ref  # Already built with e1_res, e2_res
    cat_eT = treecorr.Catalog(ra=ra, dec=dec, g1=e1_size_res, g2=e2_size_res,
                              ra_units='deg', dec_units='deg',
                              patch_centers=patch_centers)
    cat_T = treecorr.Catalog(ra=ra, dec=dec, k=size_res,
                             ra_units='deg', dec_units='deg',
                             patch_centers=patch_centers)

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

    return rho_stats, patch_centers


def plot_rho_statistics(rho_stats, output_file, title=None, ylims=None, des_rho=None):
    """
    Plot all rho statistics.

    Parameters
    ----------
    rho_stats : dict
        Dict with rho1-5 and rho3alt, either treecorr objects or dicts with meanr/xip/varxip
    output_file : str
        Output filename
    title : str
        Plot title
    ylims : dict
        Y-axis limits per rho stat, e.g. {'rho1': 1e-6, 'rho2': 1e-7, 'rho3alt': (0, 1e-4)}
        For rho1-5: single value means symmetric [-val, val]
        For rho3alt: tuple (min, max)
    des_rho : dict
        DES Y6 rho statistics for comparison (from load_des_y6_rho). Contains rho1-5, not rho3alt.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    rho_labels = {
        'rho1': r"$\rho_{1}(\theta) = \langle \delta e, \delta e \rangle$",
        'rho2': r"$\rho_{2}(\theta) = \langle e, \delta e \rangle$",
        'rho3': r"$\rho_{3}(\theta) = \langle e\frac{\delta T}{T} , e\frac{\delta T}{T} \rangle$",
        'rho4': r"$\rho_{4}(\theta) = \langle \delta e, e\frac{\delta T}{T} \rangle$",
        'rho5': r"$\rho_{5}(\theta) = \langle e, e\frac{\delta T}{T} \rangle$",
        'rho3alt': r"$\rho'_{3}(\theta) = \langle \frac{\delta T}{T}, \frac{\delta T}{T}\rangle$",
    }

    # LSSTCam scales
    CCD_SCALE = 13.3  # arcmin (4000 pixels * 0.2 arcsec/pixel / 60)
    FOCAL_PLANE_SCALE = 210.0  # arcmin (3.5 deg * 60)

    if ylims is None:
        ylims = {}

    plot_order = ['rho1', 'rho2', 'rho3', 'rho4', 'rho5', 'rho3alt']

    for idx, rho_name in enumerate(plot_order):
        ax = axes.flat[idx]
        rho = rho_stats[rho_name]

        # Handle both treecorr objects and dicts from pkl
        if hasattr(rho, 'meanr'):
            theta = rho.meanr
            if rho_name == 'rho3alt':
                y = rho.xi
                yerr = np.sqrt(rho.varxi)
            else:
                y = rho.xip
                yerr = np.sqrt(rho.varxip)
        else:
            theta = rho['meanr']
            y = rho['xip']
            yerr = np.sqrt(rho['varxip'])

        ax.errorbar(theta, y, yerr=yerr, fmt='o-', capsize=2, markersize=4, color='blue', label='Rubin DP2')

        # Overlay DES Y6 if provided (not for rho3alt)
        if des_rho is not None and rho_name in des_rho:
            des = des_rho[rho_name]
            ax.errorbar(des['meanr'], des['xip'], yerr=np.sqrt(des['varxip']),
                        fmt='s--', capsize=2, markersize=4, alpha=0.7, color='black', label='DES Y6 riz')

        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)

        # Add LSSTCam scale markers
        ax.axvline(CCD_SCALE, color='red', linestyle=':', alpha=0.7, label='CCD scale' if idx == 0 else None)
        ax.axvline(FOCAL_PLANE_SCALE, color='green', linestyle=':', alpha=0.7, label='Focal plane' if idx == 0 else None)

        ax.set_xscale('log')
        if rho_name != 'rho3alt':
            ax.set_yscale('symlog', linthresh=1e-8)
            if rho_name in ylims:
                val = ylims[rho_name]
                ax.set_ylim(-val, val)
        else:
            if rho_name in ylims:
                ax.set_ylim(ylims[rho_name])
        ax.set_xlabel('Separation [arcmin]')
        ax.set_ylabel(rho_labels[rho_name])
        ax.grid(True, alpha=0.3)

        if idx == 0:
            ax.legend(loc='best', fontsize=8)

    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"Saved plot: {output_file}")


def replot_from_pkl(pkl_file, output_file=None, title=None, ylims=None, des_file=None):
    """Replot rho statistics from saved pkl file."""
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)

    rho_stats = data['rho_stats']

    if output_file is None:
        output_file = pkl_file.replace('.pkl', '.png')

    if title is None:
        band = data.get('band', '?')
        n_sources = data.get('n_sources', '?')
        title = f"Rho Statistics - {band}-band ({n_sources} sources)"

    des_rho = None
    if des_file is not None:
        des_rho = load_des_y6_rho(des_file)

    plot_rho_statistics(rho_stats, output_file, title=title, ylims=ylims, des_rho=des_rho)


def run_coadd(band, tractMappingFile, repOut, exclude_crowded=True, galactic_b_min=25., max_tracts=None,
              npatch=25, patch_centers=None, ellipticity_type='distortion',
              min_sep=0.1, max_sep=900.0, nbins=40):
    """Run rho statistics on coadd data."""
    print(f"Computing rho statistics for COADD, band={band}, ellipticity_type={ellipticity_type}")
    print(f"  Angular bins: {nbins} bins from {min_sep} to {max_sep} arcmin")

    with open(tractMappingFile, 'rb') as f:
        tract_mapping = pickle.load(f)

    # Limit number of tracts for testing
    tract_items = list(tract_mapping.items())
    if max_tracts is not None and len(tract_items) > max_tracts:
        tract_items = tract_items[:max_tracts]

    print(f"  Loading {len(tract_items)} tracts...")

    all_data = {k: [] for k in ['ixx', 'iyy', 'ixy', 'ixx_psf', 'iyy_psf', 'ixy_psf', 'ra', 'dec']}

    for tract, info in tqdm(tract_items, desc="Loading tracts"):
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
    inputs = compute_rho_inputs(all_data, ellipticity_type=ellipticity_type)

    # Treecorr config
    treecorr_config = {
        'sep_units': 'arcmin',
        'min_sep': min_sep,
        'max_sep': max_sep,
        'nbins': nbins,
    }

    # Build suffix for output files
    suffix = f'coadd_{band}_{ellipticity_type}'

    # Compute rho stats with consistent patch centers
    os.makedirs(repOut, exist_ok=True)
    patch_centers_file = os.path.join(repOut, f'patch_centers_{suffix}.txt')
    rho_stats, patch_centers_out = compute_rho_statistics(
        inputs, treecorr_config,
        npatch=npatch, patch_centers=patch_centers,
        patch_centers_file=patch_centers_file if patch_centers is None else None
    )

    # Save treecorr Corr2 results using TreeCorr's write method
    treecorr_dir = os.path.join(repOut, f'treecorr_{suffix}')
    os.makedirs(treecorr_dir, exist_ok=True)
    for rho_name, rho_corr in rho_stats.items():
        output_file = os.path.join(treecorr_dir, f'{rho_name}.fits')
        rho_corr.write(output_file)
    print(f"Saved TreeCorr Corr2 files: {treecorr_dir}/")

    # Save summary results (for plotting)
    output_pkl = os.path.join(repOut, f'rho_stats_{suffix}.pkl')
    with open(output_pkl, 'wb') as f:
        pickle.dump({
            'rho_stats': {k: {'meanr': v.meanr, 'xip': v.xip if hasattr(v, 'xip') else v.xi,
                              'varxip': v.varxip if hasattr(v, 'varxip') else v.varxi,
                              'npairs': v.npairs}
                         for k, v in rho_stats.items()},
            'band': band,
            'n_sources': len(inputs['ra']),
            'treecorr_config': treecorr_config,
            'patch_centers_file': patch_centers_file,
            'ellipticity_type': ellipticity_type,
        }, f)
    print(f"Saved summary: {output_pkl}")

    # Plot
    output_plot = os.path.join(repOut, f'rho_stats_{suffix}.png')
    plot_rho_statistics(rho_stats, output_plot, title=f"Rho Statistics - Coadd {band}-band ({ellipticity_type})")


def run_single_visit(band, visitMappingFile, repOut, exclude_crowded=True, galactic_b_min=25., max_visits=None,
                     npatch=25, patch_centers=None, coaddDetectorFile=None, ellipticity_type='distortion',
                     min_sep=0.1, max_sep=900.0, nbins=40):
    """Run rho statistics on single visit data."""
    print(f"Computing rho statistics for SINGLE VISIT, band={band}, ellipticity_type={ellipticity_type}")
    print(f"  Angular bins: {nbins} bins from {min_sep} to {max_sep} arcmin")

    with open(visitMappingFile, 'rb') as f:
        visit_mapping = pickle.load(f)

    # Load coadd detector mapping if specified
    coadd_detector_mapping = None
    if coaddDetectorFile is not None:
        with open(coaddDetectorFile, 'rb') as f:
            coadd_detector_mapping = pickle.load(f)
        print(f"  Loaded coadd detector mapping: {len(coadd_detector_mapping)} visits")

    # Filter by band
    selected_visits = [(v, info) for v, info in visit_mapping.items() if info['band'] == band]

    # Limit number of visits for testing
    if max_visits is not None and len(selected_visits) > max_visits:
        selected_visits = selected_visits[:max_visits]

    print(f"  Loading {len(selected_visits)} visits for band {band}...")

    all_data = {k: [] for k in ['ixx', 'iyy', 'ixy', 'ixx_psf', 'iyy_psf', 'ixy_psf', 'ra', 'dec']}

    n_skipped_no_coadd = 0
    for visit, info in tqdm(selected_visits, desc="Loading visits"):
        # Get coadd detectors for this visit (if filtering enabled)
        coadd_detectors = None
        if coadd_detector_mapping is not None:
            if visit not in coadd_detector_mapping:
                n_skipped_no_coadd += 1
                continue
            coadd_detectors = coadd_detector_mapping[visit]

        try:
            data = load_single_visit_data(info['parquet_path'], coadd_detectors=coadd_detectors)
            for k in all_data:
                all_data[k].append(data[k])
        except Exception as e:
            print(f"  Warning: failed visit {visit}: {e}")

    if n_skipped_no_coadd > 0:
        print(f"  Skipped {n_skipped_no_coadd} visits not in coadd")

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
    inputs = compute_rho_inputs(all_data, ellipticity_type=ellipticity_type)

    # Treecorr config
    treecorr_config = {
        'sep_units': 'arcmin',
        'min_sep': min_sep,
        'max_sep': max_sep,
        'nbins': nbins,
    }

    # Build suffix for output files
    suffix = f'single_visit_{band}_{ellipticity_type}'
    if coaddDetectorFile is not None:
        suffix += '_coaddOnly'

    # Compute rho stats with consistent patch centers
    os.makedirs(repOut, exist_ok=True)
    patch_centers_file = os.path.join(repOut, f'patch_centers_{suffix}.txt')
    rho_stats, patch_centers_out = compute_rho_statistics(
        inputs, treecorr_config,
        npatch=npatch, patch_centers=patch_centers,
        patch_centers_file=patch_centers_file if patch_centers is None else None
    )

    # Save treecorr Corr2 results using TreeCorr's write method
    treecorr_dir = os.path.join(repOut, f'treecorr_{suffix}')
    os.makedirs(treecorr_dir, exist_ok=True)
    for rho_name, rho_corr in rho_stats.items():
        output_file = os.path.join(treecorr_dir, f'{rho_name}.fits')
        rho_corr.write(output_file)
    print(f"Saved TreeCorr Corr2 files: {treecorr_dir}/")

    # Save summary results (for plotting)
    output_pkl = os.path.join(repOut, f'rho_stats_{suffix}.pkl')
    with open(output_pkl, 'wb') as f:
        pickle.dump({
            'rho_stats': {k: {'meanr': v.meanr, 'xip': v.xip if hasattr(v, 'xip') else v.xi,
                              'varxip': v.varxip if hasattr(v, 'varxip') else v.varxi,
                              'npairs': v.npairs}
                         for k, v in rho_stats.items()},
            'band': band,
            'n_sources': len(inputs['ra']),
            'n_visits': len(selected_visits) - n_skipped_no_coadd,
            'treecorr_config': treecorr_config,
            'patch_centers_file': patch_centers_file,
            'coadd_only': coaddDetectorFile is not None,
            'ellipticity_type': ellipticity_type,
        }, f)
    print(f"Saved summary: {output_pkl}")

    # Plot
    title = f"Rho Statistics - Single Visit {band}-band ({ellipticity_type})"
    if coaddDetectorFile is not None:
        title += " (coadd detectors only)"
    output_plot = os.path.join(repOut, f'rho_stats_{suffix}.png')
    plot_rho_statistics(rho_stats, output_plot, title=title)


def main():
    parser = argparse.ArgumentParser(description="Compute Rho statistics for PSF modeling")
    parser.add_argument('--mode', type=str, required=True, choices=['coadd', 'single_visit', 'replot'],
                        help='Data mode: coadd, single_visit, or replot')
    parser.add_argument('--band', type=str, default=None, help='Band to process (u, g, r, i, z, y)')
    parser.add_argument('--tractMappingFile', type=str, default=None,
                        help='Path to tract_parquet_mapping.pkl (for coadd)')
    parser.add_argument('--visitMappingFile', type=str, default=None,
                        help='Path to visit_parquet_mapping_skycoord.pkl (for single_visit)')
    parser.add_argument('--repOut', type=str, default='rho_stats/', help='Output directory')
    parser.add_argument('--galactic_b_min', type=float, default=25.,
                        help='Minimum |b| for MW exclusion (degrees)')
    parser.add_argument('--no_exclude_crowded', action='store_true',
                        help='Do not exclude crowded regions')
    parser.add_argument('--max', type=int, default=None,
                        help='Maximum number of tracts/visits to process (for testing)')
    parser.add_argument('--pklInput', type=str, default=None,
                        help='Path to pkl file for replot mode')
    parser.add_argument('--ylim_rho1', type=float, default=None, help='Y-axis limit for rho1 (symmetric)')
    parser.add_argument('--ylim_rho2', type=float, default=None, help='Y-axis limit for rho2 (symmetric)')
    parser.add_argument('--ylim_rho3', type=float, default=None, help='Y-axis limit for rho3 (symmetric)')
    parser.add_argument('--ylim_rho4', type=float, default=None, help='Y-axis limit for rho4 (symmetric)')
    parser.add_argument('--ylim_rho5', type=float, default=None, help='Y-axis limit for rho5 (symmetric)')
    parser.add_argument('--ylim_rho3alt', type=float, nargs=2, default=None,
                        help='Y-axis limits for rho3alt (min max, e.g. 0 1e-4)')
    parser.add_argument('--desFile', type=str, default=None,
                        help='Path to DES Y6 rho stats pkl file for comparison overlay')
    parser.add_argument('--npatch', type=int, default=25,
                        help='Number of patches for jackknife covariance (default: 25)')
    parser.add_argument('--patchCenters', type=str, default=None,
                        help='Path to patch centers file (for consistent patches across runs)')
    parser.add_argument('--coaddDetectorFile', type=str, default=None,
                        help='Path to coadd_detector_mapping.pkl to filter only detectors in coadd (single_visit mode)')
    parser.add_argument('--ellipticityType', type=str, default='distortion', choices=['distortion', 'shear'],
                        help='Ellipticity definition: distortion (default) or shear')
    parser.add_argument('--min_sep', type=float, default=0.1,
                        help='Minimum separation in arcmin (default: 0.1)')
    parser.add_argument('--max_sep', type=float, default=900.0,
                        help='Maximum separation in arcmin (default: 900)')
    parser.add_argument('--nbins', type=int, default=40,
                        help='Number of separation bins (default: 40)')

    args = parser.parse_args()

    if args.mode == 'replot':
        if args.pklInput is None:
            raise ValueError("--pklInput required for replot mode")
        ylims = {}
        if args.ylim_rho1 is not None:
            ylims['rho1'] = args.ylim_rho1
        if args.ylim_rho2 is not None:
            ylims['rho2'] = args.ylim_rho2
        if args.ylim_rho3 is not None:
            ylims['rho3'] = args.ylim_rho3
        if args.ylim_rho4 is not None:
            ylims['rho4'] = args.ylim_rho4
        if args.ylim_rho5 is not None:
            ylims['rho5'] = args.ylim_rho5
        if args.ylim_rho3alt is not None:
            ylims['rho3alt'] = tuple(args.ylim_rho3alt)
        replot_from_pkl(args.pklInput, ylims=ylims if ylims else None, des_file=args.desFile)
    elif args.mode == 'coadd':
        if args.tractMappingFile is None:
            raise ValueError("--tractMappingFile required for coadd mode")
        if args.band is None:
            raise ValueError("--band required for coadd mode")
        exclude_crowded = not args.no_exclude_crowded
        run_coadd(args.band, args.tractMappingFile, args.repOut,
                  exclude_crowded=exclude_crowded, galactic_b_min=args.galactic_b_min,
                  max_tracts=args.max, npatch=args.npatch, patch_centers=args.patchCenters,
                  ellipticity_type=args.ellipticityType,
                  min_sep=args.min_sep, max_sep=args.max_sep, nbins=args.nbins)
    else:
        if args.visitMappingFile is None:
            raise ValueError("--visitMappingFile required for single_visit mode")
        if args.band is None:
            raise ValueError("--band required for single_visit mode")
        exclude_crowded = not args.no_exclude_crowded
        run_single_visit(args.band, args.visitMappingFile, args.repOut,
                         exclude_crowded=exclude_crowded, galactic_b_min=args.galactic_b_min,
                         max_visits=args.max, npatch=args.npatch, patch_centers=args.patchCenters,
                         coaddDetectorFile=args.coaddDetectorFile, ellipticity_type=args.ellipticityType,
                         min_sep=args.min_sep, max_sep=args.max_sep, nbins=args.nbins)


if __name__ == "__main__":
    main()
