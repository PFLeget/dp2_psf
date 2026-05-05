#!/usr/bin/env python
"""
Animated Rho Statistics for COSMOS Deep Drilling Field.

Creates frames showing evolution of rho statistics and sky projections
as visits accumulate in the COSMOS DDF.

Layout (3x3 grid):
  Row 1: dT/T skymap  | rho1        | rho2
  Row 2: de1 skymap   | rho3        | rho4
  Row 3: de2 skymap   | rho5        | rho3alt
"""

import numpy as np
import treecorr
import treegp
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import hpgeom as hpg
import os
os.environ["POLARS_MAX_THREADS"] = "1"
import polars

import pickle
import argparse

from astropy.coordinates import SkyCoord
import astropy.units as u

from skyproj import GnomonicSkyproj


# COSMOS DDF center and radius
COSMOS_RA = 150.12  # degrees
COSMOS_DEC = 2.21   # degrees
COSMOS_RADIUS = 1.75 * 2  # degrees (focal plane diameter)

# LSSTCam scales for rho stat plots
CCD_SCALE = 13.3  # arcmin
FOCAL_PLANE_SCALE = 210.0  # arcmin (3.5 deg)


def visit_overlaps_cosmos(visit_ra, visit_dec, overlap_radius=1.75):
    """
    Check if a visit overlaps with COSMOS DDF.

    Parameters
    ----------
    visit_ra, visit_dec : float
        Visit center coordinates in degrees
    overlap_radius : float
        Radius around visit center (focal plane radius in degrees)

    Returns
    -------
    bool
        True if visit overlaps with COSMOS
    """
    visit_center = SkyCoord(ra=visit_ra * u.degree, dec=visit_dec * u.degree)
    cosmos_center = SkyCoord(ra=COSMOS_RA * u.degree, dec=COSMOS_DEC * u.degree)
    sep = visit_center.separation(cosmos_center).degree
    return sep < (overlap_radius + COSMOS_RADIUS)


def load_single_visit_data(parquet_path):
    """Load single visit data with sky coordinate moments."""
    columns = [
        'coord_ra', 'coord_dec',
        'shape_Iuu', 'shape_Ivv', 'shape_Iuv',
        'psfShape_Iuu', 'psfShape_Ivv', 'psfShape_Iuv',
    ]

    table = polars.scan_parquet(parquet_path).select(columns).collect()

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


def filter_to_cosmos_region(data, radius_margin=0.5):
    """Filter data to keep only sources within COSMOS region."""
    coords = SkyCoord(ra=data['ra'] * u.degree, dec=data['dec'] * u.degree)
    cosmos_center = SkyCoord(ra=COSMOS_RA * u.degree, dec=COSMOS_DEC * u.degree)
    sep = coords.separation(cosmos_center).degree
    mask = sep < (COSMOS_RADIUS + radius_margin)
    return {k: v[mask] for k, v in data.items()}


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
        'dT_T': (T - T_psf) / T,
    }


def compute_rho_statistics(inputs, treecorr_config):
    """Compute all rho statistics (no patches for animation)."""
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

    rho1 = treecorr.GGCorrelation(config=treecorr_config)
    rho1.process(cat_de)
    rho_stats['rho1'] = rho1

    rho2 = treecorr.GGCorrelation(config=treecorr_config)
    rho2.process(cat_e, cat_de)
    rho_stats['rho2'] = rho2

    rho3 = treecorr.GGCorrelation(config=treecorr_config)
    rho3.process(cat_eT)
    rho_stats['rho3'] = rho3

    rho4 = treecorr.GGCorrelation(config=treecorr_config)
    rho4.process(cat_de, cat_eT)
    rho_stats['rho4'] = rho4

    rho5 = treecorr.GGCorrelation(config=treecorr_config)
    rho5.process(cat_e, cat_eT)
    rho_stats['rho5'] = rho5

    rho3alt = treecorr.KKCorrelation(config=treecorr_config)
    rho3alt.process(cat_T)
    rho_stats['rho3alt'] = rho3alt

    return rho_stats


def compute_healpix_maps(inputs, bin_spacing=30):
    """Compute HEALPix binned sky maps for dT/T, de1, de2."""
    coord = np.array([inputs['ra'], inputs['dec']]).T

    maps = {}
    for key in ['dT_T', 'e1_res', 'e2_res']:
        meanify = treegp.meanify_healpix(bin_spacing=bin_spacing)
        meanify.add_field(coord, inputs[key])
        meanify.meanify()
        maps[key] = {
            'coords0': meanify.coords0,
            'params0': meanify.params0,
            'nside': meanify.nside,
            'valid_pixels': meanify._valid_pixels,
        }
    return maps


def plot_frame(rho_stats, healpix_maps, n_visits, n_sources, output_file, band,
               ylims=None, sky_color_scales=None):
    """
    Plot a single frame with 3x3 grid layout.

    Layout:
      Row 1: dT/T skymap  | rho1        | rho2
      Row 2: de1 skymap   | rho3        | rho4
      Row 3: de2 skymap   | rho5        | rho3alt
    """
    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(3, 3, figure=fig, wspace=0.25, hspace=0.3)

    # Default y-axis limits
    if ylims is None:
        ylims = {
            'rho1': 1e-5,
            'rho2': 1e-5,
            'rho3': 1e-7,
            'rho4': 1e-6,
            'rho5': 1e-6,
            'rho3alt': (0, 2e-5),
        }

    # Default sky color scales
    if sky_color_scales is None:
        sky_color_scales = {
            'dT_T': 0.02,
            'e1_res': 0.01,
            'e2_res': 0.01,
        }

    rho_labels = {
        'rho1': r"$\rho_{1}(\theta) = \langle \delta e, \delta e \rangle$",
        'rho2': r"$\rho_{2}(\theta) = \langle e, \delta e \rangle$",
        'rho3': r"$\rho_{3}(\theta) = \langle e\frac{\delta T}{T} , e\frac{\delta T}{T} \rangle$",
        'rho4': r"$\rho_{4}(\theta) = \langle \delta e, e\frac{\delta T}{T} \rangle$",
        'rho5': r"$\rho_{5}(\theta) = \langle e, e\frac{\delta T}{T} \rangle$",
        'rho3alt': r"$\rho'_{3}(\theta) = \langle \frac{\delta T}{T}, \frac{\delta T}{T}\rangle$",
    }

    sky_labels = {
        'dT_T': r'$\delta T / T$',
        'e1_res': r'$\delta e_1$',
        'e2_res': r'$\delta e_2$',
    }

    # Grid layout mapping
    sky_panels = [(0, 0, 'dT_T'), (1, 0, 'e1_res'), (2, 0, 'e2_res')]
    rho_panels = [
        (0, 1, 'rho1'), (0, 2, 'rho2'),
        (1, 1, 'rho3'), (1, 2, 'rho4'),
        (2, 1, 'rho5'), (2, 2, 'rho3alt'),
    ]

    CMAP = plt.cm.seismic

    # Plot sky maps (column 0)
    for row, col, key in sky_panels:
        ax = fig.add_subplot(gs[row, col])

        sp = GnomonicSkyproj(ax=ax, lon_0=COSMOS_RA, lat_0=COSMOS_DEC)
        sp.set_extent([COSMOS_RA - COSMOS_RADIUS - 0.5, COSMOS_RA + COSMOS_RADIUS + 0.5,
                       COSMOS_DEC - COSMOS_RADIUS - 0.5, COSMOS_DEC + COSMOS_RADIUS + 0.5])

        hpx_data = healpix_maps[key]
        npix = hpg.nside_to_npixel(hpx_data['nside'])
        healpix_map = np.full(npix, hpg.UNSEEN)
        healpix_map[hpx_data['valid_pixels']] = hpx_data['params0']

        vmin, vmax = -sky_color_scales[key], sky_color_scales[key]
        im, _, _, _ = sp.draw_hpxmap(healpix_map, nest=True, zoom=True,
                                      vmin=vmin, vmax=vmax, cmap=CMAP)
        sp.draw_colorbar(label=sky_labels[key], fontsize=10, pad=0.02)
        ax.set_title(f'COSMOS {sky_labels[key]}', fontsize=11)

    # Plot rho statistics (columns 1-2)
    for row, col, rho_name in rho_panels:
        ax = fig.add_subplot(gs[row, col])
        rho = rho_stats[rho_name]

        theta = rho.meanr
        if rho_name == 'rho3alt':
            y = rho.xi
            yerr = np.sqrt(rho.varxi) if rho.varxi is not None else np.zeros_like(y)
        else:
            y = rho.xip
            yerr = np.sqrt(rho.varxip) if rho.varxip is not None else np.zeros_like(y)

        ax.errorbar(theta, y, yerr=yerr, fmt='o-', capsize=2, markersize=3, color='blue')
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)

        # LSSTCam scale markers
        ax.axvline(CCD_SCALE, color='k', linestyle='--', alpha=0.7, lw=1)
        ax.axvline(FOCAL_PLANE_SCALE, color='k', linestyle=':', alpha=0.7, lw=1)

        ax.set_xscale('log')
        if rho_name != 'rho3alt':
            ax.set_yscale('symlog', linthresh=1e-8)
            if rho_name in ylims:
                val = ylims[rho_name]
                ax.set_ylim(-val, val)
        else:
            if rho_name in ylims:
                ax.set_ylim(ylims[rho_name])

        ax.set_xlabel('Separation [arcmin]', fontsize=9)
        ax.set_ylabel(rho_labels[rho_name], fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)

    # Main title
    fig.suptitle(f'COSMOS DDF Rho Statistics | Band: {band} | Visits: {n_visits} | Sources: {n_sources:,}',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()


def run_animation(band, visitMappingFile, repOut, ellipticity_type='distortion',
                  min_sep=0.5, max_sep=250.0, nbins=20, bin_spacing=30,
                  ylims=None, sky_color_scales=None, frame_interval=10, max_visits=None):
    """
    Run animated rho statistics for COSMOS DDF.

    Parameters
    ----------
    band : str
        Band to process
    visitMappingFile : str
        Path to visit_parquet_mapping_skycoord.pkl
    repOut : str
        Output directory for frames
    ellipticity_type : str
        'distortion' or 'shear'
    min_sep, max_sep : float
        Angular separation range in arcmin
    nbins : int
        Number of separation bins
    bin_spacing : float
        HEALPix bin spacing in arcsec for sky maps
    ylims : dict
        Y-axis limits for rho stats
    sky_color_scales : dict
        Color scale limits for sky maps
    frame_interval : int
        Save a frame every N visits
    max_visits : int
        Maximum number of visits to process (for testing). None = all visits.
    """
    print(f"COSMOS DDF Animated Rho Statistics")
    print(f"  Band: {band}")
    print(f"  Ellipticity type: {ellipticity_type}")
    print(f"  Angular bins: {nbins} bins from {min_sep} to {max_sep} arcmin")
    print(f"  Frame interval: every {frame_interval} visits")

    # Load visit mapping
    with open(visitMappingFile, 'rb') as f:
        visit_mapping = pickle.load(f)

    # Filter visits by band and COSMOS overlap
    print("\nFiltering visits that overlap COSMOS...")
    cosmos_visits = []
    for visit, info in visit_mapping.items():
        if info['band'] != band:
            continue
        # Check if visit overlaps COSMOS
        if 'ra' in info and 'dec' in info:
            if visit_overlaps_cosmos(info['ra'], info['dec']):
                cosmos_visits.append((visit, info))

    # Sort by visit number for temporal ordering
    cosmos_visits.sort(key=lambda x: x[0])
    print(f"Found {len(cosmos_visits)} visits overlapping COSMOS in {band}-band")

    # Limit visits for testing
    if max_visits is not None and len(cosmos_visits) > max_visits:
        cosmos_visits = cosmos_visits[:max_visits]
        print(f"Limited to first {max_visits} visits (testing mode)")

    if len(cosmos_visits) == 0:
        print("No visits found. Exiting.")
        return

    # Create output directory
    frames_dir = os.path.join(repOut, f'cosmos_frames_{band}_{ellipticity_type}')
    os.makedirs(frames_dir, exist_ok=True)

    # TreeCorr config (smaller range for DDF)
    treecorr_config = {
        'sep_units': 'arcmin',
        'min_sep': min_sep,
        'max_sep': max_sep,
        'nbins': nbins,
    }

    # Accumulate data across visits
    all_data = {k: [] for k in ['ixx', 'iyy', 'ixy', 'ixx_psf', 'iyy_psf', 'ixy_psf', 'ra', 'dec']}
    frame_count = 0

    for i, (visit, info) in enumerate(tqdm(cosmos_visits, desc="Processing visits")):
        try:
            data = load_single_visit_data(info['parquet_path'])
            # Filter to COSMOS region only
            data = filter_to_cosmos_region(data)

            if len(data['ra']) == 0:
                continue

            for k in all_data:
                all_data[k].append(data[k])
        except Exception as e:
            print(f"  Warning: failed visit {visit}: {e}")
            continue

        # Check if we should save a frame
        n_visits = i + 1
        if n_visits % frame_interval != 0 and n_visits != len(cosmos_visits):
            continue

        # Concatenate all accumulated data
        combined = {k: np.concatenate(v) for k, v in all_data.items()}

        # Filter NaN
        valid = np.isfinite(combined['ixx']) & np.isfinite(combined['iyy']) & np.isfinite(combined['ixy'])
        valid &= np.isfinite(combined['ixx_psf']) & np.isfinite(combined['iyy_psf']) & np.isfinite(combined['ixy_psf'])
        combined = {k: v[valid] for k, v in combined.items()}

        n_sources = len(combined['ra'])
        if n_sources < 100:
            print(f"  Skipping frame {n_visits}: only {n_sources} sources")
            continue

        # Compute rho inputs
        inputs = compute_rho_inputs(combined, ellipticity_type=ellipticity_type)

        # Compute rho statistics
        try:
            rho_stats = compute_rho_statistics(inputs, treecorr_config)
        except Exception as e:
            print(f"  Warning: rho stat computation failed at visit {n_visits}: {e}")
            continue

        # Compute HEALPix maps
        healpix_maps = compute_healpix_maps(inputs, bin_spacing=bin_spacing)

        # Plot frame
        frame_file = os.path.join(frames_dir, f'frame_{frame_count:04d}.png')
        plot_frame(rho_stats, healpix_maps, n_visits, n_sources, frame_file, band,
                   ylims=ylims, sky_color_scales=sky_color_scales)
        frame_count += 1

    print(f"\nSaved {frame_count} frames to: {frames_dir}/")
    print(f"\nTo create animation, run:")
    print(f"  ffmpeg -framerate 5 -i {frames_dir}/frame_%04d.png -c:v libx264 -pix_fmt yuv420p {repOut}/cosmos_rho_anim_{band}.mp4")


def main():
    parser = argparse.ArgumentParser(description="Animated Rho Statistics for COSMOS DDF")
    parser.add_argument('--band', type=str, required=True, help='Band to process (u, g, r, i, z, y)')
    parser.add_argument('--visitMappingFile', type=str, required=True,
                        help='Path to visit_parquet_mapping_skycoord.pkl')
    parser.add_argument('--repOut', type=str, default='cosmos_rho_anim/', help='Output directory')
    parser.add_argument('--ellipticityType', type=str, default='distortion', choices=['distortion', 'shear'],
                        help='Ellipticity definition')
    parser.add_argument('--min_sep', type=float, default=0.5, help='Min separation in arcmin')
    parser.add_argument('--max_sep', type=float, default=250.0, help='Max separation in arcmin')
    parser.add_argument('--nbins', type=int, default=20, help='Number of separation bins')
    parser.add_argument('--bin_spacing', type=float, default=30, help='HEALPix bin spacing in arcsec')
    parser.add_argument('--frame_interval', type=int, default=10, help='Save frame every N visits')
    parser.add_argument('--max_visits', type=int, default=None, help='Max visits to process (for testing)')

    # Y-axis limits
    parser.add_argument('--ylim_rho1', type=float, default=1e-5)
    parser.add_argument('--ylim_rho2', type=float, default=1e-5)
    parser.add_argument('--ylim_rho3', type=float, default=1e-7)
    parser.add_argument('--ylim_rho4', type=float, default=1e-6)
    parser.add_argument('--ylim_rho5', type=float, default=1e-6)
    parser.add_argument('--ylim_rho3alt_min', type=float, default=0)
    parser.add_argument('--ylim_rho3alt_max', type=float, default=2e-5)

    # Sky color scales
    parser.add_argument('--sky_scale_dT', type=float, default=0.02, help='Color scale for dT/T')
    parser.add_argument('--sky_scale_de', type=float, default=0.01, help='Color scale for de1, de2')

    args = parser.parse_args()

    ylims = {
        'rho1': args.ylim_rho1,
        'rho2': args.ylim_rho2,
        'rho3': args.ylim_rho3,
        'rho4': args.ylim_rho4,
        'rho5': args.ylim_rho5,
        'rho3alt': (args.ylim_rho3alt_min, args.ylim_rho3alt_max),
    }

    sky_color_scales = {
        'dT_T': args.sky_scale_dT,
        'e1_res': args.sky_scale_de,
        'e2_res': args.sky_scale_de,
    }

    run_animation(
        band=args.band,
        visitMappingFile=args.visitMappingFile,
        repOut=args.repOut,
        ellipticity_type=args.ellipticityType,
        min_sep=args.min_sep,
        max_sep=args.max_sep,
        nbins=args.nbins,
        bin_spacing=args.bin_spacing,
        ylims=ylims,
        sky_color_scales=sky_color_scales,
        frame_interval=args.frame_interval,
        max_visits=args.max_visits,
    )


if __name__ == "__main__":
    main()
