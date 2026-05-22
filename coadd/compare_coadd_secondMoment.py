#!/usr/bin/env python
"""
Compare coadd second moments between two collections.
Creates 3 separate plots (A, B, diff) with fixed xlim/ylim.
Only loads tracts that exist in both collections.
"""

import numpy as np
import treegp
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import hpgeom as hpg

import os
os.environ["POLARS_MAX_THREADS"] = "1"
import polars

import pickle
import argparse

from lsst.daf.butler import Butler

from skyproj import McBrydeSkyproj
from skyproj.survey import _Survey


class SurveyMcBrydeSkyproj(_Survey, McBrydeSkyproj):
    """McBryde projection with survey footprint drawing capabilities."""
    pass


def get_parquet_columns(band):
    """Get the columns to read from parquet for a given band."""
    return [
        f"{band}_calib_psf_used", "detect_isPrimary", "refExtendedness",
        f"{band}_pixelFlags_inexact_psfCenter",
        f"{band}_ixxPSF", f"{band}_iyyPSF", f"{band}_ixyPSF",
        f"{band}_ixx", f"{band}_iyy", f"{band}_ixy",
        f"{band}_ra", f"{band}_dec",
    ]


def load_tract_data(parquet_path, band):
    """Load tract data from parquet file and compute derived columns."""
    columns = get_parquet_columns(band)
    table = polars.scan_parquet(parquet_path).select(columns).collect()

    # Apply quality filters
    mask = table['detect_isPrimary'].to_numpy() == True
    mask &= table['refExtendedness'].to_numpy() == 0.0
    mask &= table[f'{band}_calib_psf_used'].to_numpy() == True
    mask &= table[f'{band}_pixelFlags_inexact_psfCenter'].to_numpy() == False

    ixx_src = table[f"{band}_ixx"].to_numpy()[mask]
    iyy_src = table[f"{band}_iyy"].to_numpy()[mask]
    ixy_src = table[f"{band}_ixy"].to_numpy()[mask]
    ixx_psf = table[f"{band}_ixxPSF"].to_numpy()[mask]
    iyy_psf = table[f"{band}_iyyPSF"].to_numpy()[mask]
    ixy_psf = table[f"{band}_ixyPSF"].to_numpy()[mask]
    ra = table[f"{band}_ra"].to_numpy()[mask]
    dec = table[f"{band}_dec"].to_numpy()[mask]

    T_src = ixx_src + iyy_src
    T_psf = ixx_psf + iyy_psf

    e1_src = (ixx_src - iyy_src) / T_src
    e2_src = 2 * ixy_src / T_src
    e1_psf = (ixx_psf - iyy_psf) / T_psf
    e2_psf = 2 * ixy_psf / T_psf

    return {
        'dT_T': (T_src - T_psf) / T_src,
        'de1': e1_src - e1_psf,
        'de2': e2_src - e2_psf,
        'ra': ra,
        'dec': dec,
    }


def get_tracts_from_collection(butler, collection):
    """Get list of tracts available in a collection."""
    refs = list(butler.registry.queryDatasets("object_all", collections=collection))
    tracts = sorted(set(ref.dataId["tract"] for ref in refs))
    return tracts


def process_collection(butler, collection, tracts, band, key, bin_spacing):
    """Process a collection and return HEALPix map data."""
    meanify_obj = treegp.meanify_healpix(bin_spacing=bin_spacing)

    n_loaded = 0
    n_stars = 0
    for tract in tqdm(tracts, desc=f"Loading {collection[:40]}"):
        try:
            uri = butler.getURI("object_all", instrument="LSSTCam",
                               skymap="lsst_cells_v2", tract=tract,
                               collections=collection)
            data = load_tract_data(uri.geturl(), band)

            if len(data['ra']) == 0:
                continue

            coord = np.array([data['ra'], data['dec']]).T
            valid = np.isfinite(data[key])

            if np.sum(valid) > 0:
                meanify_obj.add_field(coord[valid], data[key][valid])
                n_loaded += 1
                n_stars += np.sum(valid)
        except Exception as e:
            pass

    meanify_obj.meanify()

    return {
        'coords0': meanify_obj.coords0,
        'params0': meanify_obj.params0,
        'nside': meanify_obj.nside,
        'valid_pixels': meanify_obj._valid_pixels,
        'n_tracts': n_loaded,
        'n_stars': n_stars,
    }


def compute_extent(result_A, result_B):
    """Compute common extent (xlim, ylim) from both results."""
    # Get all RA/Dec from both results
    all_ra = np.concatenate([result_A['coords0'][:, 0], result_B['coords0'][:, 0]])
    all_dec = np.concatenate([result_A['coords0'][:, 1], result_B['coords0'][:, 1]])

    ra_min, ra_max = np.nanmin(all_ra), np.nanmax(all_ra)
    dec_min, dec_max = np.nanmin(all_dec), np.nanmax(all_dec)

    # Add small margin
    ra_margin = (ra_max - ra_min) * 0.05
    dec_margin = (dec_max - dec_min) * 0.05

    return (ra_max + ra_margin, ra_min - ra_margin), (dec_min - dec_margin, dec_max + dec_margin)


def plot_single_map(result, key, band, label, repOutPlot, colorScale, xlim, ylim, suffix):
    """Create a single sky map plot."""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111)

    CMAP = plt.cm.seismic
    vmin, vmax = -colorScale, colorScale

    key_labels = {
        'dT_T': r'$\delta T / T$',
        'de1': r'$\delta e_1$',
        'de2': r'$\delta e_2$',
    }
    ksm = key_labels.get(key, key)

    # Build HEALPix map
    nside = result['nside']
    npix = hpg.nside_to_npixel(nside)
    healpix_map = np.full(npix, hpg.UNSEEN)
    healpix_map[result['valid_pixels']] = result['params0']

    sp = SurveyMcBrydeSkyproj(ax=ax, lon_0=0.0)
    sp.draw_hpxmap(healpix_map, nest=True, zoom=False, vmin=vmin, vmax=vmax, cmap=CMAP)

    # Set fixed extent
    sp.ax.set_xlim(xlim)
    sp.ax.set_ylim(ylim)

    sp.draw_colorbar(label=ksm, fontsize=14)
    ax.set_title(f"{label}\n{band}-band | N_tracts={result['n_tracts']} | N_stars={result['n_stars']:,}",
                 fontsize=14)

    plt.tight_layout()

    os.makedirs(repOutPlot, exist_ok=True)
    output_file = os.path.join(repOutPlot, f'compare_coadd_{key}_{band}_{suffix}.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")


def plot_difference_map(result_A, result_B, key, band, label_A, label_B, repOutPlot, colorScale, xlim, ylim):
    """Create difference map plot."""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111)

    CMAP = plt.cm.seismic
    diff_scale = colorScale / 2

    key_labels = {
        'dT_T': r'$\delta T / T$',
        'de1': r'$\delta e_1$',
        'de2': r'$\delta e_2$',
    }
    ksm = key_labels.get(key, key)

    # Build difference HEALPix map
    nside = result_A['nside']
    npix = hpg.nside_to_npixel(nside)
    map_diff = np.full(npix, hpg.UNSEEN)

    # Find common pixels
    common_pixels = np.intersect1d(result_A['valid_pixels'], result_B['valid_pixels'])

    # Create index mapping for params0
    idx_A = {p: i for i, p in enumerate(result_A['valid_pixels'])}
    idx_B = {p: i for i, p in enumerate(result_B['valid_pixels'])}

    for pix in common_pixels:
        map_diff[pix] = result_B['params0'][idx_B[pix]] - result_A['params0'][idx_A[pix]]

    sp = SurveyMcBrydeSkyproj(ax=ax, lon_0=0.0)
    sp.draw_hpxmap(map_diff, nest=True, zoom=False, vmin=-diff_scale, vmax=diff_scale, cmap=CMAP)

    # Set fixed extent
    sp.ax.set_xlim(xlim)
    sp.ax.set_ylim(ylim)

    sp.draw_colorbar(label=f'{ksm} (B - A)', fontsize=14)
    ax.set_title(f"Difference: {label_B} - {label_A}\n{band}-band | N_common_pixels={len(common_pixels)}",
                 fontsize=14)

    plt.tight_layout()

    os.makedirs(repOutPlot, exist_ok=True)
    output_file = os.path.join(repOutPlot, f'compare_coadd_{key}_{band}_diff.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Compare coadd second moments between two collections")
    parser.add_argument('--repo', type=str, default='dp2_prep')
    parser.add_argument('--collection_A', type=str,
                        default='LSSTCam/runs/DRP/DP2/v30_0_6_rc1/DM-53881/stage3',
                        help='Collection A (reference)')
    parser.add_argument('--collection_B', type=str,
                        default='u/mullaney/DM-54957_subregion/20260521T063810Z',
                        help='Collection B (test)')
    parser.add_argument('--label_A', type=str, default='DP2 v30_0_6_rc1',
                        help='Label for collection A')
    parser.add_argument('--label_B', type=str, default='DM-54957',
                        help='Label for collection B')
    parser.add_argument('--band', type=str, default='i', help='Band to process')
    parser.add_argument('--key', type=str, default='dT_T',
                        choices=['dT_T', 'de1', 'de2'],
                        help='Second moment key to plot')
    parser.add_argument('--bin_spacing', type=float, default=3600,
                        help='HEALPix bin spacing in arcsec')
    parser.add_argument('--colorScale', type=float, default=0.005,
                        help='Color scale for plots')
    parser.add_argument('--repOutPlot', type=str, default='plots/',
                        help='Output directory')

    args = parser.parse_args()

    print(f"Comparing coadd {args.key} in {args.band}-band")
    print(f"  Collection A: {args.collection_A}")
    print(f"  Collection B: {args.collection_B}")

    butler = Butler(args.repo)

    # Get tracts from both collections and find intersection
    print(f"\nQuerying tracts from both collections...")
    tracts_A = set(get_tracts_from_collection(butler, args.collection_A))
    tracts_B = set(get_tracts_from_collection(butler, args.collection_B))
    common_tracts = sorted(tracts_A & tracts_B)
    print(f"  Collection A: {len(tracts_A)} tracts")
    print(f"  Collection B: {len(tracts_B)} tracts")
    print(f"  Intersection: {len(common_tracts)} tracts")

    if len(common_tracts) == 0:
        print("No common tracts found!")
        return

    # Process both collections (only common tracts)
    print(f"\nProcessing collection A...")
    result_A = process_collection(butler, args.collection_A, common_tracts, args.band,
                                   args.key, args.bin_spacing)

    print(f"\nProcessing collection B...")
    result_B = process_collection(butler, args.collection_B, common_tracts, args.band,
                                   args.key, args.bin_spacing)

    # Compute common extent
    xlim, ylim = compute_extent(result_A, result_B)
    print(f"\nFixed extent: xlim={xlim}, ylim={ylim}")

    # Create 3 separate plots with same extent
    print(f"\nCreating plots...")
    plot_single_map(result_A, args.key, args.band, args.label_A, args.repOutPlot,
                    args.colorScale, xlim, ylim, 'A')
    plot_single_map(result_B, args.key, args.band, args.label_B, args.repOutPlot,
                    args.colorScale, xlim, ylim, 'B')
    plot_difference_map(result_A, result_B, args.key, args.band,
                        args.label_A, args.label_B, args.repOutPlot, args.colorScale, xlim, ylim)

    # Save results
    results = {
        'A': result_A,
        'B': result_B,
        'collection_A': args.collection_A,
        'collection_B': args.collection_B,
        'label_A': args.label_A,
        'label_B': args.label_B,
        'key': args.key,
        'band': args.band,
        'common_tracts': common_tracts,
        'xlim': xlim,
        'ylim': ylim,
    }
    pkl_file = os.path.join(args.repOutPlot, f'compare_coadd_{args.key}_{args.band}.pkl')
    with open(pkl_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"Saved: {pkl_file}")


if __name__ == "__main__":
    main()
