#!/usr/bin/env python
"""
Average star images as a function of Zernike coefficient.

For each bin of a given Zernike coefficient (e.g., z4), grab all stars
from visits in that bin (on a single detector) and compute the average
star image.
"""

import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import os
os.environ["POLARS_MAX_THREADS"] = "1"
import polars
import argparse

from lsst.daf.butler import Butler
import lsst.geom as geom


# Default parameters
DEFAULT_DETECTOR = 94
DEFAULT_STAMP_SIZE = 41
PSF_MAX_MIN = 10000
PSF_MAX_MAX = 40000

PARQUET_COLUMNS = [
    'slot_Centroid_x', 'slot_Centroid_y', 'detector', 'psf_max_value',
]


def get_zernike_binning(zernikeKey):
    """Get binning parameters for a given Zernike coefficient."""
    if zernikeKey == 'z4':
        z_central = np.linspace(-1, 1, 41)[5:36]
        half_bin_size = 0.15
    elif zernikeKey in ['z5', 'z6']:
        z_central = np.linspace(-0.5, 0.5, 41)
        half_bin_size = 0.1
    elif zernikeKey == 'z7':
        z_central = np.linspace(-0.25, 0.5, 41)
        half_bin_size = 0.03 * 3
    elif zernikeKey in ['z8', 'z9', 'z10']:
        z_central = np.linspace(-0.25, 0.25, 41)
        half_bin_size = 0.03 * 3
    elif zernikeKey == 'z11':
        z_central = np.linspace(-0.25, 0.25, 41)
        half_bin_size = 0.03
    else:
        # Default binning for other Zernikes
        z_central = np.linspace(-0.5, 0.5, 21)
        half_bin_size = 0.1

    return z_central, half_bin_size


def load_star_stamps(butler, visit, detector, stampSize=DEFAULT_STAMP_SIZE):
    """
    Load and return normalized star stamps for a given visit/detector.

    Returns
    -------
    stars : list of 2D arrays
        Normalized star images
    weights : list of 2D arrays
        Inverse variance weights
    """
    dataID = {
        "instrument": "LSSTCam",
        "visit": visit,
        "detector": detector,
    }

    # Get star positions from parquet
    uri = butler.getURI("refit_psf_star", **dataID)
    parquet_path = uri.geturl()
    psfTable = polars.scan_parquet(parquet_path).select(PARQUET_COLUMNS).collect()
    psfTable = psfTable.filter(polars.col("detector") == detector)

    # Get the calibrated exposure
    calexp = butler.get("preliminary_visit_image", **dataID)

    stars = []
    weights = []

    for i in range(len(psfTable['slot_Centroid_x'])):
        psf_max = psfTable['psf_max_value'][i]
        if psf_max > PSF_MAX_MIN and psf_max < PSF_MAX_MAX:
            try:
                positionStar = geom.Point2D(
                    psfTable['slot_Centroid_x'][i],
                    psfTable['slot_Centroid_y'][i]
                )
                srcImg = calexp.getCutout(positionStar, geom.Extent2I(stampSize, stampSize))
                im = srcImg.getMaskedImage()

                # Normalize by total flux
                total_flux = np.sum(im.image.array)
                if total_flux > 0:
                    star = im.image.array / total_flux
                    var = im.variance.array / total_flux
                    # Avoid division by zero in weights
                    var = np.where(var > 0, var, np.inf)
                    stars.append(star)
                    weights.append(1. / var)
            except Exception:
                continue

    return stars, weights


def compute_average_star(all_stars, all_weights):
    """Compute weighted average of star images."""
    if len(all_stars) == 0:
        return None

    stars = np.array(all_stars)
    weights = np.array(all_weights)

    # Replace inf weights with 0
    weights = np.where(np.isfinite(weights), weights, 0)

    # Weighted average
    sum_weights = np.sum(weights, axis=0)
    sum_weights = np.where(sum_weights > 0, sum_weights, 1)  # Avoid division by zero
    mean_star = np.sum(stars * weights, axis=0) / sum_weights

    return mean_star


def plot_zernike_bin(mean_star, z_all, z_min, z_max, z_median, zernikeKey, n_stars,
                     output_file, bin_idx):
    """Plot the averaged star and Zernike distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plt.subplots_adjust(wspace=0.3)

    # Left: averaged star image
    ax1 = axes[0]
    if mean_star is not None:
        im = ax1.imshow(mean_star, cmap=plt.cm.Greys_r, vmin=0, vmax=np.max(mean_star),
                        origin='lower')
        plt.colorbar(im, ax=ax1, label='Normalized flux')
    else:
        ax1.text(0.5, 0.5, 'No stars', ha='center', va='center', transform=ax1.transAxes)
    ax1.set_title(f'Average PSF ({n_stars} stars)\n{zernikeKey} = {z_median:.3f}', fontsize=14)
    ax1.set_xlabel('x (pixels)')
    ax1.set_ylabel('y (pixels)')

    # Right: Zernike distribution
    ax2 = axes[1]
    if zernikeKey == 'z4':
        binning = np.linspace(-1, 1, 50)
    else:
        binning = np.linspace(-0.6, 0.6, 50)

    ax2.hist(z_all, bins=binning, color='blue', alpha=0.7, edgecolor='black')
    ylim = ax2.get_ylim()
    ax2.fill_betweenx(ylim, z_min, z_max, color='red', alpha=0.3, label='Current bin')
    ax2.axvline(z_median, color='red', linestyle='--', linewidth=2, label=f'Median = {z_median:.3f}')
    ax2.set_ylim(ylim)

    # Zernike label formatting
    if len(zernikeKey) == 2:
        zernike_label = f'${zernikeKey[0]}_{zernikeKey[1]}$'
    else:
        zernike_label = f'${zernikeKey[0]}_{{{zernikeKey[1:]}}}$'

    ax2.set_xlabel(zernike_label, fontsize=16)
    ax2.set_ylabel('Number of visits', fontsize=14)
    ax2.legend(loc='upper right')
    ax2.set_title(f'Zernike distribution (bin {bin_idx})', fontsize=14)

    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_file}")


def avgPSF_vs_Zernike(band='r', zernikeKey='z4', detector=DEFAULT_DETECTOR,
                      visitMappingFile="data/visit_parquet_mapping.pkl",
                      dicZernike="data/visit_to_band_mapv2.pkl",
                      repoButler="dp2_prep",
                      collectionButler="LSSTCam/runs/DRP/DP2/v30_0_0/DM-53881/stage2",
                      repOutPlot='plots/',
                      repOutFile='data/',
                      stampSize=DEFAULT_STAMP_SIZE,
                      max_visits_per_bin=None):
    """
    Compute average PSF as a function of Zernike coefficient.

    Parameters
    ----------
    band : str
        Band to process (default: 'r')
    zernikeKey : str
        Zernike coefficient to bin by (e.g., 'z4', 'z5', ...)
    detector : int
        Detector ID (default: 94)
    visitMappingFile : str
        Path to visit_parquet_mapping.pkl
    dicZernike : str
        Path to Zernike dictionary pickle
    repoButler : str
        Butler repository
    collectionButler : str
        Butler collection
    repOutPlot : str
        Output directory for plots
    repOutFile : str
        Output directory for pickle files
    stampSize : int
        Size of star stamps in pixels
    max_visits_per_bin : int, optional
        Maximum visits to process per bin (for testing)
    """

    print(f"Computing average PSF vs {zernikeKey} for {band}-band, detector {detector}")

    # Initialize butler
    butler = Butler(repoButler, collections=collectionButler)

    # Get valid visits from butler
    refit_psf_star_dsrs = list(butler.registry.queryDatasets("refit_psf_star"))
    visitsDP2 = set()
    for dsr in refit_psf_star_dsrs:
        visitsDP2.add(dsr.dataId["visit"])

    # Load visit mapping
    with open(visitMappingFile, 'rb') as f:
        visit_mapping = pickle.load(f)

    # Load Zernike dictionary
    with open(dicZernike, 'rb') as f:
        zernike_table = pickle.load(f)

    # Build Zernike dictionary for this band
    zernikeDic = {}
    for visit in zernike_table:
        if visit in visitsDP2 and zernike_table[visit]['band'] == band:
            if visit not in zernikeDic:
                zernikeDic[visit] = zernike_table[visit][zernikeKey]

    # Get all Zernike values for histogram
    z_all = [np.median(zernikeDic[visit]) for visit in zernikeDic]
    print(f"  Found {len(zernikeDic)} visits in {band}-band")

    # Get binning
    z_central, half_bin_size = get_zernike_binning(zernikeKey)
    z_min_arr = z_central - half_bin_size
    z_max_arr = z_central + half_bin_size

    # Create output directories
    os.makedirs(repOutPlot, exist_ok=True)
    os.makedirs(repOutFile, exist_ok=True)

    # Results storage
    all_results = {
        'band': band,
        'zernikeKey': zernikeKey,
        'detector': detector,
        'stampSize': stampSize,
        'bins': [],
    }

    # Loop over Zernike bins
    for bin_idx, (z_min, z_max) in enumerate(zip(z_min_arr, z_max_arr)):
        print(f"\nBin {bin_idx}: {zernikeKey} in [{z_min:.3f}, {z_max:.3f}]")

        # Find visits in this bin
        visits_in_bin = []
        z_values_in_bin = []
        for visit in zernikeDic:
            z_med = np.median(zernikeDic[visit])
            if z_min < z_med < z_max:
                if visit in visit_mapping:
                    visits_in_bin.append(visit)
                    z_values_in_bin.append(z_med)

        if len(visits_in_bin) == 0:
            print("  No visits in this bin, skipping...")
            continue

        # Limit visits for testing
        if max_visits_per_bin is not None and len(visits_in_bin) > max_visits_per_bin:
            visits_in_bin = visits_in_bin[:max_visits_per_bin]
            z_values_in_bin = z_values_in_bin[:max_visits_per_bin]

        print(f"  Processing {len(visits_in_bin)} visits...")

        # Collect all stars from visits in this bin
        all_stars = []
        all_weights = []

        for visit in tqdm(visits_in_bin, desc=f"  Loading stars"):
            try:
                stars, weights = load_star_stamps(butler, visit, detector, stampSize)
                all_stars.extend(stars)
                all_weights.extend(weights)
            except Exception as e:
                print(f"    Warning: failed visit {visit}: {e}")

        print(f"  Total stars collected: {len(all_stars)}")

        if len(all_stars) == 0:
            print("  No stars found, skipping...")
            continue

        # Compute average star
        mean_star = compute_average_star(all_stars, all_weights)
        z_median = np.median(z_values_in_bin)

        # Store results
        bin_result = {
            'z_min': z_min,
            'z_max': z_max,
            'z_median': z_median,
            'n_visits': len(visits_in_bin),
            'n_stars': len(all_stars),
            'mean_star': mean_star,
        }
        all_results['bins'].append(bin_result)

        # Plot
        output_plot = os.path.join(repOutPlot, f'avgPSF_{zernikeKey}_bin{bin_idx:02d}_{band}_det{detector}.png')
        plot_zernike_bin(mean_star, z_all, z_min, z_max, z_median, zernikeKey,
                         len(all_stars), output_plot, bin_idx)

    # Save all results
    output_pkl = os.path.join(repOutFile, f'avgPSF_vs_{zernikeKey}_{band}_det{detector}.pkl')
    with open(output_pkl, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"\nSaved results: {output_pkl}")


def main():
    defaultCollectionButler = "LSSTCam/runs/DRP/DP2/v30_0_0/DM-53881/stage2"
    defaultDicZernike = "/sdf/home/l/leget/rubin-user/lsst_dev/tickets/dp2_psf/data/visit_to_band_mapv2.pkl"
    defaultVisitMappingFile = "/sdf/home/l/leget/rubin-user/lsst_dev/tickets/dp2_psf/data/visit_parquet_mapping.pkl"
    defaultRepOutPlot = "/sdf/home/l/leget/rubin-user/lsst_dev/tickets/dp2_psf/plots/avgPSF/"
    defaultRepOutFile = "/sdf/home/l/leget/rubin-user/lsst_dev/tickets/dp2_psf/data/"

    parser = argparse.ArgumentParser(description="Average PSF as function of Zernike coefficient")
    parser.add_argument('--band', type=str, default='r', help="Band to process (default: r)")
    parser.add_argument('--zernikeKey', type=str, default='z4', help="Zernike coefficient (default: z4)")
    parser.add_argument('--detector', type=int, default=DEFAULT_DETECTOR,
                        help=f"Detector ID (default: {DEFAULT_DETECTOR})")
    parser.add_argument('--visitMappingFile', type=str, default=defaultVisitMappingFile,
                        help="Path to visit_parquet_mapping.pkl")
    parser.add_argument('--dicZernike', type=str, default=defaultDicZernike,
                        help="Path to Zernike dictionary pickle")
    parser.add_argument('--repoButler', type=str, default='dp2_prep', help="Butler repository")
    parser.add_argument('--collectionButler', type=str, default=defaultCollectionButler,
                        help="Butler collection")
    parser.add_argument('--repOutPlot', type=str, default=defaultRepOutPlot,
                        help="Output directory for plots")
    parser.add_argument('--repOutFile', type=str, default=defaultRepOutFile,
                        help="Output directory for pickle files")
    parser.add_argument('--stampSize', type=int, default=DEFAULT_STAMP_SIZE,
                        help=f"Stamp size in pixels (default: {DEFAULT_STAMP_SIZE})")
    parser.add_argument('--max_visits_per_bin', type=int, default=None,
                        help="Max visits per bin (for testing)")

    args = parser.parse_args()

    avgPSF_vs_Zernike(
        band=args.band,
        zernikeKey=args.zernikeKey,
        detector=args.detector,
        visitMappingFile=args.visitMappingFile,
        dicZernike=args.dicZernike,
        repoButler=args.repoButler,
        collectionButler=args.collectionButler,
        repOutPlot=args.repOutPlot,
        repOutFile=args.repOutFile,
        stampSize=args.stampSize,
        max_visits_per_bin=args.max_visits_per_bin,
    )


if __name__ == "__main__":
    main()
