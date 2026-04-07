#!/usr/bin/env python
"""
Generate star images for annotation.

For a given visit, generates a 3x3 grid of star stamps per detector
and saves as plotForAnnotation/{visit}_{detector}_{FLAG}.png

FLAG is determined by z4:
- abs(z4 + 0.25) > 1.5 → likely_bad
- abs(z4 + 0.25) <= 1.5 → likely_good
"""

import numpy as np
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
DEFAULT_STAMP_SIZE = 41
Z4_THRESHOLD = 1.5

PARQUET_COLUMNS = [
    'slot_Centroid_x', 'slot_Centroid_y', 'detector',
]


def get_z4_flag(z4_median):
    """Determine flag based on z4 value."""
    if abs(z4_median + 0.25) > Z4_THRESHOLD:
        return "likely_bad"
    else:
        return "likely_good"


def generate_star_grid(butler, visit, detector, stampSize=DEFAULT_STAMP_SIZE):
    """
    Generate a 3x3 grid of star stamps for a given visit/detector.

    Returns the figure or None if failed.
    """
    dataID = {
        "instrument": "LSSTCam",
        "visit": visit,
        "detector": detector,
    }

    try:
        # Get star positions from parquet
        uri = butler.getURI("refit_psf_star", **dataID)
        parquet_path = uri.geturl()
        psfTable = polars.scan_parquet(parquet_path).select(PARQUET_COLUMNS).collect()
        psfTable = psfTable.filter(polars.col("detector") == detector)

        if len(psfTable) < 9:
            print(f"    Not enough stars ({len(psfTable)}) for detector {detector}")
            return None

        # Get the calibrated exposure
        calexp = butler.get("preliminary_visit_image", **dataID)

        fig, axes = plt.subplots(3, 3, figsize=(6, 6))
        plt.subplots_adjust(wspace=0.02, hspace=0.02)

        for i in range(9):
            ax = axes.flat[i]
            try:
                positionStar = geom.Point2D(
                    psfTable['slot_Centroid_x'][i],
                    psfTable['slot_Centroid_y'][i]
                )
                srcImg = calexp.getCutout(positionStar, geom.Extent2I(stampSize, stampSize))
                im = srcImg.getMaskedImage()
                star = im.image.array / np.sum(im.image.array)

                ax.imshow(star, cmap=plt.cm.Greys_r, vmin=0, vmax=np.max(star), origin='lower')
            except Exception:
                ax.text(0.5, 0.5, 'X', ha='center', va='center', fontsize=20)

            ax.set_xticks([])
            ax.set_yticks([])

        return fig

    except Exception as e:
        print(f"    Failed detector {detector}: {e}")
        return None


def generate_visit_images(visit, dicZernike, repoButler="dp2_prep",
                          collectionButler="LSSTCam/runs/DRP/DP2/v30_0_0/DM-53881/stage2",
                          repOut='plotForAnnotation/',
                          stampSize=DEFAULT_STAMP_SIZE):
    """
    Generate annotation images for all detectors of a visit.

    Parameters
    ----------
    visit : int
        Visit ID
    dicZernike : str
        Path to Zernike dictionary pickle
    repoButler : str
        Butler repository
    collectionButler : str
        Butler collection
    repOut : str
        Output directory for plots
    stampSize : int
        Size of star stamps in pixels
    """

    print(f"Processing visit {visit}")

    # Load Zernike dictionary to get z4 and determine flag
    with open(dicZernike, 'rb') as f:
        zernike_table = pickle.load(f)

    if visit not in zernike_table:
        print(f"  Visit {visit} not found in Zernike table, skipping...")
        return

    z4_median = np.nanmedian(zernike_table[visit]['z4'])
    flag = get_z4_flag(z4_median)
    print(f"  z4 = {z4_median:.3f}, flag = {flag}")

    # Initialize butler
    butler = Butler(repoButler, collections=collectionButler)

    # Get all detectors for this visit
    refit_psf_star_dsrs = list(butler.registry.queryDatasets(
        "refit_psf_star",
        instrument="LSSTCam",
        visit=visit
    ))

    detectors = sorted(set(dsr.dataId["detector"] for dsr in refit_psf_star_dsrs))
    print(f"  Found {len(detectors)} detectors")

    # Create output directory
    os.makedirs(repOut, exist_ok=True)

    # Generate images for each detector
    for detector in detectors:
        fig = generate_star_grid(butler, visit, detector, stampSize)

        if fig is not None:
            # Add title with visit, detector, and flag
            fig.suptitle(f"Visit {visit} | Det {detector} | {flag} (z4={z4_median:.2f})",
                         fontsize=10, y=0.98)

            output_file = os.path.join(repOut, f"{visit}_{detector}_{flag}.png")
            fig.savefig(output_file, dpi=100, bbox_inches='tight')
            plt.close(fig)
            print(f"    Saved: {output_file}")


def main():
    defaultCollectionButler = "LSSTCam/runs/DRP/DP2/v30_0_0/DM-53881/stage2"
    defaultDicZernike = "/sdf/home/l/leget/rubin-user/lsst_dev/tickets/dp2_psf/data/visit_to_band_mapv2.pkl"
    defaultRepOut = "/sdf/home/l/leget/rubin-user/lsst_dev/tickets/dp2_psf/miniDonutsClassifier/plotForAnnotation/"

    parser = argparse.ArgumentParser(description="Generate star images for annotation")
    parser.add_argument('--visit', type=int, required=True, help="Visit ID to process")
    parser.add_argument('--dicZernike', type=str, default=defaultDicZernike,
                        help="Path to Zernike dictionary pickle")
    parser.add_argument('--repoButler', type=str, default='dp2_prep', help="Butler repository")
    parser.add_argument('--collectionButler', type=str, default=defaultCollectionButler,
                        help="Butler collection")
    parser.add_argument('--repOut', type=str, default=defaultRepOut,
                        help="Output directory for plots")
    parser.add_argument('--stampSize', type=int, default=DEFAULT_STAMP_SIZE,
                        help=f"Stamp size in pixels (default: {DEFAULT_STAMP_SIZE})")

    args = parser.parse_args()

    generate_visit_images(
        visit=args.visit,
        dicZernike=args.dicZernike,
        repoButler=args.repoButler,
        collectionButler=args.collectionButler,
        repOut=args.repOut,
        stampSize=args.stampSize,
    )


if __name__ == "__main__":
    main()
