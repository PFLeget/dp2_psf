#!/usr/bin/env python
"""
Process astrometric residuals for a single healpix region.
Designed to be run as a SLURM job.

Usage:
    python getAstro_single.py --healpix 123
    python getAstro_single.py --healpix 123 456 789  # multiple healpix
"""
import numpy as np
from lsst.daf.butler import Butler
from lsst.sphgeom import HealpixPixelization
from lsst.afw.cameraGeom import FOCAL_PLANE, PIXELS
from astropy.table import Table
import os
import argparse


pixelization = HealpixPixelization(3)


def getFpResids(healpix, fitStarsRef, butler, camera, output_dir):
    """
    Compute focal plane astrometric residuals for a single healpix region.

    Parameters
    ----------
    healpix : int
        Healpix ID to process
    fitStarsRef : DatasetRef
        Butler dataset reference for fitStars
    butler : Butler
        Data butler
    camera : Camera
        LSSTCam camera object
    output_dir : str
        Directory to save output parquet files
    """
    print(f"Starting healpix {healpix}")

    # Check if output already exists
    output_file = os.path.join(output_dir, f'healpix_{healpix}.parq')
    if os.path.exists(output_file):
        print(f"Output already exists, skipping: {output_file}")
        return

    fitStars = butler.get(fitStarsRef, storageClass='ArrowAstropy')
    gaiaInd = fitStars['exposureName'] == 'REFERENCE'
    visits = np.unique(fitStars['exposureName'][~gaiaInd])

    if len(visits) <= 12:
        print(f"Skipping healpix {healpix} - only {len(visits)} visits")
        return

    starCat = butler.get('gbdesHealpix3AstrometricFit_starCatalog',
                         dataId=fitStarsRef.dataId, storageClass='ArrowAstropy')
    starCat.add_index('starMatchID')
    healpixRegion = pixelization.pixel(healpix)

    print(f"Healpix {healpix}: {len(visits)} visits")

    outDX = []
    outDY = []
    fpX = []
    fpY = []

    for visit in visits:
        try:
            wcsCat = butler.get('visit_summary', visit=int(visit))
        except Exception as e:
            print(f"Skip visit {visit}: {e}")
            continue

        for row in wcsCat:
            detector = row.getId()
            wcs = row.wcs
            if wcs is None:
                continue

            detSources = fitStars[(fitStars['exposureName'] == visit) &
                                  (fitStars['deviceName'] == str(detector))]
            detStars = starCat.loc[detSources['matchID']]

            # Only use stars actually in this healpix to avoid double counting
            inRegion = healpixRegion.contains(detStars['ra'] * np.pi / 180,
                                               detStars['dec'] * np.pi / 180)
            starX, starY = wcs.skyToPixelArray(detStars['ra'][inRegion],
                                                detStars['dec'][inRegion], degrees=True)
            dX = detSources['xpix'][inRegion] - starX
            dY = detSources['ypix'][inRegion] - starY

            mapping = camera[detector].getTransform(PIXELS, FOCAL_PLANE).getMapping()
            fpXY = mapping.applyForward(np.array([detSources['xpix'][inRegion],
                                                   detSources['ypix'][inRegion]]))
            outDX.append(dX)
            outDY.append(dY)
            fpX.append(fpXY[0])
            fpY.append(fpXY[1])

    if len(outDX) == 0:
        print(f"No data for healpix {healpix}")
        return

    outDX = np.concatenate(outDX)
    outDY = np.concatenate(outDY)
    fpX = np.concatenate(fpX)
    fpY = np.concatenate(fpY)

    tab = Table({'fpX': fpX, 'fpY': fpY, 'dx': outDX, 'dy': outDY})
    tab.write(output_file, overwrite=True)
    print(f"Saved: {output_file} ({len(tab)} stars)")


def main():
    parser = argparse.ArgumentParser(description="Process astrometric residuals for healpix regions")
    parser.add_argument('--healpix', type=int, nargs='+', required=True,
                        help="Healpix ID(s) to process")
    parser.add_argument('--output_dir', type=str,
                        default='/sdf/scratch/users/c/csaunder/DP2_fp_resids',
                        help="Output directory for parquet files")
    parser.add_argument('--physical_filter', type=str, default='i_39',
                        help="Physical filter to process (default: i_39)")
    parser.add_argument('--repo', type=str, default='dp2_prep',
                        help="Butler repository (default: dp2_prep)")
    parser.add_argument('--collection', type=str,
                        default='LSSTCam/runs/DRP/DP2/v30_0_0/DM-53881/stage2',
                        help="Butler collection")

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize butler
    print(f"Initializing butler: {args.repo}, collection: {args.collection}")
    butler = Butler(args.repo, collections=args.collection)
    camera = butler.get('camera', instrument='LSSTCam')

    # Get all fitStars references
    print(f"Querying datasets for filter: {args.physical_filter}")
    fitStarsRefs = butler.query_datasets('gbdesHealpix3AstrometricFit_fitStars',
                                          physical_filter=args.physical_filter)
    fitStarsDict = {ref.dataId['healpix3']: ref for ref in fitStarsRefs}

    print(f"Found {len(fitStarsDict)} healpix regions")
    print(f"Processing {len(args.healpix)} healpix: {args.healpix}")

    # Process each requested healpix
    for healpix in args.healpix:
        if healpix not in fitStarsDict:
            print(f"Healpix {healpix} not found in dataset")
            continue
        getFpResids(healpix, fitStarsDict[healpix], butler, camera, args.output_dir)

    print("Done!")


if __name__ == "__main__":
    main()
