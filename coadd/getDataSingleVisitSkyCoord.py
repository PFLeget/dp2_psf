#!/usr/bin/env python
"""
Create a mapping file for single visit data with sky coordinate moments.
Uses the re-run collection with Iuu, Ivv, Iuv columns (arcsec^2).
"""

from lsst.daf.butler import Butler
import numpy as np
from tqdm import tqdm
import os
import argparse
import pickle


def getDataSingleVisitSkyCoord(repOut='../data/'):
    """
    Create a mapping file that stores the parquet file location for each visit.
    Uses the re-run with sky coordinate moments (shape_Iuu, shape_Ivv, shape_Iuv).
    """

    repo = "dp2_prep"
    collection = "u/leget/LSSTCam/DM-54194/DP2"

    butler = Butler(repo, collections=collection)

    # Query for refit_psf_star (single visit PSF star table with sky moments)
    refit_psf_dsrs = list(butler.registry.queryDatasets("refit_psf_star"))

    visit_mapping = {}

    for dsr in tqdm(refit_psf_dsrs, desc="Building visit mapping"):
        visit = dsr.dataId["visit"]
        band = dsr.dataId["band"]

        if visit not in visit_mapping:
            # Get the URI to the parquet file
            uri = butler.getURI("refit_psf_star", instrument="LSSTCam", visit=visit)
            parquet_path = uri.geturl()

            visit_mapping[visit] = {
                'parquet_path': parquet_path,
                'band': band,
            }

    # Save the mapping
    os.makedirs(repOut, exist_ok=True)
    output_file = os.path.join(repOut, 'visit_parquet_mapping_skycoord.pkl')
    with open(output_file, 'wb') as f:
        pickle.dump(visit_mapping, f)

    print(f"Saved visit mapping to {output_file}")
    print(f"Total visits: {len(visit_mapping)}")

    # Print summary by band
    bands = {}
    for visit, info in visit_mapping.items():
        band = info['band']
        bands[band] = bands.get(band, 0) + 1
    for band, count in sorted(bands.items()):
        print(f"  {band}: {count} visits")


def main():
    parser = argparse.ArgumentParser(description="Create visit to parquet file mapping for sky coord data.")
    parser.add_argument("--repOut", type=str, default='../data/', help="Output directory for the mapping pickle file")
    args = parser.parse_args()

    getDataSingleVisitSkyCoord(repOut=args.repOut)


if __name__ == "__main__":
    main()
