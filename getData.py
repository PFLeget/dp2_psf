from lsst.daf.butler import Butler
import numpy as np
from tqdm import tqdm
import os
import argparse
import pickle


def getData(repOut='data/'):
    """
    Create a mapping file that stores the parquet file location for each visit.
    This avoids copying data locally - instead we just reference the original files.
    """

    repo = "dp2_prep"
    collection = "LSSTCam/runs/DRP/DP2/v30_0_0/DM-53881/stage2"

    butler = Butler(repo, collections=collection)

    PSFTable_visit_dsrs = list(butler.registry.queryDatasets("refit_psf_star"))

    visit_mapping = {}

    for dsr in tqdm(PSFTable_visit_dsrs, desc="Building visit mapping"):
        visit = dsr.dataId["visit"]
        band = dsr.dataId["band"]

        # Get the URI to the parquet file (fast operation)
        uri = butler.getURI("refit_psf_star", instrument="LSSTCam", visit=visit)
        parquet_path = uri.geturl()

        visit_mapping[visit] = {
            'parquet_path': parquet_path,
            'band': band,
        }

    # Save the mapping
    output_file = os.path.join(repOut, 'visit_parquet_mapping.pkl')
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
    parser = argparse.ArgumentParser(description="Create visit to parquet file mapping.")
    parser.add_argument("--repOut", type=str, default='data/', help="Output directory for the mapping pickle file")
    args = parser.parse_args()

    getData(repOut=args.repOut)


if __name__ == "__main__":
    main()
