#!/usr/bin/env python
"""
Create a mapping file that stores the parquet file location for each tract.
Equivalent to getData.py but for coadd data (object_all from stage3).
"""

from lsst.daf.butler import Butler
import numpy as np
from tqdm import tqdm
import os
import argparse
import pickle


def getDataCoadd(repOut='data/'):
    """
    Create a mapping file that stores the parquet file location for each tract.
    This avoids copying data locally - instead we just reference the original files.
    """

    repo = "dp2_prep"
    collection = "LSSTCam/runs/DRP/DP2/v30_0_0/DM-53881/stage3"

    butler = Butler(repo, collections=collection)

    objectAll_dsrs = list(butler.registry.queryDatasets("object_all"))

    tract_mapping = {}

    for dsr in tqdm(objectAll_dsrs, desc="Building tract mapping"):
        tract = dsr.dataId["tract"]

        # Get the URI to the parquet file (fast operation)
        uri = butler.getURI("object_all", skymap='lsst_cells_v2', tract=tract)
        parquet_path = uri.geturl()

        tract_mapping[tract] = {
            'parquet_path': parquet_path,
        }

    # Save the mapping
    os.makedirs(repOut, exist_ok=True)
    output_file = os.path.join(repOut, 'tract_parquet_mapping.pkl')
    with open(output_file, 'wb') as f:
        pickle.dump(tract_mapping, f)

    print(f"Saved tract mapping to {output_file}")
    print(f"Total tracts: {len(tract_mapping)}")


def main():
    parser = argparse.ArgumentParser(description="Create tract to parquet file mapping for coadd data.")
    parser.add_argument("--repOut", type=str, default='data/', help="Output directory for the mapping pickle file")
    args = parser.parse_args()

    getDataCoadd(repOut=args.repOut)


if __name__ == "__main__":
    main()
