#!/usr/bin/env python
"""
Generate a mapping of visit -> set of detectors that went into coadd.

This is used to filter single visit data for apples-to-apples comparison with coadd.
"""

import pickle
import argparse
from tqdm import tqdm

from lsst.daf.butler import Butler


def get_coadd_detector_mapping(repo, collection, skymap="lsst_cells_v2"):
    """
    Get mapping of visits to detectors that went into the coadd.

    Parameters
    ----------
    repo : str
        Butler repository path
    collection : str
        Collection name
    skymap : str
        Skymap name

    Returns
    -------
    dict
        {visit: set of detector IDs}
    """
    butler = Butler(repo, collections=collection)
    table = butler.get("deep_coadd_input_summary", skymap=skymap)

    dic = {}
    for visit, detector in tqdm(zip(table['visit'], table['detector']),
                                 total=len(table['visit']),
                                 desc="Building coadd detector mapping"):
        visit = int(visit)
        detector = int(detector)
        if visit not in dic:
            dic[visit] = set({detector})
        else:
            dic[visit].add(detector)

    return dic


def main():
    parser = argparse.ArgumentParser(description="Generate coadd detector mapping")
    parser.add_argument('--repo', type=str, default='dp2_prep',
                        help='Butler repository')
    parser.add_argument('--collection', type=str,
                        default='LSSTCam/runs/DRP/DP2/v30_0_6_rc1/DM-53881/stage3',
                        help='Collection name')
    parser.add_argument('--skymap', type=str, default='lsst_cells_v2',
                        help='Skymap name')
    parser.add_argument('--output', type=str, default='data/coadd_detector_mapping.pkl',
                        help='Output pickle file')

    args = parser.parse_args()

    mapping = get_coadd_detector_mapping(args.repo, args.collection, args.skymap)

    print(f"Total visits: {len(mapping)}")
    total_detectors = sum(len(v) for v in mapping.values())
    print(f"Total visit-detector pairs: {total_detectors}")

    with open(args.output, 'wb') as f:
        pickle.dump(mapping, f)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
