#!/usr/bin/env python
"""
Export visit data to parquet files for fast access.
One parquet file per visit with all detector data.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from lsst.daf.butler import Butler
import lsst.afw.cameraGeom as cameraGeom
from lsst.obs.lsst import LsstCam
import os
import argparse

camera = LsstCam.getCamera()

SELECTED_COLUMNS = [
    'slot_Shape_xx', 'slot_Shape_yy', 'slot_Shape_xy',
    'slot_Centroid_x', 'slot_Centroid_y',
    'calib_psf_candidate',
]


def pixel_to_focal(x, y, det):
    """Convert pixel coordinates to focal plane coordinates (mm)."""
    tx = det.getTransform(cameraGeom.PIXELS, cameraGeom.FOCAL_PLANE)
    fpx, fpy = tx.getMapping().applyForward(np.vstack((x, y)))
    return fpx.ravel(), fpy.ravel()


def export_visit_to_parquet(butler, visit, collection, output_dir):
    """Export a single visit to parquet file."""

    # Get list of available detectors
    dsrefs = list(butler.registry.queryDatasets(
        "single_visit_star_unstandardized",
        instrument="LSSTCam", visit=visit,
        collections=collection
    ))
    ccdIds = sorted(set(dsr.dataId["detector"] for dsr in dsrefs))

    if len(ccdIds) == 0:
        return False

    all_rows = []

    for ccd in ccdIds:
        try:
            ref = butler.query_datasets("single_visit_star_unstandardized",
                                        data_id={"instrument": 'LSSTCam', "visit": visit, "detector": ccd},
                                        collections=collection)[0]
            table = butler.get(ref, parameters={"columns": SELECTED_COLUMNS}, storageClass="DataFrame")
            table = table[table['calib_psf_candidate']]

            if len(table) == 0:
                continue

            slot_Shape_xx = table['slot_Shape_xx'].to_numpy()
            slot_Shape_yy = table['slot_Shape_yy'].to_numpy()
            slot_Shape_xy = table['slot_Shape_xy'].to_numpy()
            xCCD = table['slot_Centroid_x'].to_numpy()
            yCCD = table['slot_Centroid_y'].to_numpy()

            # Compute second moments
            T_src = slot_Shape_xx + slot_Shape_yy
            e1_src = (slot_Shape_xx - slot_Shape_yy) / T_src
            e2_src = 2 * slot_Shape_xy / T_src

            # Convert to focal plane coordinates
            fpx, fpy = pixel_to_focal(xCCD, yCCD, camera[ccd])

            for i in range(len(xCCD)):
                all_rows.append({
                    'detector': ccd,
                    'xCCD': xCCD[i],
                    'yCCD': yCCD[i],
                    'fpx': fpx[i],
                    'fpy': fpy[i],
                    'T_src': T_src[i],
                    'e1_src': e1_src[i],
                    'e2_src': e2_src[i],
                })
        except Exception as e:
            continue

    if len(all_rows) == 0:
        return False

    df = pd.DataFrame(all_rows)
    output_file = os.path.join(output_dir, f'visit_{visit}.parquet')
    df.to_parquet(output_file, index=False)
    return True


def main():
    defaultCollection = "u/leget/LSSTCam/HeightMapCorrelation20260310"
    defaultVisitIdsFile = "/sdf/home/l/leget/rubin-user/lsst_dev/tickets/dp2_psf/test_night_20260310/visitIds.txt"
    defaultOutputDir = "/sdf/home/l/leget/rubin-user/lsst_dev/tickets/dp2_psf/test_night_20260310/parquet_cache"

    parser = argparse.ArgumentParser(description="Export visit data to parquet files")
    parser.add_argument('--visitIds', type=str, default=defaultVisitIdsFile,
                        help="File with visit IDs (one per line)")
    parser.add_argument('--collection', type=str, default=defaultCollection,
                        help="Butler collection")
    parser.add_argument('--output_dir', type=str, default=defaultOutputDir,
                        help="Output directory for parquet files")
    parser.add_argument('--start_idx', type=int, default=0,
                        help="Start index in visit list (for parallel jobs)")
    parser.add_argument('--end_idx', type=int, default=None,
                        help="End index in visit list (for parallel jobs)")

    args = parser.parse_args()

    # Load visit IDs
    with open(args.visitIds, 'r') as f:
        visits = [int(line.strip()) for line in f if line.strip()]

    # Slice for parallel processing
    if args.end_idx is not None:
        visits = visits[args.start_idx:args.end_idx]
    else:
        visits = visits[args.start_idx:]

    print(f"Processing {len(visits)} visits (index {args.start_idx} to {args.start_idx + len(visits)})")

    os.makedirs(args.output_dir, exist_ok=True)

    butler = Butler('/repo/embargo')

    success_count = 0
    for visit in tqdm(visits, desc="Exporting visits"):
        output_file = os.path.join(args.output_dir, f'visit_{visit}.parquet')
        if os.path.exists(output_file):
            success_count += 1
            continue

        if export_visit_to_parquet(butler, visit, args.collection, args.output_dir):
            success_count += 1
        else:
            print(f"  Failed: visit {visit}")

    print(f"\nExported {success_count}/{len(visits)} visits to {args.output_dir}")


if __name__ == "__main__":
    main()
