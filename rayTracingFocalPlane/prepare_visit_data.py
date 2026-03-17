#!/usr/bin/env python
"""
Prepare visit data for optical Zernike fitting.
Extracts PSF star second moments and converts to focal plane coordinates.
"""

import numpy as np
import polars as pl
import argparse
from lsst.daf.butler import Butler
import lsst.afw.cameraGeom as cameraGeom
from lsst.obs.lsst import LsstCam

camera = LsstCam.getCamera()

PARQUET_COLUMNS = [
    'slot_Shape_xx', 'slot_Shape_yy', 'slot_Shape_xy',
    'slot_PsfShape_xx', 'slot_PsfShape_xy', 'slot_PsfShape_yy',
    'slot_Centroid_x', 'slot_Centroid_y',
    'detector', 'calib_psf_candidate',
]


def pixel_to_focal(x, y, det):
    """
    Convert pixel coordinates to focal plane coordinates.

    Parameters
    ----------
    x, y : array
        Pixel coordinates.
    det : lsst.afw.cameraGeom.Detector
        Detector object.

    Returns
    -------
    fpx, fpy : array
        Focal plane position in millimeters.
    """
    tx = det.getTransform(cameraGeom.PIXELS, cameraGeom.FOCAL_PLANE)
    fpx, fpy = tx.getMapping().applyForward(np.vstack((x, y)))
    return fpx.ravel(), fpy.ravel()


def prepare_visit_data(visit, collection, output_file, repo='/repo/main'):
    """
    Prepare visit data for Zernike fitting.

    Parameters
    ----------
    visit : int
        Visit ID
    collection : str
        Butler collection
    output_file : str
        Output parquet file path
    repo : str
        Butler repository path
    """
    butler = Butler(repo, collections=collection)

    rotator_angle =  butler.get("raw.visitInfo", exposure=visit, detector=42).getBoresightRotAngle()
    rotator_angle_radian = rotator_angle.asRadians()

    # Get URI to the parquet file
    uri = butler.getURI("refit_psf_star", instrument="LSSTCam", visit=visit)
    print(f"Loading data from: {uri.geturl()}")

    # Load and filter data
    table = pl.scan_parquet(uri.geturl()).select(PARQUET_COLUMNS).collect()
    table = table.filter(pl.col('calib_psf_candidate'))
    print(f"Loaded {len(table)} PSF candidates")

    # Convert to numpy for processing
    x_ccd = table['slot_Centroid_x'].to_numpy()
    y_ccd = table['slot_Centroid_y'].to_numpy()
    detector = table['detector'].to_numpy()

    ixx = table['slot_Shape_xx'].to_numpy()
    iyy = table['slot_Shape_yy'].to_numpy()
    ixy = table['slot_Shape_xy'].to_numpy()

    ixx_psf = table['slot_PsfShape_xx'].to_numpy()
    iyy_psf = table['slot_PsfShape_yy'].to_numpy()
    ixy_psf = table['slot_PsfShape_xy'].to_numpy()

    # Convert pixel to focal plane coordinates per detector
    x_fp = np.zeros_like(x_ccd)
    y_fp = np.zeros_like(y_ccd)

    unique_detectors = np.unique(detector)
    print(f"Processing {len(unique_detectors)} detectors...")

    for det_id in unique_detectors:
        mask = detector == det_id
        det = camera[int(det_id)]
        x_fp[mask], y_fp[mask] = pixel_to_focal(x_ccd[mask], y_ccd[mask], det)

    # Create output dataframe
    output = pl.DataFrame({
        'rotator_angle_radian': [rotator_angle_radian] * len(x_fp),
        'visit': [visit] * len(x_fp),
        'detector': detector,
        'x_ccd': x_ccd,
        'y_ccd': y_ccd,
        'x_fp': x_fp,
        'y_fp': y_fp,
        'ixx': ixx,
        'iyy': iyy,
        'ixy': ixy,
        'ixx_psf': ixx_psf,
        'iyy_psf': iyy_psf,
        'ixy_psf': ixy_psf,
    })

    # Add derived quantities
    output = output.with_columns([
        (pl.col('ixx') + pl.col('iyy')).alias('T'),
        ((pl.col('ixx') - pl.col('iyy')) / (pl.col('ixx') + pl.col('iyy'))).alias('e1'),
        (2 * pl.col('ixy') / (pl.col('ixx') + pl.col('iyy'))).alias('e2'),
        (pl.col('ixx_psf') + pl.col('iyy_psf')).alias('T_psf'),
        ((pl.col('ixx_psf') - pl.col('iyy_psf')) / (pl.col('ixx_psf') + pl.col('iyy_psf'))).alias('e1_psf'),
        (2 * pl.col('ixy_psf') / (pl.col('ixx_psf') + pl.col('iyy_psf'))).alias('e2_psf'),
    ])

    # Save to parquet
    output.write_parquet(output_file)
    print(f"Saved {len(output)} rows to {output_file}")

    # Print summary statistics
    print("\nSummary statistics:")
    print(f"  x_fp range: [{x_fp.min():.1f}, {x_fp.max():.1f}] mm")
    print(f"  y_fp range: [{y_fp.min():.1f}, {y_fp.max():.1f}] mm")
    print(f"  T range: [{output['T'].min():.2f}, {output['T'].max():.2f}] pixel²")
    print(f"  e1 range: [{output['e1'].min():.3f}, {output['e1'].max():.3f}]")
    print(f"  e2 range: [{output['e2'].min():.3f}, {output['e2'].max():.3f}]")


def main():
    parser = argparse.ArgumentParser(description="Prepare visit data for Zernike fitting")
    parser.add_argument('--visit', type=int, required=True, help="Visit ID")
    parser.add_argument('--collection', type=str,
                        default="LSSTCam/runs/DRP/DP2/v30_0_0/DM-53881/stage2",
                        help="Butler collection")
    parser.add_argument('--output', type=str, default=None,
                        help="Output parquet file (default: visit_<ID>_psf.parquet)")
    parser.add_argument('--repo', type=str, default='dp2_prep',
                        help="Butler repository")

    args = parser.parse_args()

    if args.output is None:
        args.output = f"visit_{args.visit}_psf.parquet"

    prepare_visit_data(args.visit, args.collection, args.output, args.repo)


if __name__ == "__main__":
    main()
