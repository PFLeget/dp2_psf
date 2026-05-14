#!/usr/bin/env python
"""
Create a FITS catalog of all stars in COSMOS r-band visits.

Includes CCD, focal plane, and sky coordinates, plus all PSF shape measurements.
"""

import numpy as np
from tqdm import tqdm
import os
os.environ["POLARS_MAX_THREADS"] = "1"
import polars
import pickle
import argparse

from lsst.daf.butler import Butler
from lsst.obs.lsst import LsstCam
import lsst.afw.cameraGeom as cameraGeom
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
import astropy.units as u


COSMOS_RA = 150.12  # degrees
COSMOS_DEC = 2.21   # degrees

camera = LsstCam.getCamera()

# Columns to read from parquet files
PARQUET_COLUMNS = [
    # Sky coordinate moments (arcsec^2)
    'shape_Iuu', 'shape_Ivv', 'shape_Iuv',
    'psfShape_Iuu', 'psfShape_Ivv', 'psfShape_Iuv',
    # CCD coordinate moments (pixel^2)
    'slot_Shape_xx', 'slot_Shape_yy', 'slot_Shape_xy',
    'slot_PsfShape_xx', 'slot_PsfShape_yy', 'slot_PsfShape_xy',
    # Coordinates
    'coord_ra', 'coord_dec',
    'slot_Centroid_x', 'slot_Centroid_y',
    # Flux
    'base_GaussianFlux_instFlux', 'base_GaussianFlux_instFluxErr',
    # Metadata
    'detector', 'psf_max_value',
    'calib_psf_used', 'calib_psf_reserved', 'calib_psf_candidate',
]


def angular_distance(ra, dec, ra_center=0, dec_center=0, unit='deg'):
    """Compute angular distance from a center point."""
    coords = SkyCoord(ra=ra, dec=dec, unit='deg')
    center = SkyCoord(ra=ra_center, dec=dec_center, unit='deg')
    sep = coords.separation(center)
    if unit == 'deg':
        return sep.deg
    elif unit == 'arcmin':
        return sep.arcmin
    else:
        return sep.arcsec


def get_cosmos_visits(repo, collection, band, max_sep_deg=0.5):
    """Get visit IDs that overlap COSMOS using Butler visit_table."""
    butler = Butler(repo, collections=collection)
    visitTable = butler.get('visit_table', instrument="LSSTCam")

    angular_sep = angular_distance(
        visitTable['ra'], visitTable['dec'],
        ra_center=COSMOS_RA, dec_center=COSMOS_DEC, unit='deg')

    mask = angular_sep < max_sep_deg
    mask &= visitTable['band'] == band
    visitId_cosmos = visitTable['visitId'][mask]

    return list(visitId_cosmos)


def pixel_to_focal(x, y, det):
    """
    Transform pixel coordinates to focal plane coordinates.

    Parameters
    ----------
    x, y : array
        Pixel coordinates.
    det : lsst.afw.cameraGeom.Detector
        Detector of interest.

    Returns
    -------
    fpx, fpy : array
        Focal plane position in millimeters in DVCS.
    """
    tx = det.getTransform(cameraGeom.PIXELS, cameraGeom.FOCAL_PLANE)
    fpx, fpy = tx.getMapping().applyForward(np.vstack((x, y)))
    return fpx.ravel(), fpy.ravel()


def load_visit_data(parquet_path, visit_id):
    """Load visit data and compute all derived quantities."""
    table = polars.scan_parquet(parquet_path).select(PARQUET_COLUMNS).collect()
    n = len(table)

    if n == 0:
        return None

    # Extract raw columns
    data = {col: table[col].to_numpy() for col in PARQUET_COLUMNS}

    # Add visit ID
    data['visit'] = np.full(n, visit_id, dtype=np.int64)

    # --- Sky coordinates ---
    data['ra'] = np.degrees(data['coord_ra'])
    data['dec'] = np.degrees(data['coord_dec'])

    # --- CCD coordinates ---
    data['x_ccd'] = data['slot_Centroid_x']
    data['y_ccd'] = data['slot_Centroid_y']

    # --- Focal plane coordinates ---
    data['x_fp'] = np.zeros(n)
    data['y_fp'] = np.zeros(n)

    # Transform CCD to focal plane for each detector
    for det_id in np.unique(data['detector']):
        mask = data['detector'] == det_id
        x_fp, y_fp = pixel_to_focal(data['x_ccd'][mask], data['y_ccd'][mask], camera[det_id])
        data['x_fp'][mask] = x_fp
        data['y_fp'][mask] = y_fp

    # --- Sky coordinate derived quantities (arcsec^2) ---
    iuu_src = data['shape_Iuu']
    ivv_src = data['shape_Ivv']
    iuv_src = data['shape_Iuv']
    iuu_psf = data['psfShape_Iuu']
    ivv_psf = data['psfShape_Ivv']
    iuv_psf = data['psfShape_Iuv']

    T_src_sky = iuu_src + ivv_src
    T_psf_sky = iuu_psf + ivv_psf

    data['T_src_sky'] = T_src_sky
    data['T_psf_sky'] = T_psf_sky
    data['dT_T_sky'] = (T_src_sky - T_psf_sky) / T_src_sky

    data['e1_src_sky'] = (iuu_src - ivv_src) / T_src_sky
    data['e2_src_sky'] = 2 * iuv_src / T_src_sky
    data['e1_psf_sky'] = (iuu_psf - ivv_psf) / T_psf_sky
    data['e2_psf_sky'] = 2 * iuv_psf / T_psf_sky

    data['de1_sky'] = data['e1_src_sky'] - data['e1_psf_sky']
    data['de2_sky'] = data['e2_src_sky'] - data['e2_psf_sky']

    # --- CCD coordinate derived quantities (pixel^2) ---
    ixx_src = data['slot_Shape_xx']
    iyy_src = data['slot_Shape_yy']
    ixy_src = data['slot_Shape_xy']
    ixx_psf = data['slot_PsfShape_xx']
    iyy_psf = data['slot_PsfShape_yy']
    ixy_psf = data['slot_PsfShape_xy']

    T_src_ccd = ixx_src + iyy_src
    T_psf_ccd = ixx_psf + iyy_psf

    data['T_src_ccd'] = T_src_ccd
    data['T_psf_ccd'] = T_psf_ccd
    data['dT_T_ccd'] = (T_src_ccd - T_psf_ccd) / T_src_ccd

    data['e1_src_ccd'] = (ixx_src - iyy_src) / T_src_ccd
    data['e2_src_ccd'] = 2 * ixy_src / T_src_ccd
    data['e1_psf_ccd'] = (ixx_psf - iyy_psf) / T_psf_ccd
    data['e2_psf_ccd'] = 2 * ixy_psf / T_psf_ccd

    data['de1_ccd'] = data['e1_src_ccd'] - data['e1_psf_ccd']
    data['de2_ccd'] = data['e2_src_ccd'] - data['e2_psf_ccd']

    # --- SNR ---
    flux = data['base_GaussianFlux_instFlux']
    flux_err = data['base_GaussianFlux_instFluxErr']
    data['snr'] = flux / flux_err

    return data


def main():
    parser = argparse.ArgumentParser(description="Create COSMOS star catalog for PSF debugging")
    parser.add_argument('--repo', type=str, default='dp2_prep')
    parser.add_argument('--collection', type=str, default='LSSTCam/runs/DRP/DP2/v30_0_0/DM-53881/stage2')
    parser.add_argument('--visitMappingFile', type=str, default='../data/visit_parquet_mapping_skycoord.pkl',
                        help='Path to visit_parquet_mapping_skycoord.pkl')
    parser.add_argument('--output', type=str, default='cosmos_stars_r.fits',
                        help='Output FITS file')
    parser.add_argument('--max_visits', type=int, default=None,
                        help='Max visits to process (for testing)')

    args = parser.parse_args()

    print("COSMOS Star Catalog Generator")
    print("=" * 50)

    # Get COSMOS visits
    print("\nQuerying COSMOS visits from Butler...")
    cosmos_visit_ids = get_cosmos_visits(args.repo, args.collection, 'r')
    print(f"Found {len(cosmos_visit_ids)} COSMOS visits in r-band")

    if args.max_visits is not None:
        cosmos_visit_ids = cosmos_visit_ids[:args.max_visits]
        print(f"Limited to {args.max_visits} visits")

    # Load visit mapping
    print(f"\nLoading visit mapping from {args.visitMappingFile}...")
    with open(args.visitMappingFile, 'rb') as f:
        visit_mapping = pickle.load(f)

    # Process all visits
    print(f"\nProcessing {len(cosmos_visit_ids)} visits...")
    all_data = None

    for visit_id in tqdm(cosmos_visit_ids, desc="Loading visits"):
        if visit_id not in visit_mapping:
            continue

        info = visit_mapping[visit_id]
        try:
            data = load_visit_data(info['parquet_path'], visit_id)
            if data is None:
                continue

            if all_data is None:
                all_data = {k: [v] for k, v in data.items()}
            else:
                for k, v in data.items():
                    all_data[k].append(v)
        except Exception as e:
            print(f"Warning: failed visit {visit_id}: {e}")

    # Concatenate all data
    print("\nConcatenating data...")
    for k in all_data:
        all_data[k] = np.concatenate(all_data[k])

    n_total = len(all_data['visit'])
    print(f"Total stars: {n_total:,}")

    # Select columns for output (clean names)
    output_columns = [
        # Identifiers
        'visit', 'detector',
        # Sky coordinates
        'ra', 'dec',
        # CCD coordinates (pixels)
        'x_ccd', 'y_ccd',
        # Focal plane coordinates (mm)
        'x_fp', 'y_fp',
        # Raw sky moments (arcsec^2)
        'shape_Iuu', 'shape_Ivv', 'shape_Iuv',
        'psfShape_Iuu', 'psfShape_Ivv', 'psfShape_Iuv',
        # Raw CCD moments (pixel^2)
        'slot_Shape_xx', 'slot_Shape_yy', 'slot_Shape_xy',
        'slot_PsfShape_xx', 'slot_PsfShape_yy', 'slot_PsfShape_xy',
        # Derived sky quantities
        'T_src_sky', 'T_psf_sky', 'dT_T_sky',
        'e1_src_sky', 'e2_src_sky', 'e1_psf_sky', 'e2_psf_sky',
        'de1_sky', 'de2_sky',
        # Derived CCD quantities
        'T_src_ccd', 'T_psf_ccd', 'dT_T_ccd',
        'e1_src_ccd', 'e2_src_ccd', 'e1_psf_ccd', 'e2_psf_ccd',
        'de1_ccd', 'de2_ccd',
        # Flux/SNR
        'base_GaussianFlux_instFlux', 'base_GaussianFlux_instFluxErr', 'snr',
        # Quality flags
        'psf_max_value', 'calib_psf_used', 'calib_psf_reserved', 'calib_psf_candidate',
    ]

    # Build astropy table
    print("\nBuilding FITS table...")
    table_data = {col: all_data[col] for col in output_columns}
    astro_table = Table(table_data)

    # Write FITS
    print(f"Writing to {args.output}...")
    astro_table.write(args.output, format='fits', overwrite=True)
    print(f"Done! Wrote {n_total:,} stars to {args.output}")

    # Print summary
    print("\n" + "=" * 50)
    print("Summary:")
    print(f"  Visits processed: {len(np.unique(all_data['visit']))}")
    print(f"  Detectors: {len(np.unique(all_data['detector']))}")
    print(f"  Total stars: {n_total:,}")
    print(f"  File size: {os.path.getsize(args.output) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
