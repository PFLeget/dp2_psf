"""
Plot height map and second moments averaged over all DP2 visits.

Produces two separate figures:
1. Height map from SLAC metrology
2. Second moments averaged over all visits
"""
import numpy as np
import re
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import treegp
import lsst.afw.cameraGeom as cameraGeom
from lsst.obs.lsst import LsstCam
from astropy.io import fits
from astropy.table import Table
import os
os.environ["POLARS_MAX_THREADS"] = "1"
import polars
import argparse

camera = LsstCam.getCamera()

PARQUET_COLUMNS = [
    'slot_Shape_xx', 'slot_Shape_yy', 'slot_Shape_xy',
    'slot_Centroid_x', 'slot_Centroid_y',
    'detector',
]


def load_visit_data(parquet_path):
    """Load visit data from parquet file."""
    table = polars.scan_parquet(parquet_path).select(PARQUET_COLUMNS).collect()

    slot_Shape_xx = table['slot_Shape_xx'].to_numpy()
    slot_Shape_yy = table['slot_Shape_yy'].to_numpy()
    slot_Shape_xy = table['slot_Shape_xy'].to_numpy()

    T_src = slot_Shape_xx + slot_Shape_yy
    e1_src = (slot_Shape_xx - slot_Shape_yy) / T_src
    e2_src = 2 * slot_Shape_xy / T_src

    return {
        'T_src': T_src,
        'e1_src': e1_src,
        'e2_src': e2_src,
        'xCCD': table['slot_Centroid_x'].to_numpy(),
        'yCCD': table['slot_Centroid_y'].to_numpy(),
        'detector': table['detector'].to_numpy(),
    }


def pixel_to_focal(x, y, det):
    """Convert pixel coordinates to focal plane coordinates (mm)."""
    tx = det.getTransform(cameraGeom.PIXELS, cameraGeom.FOCAL_PLANE)
    fpx, fpy = tx.getMapping().applyForward(np.vstack((x, y)))
    return fpx.ravel(), fpy.ravel()


def make_metrology_table(file):
    """Make astropy table of height measurement data."""
    rows = []
    with fits.open(file) as hdulist:
        for hdu in hdulist:
            if isinstance(hdu, fits.BinTableHDU):
                table = Table(hdu.data)
                extname = hdu.header['EXTNAME']
                if re.fullmatch(r'R\d\dS\d\d', extname):
                    extname = re.sub(r'(R\d\d)(S\d\d)', r'\1_\2', extname)
                    for x, y, z_mod, z_meas in zip(table['X_CCS'], table['Y_CCS'],
                                                    table['Z_CCS_MODEL'], table['Z_CCS_MEASURED']):
                        rows.append([y, x, z_mod, z_meas, extname])
    return Table(rows=rows, names=['fpx', 'fpy', 'z_mod', 'z_meas', 'det'])


def get_field_for_key(dic, key):
    """Get the field array for a given second moment key."""
    if key in ['T', 'dT']:
        return dic['T_src']
    elif key in ['e1', 'de1']:
        return dic['e1_src']
    elif key in ['e2', 'de2']:
        return dic['e2_src']
    else:
        raise ValueError(f'Invalid key: {key}')


def plot_height_map_and_moments(
    visitMappingFile,
    fitHeightMap,
    secondMomentKey='dT',
    bands=None,
    max_visits=None,
    repOutPlot='plots/',
    subtract_focal_plane_mean=True,
):
    if secondMomentKey not in ['T', 'e1', 'e2', 'dT', 'de1', 'de2']:
        raise ValueError(f'Invalid secondMomentKey: {secondMomentKey}')

    # Load visit mapping
    print(f"Loading visit mapping from {visitMappingFile}...")
    with open(visitMappingFile, 'rb') as f:
        visit_mapping = pickle.load(f)

    # Filter by band if specified
    if bands is not None:
        bands_set = set(bands)
        visit_mapping = {v: info for v, info in visit_mapping.items()
                        if info.get('band') in bands_set}
        print(f"Filtered to bands {bands}: {len(visit_mapping)} visits")
    else:
        print(f"All bands: {len(visit_mapping)} visits")

    # Apply visit limit
    visit_list = list(visit_mapping.keys())
    if max_visits is not None and max_visits < len(visit_list):
        visit_list = visit_list[:max_visits]
        print(f"Limited to {max_visits} visits for testing")

    # Load height map
    print(f"Loading height map from {fitHeightMap}...")
    tableSLAC = make_metrology_table(file=fitHeightMap)

    # Initialize meanify objects per CCD
    meanify = {}
    n_visits_processed = 0

    # Process all visits
    for visit in tqdm(visit_list, desc="Processing visits"):
        try:
            parquet_path = visit_mapping[visit]['parquet_path']
            dic = load_visit_data(parquet_path)
            ccdIds = list(set(dic['detector']))

            field_full = get_field_for_key(dic, secondMomentKey)

            # Subtract focal plane mean per visit (default), not per-CCD mean
            if subtract_focal_plane_mean:
                visit_mean = np.nanmean(field_full)
            else:
                visit_mean = 0.0

            for ccd in ccdIds:
                if ccd not in meanify:
                    meanify[ccd] = treegp.meanify(bin_spacing=150, statistics="median")

                mask = dic['detector'] == ccd
                coord = np.array([dic['xCCD'][mask], dic['yCCD'][mask]]).T
                field = field_full[mask] - visit_mean

                meanify[ccd].add_field(coord, field)

            n_visits_processed += 1

        except Exception as e:
            print(f"Warning: Failed to process visit {visit}: {e}")
            continue

    print(f"\nProcessed {n_visits_processed} visits")

    if n_visits_processed == 0:
        print("No visits processed, exiting.")
        return

    # Meanify all CCDs
    for ccd in tqdm(meanify, desc="Computing means"):
        meanify[ccd].meanify()

    os.makedirs(repOutPlot, exist_ok=True)
    band_str = bands if bands else 'all'
    meanHeightFocalPlane = np.mean(np.array(tableSLAC['z_meas']))

    # ============ Figure 1: Height map ============
    fig1, ax1 = plt.subplots(figsize=(12, 10))

    for ccd in meanify:
        FiltDet = np.array(tableSLAC['det']) == camera[ccd].getName()
        coordSLAC = np.array([np.array(tableSLAC['fpx'])[FiltDet],
                              np.array(tableSLAC['fpy'])[FiltDet]]).T
        heightSLAC = np.array(tableSLAC['z_meas'])[FiltDet] - meanHeightFocalPlane
        ax1.scatter(coordSLAC[:, 0], coordSLAC[:, 1], s=8, marker='s',
                    c=heightSLAC, cmap=plt.cm.seismic, vmin=-0.005, vmax=0.005)

    ax1.set_aspect('equal')
    ax1.set_xlabel('x (mm)', size=16)
    ax1.set_ylabel('y (mm)', size=16)
    ax1.tick_params(labelsize=14)
    ax1.set_title("Height map from SLAC metrology", size=18)

    sm = plt.cm.ScalarMappable(cmap=plt.cm.seismic, norm=plt.Normalize(-0.005, 0.005))
    cb = fig1.colorbar(sm, ax=ax1)
    cb.set_label("z - <z> (mm)", size=16)
    cb.ax.tick_params(labelsize=14)

    outfile1 = os.path.join(repOutPlot, f'heightMap_{band_str}_n{n_visits_processed}.png')
    fig1.savefig(outfile1, dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print(f"Saved: {outfile1}")

    # ============ Figure 2: Second moments ============
    if secondMomentKey in ['T', 'dT']:
        vmax = 0.5
        colorlabel = "T - <T> (pixel$^2$)"
    else:
        vmax = 0.05
        colorlabel = f"{secondMomentKey} - <{secondMomentKey}>"
    vmin = -vmax

    fig2, ax2 = plt.subplots(figsize=(12, 10))

    for ccd in meanify:
        x, y = np.meshgrid(meanify[ccd]._xedge, meanify[ccd]._yedge)
        nBin0, nBin1 = np.shape(x)[0], np.shape(x)[1]
        x = x.reshape(nBin0 * nBin1)
        y = y.reshape(nBin0 * nBin1)
        x, y = pixel_to_focal(x, y, camera[ccd])
        x = x.reshape((nBin0, nBin1))
        y = y.reshape((nBin0, nBin1))
        ax2.pcolormesh(x, y, meanify[ccd]._average, vmin=vmin, vmax=vmax, cmap=plt.cm.seismic)

    ax2.set_aspect('equal')
    ax2.set_xlabel('x (mm)', size=16)
    ax2.set_ylabel('y (mm)', size=16)
    ax2.tick_params(labelsize=14)
    ax2.set_title(f"{secondMomentKey} - {n_visits_processed} visits ({band_str} bands)", size=18)

    sm = plt.cm.ScalarMappable(cmap=plt.cm.seismic, norm=plt.Normalize(vmin, vmax))
    cb = fig2.colorbar(sm, ax=ax2)
    cb.set_label(colorlabel, size=16)
    cb.ax.tick_params(labelsize=14)

    outfile2 = os.path.join(repOutPlot, f'{secondMomentKey}_{band_str}_n{n_visits_processed}.png')
    fig2.savefig(outfile2, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"Saved: {outfile2}")


def main():
    defaultVisitMappingFile = "/sdf/home/l/leget/rubin-user/lsst_dev/tickets/dp2_psf/data/visit_parquet_mapping.pkl"
    defaultFitHeightMap = "/sdf/home/l/leget/rubin-user/lsst_dev/tickets/dp2_psf/data/LSST_FP_cold_b_measurement_4col_bysurface.fits"
    defaultRepOutPlot = "/sdf/home/l/leget/rubin-user/lsst_dev/tickets/dp2_psf/plots/"

    parser = argparse.ArgumentParser(description="Height map vs second moments for all DP2 visits")
    parser.add_argument('--visitMappingFile', type=str, default=defaultVisitMappingFile)
    parser.add_argument('--fitHeightMap', type=str, default=defaultFitHeightMap)
    parser.add_argument('--secondMomentKey', type=str, default='dT',
                        choices=['T', 'e1', 'e2', 'dT', 'de1', 'de2'])
    parser.add_argument('--bands', type=str, default=None,
                        help="Filter bands, e.g. 'gri' or None for all")
    parser.add_argument('--max_visits', type=int, default=None,
                        help="Maximum visits to process (for testing)")
    parser.add_argument('--repOutPlot', type=str, default=defaultRepOutPlot)
    parser.add_argument('--subtract_focal_plane_mean', action='store_true', default=True,
                        help="Subtract focal plane mean per visit (default: True)")
    parser.add_argument('--no_subtract_mean', action='store_false', dest='subtract_focal_plane_mean',
                        help="Do not subtract any mean")

    args = parser.parse_args()

    plot_height_map_and_moments(
        visitMappingFile=args.visitMappingFile,
        fitHeightMap=args.fitHeightMap,
        secondMomentKey=args.secondMomentKey,
        bands=args.bands,
        max_visits=args.max_visits,
        repOutPlot=args.repOutPlot,
        subtract_focal_plane_mean=args.subtract_focal_plane_mean,
    )


if __name__ == "__main__":
    main()
