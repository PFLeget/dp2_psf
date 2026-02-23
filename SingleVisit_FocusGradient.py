import numpy as np
import re
from sklearn.neighbors import KNeighborsRegressor
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import pickle
from lsst.daf.butler import Butler
import lsst.afw.cameraGeom as cameraGeom
from lsst.obs.lsst import LsstCam
from astropy.io import fits
from astropy.table import Table
import os
os.environ["POLARS_MAX_THREADS"] = "1"
import polars
import argparse

camera = LsstCam.getCamera()


# Columns to read from parquet files
PARQUET_COLUMNS = [
    'slot_Shape_xx', 'slot_Shape_yy', 'slot_Shape_xy',
    'slot_PsfShape_xx', 'slot_PsfShape_xy', 'slot_PsfShape_yy',
    'coord_ra', 'coord_dec', 'slot_Centroid_x', 'slot_Centroid_y',
    'detector', 'psf_max_value', 'calib_psf_reserved',
]


def load_visit_data(parquet_path):
    """
    Load visit data from parquet file and compute derived columns.
    """
    table = polars.scan_parquet(parquet_path).select(PARQUET_COLUMNS).collect()

    slot_Shape_xx = table['slot_Shape_xx'].to_numpy()
    slot_Shape_yy = table['slot_Shape_yy'].to_numpy()
    slot_Shape_xy = table['slot_Shape_xy'].to_numpy()

    T_src = slot_Shape_xx + slot_Shape_yy

    return {
        'T': T_src,
        'ixx_src': slot_Shape_xx,
        'iyy_src': slot_Shape_yy,
        'ixy_src': slot_Shape_xy,
        'xCCD': table['slot_Centroid_x'].to_numpy(),
        'yCCD': table['slot_Centroid_y'].to_numpy(),
        'detector': table['detector'].to_numpy(),
    }


def pixel_to_focal(x, y, det):
    """
    Convert pixel coordinates to focal plane coordinates.
    """
    tx = det.getTransform(cameraGeom.PIXELS, cameraGeom.FOCAL_PLANE)
    fpx, fpy = tx.getMapping().applyForward(np.vstack((x, y)))
    return fpx.ravel(), fpy.ravel()


def make_metrology_table(file="LSST_FP_cold_b_measurement_4col_bysurface.fits", rsid=None):
    """
    Make an astropy table of the height measurement data.
    """
    rows = []
    with fits.open(file) as hdulist:
        for hdu in hdulist:
            if isinstance(hdu, fits.BinTableHDU):
                table = Table(hdu.data)
                extname = hdu.header['EXTNAME']
                if rsid is not None:
                    if extname == rsid:
                        extname = re.sub(r'(R\d\d)(S\d\d)', r'\1_\2', extname)
                        for x, y, z_mod, z_meas in zip(table['X_CCS'], table['Y_CCS'],
                                                        table['Z_CCS_MODEL'], table['Z_CCS_MEASURED']):
                            fpx = y
                            fpy = x
                            rows.append([fpx, fpy, z_mod, z_meas, extname])
                else:
                    if re.fullmatch(r'R\d\dS\d\d', extname):
                        extname = re.sub(r'(R\d\d)(S\d\d)', r'\1_\2', extname)
                        for x, y, z_mod, z_meas in zip(table['X_CCS'], table['Y_CCS'],
                                                        table['Z_CCS_MODEL'], table['Z_CCS_MEASURED']):
                            fpx = y
                            fpy = x
                            rows.append([fpx, fpy, z_mod, z_meas, extname])

        bigtable = Table(rows=rows, names=['fpx', 'fpy', 'z_mod', 'z_meas', 'det'])

    return bigtable


def get_ccd_corners_fp(det):
    """
    Get the 4 corners of a CCD in focal plane coordinates.
    Returns list of (x, y) tuples for each corner.
    """
    bbox = det.getBBox()
    corners_pix_x = np.array([bbox.getMinX(), bbox.getMaxX(), bbox.getMaxX(), bbox.getMinX()], dtype=float)
    corners_pix_y = np.array([bbox.getMinY(), bbox.getMinY(), bbox.getMaxY(), bbox.getMaxY()], dtype=float)

    fpx, fpy = pixel_to_focal(corners_pix_x, corners_pix_y, det)
    corners_fp = list(zip(fpx, fpy))
    return corners_fp


def analyze_single_visit(visit, visitMappingFile, fitHeightMap, repOutPlot):
    """
    Analyze focus gradient for a single visit.
    """
    # Load visit mapping
    with open(visitMappingFile, 'rb') as f:
        visit_mapping = pickle.load(f)

    if visit not in visit_mapping:
        raise ValueError(f"Visit {visit} not found in mapping file")

    # Load height map from SLAC
    tableSLAC = make_metrology_table(file=fitHeightMap)

    # Load visit data
    print(f"Loading visit {visit}...")
    dic = load_visit_data(visit_mapping[visit]['parquet_path'])
    band = visit_mapping[visit]['band']

    ccdIds = list(set(dic['detector']))
    print(f"Found {len(ccdIds)} CCDs in visit {visit}")

    # Prepare figure
    fig = plt.figure(figsize=(20, 20))
    plt.subplots_adjust(top=0.95, wspace=0.25, hspace=0.15, right=0.95, left=0.07, bottom=0.05)

    # Store correlation coefficients per CCD
    ccd_correlations = {}
    ccd_corners = {}

    # ============================================================
    # Panel 1 (2,2,1): Height map from SLAC
    # ============================================================
    plt.subplot(2, 2, 1)

    for ccd in ccdIds:
        det = camera[ccd]
        FiltDet = np.array(tableSLAC['det']) == det.getName()
        if np.sum(FiltDet) == 0:
            continue
        meanHeightDet = np.mean(np.array(tableSLAC['z_meas'])[FiltDet])
        coordSLAC = np.array([np.array(tableSLAC['fpx'])[FiltDet],
                              np.array(tableSLAC['fpy'])[FiltDet]]).T
        heightSLAC = np.array(tableSLAC['z_meas'])[FiltDet] - meanHeightDet
        plt.scatter(coordSLAC[:, 0], coordSLAC[:, 1], s=12, marker='s',
                    c=heightSLAC, cmap=plt.cm.seismic, vmin=-0.005, vmax=0.005)

    cb = plt.colorbar()
    cb.set_label("z - <z> (mm)", size=22)
    cb.ax.tick_params(labelsize=18)
    plt.xlabel('x (mm)', size=22)
    plt.ylabel('y (mm)', size=22)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.title("Height map from SLAC", size=18)
    plt.axis('equal')

    # ============================================================
    # Panel 2 (2,2,2): Placeholder for z4 corners
    # ============================================================
    plt.subplot(2, 2, 2)
    plt.text(0.5, 0.5, "z4 corner map\n(placeholder)",
             ha='center', va='center', fontsize=20, transform=plt.gca().transAxes)
    plt.xlabel('x (mm)', size=22)
    plt.ylabel('y (mm)', size=22)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.title("Wavefront z4 at corners", size=18)

    # ============================================================
    # Panel 3 (2,2,3): Single visit T - <T> focal plane map
    # ============================================================
    plt.subplot(2, 2, 3)

    for ccd in tqdm(ccdIds, desc="Processing CCDs"):
        det = camera[ccd]
        filtreDetector = dic['detector'] == ccd

        if np.sum(filtreDetector) < 10:
            continue

        # Get star positions and sizes
        xCCD = dic['xCCD'][filtreDetector]
        yCCD = dic['yCCD'][filtreDetector]
        T = dic['T'][filtreDetector]

        # Subtract mean T for this CCD
        T_centered = T - np.mean(T)

        # Convert to focal plane coordinates
        fpx, fpy = pixel_to_focal(xCCD, yCCD, det)

        # Plot T - <T>
        plt.scatter(fpx, fpy, s=8, c=T_centered,
                    cmap=plt.cm.seismic, vmin=-0.5, vmax=0.5)

        # Get height map for this CCD
        FiltDet = np.array(tableSLAC['det']) == det.getName()
        if np.sum(FiltDet) == 0:
            continue

        meanHeightDet = np.mean(np.array(tableSLAC['z_meas'])[FiltDet])
        coordSLAC = np.array([np.array(tableSLAC['fpx'])[FiltDet],
                              np.array(tableSLAC['fpy'])[FiltDet]]).T
        heightSLAC = np.array(tableSLAC['z_meas'])[FiltDet] - meanHeightDet

        # Get CCD corners for later use
        corners = get_ccd_corners_fp(det)

        # Use KNN to interpolate height at star positions
        try:
            knn = KNeighborsRegressor(n_neighbors=min(20, len(coordSLAC)))
            knn.fit(coordSLAC, heightSLAC)
            star_coords_fp = np.array([fpx, fpy]).T
            height_at_stars = knn.predict(star_coords_fp)

            # Compute correlation coefficient
            valid = np.isfinite(T_centered) & np.isfinite(height_at_stars)
            if np.sum(valid) > 10:
                rho = np.corrcoef(height_at_stars[valid], T_centered[valid])[0, 1]
                ccd_correlations[ccd] = rho
                ccd_corners[ccd] = corners
        except Exception as e:
            print(f"KNN failed for CCD {ccd}: {e}")

    cb = plt.colorbar()
    cb.set_label("T - <T> (pixel$^2$)", size=22)
    cb.ax.tick_params(labelsize=18)
    plt.xlabel('x (mm)', size=22)
    plt.ylabel('y (mm)', size=22)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.title(f"Visit {visit} ({band}-band) | Star size residuals", size=18)
    plt.axis('equal')

    # ============================================================
    # Panel 4 (2,2,4): Correlation coefficient map per CCD
    # ============================================================
    ax4 = plt.subplot(2, 2, 4)

    patches = []
    colors = []

    for ccd, rho in ccd_correlations.items():
        corners = ccd_corners[ccd]
        # Create polygon from corners
        polygon = mpatches.Polygon(corners, closed=True)
        patches.append(polygon)
        colors.append(rho)

    if patches:
        collection = PatchCollection(patches, cmap=plt.cm.seismic,
                                     edgecolor='black', linewidth=0.5)
        collection.set_array(np.array(colors))
        collection.set_clim(-1, 1)
        ax4.add_collection(collection)

        # Set axis limits based on focal plane extent
        all_x = [c[0] for corners in ccd_corners.values() for c in corners]
        all_y = [c[1] for corners in ccd_corners.values() for c in corners]
        ax4.set_xlim(min(all_x) - 10, max(all_x) + 10)
        ax4.set_ylim(min(all_y) - 10, max(all_y) + 10)

        cb = plt.colorbar(collection)
        cb.set_label(r"$\rho$ (T, height)", size=22)
        cb.ax.tick_params(labelsize=18)

    plt.xlabel('x (mm)', size=22)
    plt.ylabel('y (mm)', size=22)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.title("Correlation: star size vs height per CCD", size=18)
    plt.axis('equal')

    # Add global correlation info
    all_rho = list(ccd_correlations.values())
    if all_rho:
        median_rho = np.median(all_rho)
        fig.suptitle(f"Visit {visit} | Band: {band} | Median $\\rho$ = {median_rho:.3f}",
                     fontsize=22, y=0.98)

    # Save figure
    output_file = os.path.join(repOutPlot, f'focus_gradient_visit_{visit}.png')
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"Saved: {output_file}")

    return ccd_correlations


def main():
    defaultFitHeightMap = "/sdf/home/l/leget/rubin-user/lsst_dev/tickets/dp2_psf/data/LSST_FP_cold_b_measurement_4col_bysurface.fits"
    defaultVisitMappingFile = "/sdf/home/l/leget/rubin-user/lsst_dev/tickets/dp2_psf/data/visit_parquet_mapping.pkl"
    defaultRepOutPlot = "/sdf/home/l/leget/rubin-user/lsst_dev/tickets/dp2_psf/plots/"

    parser = argparse.ArgumentParser(description="Single visit focus gradient analysis")
    parser.add_argument('--visit', type=int, required=True, help="Visit ID to analyze")
    parser.add_argument('--visitMappingFile', type=str, default=defaultVisitMappingFile,
                        help="Path to visit_parquet_mapping.pkl file")
    parser.add_argument('--fitHeightMap', type=str, default=defaultFitHeightMap,
                        help="Path to SLAC height map FITS file")
    parser.add_argument('--repOutPlot', type=str, default=defaultRepOutPlot,
                        help="Output directory for plots")

    args = parser.parse_args()

    analyze_single_visit(
        visit=args.visit,
        visitMappingFile=args.visitMappingFile,
        fitHeightMap=args.fitHeightMap,
        repOutPlot=args.repOutPlot,
    )


if __name__ == "__main__":
    main()
