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
import treegp
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


def analyze_single_visit(visit, visitMappingFile, fitHeightMap, repOutPlot, zernikeCornersFile=None):
    """
    Analyze focus gradient for a single visit.
    """
    # Load visit mapping
    with open(visitMappingFile, 'rb') as f:
        visit_mapping = pickle.load(f)

    # Load zernike corners if provided
    zernike_corners = None
    if zernikeCornersFile is not None:
        with open(zernikeCornersFile, 'rb') as f:
            zernike_corners = pickle.load(f)

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
    # Panel 2 (2,2,2): z4 at corner wavefront sensors
    # ============================================================
    plt.subplot(2, 2, 2)

    # Draw CCD outlines from the visit
    for ccd in ccdIds:
        det = camera[ccd]
        corners = get_ccd_corners_fp(det)
        # Close the polygon
        corners_closed = corners + [corners[0]]
        xs = [c[0] for c in corners_closed]
        ys = [c[1] for c in corners_closed]
        plt.plot(xs, ys, 'k-', linewidth=0.5, alpha=0.5)

    if zernike_corners is not None and visit in zernike_corners:
        z4_corners = zernike_corners[visit].get('z4_corners', [])
        if z4_corners:
            fpx_corners = [c['fpx'] for c in z4_corners]
            fpy_corners = [c['fpy'] for c in z4_corners]
            z4_vals = [c['value'] for c in z4_corners]

            # Set color scale based on band
            # Optimal z4: -0.2 for ugriz (range -0.6 to 0.2), 0 for y (range -0.4 to 0.4)
            if band == 'y':
                z4_vmin, z4_vmax = -0.4, 0.4
            else:
                z4_vmin, z4_vmax = -0.6, 0.2

            sc = plt.scatter(fpx_corners, fpy_corners, c=z4_vals, s=800,
                             cmap=plt.cm.seismic, vmin=z4_vmin, vmax=z4_vmax,
                             edgecolor='black', linewidth=2, marker='s')

            # Add labels with detector name and z4 value
            for corner in z4_corners:
                plt.annotate(f"{corner['det_name']}\n$z_4$={corner['value']:.3f}",
                             (corner['fpx'], corner['fpy']),
                             textcoords="offset points", xytext=(0, 35),
                             ha='center', fontsize=10, fontweight='bold')

            cb = plt.colorbar(sc)
            cb.set_label("$z_4$ ($\\mu$m)", size=22)
            cb.ax.tick_params(labelsize=18)

            plt.xlim(-350, 350)
            plt.ylim(-350, 350)
        else:
            plt.text(0.5, 0.5, "No z4 corner data\nfor this visit",
                     ha='center', va='center', fontsize=16, transform=plt.gca().transAxes)
    else:
        plt.text(0.5, 0.5, "z4 corner file\nnot provided",
                 ha='center', va='center', fontsize=16, transform=plt.gca().transAxes)

    plt.xlabel('x (mm)', size=22)
    plt.ylabel('y (mm)', size=22)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    # Set title based on whether we have z4 data
    if zernike_corners is not None and visit in zernike_corners:
        z4_corners_data = zernike_corners[visit].get('z4_corners', [])
        if z4_corners_data:
            z4_mean_title = np.nanmean([c['value'] for c in z4_corners_data])
            plt.title(f"Wavefront $z_4$ at corners | $<z_4>$={z4_mean_title:.3f} $\\mu$m", size=18)
        else:
            plt.title("Wavefront $z_4$ at corners", size=18)
    else:
        plt.title("Wavefront $z_4$ at corners", size=18)
    plt.axis('equal')

    # ============================================================
    # Panel 3 (2,2,3): Single visit T - <T> focal plane map
    # ============================================================
    plt.subplot(2, 2, 3)

    # First pass: build meanify for each CCD
    meanify_dict = {}
    for ccd in ccdIds:
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

        # Create meanify for this CCD
        meanify_dict[ccd] = treegp.meanify(bin_spacing=500, statistics="median")
        coord = np.array([xCCD, yCCD]).T
        meanify_dict[ccd].add_field(coord, T_centered)

    # Second pass: meanify, plot, and compute correlations
    for ccd in tqdm(meanify_dict.keys(), desc="Processing CCDs"):
        det = camera[ccd]
        meanify_dict[ccd].meanify()

        # Get grid in focal plane coordinates for pcolormesh
        x, y = np.meshgrid(meanify_dict[ccd]._xedge, meanify_dict[ccd]._yedge)
        nBin0, nBin1 = np.shape(x)[0], np.shape(x)[1]
        x = x.reshape(nBin0 * nBin1).astype(float)
        y = y.reshape(nBin0 * nBin1).astype(float)
        x, y = pixel_to_focal(x, y, det)
        x = x.reshape((nBin0, nBin1))
        y = y.reshape((nBin0, nBin1))

        # Plot with pcolormesh
        plt.pcolormesh(x, y, meanify_dict[ccd]._average,
                       vmin=-0.5, vmax=0.5, cmap=plt.cm.seismic)

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

        # Get meanified coordinates in focal plane
        CoordSubmit = meanify_dict[ccd].coords0
        csx, csy = pixel_to_focal(CoordSubmit[:, 0].astype(float),
                                   CoordSubmit[:, 1].astype(float), det)
        CoordSubmit_fp = np.array([csx, csy]).T
        T_binned = meanify_dict[ccd].params0

        # Use KNN to interpolate height at binned star positions
        try:
            knn = KNeighborsRegressor(n_neighbors=min(20, len(coordSLAC)))
            knn.fit(coordSLAC, heightSLAC)
            height_at_bins = knn.predict(CoordSubmit_fp)

            # Compute correlation coefficient
            valid = np.isfinite(T_binned) & np.isfinite(height_at_bins)
            if np.sum(valid) > 10:
                rho = np.corrcoef(height_at_bins[valid], T_binned[valid])[0, 1]
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

        # Compute gradient of correlation coefficient across focal plane
        # Get CCD centers and their correlation values
        ccd_centers_x = []
        ccd_centers_y = []
        ccd_rho_vals = []
        for ccd, rho in ccd_correlations.items():
            corners = ccd_corners[ccd]
            center_x = np.mean([c[0] for c in corners])
            center_y = np.mean([c[1] for c in corners])
            ccd_centers_x.append(center_x)
            ccd_centers_y.append(center_y)
            ccd_rho_vals.append(rho)

        ccd_centers_x = np.array(ccd_centers_x)
        ccd_centers_y = np.array(ccd_centers_y)
        ccd_rho_vals = np.array(ccd_rho_vals)

        # Fit a plane: rho = a*x + b*y + c using least squares
        # Design matrix: [x, y, 1]
        valid_fit = np.isfinite(ccd_rho_vals)
        if np.sum(valid_fit) > 3:
            A = np.column_stack([ccd_centers_x[valid_fit],
                                 ccd_centers_y[valid_fit],
                                 np.ones(np.sum(valid_fit))])
            coeffs, _, _, _ = np.linalg.lstsq(A, ccd_rho_vals[valid_fit], rcond=None)
            grad_x, grad_y = coeffs[0], coeffs[1]

            # Normalize the gradient
            grad_norm = np.sqrt(grad_x**2 + grad_y**2)
            if grad_norm > 0:
                grad_x_norm = grad_x / grad_norm
                grad_y_norm = grad_y / grad_norm

                # Draw arrow at center of focal plane
                arrow_scale = 100  # Length of arrow in mm
                ax4.arrow(0, 0, grad_x_norm * arrow_scale, grad_y_norm * arrow_scale,
                          head_width=20, head_length=15, fc='black', ec='black',
                          linewidth=3, zorder=10)

                # Add text showing gradient magnitude
                ax4.text(0, -50, f"|$\\nabla \\rho$| = {grad_norm*1000:.2f} mm$^{{-1}}$",
                         ha='center', va='top', fontsize=12, fontweight='bold')

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
    defaultZernikeCornersFile = "/sdf/home/l/leget/rubin-user/lsst_dev/tickets/dp2_psf/data/visit_zernike_corners.pkl"
    defaultRepOutPlot = "/sdf/home/l/leget/rubin-user/lsst_dev/tickets/dp2_psf/plots/"

    parser = argparse.ArgumentParser(description="Single visit focus gradient analysis")
    parser.add_argument('--visit', type=int, required=True, help="Visit ID to analyze")
    parser.add_argument('--visitMappingFile', type=str, default=defaultVisitMappingFile,
                        help="Path to visit_parquet_mapping.pkl file")
    parser.add_argument('--fitHeightMap', type=str, default=defaultFitHeightMap,
                        help="Path to SLAC height map FITS file")
    parser.add_argument('--zernikeCornersFile', type=str, default=defaultZernikeCornersFile,
                        help="Path to visit_zernike_corners.pkl file")
    parser.add_argument('--repOutPlot', type=str, default=defaultRepOutPlot,
                        help="Output directory for plots")

    args = parser.parse_args()

    analyze_single_visit(
        visit=args.visit,
        visitMappingFile=args.visitMappingFile,
        fitHeightMap=args.fitHeightMap,
        repOutPlot=args.repOutPlot,
        zernikeCornersFile=args.zernikeCornersFile,
    )


if __name__ == "__main__":
    main()
