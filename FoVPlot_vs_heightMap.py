import numpy as np
import re
from sklearn.neighbors import KNeighborsRegressor
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import treegp
from lsst.daf.butler import Butler
import lsst.afw.cameraGeom as cameraGeom
import lsst.geom as geom
from lsst.obs.lsst import LsstCam
from lsst.afw.cameraGeom import PIXELS, FOCAL_PLANE
from astropy.io import fits
from astropy.table import Table
from scipy.stats import binned_statistic
import warnings
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

    Parameters
    ----------
    parquet_path : str
        Path to the parquet file

    Returns
    -------
    dict
        Dictionary with all necessary columns including derived ones
    """
    # Read parquet file with polars (fast!)
    table = polars.scan_parquet(parquet_path).select(PARQUET_COLUMNS).collect()

    # Convert to numpy arrays
    slot_Shape_xx = table['slot_Shape_xx'].to_numpy()
    slot_Shape_yy = table['slot_Shape_yy'].to_numpy()
    slot_Shape_xy = table['slot_Shape_xy'].to_numpy()
    slot_PsfShape_xx = table['slot_PsfShape_xx'].to_numpy()
    slot_PsfShape_yy = table['slot_PsfShape_yy'].to_numpy()
    slot_PsfShape_xy = table['slot_PsfShape_xy'].to_numpy()

    # Compute derived quantities
    T_src = slot_Shape_xx + slot_Shape_yy
    e1_src = (slot_Shape_xx - slot_Shape_yy) / T_src
    e2_src = 2 * slot_Shape_xy / T_src

    T_psf = slot_PsfShape_xx + slot_PsfShape_yy
    e1_psf = (slot_PsfShape_xx - slot_PsfShape_yy) / T_psf
    e2_psf = 2 * slot_PsfShape_xy / T_psf

    return {
        'ixx_src': slot_Shape_xx,
        'iyy_src': slot_Shape_yy,
        'ixy_src': slot_Shape_xy,
        'ixx_psf': slot_PsfShape_xx,
        'iyy_psf': slot_PsfShape_yy,
        'ixy_psf': slot_PsfShape_xy,
        'dT_T': (T_src - T_psf) / T_src,
        'de1': e1_src - e1_psf,
        'de2': e2_src - e2_psf,
        'ra': table['coord_ra'].to_numpy(),
        'dec': table['coord_dec'].to_numpy(),
        'xCCD': table['slot_Centroid_x'].to_numpy(),
        'yCCD': table['slot_Centroid_y'].to_numpy(),
        'detector': table['detector'].to_numpy(),
        'psf_max_value': table['psf_max_value'].to_numpy(),
        'calib_psf_reserved': table['calib_psf_reserved'].to_numpy(),
    }


def pixel_to_focal(x, y, det):
    """
    Parameters
    ----------
    x, y : array
        Pixel coordinates.
    det : lsst.afw.cameraGeom.Detector
        Detector of interest.

    Returns
    -------
    fpx, fpy : array
        Focal plane position in millimeters in DVCS
        See https://lse-349.lsst.io/
    """
    tx = det.getTransform(cameraGeom.PIXELS, cameraGeom.FOCAL_PLANE)
    fpx, fpy = tx.getMapping().applyForward(np.vstack((x, y)))

    return fpx.ravel(), fpy.ravel()


def make_metrology_table(file="LSST_FP_cold_b_measurement_4col_bysurface.fits", rsid=None, write=False):
    """
    Make an astropy table of the height measurement data.
    Inputs:
    file: string, file path for measurement file
    rsid: string (optional) like R##_S## if you want data for just one sensor
    write: bool (default False), whether to write out the table as a fits file
    Outputs:
    bigtable: One large astropy table with focal plane x and y coordinates, modeled and measured z values, and the RSID for which detector each fpx,fpy coord pair is on
    """

    rows = []
    with fits.open(file) as hdulist:
        for hdu in tqdm(hdulist):
            if isinstance(hdu, fits.BinTableHDU):
                table = Table(hdu.data)
                extname = hdu.header['EXTNAME']
                if rsid is not None:
                    if extname == rsid:  # filter to the single det , 172
                        extname = re.sub(r'(R\d\d)(S\d\d)', r'\1_\2', extname)
                        for x, y, z_mod, z_meas in zip(table['X_CCS'], table['Y_CCS'], table['Z_CCS_MODEL'], table['Z_CCS_MEASURED']):
                            fpx = y
                            fpy = x
                            rows.append([fpx, fpy, z_mod, z_meas, extname])
                else:
                    if re.fullmatch(r'R\d\dS\d\d', extname):
                        extname = re.sub(r'(R\d\d)(S\d\d)', r'\1_\2', extname)
                        for x, y, z_mod, z_meas in zip(table['X_CCS'], table['Y_CCS'], table['Z_CCS_MODEL'], table['Z_CCS_MEASURED']):
                                fpx = y
                                fpy = x
                                rows.append([fpx, fpy, z_mod, z_meas, extname])

        bigtable = Table(rows=rows, names=['fpx', 'fpy', 'z_mod', 'z_meas', 'det'])
        if write:
            bigtable.write('metrology_fp.fits', format='fits', overwrite=True)

    return bigtable


def getHeightMap_vs_FoV(band='g', zernikeKey="z4", repoButler="dp2_prep",
                        secondMomentKey='dT',
                        visitMappingFile="data/visit_parquet_mapping.pkl",
                        dicZernike="visit_to_band_map.pkl",
                        fitHeightMap="data/LSST_FP_cold_b_measurement_4col_bysurface.fits",
                        collectionButler="LSSTCam/runs/DRP/DP2/v30_0_0/DM-53881/stage2",
                        repOutPlot='plots/',
                        repOutFile='data/',
                        subtract_focal_plane_mean=False,
                        ):

    if secondMomentKey not in ['T', 'e1', 'e2', 'dT', 'de1', 'de2']:
        raise ValueError('Not a valid key')

    print('read butler')

    butler = Butler(repoButler, collections=collectionButler)
    refit_psf_star_visit_dsrs = list(butler.registry.queryDatasets("refit_psf_star"))
    visitsDP2 = set()

    for dsr in refit_psf_star_visit_dsrs:
        visitsDP2.update({dsr.dataId["visit"]})
    print('Done read butler')

    # Load visit mapping
    with open(visitMappingFile, 'rb') as f:
        visit_mapping = pickle.load(f)

    tableSLAC = make_metrology_table(file=fitHeightMap, rsid=None, write=False)
    table = pickle.load(open(dicZernike, 'rb'))

    zernikeDic = {}

    for visit in table:
        if visit in visitsDP2 and table[visit]['band'] == band:
            if visit not in zernikeDic:
                zernikeDic.update({visit: table[visit][zernikeKey]})

    z_i = [np.median(zernikeDic[visit]) for visit in zernikeDic]
    Z_i_SWEEP = []
    RHOSWEEP = []
    DTSWEEP = []
    HEIGHTSWEEP = []

    z_i_central = np.linspace(-1, 1, 41)
    half_bin_zise = 0.15
    if zernikeKey == 'z4':
        z_i_central = z_i_central[5:36]
    if zernikeKey in ['z5', 'z6']:
        z_i_central = np.linspace(-0.5, 0.5, 41)
        half_bin_zise = 0.1
    if zernikeKey == 'z7':
        z_i_central = np.linspace(-0.25, 0.5, 41)
        half_bin_zise = 0.03 * 3
    if zernikeKey in ['z8', 'z9', 'z10']:
        z_i_central = np.linspace(-0.25, 0.25, 41)
        half_bin_zise = 0.03 * 3
    if zernikeKey == 'z11':
        z_i_central = np.linspace(-0.25, 0.25, 41)
        half_bin_zise = 0.03

    # to remove
    z_i_central = np.array([z_i_central[10]])

    SLOPES = []
    HEIGHTMIN = []
    z_i_min = z_i_central - half_bin_zise
    z_i_max = z_i_central + half_bin_zise

    MAX = 0.5
    MIN = -MAX
    MAX = MAX
    CMAP = plt.cm.seismic
    colorlabel = "T - <T> (pixel$^2$)"
    if secondMomentKey == 'de1':
        MAX = 0.05
        MIN = -MAX
        MAX = MAX
        CMAP = plt.cm.seismic
        colorlabel = "de1 - <de1> (pixel$^2$)"

    if secondMomentKey == 'de2':
        MAX = 0.05
        MIN = -MAX
        MAX = MAX
        CMAP = plt.cm.seismic
        colorlabel = "de1 - <de1> (pixel$^2$)"

    for i in range(len(z_i_min)):

        plt.figure(figsize=(20, 20))
        plt.subplots_adjust(top=0.98, wspace=0.3, hspace=0.2, right=0.99, left=0.07, bottom=0.05)

        TTFoV = []
        ZZFoV = []
        z_i_list = []

        meanify = {}
        N_visit = 0

        for visit in tqdm(zernikeDic, desc=f"Loading meanify | loop over visit in {band}-band"):

            if np.nanmedian(zernikeDic[visit]) > z_i_min[i] and np.nanmedian(zernikeDic[visit]) < z_i_max[i]:
                # Check if visit exists in the mapping
                if visit in visit_mapping:
                    # Load data directly from parquet
                    dic = load_visit_data(visit_mapping[visit]['parquet_path'])
                    ccdIds = list(set(dic['detector']))

                    # Compute field for entire focal plane first
                    T_full = dic['ixx_src'] + dic['iyy_src']
                    if secondMomentKey in ['T', 'dT']:
                        field_full = T_full
                    if secondMomentKey in ['e1', 'de1']:
                        field_full = (dic['ixx_src'] - dic['iyy_src']) / T_full
                    if secondMomentKey in ['e2', 'de2']:
                        field_full = 2 * dic['ixy_src'] / T_full

                    # Compute focal plane mean for this visit
                    if secondMomentKey in ['dT', 'de1', 'de2']:
                        if subtract_focal_plane_mean:
                            visit_focal_plane_mean = np.mean(field_full)
                        else:
                            visit_focal_plane_mean = None

                    for ccd in ccdIds:
                        if ccd not in meanify:
                            meanify.update({ccd: treegp.meanify(bin_spacing=150, statistics="median")})

                        filtreDetector = dic['detector'] == ccd
                        coord = np.array([dic['xCCD'][filtreDetector], dic['yCCD'][filtreDetector]]).T
                        field = field_full[filtreDetector]
                        if secondMomentKey in ['dT', 'de1', 'de2']:
                            if subtract_focal_plane_mean:
                                field = field - visit_focal_plane_mean
                            else:
                                field = field - np.mean(field)
                        meanify[ccd].add_field(coord, field)
                    z_i_list.append(np.nanmedian(zernikeDic[visit]))
                    N_visit += 1

        if N_visit != 0:
            # Compute focal plane mean height if needed
            if subtract_focal_plane_mean:
                meanHeightFocalPlane = np.mean(np.array(tableSLAC['z_meas']))

            for ccd in tqdm(meanify, desc=f"Building meanify | loop over ccds in {band}-band"):

                meanify[ccd].meanify()

                plt.subplot(2, 2, 1)

                FiltDet = np.array(tableSLAC['det']) == camera[ccd].getName()
                coordSLAC = np.array([np.array(tableSLAC['fpx'])[FiltDet], np.array(tableSLAC['fpy'])[FiltDet]]).T
                if subtract_focal_plane_mean:
                    heightSLAC = np.array(tableSLAC['z_meas'])[FiltDet] - meanHeightFocalPlane
                else:
                    meanHeightDet = np.mean(np.array(tableSLAC['z_meas'])[FiltDet])
                    heightSLAC = np.array(tableSLAC['z_meas'])[FiltDet] - meanHeightDet
                plt.scatter(coordSLAC[:, 0], coordSLAC[:, 1], s=12, marker='s',
                            c=heightSLAC, cmap=plt.cm.seismic, vmin=-0.005, vmax=0.005)

                plt.subplot(2, 2, 3)

                x, y = np.meshgrid(meanify[ccd]._xedge, meanify[ccd]._yedge)
                nBin0, nBin1 = np.shape(x)[0], np.shape(x)[1]
                x = x.reshape(nBin0*nBin1)
                y = y.reshape(nBin0*nBin1)
                x, y = pixel_to_focal(x, y, camera[ccd])
                x = x.reshape((nBin0, nBin1))
                y = y.reshape((nBin0, nBin1))

                _ = plt.xticks(fontsize=18)
                _ = plt.yticks(fontsize=18)

                plt.pcolormesh(x, y, meanify[ccd]._average, vmin=MIN, vmax=MAX, cmap=CMAP)

                CoordSubmit = meanify[ccd].coords0
                csx, csy = pixel_to_focal(CoordSubmit[:, 0], CoordSubmit[:, 1], camera[ccd])
                CoordSubmit = np.array([csx, csy]).T
                PsfSubmit = meanify[ccd].params0

                plt.subplot(2, 2, 4)
                try:
                    knn = KNeighborsRegressor(n_neighbors=20)
                    knn.fit(coordSLAC, heightSLAC)
                    predict = knn.predict(CoordSubmit)

                    TTFoV.append(PsfSubmit)
                    ZZFoV.append(predict)

                    plt.scatter(predict, PsfSubmit, color='b', s=2)
                except:
                    print("KNN failed")

                plt.ylabel(colorlabel, fontsize=22)
                plt.xlabel("z - <z> (mm)", fontsize=22)

                _ = plt.xticks(fontsize=18)
                _ = plt.yticks(fontsize=18)

                if not subtract_focal_plane_mean:
                    plt.ylim(1.5*MIN, 1.5*MAX)
                    plt.xlim(-7e-3, 7e-3)
                else:
                    cstPlot = 3
                    plt.ylim(cstPlot*1.5*MIN, cstPlot*1.5*MAX)
                    plt.xlim(-7e-3*cstPlot, 7e-3*cstPlot)

                ylim = plt.ylim()
                xlim = plt.xlim()
                plt.plot([0, 0], ylim, 'k--')
                plt.plot(xlim, [0, 0], 'k--')

                plt.xlim(xlim)
                plt.ylim(ylim)

            plt.subplot(2, 2, 1)
            cb = plt.colorbar()
            plt.axis('equal')
            cb.set_label("z - <z> (mm)", size=22)
            cb.ax.tick_params(labelsize=18)
            plt.xlabel('x (mm)', size=22)
            plt.ylabel('y (mm)', size=22)
            _ = plt.xticks(fontsize=18)
            _ = plt.yticks(fontsize=18)
            plt.title("Height map from SLAC", size=18)

            plt.subplot(2, 2, 3)
            cb = plt.colorbar()

            cb.set_label(colorlabel, size=18)
            cb.ax.tick_params(labelsize=18)
            plt.xlabel('x (mm)', size=22)
            plt.ylabel('y (mm)', size=22)

            plt.title(f"<{zernikeKey}> = %.3f" % ((np.nanmedian(z_i_list))), size=18)
            plt.axis('equal')
            _ = plt.xticks(fontsize=18)
            _ = plt.yticks(fontsize=18)

            plt.subplot(2, 2, 2)

            binning = np.linspace(-1, 1, 100)

            _ = plt.hist(z_i, color='b', bins=binning)
            ylim = plt.ylim()
            xlim = plt.xlim()
            plt.fill_betweenx(ylim, z_i_min[i], x2=z_i_max[i], color='r', alpha=0.3)
            plt.ylim(ylim)
            plt.xlim(xlim)
            if len(zernikeKey) == 2:
                zernike_label = f'${zernikeKey[0]}_{zernikeKey[1]}$'
            if len(zernikeKey) == 3:
                zernike_label = '$%s_{%s%s}$' % ((zernikeKey[0], zernikeKey[1], zernikeKey[2]))
            plt.xlabel(zernike_label, fontsize=22)
            _ = plt.xticks(fontsize=18)
            _ = plt.yticks(fontsize=18)

            PsfSubmit = np.concatenate(TTFoV)
            predict = np.concatenate(ZZFoV)

            FILTRESWEEP = np.isfinite(PsfSubmit) & np.isfinite(predict)
            Z_i_SWEEP.append(np.median(z_i_list))
            RHOSWEEP.append(np.corrcoef(predict, PsfSubmit)[0, 1])
            DTSWEEP.append(PsfSubmit)
            HEIGHTSWEEP.append(predict)

            plt.savefig(os.path.join(repOutPlot, f'{band}/{zernikeKey}_FoV_{i}_{band}_{secondMomentKey}.png'))
            plt.close()

    dicSweep = {'Z_i_SWEEP': Z_i_SWEEP,
                'RHOSWEEP': RHOSWEEP,
                'DTSWEEP': DTSWEEP,
                'HEIGHTSWEEP': HEIGHTSWEEP,
                'band': band,
                'zernike': zernikeKey}

    FPKL = open(os.path.join(repOutFile, f'rho_sweep_{band}_{zernikeKey}_{secondMomentKey}.pkl'), 'wb')
    pickle.dump(dicSweep, FPKL)
    FPKL.close()


def main():

    defaultCollectionButler = "LSSTCam/runs/DRP/DP2/v30_0_0/DM-53881/stage2"
    defaultFitHeightMap = "/sdf/home/l/leget/rubin-user/lsst_dev/tickets/dp2_psf/data/LSST_FP_cold_b_measurement_4col_bysurface.fits"
    defaultDicZernike = "/sdf/home/l/leget/rubin-user/lsst_dev/tickets/dp2_psf/data/visit_to_band_mapv2.pkl"
    defaultVisitMappingFile = "/sdf/home/l/leget/rubin-user/lsst_dev/tickets/dp2_psf/data/visit_parquet_mapping.pkl"
    defaultRepOutPlot = "/sdf/home/l/leget/rubin-user/lsst_dev/tickets/dp2_psf/plots/"
    defaultRepOutFile = "/sdf/home/l/leget/rubin-user/lsst_dev/tickets/dp2_psf/data/"

    parser = argparse.ArgumentParser(description="Height map vs second moment analysis")
    parser.add_argument('--band', type=str, required=True, help="The band to process (e.g., y, g, r, i, z, u)")
    parser.add_argument('--visitMappingFile', type=str, default=defaultVisitMappingFile, help="Path to visit_parquet_mapping.pkl file")

    parser.add_argument('--secondMomentKey', type=str, default='dT', help='key second moment')
    parser.add_argument('--zernikeKey', type=str, default='z4', help='Zernike coeff where the sweep is done')
    parser.add_argument('--repoButler', type=str, default='dp2_prep', help='Rep Butler')
    parser.add_argument('--repoCollectionButler', type=str, default=defaultCollectionButler, help='Collection DP2')
    parser.add_argument('--dicZernike', type=str, default=defaultDicZernike, help='dic zernike')
    parser.add_argument('--fitHeightMap', type=str, default=defaultFitHeightMap, help='Height map')
    parser.add_argument('--repOutPlot', type=str, default=defaultRepOutPlot, help='Rep out plot')
    parser.add_argument('--repOutFile', type=str, default=defaultRepOutFile, help='Rep out file')
    parser.add_argument('--subtract_focal_plane_mean', action='store_true',
                        help='Subtract focal plane mean per visit instead of per-CCD mean')

    args = parser.parse_args()

    getHeightMap_vs_FoV(band=args.band, zernikeKey=args.zernikeKey,
                        secondMomentKey=args.secondMomentKey,
                        visitMappingFile=args.visitMappingFile,
                        repoButler=args.repoButler,
                        collectionButler=args.repoCollectionButler,
                        dicZernike=args.dicZernike,
                        fitHeightMap=args.fitHeightMap,
                        repOutPlot=args.repOutPlot,
                        repOutFile=args.repOutFile,
                        subtract_focal_plane_mean=args.subtract_focal_plane_mean,
                        )


if __name__ == "__main__":
    main()
