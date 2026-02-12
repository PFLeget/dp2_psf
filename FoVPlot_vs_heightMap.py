import numpy as np
import re
from sklearn.neighbors import KNeighborsRegressor
from tqdm import tqdm
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
import pandas as pd
import argparse

camera = LsstCam.getCamera()

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
                    if extname == rsid: # filter to the single det , 172
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
        if write: bigtable.write('metrology_fp.fits', format='fits', overwrite=True)

    return bigtable

def getHeightMap_vs_FoV(band='g', zernikeKey="z4", repoButler="dp2_prep",
                        secondMomentKey = 'dT',
                        pathPSFFile="data/g/",
                        dicZernike="visit_to_band_map.pkl",
                        fitHeightMap="data/LSST_FP_cold_b_measurement_4col_bysurface.fits",
                        collectionButler="LSSTCam/runs/DRP/DP2/v30_0_0/DM-53881/stage2",
                        repOutPlot='plots/',
                        repOutFile='data/',
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

        z_i_central = np.linspace(-1,1,41)
        half_bin_zise = 0.15
        if zernikeKey == 'z4':
            z_i_central = z_i_central[5:36]
        if zernikeKey == 'z11':
            z_i_central = np.linspace(-0.25,0.25,41)
            #z_i_central = z_i_central[5:36]
            half_bin_zise = 0.03
        if zernikeKey == 'z7':
            z_i_central = np.linspace(-0.25,0.5,41)
            #z_i_central = z_i_central[5:36]
            half_bin_zise = 0.03 * 3

        SLOPES = []
        HEIGHTMIN = []
        z_i_min = z_i_central - half_bin_zise
        z_i_max = z_i_central + half_bin_zise

        # 14
        # z_i_min = [z_i_min[14]]
        # z_i_max = [z_i_max[14]]

        MAX = 0.5
        MIN = -MAX
        MAX = MAX
        CMAP = plt.cm.seismic
        colorlabel = "T - <T> (pixel$^2$)"

        for i in range(len(z_i_min)):

            plt.figure(figsize=(20,20))
            plt.subplots_adjust(top=0.98, wspace=0.3, hspace=0.2, right=0.99, left=0.07, bottom=0.05)

            TTFoV = []
            ZZFoV = []
            z_i_list = []

            meanify = {}
            N_visit = 0

            for visit in tqdm(zernikeDic, desc=f"Loading meanify | loop over visit in {band}-band"):

                if np.nanmedian(zernikeDic[visit])> z_i_min[i] and np.nanmedian(zernikeDic[visit])<z_i_max[i]:
                    if os.path.exists(os.path.join(pathPSFFile, f'{visit}.pkl')):
                        dic = pd.read_pickle(os.path.join(pathPSFFile, f'{visit}.pkl'))
                        ccdIds = list(set(dic[visit]['detector']))

                        for ccd in ccdIds:
                            if ccd not in meanify:
                                meanify.update({ccd: treegp.meanify(bin_spacing=150, statistics="median")})
                            
                            filtreDetector = dic[visit]['detector'] == ccd
                            coord = np.array([dic[visit]['xCCD'][filtreDetector], dic[visit]['yCCD'][filtreDetector]]).T
                            T = dic[visit]['ixx_src'][filtreDetector] + dic[visit]['iyy_src'][filtreDetector]
                            if secondMomentKey in ['T', 'dT']:
                                field = T
                            if secondMomentKey in ['e1', 'de1']:
                                field = (dic[visit]['ixx_src'][filtreDetector] - dic[visit]['iyy_src'][filtreDetector]) / T
                            if secondMomentKey in ['e2', 'de2']:
                                field = 2 * dic[visit]['ixy_src'][filtreDetector] / T
                            if secondMomentKey in ['dT', 'de1', 'de2']:
                                field -= np.mean(field)
                            #table['e1_psf'] = (table['slot_PsfShape_xx'] - table['slot_PsfShape_yy']) / table['T_psf']
                            #table['e2_psf'] = 2*table['slot_PsfShape_xy'] / table['T_psf']
                            if secondMomentKey in ['e1', 'de1']:
                                T = dic[visit]['ixx_src'][filtreDetector] + dic[visit]['iyy_src'][filtreDetector]
                                if secondMomentKey == 'dT':
                                    field = T - np.mean(T)
                            meanify[ccd].add_field(coord, field)
                        z_i_list.append(np.nanmedian(zernikeDic[visit]))
                        N_visit += 1 
            if N_visit != 0:
                for ccd in tqdm(meanify, desc=f"Building meanify | loop over ccds in {band}-band"):

                    meanify[ccd].meanify()

                    plt.subplot(2,2,1)
                
                    FiltDet = np.array(tableSLAC['det']) == camera[ccd].getName()
                    meanHeightDet = np.mean(np.array(tableSLAC['z_meas'])[FiltDet])
                    coordSLAC = np.array([np.array(tableSLAC['fpx'])[FiltDet], np.array(tableSLAC['fpy'])[FiltDet]]).T
                    heightSLAC = np.array(tableSLAC['z_meas'])[FiltDet] - meanHeightDet
                    plt.scatter(coordSLAC[:,0], coordSLAC[:,1], s=12, marker='s',
                                c=heightSLAC, cmap=plt.cm.seismic, vmin=-0.005, vmax=0.005)

                    plt.subplot(2,2,3)
            
                    x, y = np.meshgrid(meanify[ccd]._xedge, meanify[ccd]._yedge)
                    nBin0, nBin1 = np.shape(x)[0], np.shape(x)[1]
                    x = x.reshape(nBin0*nBin1)
                    y = y.reshape(nBin0*nBin1)
                    x, y = pixel_to_focal(x, y, camera[ccd])
                    x = x.reshape((nBin0, nBin1))
                    y = y.reshape((nBin0, nBin1))
            
                    _ = plt.xticks(fontsize=18)
                    _ = plt.yticks(fontsize=18)
                
                    plt.pcolormesh(x, y , meanify[ccd]._average, vmin=MIN, vmax=MAX, cmap=CMAP)
        
                    CoordSubmit = meanify[ccd].coords0
                    csx, csy = pixel_to_focal(CoordSubmit[:,0], CoordSubmit[:,1], camera[ccd])
                    CoordSubmit = np.array([csx, csy]).T
                    PsfSubmit = meanify[ccd].params0

                    plt.subplot(2,2,4)
                    try:
                        knn = KNeighborsRegressor(n_neighbors=20)#, weights='distance')
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
            
                    plt.ylim(1.5*MIN, 1.5*MAX)
                    plt.xlim(-7e-3, 7e-3)
            
                    ylim = plt.ylim()
                    xlim = plt.xlim()
                    plt.plot([0,0], ylim, 'k--')
                    plt.plot(xlim, [0,0], 'k--')

                    plt.xlim(xlim)
                    plt.ylim(ylim)

                plt.subplot(2,2,1)
                cb = plt.colorbar()
                plt.axis('equal')    
                cb.set_label("z - <z> (mm)", size=22)
                cb.ax.tick_params(labelsize=18)
                plt.xlabel('x (mm)',size=22)
                plt.ylabel('y (mm)',size=22)
                _ = plt.xticks(fontsize=18)
                _ = plt.yticks(fontsize=18)
                plt.title("Height map from SLAC", size=18)

                plt.subplot(2,2,3)
                cb = plt.colorbar()

                cb.set_label(colorlabel, size=18)
                cb.ax.tick_params(labelsize=18)
                plt.xlabel('x (mm)',size=22)
                plt.ylabel('y (mm)',size=22)

                plt.title(f"<{zernikeKey}> = %.3f"%((np.nanmedian(z_i_list))), size=18)
                plt.axis('equal')
                _ = plt.xticks(fontsize=18)
                _ = plt.yticks(fontsize=18)

                plt.subplot(2,2,2)

                binning = np.linspace(-2.5, 2.5, 100)

                if zernikeKey == 'z11':
                    binning = np.linspace(-0.5, 0.5, 50)

                _ = plt.hist(z_i, color='b', bins=binning)
                ylim = plt.ylim()
                xlim = plt.xlim()
                plt.fill_betweenx(ylim, z_i_min[i], x2=z_i_max[i], color='r', alpha=0.3)
                plt.ylim(ylim)
                plt.xlim(xlim)
                if len(zernikeKey) == 2:
                    zernike_label = f'${zernikeKey[0]}_{zernikeKey[1]}$'
                if len(zernikeKey) == 3:
                    zernike_label = '$%s_{%s%s}$'%((zernikeKey[0], zernikeKey[1], zernikeKey[2]))
                plt.xlabel(zernike_label, fontsize=22)
                _ = plt.xticks(fontsize=18)
                _ = plt.yticks(fontsize=18)


                PsfSubmit = np.concatenate(TTFoV)
                predict= np.concatenate(ZZFoV)
        
                FILTRESWEEP = np.isfinite(PsfSubmit) & np.isfinite(predict)
                Z_i_SWEEP.append(np.median(z_i_list))
                RHOSWEEP.append(np.corrcoef(predict, PsfSubmit)[0, 1])
                DTSWEEP.append(PsfSubmit)
                HEIGHTSWEEP.append(predict)

                plt.savefig(os.path.join(repOutPlot, f'{zernikeKey}_FoV_{i}_{band}_{secondMomentKey}.png'))
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
    defaultFitHeightMap = "data/LSST_FP_cold_b_measurement_4col_bysurface.fits"
    defaultDicZernike = "data/visit_to_band_mapv2.pkl"
    defaultRepOutPlot = "plots/"
    defaultRepOutFile = "data/"


    parser = argparse.ArgumentParser(description="Height map vs second moment analysis")
    parser.add_argument('--band', type=str, required=True, help="The band to process (e.g., y, g, r, i, z, u)")
    parser.add_argument('--pathPSFFile', type=str, required=True, help="Path to PSF File")

    parser.add_argument('--secondMomentKey', type=str, default='dT', help='key second moment')
    parser.add_argument('--zernikeKey', type=str, default='z4', help='Zernike coeff where the sweep is done')
    parser.add_argument('--repoButler', type=str, default='dp2_prep', help='Rep Butler')
    parser.add_argument('--repoCollectionButler', type=str, default=defaultCollectionButler, help='Collection DP2')
    parser.add_argument('--dicZernike', type=str, default=defaultDicZernike, help='dic zernike')
    parser.add_argument('--fitHeightMap', type=str, default=defaultFitHeightMap, help='Height map')
    parser.add_argument('--repOutPlot', type=str, default=defaultRepOutPlot, help='Rep out plot')
    parser.add_argument('--repOutFile', type=str, default=defaultRepOutFile, help='Rep out file')

    args = parser.parse_args()

    getHeightMap_vs_FoV(band=args.band, zernikeKey=args.zernikeKey,
                        secondMomentKey = args.secondMomentKey,
                        pathPSFFile=args.pathPSFFile,
                        repoButler=args.repoButler,    
                        collectionButler=args.repoCollectionButler,
                        dicZernike=args.dicZernike,
                        fitHeightMap=args.fitHeightMap,
                        repOutPlot=args.repOutPlot,
                        repOutFile=args.repOutFile,
                        )


if __name__ == "__main__":
    main()
