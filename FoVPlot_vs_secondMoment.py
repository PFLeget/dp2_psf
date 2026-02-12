from lsst.daf.butler import Butler
import numpy as np
from astropy.table import Table
import astropy.units as units
import treegp
print(treegp.__version__)
from tqdm import tqdm

import matplotlib.pyplot as plt
from lsst.utils.plotting import publication_plots
publication_plots.set_rubin_plotstyle()

import pickle
import lsst.afw.cameraGeom as cameraGeom
from lsst.obs.lsst import LsstCam
from lsst.geom import SpherePoint, degrees, Point2D
import pandas as pd
import argparse

# I guess this cmap or the other can in general
# give better idea of physical variation on the
# focal plane from past experience.
# CMAP = plt.cm.inferno

import glob
import os


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


def plot_FoV_second_Moment(bands='g', rep="data/", repOutPlot='plots/', 
                           key_second_moment='dT_T', bin_spacing = 150, colorScale = 0.005, autoColorScale=False,
                           autoColorScaleCst=2., statisticsMedian=False,
                           colorlabel=None, title=None, pklInput=None, psf_max_value=0):

    CMAP = plt.cm.inferno

    if pklInput is None:

        pkls = []
        for b in bands:
            pkls.append(glob.glob(os.path.join(rep, b+'/*.pkl')))
        pkls = np.concatenate(pkls)
        # pkls = pkls[:100]


        meanifyStream = {}

        for pkl in tqdm(pkls, desc="loop over visits to compute spatial average:"):
            dic = pd.read_pickle(pkl)
            visit = list(dic.keys())[0]
            ccdIds = set(dic[visit]["detector"])
            
            for ccd in ccdIds:
                filtering = (dic[visit]["detector"] == ccd)
                filtering &= (dic[visit]["psf_max_value"] > psf_max_value)
                coord = np.array([dic[visit]['xCCD'], dic[visit]['yCCD']]).T
                if ccd not in meanifyStream:
                    if not statisticsMedian:
                        meanifyStream.update({ccd: treegp.MeanifyStream(bin_spacing=bin_spacing, bounds=(0, 4100, 0, 4100))})
                    else:
                        meanifyStream.update({ccd: treegp.meanify(bin_spacing=bin_spacing, statistics='median')})
                meanifyStream[ccd].add_field(coord[filtering], dic[visit][key_second_moment][filtering])

        for ccd in meanifyStream:
            if not statisticsMedian:
                meanifyStream[ccd].meanify()
            else:
                meanifyStream[ccd].meanify(lu_min=0, lu_max=4100, lv_min=0, lv_max=4100)
    else:
        dicInput = pd.read_pickle(pklInput)
        ccdIds = list(dicInput.keys())


    if autoColorScale:
        M = []
        for i in ccdIds:
            if pklInput is None:
                M.append(meanifyStream[i]._average)
            else:
                M.append(dicInput[i]['_average'])
        M = np.concatenate(M)
        MEAN = np.median(M[np.isfinite(M)])
        STD = np.std(M[np.isfinite(M)])

        MIN = MEAN - autoColorScaleCst * STD
        MAX = MEAN + autoColorScaleCst * STD
    else:
        MIN = -colorScale
        MAX = colorScale

    dicMeanifyPlot = {}
    plt.figure(figsize=(20,12))
    for i in ccdIds:
        if pklInput is None:
            x, y = np.meshgrid(meanifyStream[i]._xedge, meanifyStream[i]._yedge)
            nBin0, nBin1 = np.shape(x)[0], np.shape(x)[1]
            x = x.reshape(nBin0*nBin1)
            y = y.reshape(nBin0*nBin1)
            x, y = pixel_to_focal(x, y, camera[i])
            x = x.reshape((nBin0, nBin1))
            y = y.reshape((nBin0, nBin1))
            plt.pcolormesh(x, y , meanifyStream[i]._average, vmin=MIN, vmax=MAX, cmap=CMAP)
            dicMeanifyPlot.update({i: {
                'x': x,
                'y': y,
                '_average': meanifyStream[i]._average
            }})
        else:
            plt.pcolormesh(dicInput[i]['x'], dicInput[i]['y'],
                           dicInput[i]['_average'], vmin=MIN, vmax=MAX, cmap=CMAP)

    cb = plt.colorbar()
    if colorlabel is None:
        colorlabel = key_second_moment

    cb.set_label(colorlabel, size=22)
    cb.ax.tick_params(labelsize=18)
    plt.xlabel('x (mm)',size=22)
    plt.ylabel('y (mm)',size=22)
    if title is None:
        title = f"DP2 {key_second_moment} | bands: ({bands})" 
    plt.title(title, size=18)
    plt.axis('equal')
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    #plt.gcf().patch.set_facecolor('none')
    #plt.gca().patch.set_facecolor('none')
    if statisticsMedian:
        median_key = "median"
    else:
        median_key = ""
    plt.savefig(os.path.join(repOutPlot, f'{key_second_moment}_2d_{bands}_{int(bin_spacing)}_{median_key}_{int(psf_max_value)}.png'))

    if pklInput is None:
        pklFile = open(os.path.join(repOutPlot, f'{key_second_moment}_2d_{bands}_{int(bin_spacing)}_{median_key}_{int(psf_max_value)}.pkl'), 'wb')
        pickle.dump(dicMeanifyPlot, pklFile)
        pklFile.close()

def main():

    parser = argparse.ArgumentParser(description="Height map vs second moment analysis")
    parser.add_argument('--bands', type=str, required=True, help="The band to process (e.g., y, g, r, i, z, u)")
    parser.add_argument('--pathPSFRep', type=str, required=True, help="Path to PSF File")

    parser.add_argument('--key_second_moment', type=str, default='dT_T', help='second moment key')
    parser.add_argument('--bin_spacing', type=float, default=150, help='bin size')
    parser.add_argument('--psf_max_value', type=float, default=0, help='exclude all psf that has a lower value for the pixel max (e-)')
    parser.add_argument('--colorScale', type=float, default=0.005, help='Min/Max of color scale')
    parser.add_argument('--autoColorScaleCst', type=float, default=2., help='')
    parser.add_argument('--repOutPlot', type=str, default='plots/', help='Rep out plot')
    parser.add_argument('--pklInput', type=str, default=None, help='the output give as input just to redo the plot')   

    parser.add_argument('--autoColorScale', action='store_true',)
    parser.add_argument('--statisticsMedian', action='store_true',)

    args = parser.parse_args()

    plot_FoV_second_Moment(bands=args.bands, rep=args.pathPSFRep, repOutPlot=args.repOutPlot, 
                           key_second_moment=args.key_second_moment, bin_spacing = args.bin_spacing, 
                           colorScale = args.colorScale, autoColorScale=args.autoColorScale, 
                           autoColorScaleCst=args.autoColorScaleCst, statisticsMedian=args.statisticsMedian,
                           colorlabel=None, title=None, pklInput=args.pklInput, psf_max_value=args.psf_max_value)



if __name__ == "__main__":

    main()


    





    


