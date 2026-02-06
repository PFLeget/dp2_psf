from lsst.daf.butler import Butler
import numpy as np
from astropy.table import Table
import astropy.units as units
from tqdm import tqdm
import os
import argparse

import pickle


def getData(band='u', repOut='data/u/'):

    repo = "dp2_prep"
    collection = "LSSTCam/runs/DRP/DP2/v30_0_0/DM-53881/stage2"

    butler = Butler(repo, collections=collection)

    PSFTable_visit_dsrs = list(butler.registry.queryDatasets("refit_psf_star"))
    visit_ids = []
    band_ids = []
    for dsr in PSFTable_visit_dsrs:
        visit_ids.append(dsr.dataId["visit"])
        band_ids.append(dsr.dataId["band"])

    visit_ids = np.array(visit_ids)
    band_ids = np.array(band_ids)

    filterBand = band_ids == band

    visit_ids = visit_ids[filterBand]

    print(f'Number of visit in {band}: {len(visit_ids)}')

    columns_name = [
        'slot_Shape_xx', 'slot_Shape_yy', 'slot_Shape_xy',
        'slot_PsfShape_xx', 'slot_PsfShape_xy', 'slot_PsfShape_yy',
        'coord_ra', 'coord_dec', 'slot_Centroid_x', 'slot_Centroid_y',
        'detector', 'psf_max_value', 'calib_psf_reserved',
    ]
    
    for visit in tqdm(visit_ids, desc="Processing visits"):

        dic = {}
        
        table = butler.get("refit_psf_star", instrument="LSSTCam", visit=visit, parameters={"columns": columns_name})
        
        table['T_src'] = table['slot_Shape_xx'] + table['slot_Shape_yy']
        table['e1_src'] = (table['slot_Shape_xx'] - table['slot_Shape_yy']) / table['T_src']
        table['e2_src'] = 2*table['slot_Shape_xy'] / table['T_src']
        
        table['T_psf'] = table['slot_PsfShape_xx'] + table['slot_PsfShape_yy']
        table['e1_psf'] = (table['slot_PsfShape_xx'] - table['slot_PsfShape_yy']) / table['T_psf']
        table['e2_psf'] = 2*table['slot_PsfShape_xy'] / table['T_psf']

        
        dic.update({
            visit: {
                'ixx_src': np.array(table['slot_Shape_xx']),
                'iyy_src': np.array(table['slot_Shape_yy']),
                'ixy_src': np.array(table['slot_Shape_xy']),
                'ixx_psf': np.array(table['slot_PsfShape_xx']),
                'iyy_psf': np.array(table['slot_PsfShape_yy']),
                'ixy_psf': np.array(table['slot_PsfShape_xy']),
                'dT_T': np.array((table['T_src'] - table['T_psf']) / table['T_src']),
                'de1': np.array(table['e1_src'] - table['e1_psf']),
                'de2': np.array(table['e2_src'] - table['e2_psf']),
                'ra': np.array(table['coord_ra']),
                'dec': np.array(table['coord_dec']),
                'xCCD': np.array(table['slot_Centroid_x']),
                'yCCD': np.array(table['slot_Centroid_y']),
                'detector': np.array(table['detector']),
                'psf_max_value': np.array(table['psf_max_value']),
                'calib_psf_reserved': np.array(table['calib_psf_reserved']),
                'band': band,
            }
        })

        PklFile = open(os.path.join(repOut, f'{visit}.pkl'), 'wb')
        pickle.dump(dic, PklFile)
        PklFile.close()


def main():
    parser = argparse.ArgumentParser(description="Extract PSF data for all visits in a specific band.")
    parser.add_argument("--band", type=str, required=True, help="The band to process (e.g., y, g, r, i, z, u)")
    parser.add_argument("--repOut", type=str, required=True, help="Output directory for the pickle files")
    args = parser.parse_args()

    getData(band=args.band, repOut=args.repOut)


if __name__ == "__main__":
    main()