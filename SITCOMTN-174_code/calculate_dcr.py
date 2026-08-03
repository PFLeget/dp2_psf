import numpy as np
import matplotlib.pyplot as plt
import h5py
import coord
import pandas as pd

from astropy.time import Time
from astropy.coordinates import EarthLocation
import astropy.units as u

from multiprocessing import Pool, cpu_count

import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'psf_utils')))

import DCR


def process_chunk(chunk):
    
    t_lst = DCR.mjd_to_lst(chunk['mjd'])
    dcr1, dcr2 = DCR.compute_dcr(t_lst.deg, cerro_pachon.lat.deg, chunk['ra'], chunk['dec'])

    return pd.DataFrame({
        't_lst': t_lst.deg,
        'dcr1': dcr1,
        'dcr2': dcr2
    })


def process_in_parallel(data, nchunks, nworkers, out_file='../w_2025_39_DM-52645/color_fit/dcr_20251020.csv'):

    chunks = np.array_split(data, nchunks)
    
    with Pool(processes=nworkers) as pool:
        for i, result in enumerate(pool.imap(process_chunk, chunks), 1):
            print(f'processing chunk {i}', flush=True)
            result.to_csv(out_file, mode='a', header=(i == 1), index=False)



if __name__=='__main__':

    single_visit_catalog_fp = '../w_2025_39_DM-52645/color_fit/single_visit_catalog_color_w39_20251020_test.h5'

    with h5py.File(single_visit_catalog_fp, 'r') as f:
    
        mjd = f['mjd'][:]
        ra = f['ra'][:]
        dec = f['dec'][:]
    
    data = pd.DataFrame({'mjd':mjd, 'ra':ra, 'dec':dec})
    
    cerro_pachon = EarthLocation(lat=-30.2407*u.deg,
                                     lon=-70.7366*u.deg,
                                     height=2722*u.m)

    nchunks = 100
    process_in_parallel(data=data, nchunks=nchunks, nworkers=10)

    

