
import numpy as np
import h5py
import pandas as pd

from astropy.coordinates import SkyCoord
from astropy.coordinates import match_coordinates_sky
import astropy.units as u
#from astropy_healpix import HEALPix

single_visit_catalog_fp = '../w_2025_49/color/single_visit_catalog_20260111.h5'
coadd_catalog_fp = '../w_2025_49/coadd_catalog_w49.h5'

print(f'single visit cat filepath: {single_visit_catalog_fp} \n coadd cat filepath: {coadd_catalog_fp}', flush=True)

with h5py.File(single_visit_catalog_fp, 'r') as f:
    #print(f['data'].keys())
    
    single_visit_ra = np.deg2rad(f['ra'][:])
    single_visit_dec = np.deg2rad(f['dec'][:])

with h5py.File(coadd_catalog_fp, 'r') as f:
    #print(f['data'].keys())
    
    coadd_ra = f['coord_ra'][:]
    coadd_dec = f['coord_dec'][:]

print(f'The single visit catalog has: {np.size(single_visit_ra)} entries, and the coadd catalog has: {np.size(coadd_ra)} entries', flush=True) 

# nside = 256    # adjust for resolution (higher = finer)
# frame = "icrs" # or 'galactic' depending on your needs

# hp = HEALPix(nside=nside, order="nested", frame=frame)

single_visit_coords = SkyCoord(ra = single_visit_ra * u.deg, 
                            dec = single_visit_dec * u.deg)#,
                            #frame = frame)
coadd_coords = SkyCoord(ra = coadd_ra * u.deg, 
                     dec = coadd_dec * u.deg)#, 
                     #frame = frame)

print("single_visit_coords length:", len(single_visit_coords), flush=True)
print("coadd_coords length:", len(coadd_coords), flush=True)
# single_visit = pd.DataFrame({'ra':single_visit_ra, 
#                              'dec':single_visit_dec, 
#                              'healpix':hp.skycoord_to_healpix(single_visit_coords)})

# coadd = pd.DataFrame({'ra':coadd_ra, 
#                       'dec':coadd_dec, 
#                       'healpix':hp.skycoord_to_healpix(coadd_coords)})

# print(f'number of healpix pixels: {np.size(np.unique(single_visit['healpix']))}', flush=True)

print('matching', flush=True)

idx, sep2d, d3d = single_visit_coords.match_to_catalog_sky(coadd_coords)

print('idx length:', len(idx), flush=True)
# matches = []

# for i, (pix, vgroup) in enumerate(single_visit.groupby("healpix")):
#     if i % 10000 == 0:
#         print(i, flush=True)
#     #check in neighbouring pixels
#     neighbours = hp.neighbours(pix)
#     candidate_pix = np.append(neighbours, pix)
#     cgroup = coadd[coadd["healpix"].isin(candidate_pix)]
    
#     if len(cgroup) == 0:
#         print(f'{i} skipped, no coadd objects', flush=True)
#         continue

#     vcoord = SkyCoord(vgroup["ra"].values*u.deg, vgroup["dec"].values*u.deg)
#     ccoord = SkyCoord(cgroup["ra"].values*u.deg, cgroup["dec"].values*u.deg)

#     # nearest-neighbor match within the pixel
#     idx, d2d, d3d = vcoord.match_to_catalog_sky(ccoord)

#     for i_v, i_c, sep in zip(vgroup.index, idx, d2d.arcsec):
#         matches.append({
#             "single_visit_idx": i_v,
#             "coadd_idx": cgroup.index[i_c],
#             "sep_arcsec": sep
#         })

match_df = pd.DataFrame({'idx': idx, 'sep_arcsec': sep2d.arcsec})

print('saving', flush=True)

match_df.to_csv("../w_2025_49/color/matched_catalog_20260111.csv", index=False)
