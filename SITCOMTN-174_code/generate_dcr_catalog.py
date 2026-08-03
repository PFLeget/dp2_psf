import h5py
import psf_utils as pu
import DCR
import os, sys
import argparse
import numpy as np
from multiprocessing import Pool

def rotate_moments(ra, dec, ixx, iyy, ixy, model_ixx, model_iyy, model_ixy, sky_rotation, mjd):

    print('Calculating DCR', flush=True)

    lst = DCR.mjd_to_lst(mjd)
    zenith_angle, parallactic_angle = DCR.compute_zenith_and_par_angles_array(-30.244633, lst.deg, ra, dec)

    rot_tel_pos = parallactic_angle - sky_rotation*np.pi/180 - (np.pi / 2)

    tan_z2 = np.tan(zenith_angle) ** 2
    dcr_1 = tan_z2 * np.cos(2 * parallactic_angle)
    dcr_2 = tan_z2 * np.sin(2 * parallactic_angle)

    print('Transforming moments', flush=True)
    sky_ixx, sky_iyy, sky_ixy = pu.rotateXYtoNW(ixx, iyy, ixy, np.deg2rad(sky_rotation))
    azel_ixx, azel_iyy, azel_ixy = pu.rotateNWtoAA(sky_ixx, sky_iyy, sky_ixy, parallactic_angle)

    model_sky_ixx, model_sky_iyy, model_sky_ixy = pu.rotateXYtoNW(model_ixx, model_iyy, model_ixy,
                                                                  np.deg2rad(sky_rotation))
    model_azel_ixx, model_azel_iyy, model_azel_ixy = pu.rotateNWtoAA(model_sky_ixx, 
                                                                     model_sky_iyy,
                                                                     model_sky_ixy, 
                                                                     parallactic_angle)

    direct_azel_ixx, direct_azel_iyy, direct_azel_ixy = pu.rotateXYtoAA(ixx, iyy, ixy, rot_tel_pos)
    direct_model_azel_ixx, direct_model_azel_iyy, direct_model_azel_ixy = pu.rotateXYtoAA(model_ixx, model_iyy, model_ixy, rot_tel_pos)

    colnames = ['local_sidereal_time', 'zenith_angle', 'parallactic_angle', 'sky_rotation', 'rot_tel_pos', 
                'dcr_1', 'dcr_2', 'ixx', 'iyy', 'ixy', 'model_ixx', 'model_iyy', 'model_ixy',
                'sky_ixx', 'sky_iyy', 'sky_ixy', 'model_sky_ixx', 'model_sky_iyy', 'model_sky_ixy',
                'azel_ixx', 'azel_iyy', 'azel_ixy', 'model_azel_ixx', 'model_azel_iyy', 'model_azel_ixy',
               'direct_azel_ixx', 'direct_azel_iyy', 'direct_azel_ixy', 'direct_model_azel_ixx', 'direct_model_azel_iyy', 'direct_model_azel_ixy']

    moments_dict = {}

    for col, data in zip(colnames, [lst, zenith_angle, parallactic_angle, sky_rotation, rot_tel_pos, dcr_1, dcr_2,
                                    ixx, iyy, ixy, model_ixx, model_iyy, model_ixy, sky_ixx, sky_iyy, sky_ixy, 
                                    model_sky_ixx, model_sky_iyy, model_sky_ixy,
                                    azel_ixx, azel_iyy, azel_ixy, model_azel_ixx, model_azel_iyy, model_azel_ixy,
                                   direct_azel_ixx, direct_azel_iyy, direct_azel_ixy, direct_model_azel_ixx, direct_model_azel_iyy, direct_model_azel_ixy]):
        moments_dict[col] = data

    return moments_dict

    
def process_chunk(chunk, savepath, chunk_num):
    print(f'chunk {chunk_num} size {chunk.shape}', flush=True)
    
    ra, dec, ixx, iyy, ixy, model_ixx, model_iyy, model_ixy, sky_rotation, mjd = chunk.T

    
    moments_dict = rotate_moments(ra, dec, ixx, iyy, ixy, model_ixx, model_iyy, model_ixy,
                                  sky_rotation, mjd)
    print(f"Produced catalog for chunk {chunk_num}", flush=True)

    write_file(moments_dict,
               filename =  f"dcr_chunk_{chunk_num:03d}.h5",
               savepath = f"{savepath}/temp/")
    
    print(f"Written catalog for chunk {chunk_num}", flush=True)

    
def write_file(moments_dict, filename, savepath):
    path = os.path.join(savepath, filename)
    print(path, flush=True)
    
    cat = h5py.File(path, "w")
    
    for colname, data in moments_dict.items():
        #print(colname, flush=True)
        cat[colname] = data
    
    cat.close()

def compile_chunks(savepath, filename, nchunks):
    with h5py.File(f'{savepath}/temp/dcr_chunk_000.h5', 'r') as f:
        columns = list(f.keys())

    combined_catalog = {col : [] for col in columns}
    for i in range(nchunks):
        with h5py.File(f'{savepath}/temp/dcr_chunk_{i:03d}.h5', 'r') as f:
            for col in combined_catalog.keys():
                combined_catalog[col].append(f[col][:])
    
    for col in combined_catalog.keys():
        combined_catalog[col] = np.concatenate(combined_catalog[col])
    
    
    with h5py.File(f'{savepath}/{filename}', "w") as catalog:
        for col in combined_catalog.keys():
                catalog[col] = combined_catalog[col]

    print(f"Combined catalog saved to {savepath}/{filename}", flush=True)
    
def main(filepath, savepath, filename, nchunks):
    with h5py.File(filepath, 'r') as f:
        ra = f['ra'][:]
        dec = f['dec'][:]
        
        ixx = f['ixx'][:]
        iyy = f['iyy'][:]
        ixy = f['ixy'][:]
        model_ixx = f['model_ixx'][:]
        model_iyy = f['model_iyy'][:]
        model_ixy = f['model_ixy'][:]
        
        sky_rotation = f['sky_rotation'][:]
        mjd = f['mjd'][:]

    data = np.column_stack([ra, dec, ixx, iyy, ixy, model_ixx, model_iyy, model_ixy, sky_rotation, mjd])
    chunks = np.array_split(data, nchunks)
    
    args = [(chunk, savepath, i) for i, chunk in enumerate(chunks)]
    with Pool(processes=nchunks) as pool:
        for i, _ in enumerate(pool.starmap(process_chunk, args), 1):
            print(f"Processed chunk {i}/{nchunks}", flush=True)

    compile_chunks(savepath, filename, nchunks)

    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Perform second moment coordinate transformations")
    parser.add_argument(
        "--filepath", "-f", required=True,
        help="input catalog filepath"
    )
    parser.add_argument(
        "--savepath", "-o", required=True,
        help="Output file path for results"
    )
    parser.add_argument(
        "--filename", "-n", required=True,
        help="Output file name for results (e.g. DRP_catalog.h5)"
    )
    args = parser.parse_args()

    # Access the arguments
    filepath = args.filepath
    filename = args.filename
    savepath = args.savepath

    print(f"Using catalog: {filepath}", flush=True)
    print(f"Saving results to path: {savepath}, with name: {filename}", flush=True)
    os.makedirs(f"{savepath}/temp", exist_ok=True)

    
    main(filepath, savepath, filename, nchunks = 100)
