import h5py
import psf_utils as pu
import DCR

import argparse
import numpy as np


def rotate_moments(filepath):

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

    model_sky_ixx, model_sky_iyy, model_sky_ixy = pu.rotateXYtoNW(model_ixx, model_iyy, model_ixy, np.deg2rad(sky_rotation))
    model_azel_ixx, model_azel_iyy, model_azel_ixy = pu.rotateNWtoAA(model_sky_ixx, model_sky_iyy, model_sky_ixy, parallactic_angle)

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

    

    

def write_file(savepath, moments_dict):

    cat = h5py.File(savepath, "w")

    
    for colname, data in moments_dict.items():
        #print(colname, flush=True)
        cat[colname] = data
    
    cat.close()


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
    args = parser.parse_args()

    # Access the arguments
    filepath = args.filepath
    savepath = args.savepath

    print(f"Using catalog: {filepath}", flush=True)
    print(f"Saving results to path: {savepath}", flush=True)

    print('Processing catalog', flush=True)
    moments_dict = rotate_moments(filepath)
    print("Writing file", flush=True)
    write_file(savepath, moments_dict)
