import os, sys
import numpy as np
import h5py

import data_utils as du
import lsstcam_utils as lu
import psf_utils as pu

import single_visit_catalog as svc

import argparse
from multiprocessing import Pool

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

def process_chunk(dataset_refs, consdb, butler, savepath, filename, chunk_num):
    print(f'chunk {chunk_num} size {len(dataset_refs)}', flush=True)
    ## make PSF catalog
    psf_catalog = svc.make_single_visit_catalog(dataset_refs = dataset_refs,
                                   consdb = consdb,
                                   butler = butler)

    comments = [f'collection: {collection}']
    ## write PSF catalog
    svc.write_psf_catalog(psf_catalog = psf_catalog,
                         filename = f"single_visit_chunk_{chunk_num:03d}.h5", 
                         comments = comments,
                         SAVE_DIR = f"{savepath}/temp/"
                         )

def compile_chunks(savepath, filename, nchunks):
    with h5py.File(f'{savepath}/temp/single_visit_chunk_000.h5', 'r') as f:
        columns = list(f.keys())

    combined_catalog = {col : [] for col in columns}
    for i in range(nchunks):
        with h5py.File(f'{savepath}/temp/single_visit_chunk_{i:03d}.h5', 'r') as f:
            for col in combined_catalog.keys():
                combined_catalog[col].append(f[col][:])
    
    for col in combined_catalog.keys():
        combined_catalog[col] = np.concatenate(combined_catalog[col])
    
    
    with h5py.File(f'{savepath}/{filename}', "w") as catalog:
        for col in combined_catalog.keys():
                catalog[col] = combined_catalog[col]

    print(f"Combined catalog saved to {savepath}/{filename}", flush=True)


    
def main(username, collection, savepath, filename, nchunks):

    ## initialize butler
    butler_dict = dict(repo = "dp2_prep", #/repo/main
                       collections = [collection],
                       instrument = "LSSTCam"
                      )
    
    butler = du.initialize_butler(butler_dict=butler_dict)
    print('butler initialized', flush=True)
    ## get dataset refs to the Piff PSF catalog
    dataset_refs = du.get_dataset_refs(butler, data_product='refit_psf_star')
    print(f'dataset refs size {len(dataset_refs)}', flush=True)
    ## read token for ConsDB
    with open("../token.txt", "r") as file:
        token = file.readline()
    print('token read', flush=True)
    ## get exposure catalog from ConsDB
    consdb = du.get_exposure_catalog(username = username, token = token)
    print('consdb queried', f'{len(consdb)} rows', flush=True)


    chunks = np.array_split(dataset_refs, nchunks)
    # create argument tuples for starmap
    args = [(chunk, consdb, butler, savepath, filename, i) for i, chunk in enumerate(chunks)]
    
    with Pool(processes=nchunks) as pool:
        for i, _ in enumerate(pool.starmap(process_chunk, args), 1):
            print(f"Processed chunk {i}/{nchunks}", flush=True)

    compile_chunks(savepath, filename, nchunks)

    

if __name__ == '__main__':
    ## change these if necessary ##
    username = 'pai'

    
    parser = argparse.ArgumentParser(description="Generate single visit PSF catalog.")
    parser.add_argument(
        "--collection", "-c", required=True,
        help="Input collection name (e.g. u/pai/my_collection)"
    )
    parser.add_argument(
        "--savepath", "-o", required=True,
        help="Output file path for results (e.g. u/pai/rubin-user/catalogs/)"
    )
    parser.add_argument(
        "--filename", "-f", required=True,
        help="Output file name for results (e.g. DRP_catalog.h5)"
    )
    args = parser.parse_args()

    # Access the arguments
    collection = args.collection
    savepath = args.savepath
    filename = args.filename
    
    print(f"Using collection: {collection}", flush=True)
    print(f"Saving results to path: {savepath}, with name: {filename}", flush=True)
    os.makedirs(f"{savepath}/temp", exist_ok=True)
    
    #collection = "u/pai/PiffColor2025_10_17"
    #"LSSTCam/runs/DRP/20250604_20250921/w_2025_39/DM-52645"
    #"LSSTCam/runs/DRP/20250604_20250814/w_2025_33/DM-52202"
    #"LSSTCam/runs/DRP/20250501_20250609/w_2025_26/DM-51580"
    #"LSSTCam/runs/DRP/20250501_20250604/w_2025_23/DM-51284"
    #'LSSTCam/runs/DRP/20250420_20250521/w_2025_21/DM-51076'
    print('starting production', flush=True)
    main(username = username, collection = collection, savepath = savepath, filename = filename, nchunks = 100)
                         
    