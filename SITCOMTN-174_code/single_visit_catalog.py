import os, sys
import numpy as np
from astropy.table import vstack, QTable
import h5py
import astropy.units as u

import lsstcam_utils as lu
import data_utils as du
import psf_utils as pu
import DCR

COLNAMES = ['source_id', 'ra', 'dec', 'sky_rotation', 'mjd', 'slot_px_x', 'slot_px_y', 'fp_x', 'fp_y', 'visit', 'date',
            'detector', 'det_is_itl', 'det_is_e2v', 'band', 'psf_used', 'psf_reserved', 'e1', 'e2', 'T', 'ixx', 'iyy', 'ixy',
            'model_e1', 'model_e2', 'model_T', 'model_ixx', 'model_iyy', 'model_ixy', 'calib_instflux', 'calib_instflux_err', 'calib_flux',
            'calib_flux_err', 'calib_mag', 'calib_mag_err', 'seeing', 'airmass',
            'humidity', 'pressure', 'air_temp', 'wind_speed', 'wind_dir', 'obs_reason']

DTYPES = ['i8', 'f8', 'f8', 'f8', 'f8', 'f4', 'f4', 'f4', 'f4', 'i8', 'i4', 
          'i4', 'i4', 'i4', 'S1', 'i4', 'i4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 
          'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4',
          'f4', 'f4', 'f4', 'f4', 'f4', 
          'f4', 'i4', 'f4', 'f4', 'f4', 'S32']

UNITS = [None, u.deg, u.deg, u.deg, None, u.pixel, u.pixel, u.mm, u.mm, None, None, 
         None, None, None, None, None, None, None, None, u.pixel**2, u.pixel**2, u.pixel**2, u.pixel**2, 
         None, None, u.pixel**2, u.pixel**2, u.pixel**2, u.pixel**2, u.ct, u.ct, u.nJy,
         u.nJy, u.ABmag, u.ABmag, None, None,
         None, None, None, None, None, None]

def make_single_visit_catalog(dataset_refs, consdb, butler):
    
    psf_catalog = QTable()
    
    for i, ref in enumerate(dataset_refs):
        # if i % 10 == 0:
        #     print(i, flush=True)
        expid = ref.dataId['visit']
        # detid = ref.dataId['detector']
    
        try:
            source_table = butler.get("refit_psf_star", dataId={"visit":expid})
        
            source_table = pu.filter_source_table(source_table)
            e1, e2, T = pu.moment2ellipticity(source_table['slot_Shape_xx'], 
                                              source_table['slot_Shape_yy'], 
                                              source_table['slot_Shape_xy'])
    
            ixx, iyy, ixy = source_table['slot_Shape_xx'], source_table['slot_Shape_yy'], source_table['slot_Shape_xy']
            model_e1, model_e2, model_T = pu.moment2ellipticity(source_table['slot_PsfShape_xx'], 
                                                                source_table['slot_PsfShape_yy'],
                                                                source_table['slot_PsfShape_xy'])
    
            model_ixx, model_iyy, model_ixy = source_table['slot_PsfShape_xx'], source_table['slot_PsfShape_yy'], source_table['slot_PsfShape_xy']
            
            instflux = source_table['slot_CalibFlux_instFlux']
            instflux_err = source_table['slot_CalibFlux_instFluxErr']
            calibflux = source_table['slot_CalibFlux_flux']
            calibflux_err = source_table['slot_CalibFlux_fluxErr']
            calibmag = source_table['slot_CalibFlux_mag']
            calibmag_err = source_table['slot_CalibFlux_magErr']
            #ConsDB data 
            index = np.where(consdb['exposure_id'] == expid)
    
            #zeropoint = np.shape(source_table)[0]*[consdb['zero_point_median'][index][0]]
            airmass = np.shape(source_table)[0]*[consdb['airmass'][index][0]]
            mjd = np.shape(source_table)[0]*[consdb['obs_start_mjd'][index][0]]
            seeing = np.shape(source_table)[0]*[consdb['dimm_seeing'][index][0]]
            humidity = np.shape(source_table)[0]*[consdb['humidity'][index][0]]
            pressure = np.shape(source_table)[0]*[consdb['pressure'][index][0]]
            air_temp = np.shape(source_table)[0]*[consdb['air_temp'][index][0]]
            wind_speed = np.shape(source_table)[0]*[consdb['wind_speed'][index][0]]
            wind_dir = np.shape(source_table)[0]*[consdb['wind_dir'][index][0]]
            obs_reason = np.shape(source_table)[0]*[consdb['observation_reason'][index][0]]
            sky_rotation = np.shape(source_table)[0]*[consdb['sky_rotation'][index][0]]
            #make visit/source/detector ID arrays for catalog
            exposure = np.shape(source_table)[0]*[expid] #np.size(source_table)*[expid]
            detector = source_table['detector'] #np.size(source_table)*[detid]
            det_type_itl, det_type_e2v = lu.detector_type(detector)
            
            source_id = source_table['id'] 
            date = lu.visit_to_date(np.array(np.shape(source_table)[0]*[expid]))
            
            #make focal plane RA, dec, x, y arrays for catalog
            ra = np.rad2deg(source_table['coord_ra'])
            dec = np.rad2deg(source_table['coord_dec'])
            pixel_x = source_table['slot_Centroid_x']
            pixel_y = source_table['slot_Centroid_y']
        
            fp_x, fp_y = [], []
            for i, (x, y, d) in enumerate(zip(pixel_x, pixel_y, detector)):
                fpx, fpy = lu.pixel_to_focal(x, y, d)
                fp_x.append(fpx)
                fp_y.append(fpy)
    
      
            #PSF used object or reserved object
            used = source_table['calib_psf_used']
            reserved = source_table['calib_psf_reserved']
        
            #exposure band
            band = np.shape(source_table)[0]*[consdb['band'][index][0]]
            
            #save data for each exposure ID into a temporary table and 
            #concatenate vertically to the final PSF catalog
            columns = [source_id, ra, dec, sky_rotation, mjd, pixel_x, pixel_y, fp_x, fp_y, exposure, date, 
                          detector, det_type_itl, det_type_e2v, band, used, reserved, e1, e2, T, ixx, iyy, ixy, 
                          model_e1, model_e2, model_T, model_ixx, model_iyy, model_ixy, instflux, instflux_err, calibflux, 
                       calibflux_err, calibmag, calibmag_err, seeing, airmass, 
                          humidity, pressure, air_temp, wind_speed, wind_dir, obs_reason]
            
            tab = QTable(data=columns,
                        names=COLNAMES,
                        dtype=DTYPES,
                        units=UNITS)
        
            psf_catalog = vstack([psf_catalog, tab])
    
        except Exception as e:
            print(f"Warning: Failed to process - {e}", flush=True)
            continue

    return psf_catalog

def write_psf_catalog(psf_catalog, filename, comments, SAVE_DIR):

    path = os.path.join(SAVE_DIR, filename)
    print(path)
    cat = h5py.File(path, "w")

    #data = cat.create_group("data")
    #comments = cat.create_dataset("comments", shape=3, dtype=h5py.string_dtype())
    cat['comments'] = comments

    colnames = psf_catalog.colnames
    
    for colname in list(colnames):
        print(colname, flush=True)
        cat[colname] = psf_catalog[colname]
    
    cat.close()