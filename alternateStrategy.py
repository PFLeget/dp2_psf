from lsst.daf.butler import Butler
import numpy as np
import os
os.environ["POLARS_MAX_THREADS"] = "1"
import polars


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


visit = 2025050400608
selected_columns = [
        'slot_Shape_xx', 'slot_Shape_yy', 'slot_Shape_xy',
        'slot_PsfShape_xx', 'slot_PsfShape_xy', 'slot_PsfShape_yy',
        'coord_ra', 'coord_dec', 'slot_Centroid_x', 'slot_Centroid_y',
        'detector', 'psf_max_value', 'calib_psf_reserved',
    ]

# OPTION 1: SLOW
table = butler.get("refit_psf_star", instrument="LSSTCam", visit=visit, parameters={"columns": selected_columns})

# OPTION 2: FAST 
uri = butler.getURI("refit_psf_star", instrument="LSSTCam", visit=visit)
File = uri.geturl()
table = polars.scan_parquet(File).select(selected_columns).collect()