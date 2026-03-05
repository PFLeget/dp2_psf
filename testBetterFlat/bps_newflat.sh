#!/bin/bash
# A/B Test: New flat from tguillem
# detector 87, g-band, |b| > 30, after Nov 2025

visitids=$(paste -sd, visitIds_flat_test.txt)

bps submit /sdf/group/rubin/user/leget/batch/bps_generic.yaml \
    -b /repo/main \
    -i u/tguillem/test_LED_weights/flat_g_sky_w0381,LSSTCam/defaults \
    -o u/leget/LSSTCam/testFlat/ABtest_newflat \
    -p ${DRP_PIPE_DIR}/pipelines/LSSTCam/DRP.yaml#isr,calibrateImage \
    -d "instrument='LSSTCam' AND skymap='lsst_cells_v2' AND detector=87 AND exposure IN (${visitids}) AND visit IN (${visitids})"
