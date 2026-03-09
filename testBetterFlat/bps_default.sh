#!/bin/bash
# A/B Test: Default flat (control)
# detector 87, g-band visits from DP2

visitids=$(paste -sd, visitIds_flat_test.txt)

bps submit /sdf/group/rubin/user/leget/batch/bps_generic_main.yaml \
    -b /repo/main \
    -i LSSTCam/defaults \
    -o u/leget/LSSTCam/testFlat/ABtest_default \
    -p ${DRP_PIPE_DIR}/pipelines/LSSTCam/DRP.yaml#isr,calibrateImage \
    -d "instrument='LSSTCam' AND skymap='lsst_cells_v2' AND detector=87 AND exposure IN (${visitids}) AND visit IN (${visitids})"
