#!/bin/bash

ccdids=$(paste -sd, ccdIds.txt)
visitids=$(paste -sd, visitIds.txt)

bps submit /sdf/group/rubin/user/leget/batch/bps_generic_embargo.yaml \
    -b embargo \
    -i LSSTCam/defaults \
    -o u/leget/LSSTCam/HeightMapCorrelation20260311 \
    -p ${DRP_PIPE_DIR}/pipelines/LSSTCam/DRP.yaml#isr,calibrateImage \
    -d "instrument='LSSTCam' AND detector IN (${ccdids}) AND exposure IN (${visitids}) AND visit IN (${visitids})"
