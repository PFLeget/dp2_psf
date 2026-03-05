
visitids=$(paste -sd, ../../visitIds.txt)
ccdids=$(paste -sd, ../../ccdIds.txt)

bps submit /sdf/group/rubin/user/leget/batch/bps_generic.yaml \
    -b /repo/main \
    -i LSSTCam/defaults \
    -o u/leget/LSSTCam/analysis-dmtn-328/BFCoulton_w39_10152025 \
    -p ${DRP_PIPE_DIR}/pipelines/LSSTCam/DRP.yaml#isr,calibrateImage \
    --extra-qgraph-options "--config-file isr:isrLSST.py" \
    -d "instrument='LSSTCam' AND skymap='lsst_cells_v1' AND detector IN (${ccdids}) AND exposure IN (${visitids}) AND visit IN (${visitids})" \
