#!/bin/bash
#
# Re-plot coadd sky maps with custom color scales
# Edit the COLOR_SCALE values below to adjust
#

# Configuration
SCRIPT_DIR="/sdf/home/l/leget/rubin-user/lsst_dev/tickets/dp2_psf/coadd"
SCRIPT_NAME="skyPlot_coadd_secondMoment.py"
TRACT_MAPPING_FILE="/sdf/home/l/leget/rubin-user/lsst_dev/tickets/dp2_psf/data/tract_parquet_mapping.pkl"
PKL_DIR="${SCRIPT_DIR}/plots"
REP_OUT_PLOT="${SCRIPT_DIR}/plots_rescaled"

# Band to replot
BAND="r"

# ============================================
# EDIT COLOR SCALES HERE (symmetric: -val to +val)
# ============================================
COLOR_SCALE_T=0.5          # T (pixel^2)
COLOR_SCALE_e1=0.1        # e1
COLOR_SCALE_e2=0.1        # e2
COLOR_SCALE_dT_T=0.005      # dT/T
COLOR_SCALE_de1=0.001       # de1
COLOR_SCALE_de2=0.001       # de2
# ============================================

# Create output directory
mkdir -p "${REP_OUT_PLOT}"

# Load LSST environment (comment out if already loaded)
# source /sdf/group/rubin/sw/d_latest/loadLSST.bash
# setup lsst_distrib -t d_latest

cd ${SCRIPT_DIR}

echo "Re-plotting coadd sky maps for band: ${BAND}"
echo "Output directory: ${REP_OUT_PLOT}"
echo "=================================================="

# T
echo "Plotting T with colorScale=${COLOR_SCALE_T}"
python ${SCRIPT_NAME} \
    --band ${BAND} \
    --tractMappingFile "${TRACT_MAPPING_FILE}" \
    --key_second_moment T \
    --pklInput "${PKL_DIR}/coadd_T_sky_${BAND}_3600.pkl" \
    --colorScale ${COLOR_SCALE_T} \
    --repOutPlot "${REP_OUT_PLOT}"

# e1
echo "Plotting e1 with colorScale=${COLOR_SCALE_e1}"
python ${SCRIPT_NAME} \
    --band ${BAND} \
    --tractMappingFile "${TRACT_MAPPING_FILE}" \
    --key_second_moment e1 \
    --pklInput "${PKL_DIR}/coadd_e1_sky_${BAND}_3600.pkl" \
    --colorScale ${COLOR_SCALE_e1} \
    --repOutPlot "${REP_OUT_PLOT}"

# e2
echo "Plotting e2 with colorScale=${COLOR_SCALE_e2}"
python ${SCRIPT_NAME} \
    --band ${BAND} \
    --tractMappingFile "${TRACT_MAPPING_FILE}" \
    --key_second_moment e2 \
    --pklInput "${PKL_DIR}/coadd_e2_sky_${BAND}_3600.pkl" \
    --colorScale ${COLOR_SCALE_e2} \
    --repOutPlot "${REP_OUT_PLOT}"

# dT_T
echo "Plotting dT_T with colorScale=${COLOR_SCALE_dT_T}"
python ${SCRIPT_NAME} \
    --band ${BAND} \
    --tractMappingFile "${TRACT_MAPPING_FILE}" \
    --key_second_moment dT_T \
    --pklInput "${PKL_DIR}/coadd_dT_T_sky_${BAND}_3600.pkl" \
    --colorScale ${COLOR_SCALE_dT_T} \
    --repOutPlot "${REP_OUT_PLOT}"

# de1
echo "Plotting de1 with colorScale=${COLOR_SCALE_de1}"
python ${SCRIPT_NAME} \
    --band ${BAND} \
    --tractMappingFile "${TRACT_MAPPING_FILE}" \
    --key_second_moment de1 \
    --pklInput "${PKL_DIR}/coadd_de1_sky_${BAND}_3600.pkl" \
    --colorScale ${COLOR_SCALE_de1} \
    --repOutPlot "${REP_OUT_PLOT}"

# de2
echo "Plotting de2 with colorScale=${COLOR_SCALE_de2}"
python ${SCRIPT_NAME} \
    --band ${BAND} \
    --tractMappingFile "${TRACT_MAPPING_FILE}" \
    --key_second_moment de2 \
    --pklInput "${PKL_DIR}/coadd_de2_sky_${BAND}_3600.pkl" \
    --colorScale ${COLOR_SCALE_de2} \
    --repOutPlot "${REP_OUT_PLOT}"

echo "=================================================="
echo "Done! Plots saved to: ${REP_OUT_PLOT}"
