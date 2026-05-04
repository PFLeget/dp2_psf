#!/bin/bash
#
# Replot SkyPlot_vs_secondMoment from pkl files with custom color scales
#

SCRIPT_DIR="/sdf/home/l/leget/rubin-user/lsst_dev/tickets/dp2_psf"
PKL_DIR="${SCRIPT_DIR}/plots"
VISIT_MAPPING_FILE="${SCRIPT_DIR}/data/visit_parquet_mapping_skycoord.pkl"
COADD_DETECTOR_FILE="/sdf/home/l/leget/rubin-user/lsst_dev/tickets/dp2_psf/data/coadd_detector_mapping.pkl"

# Color scales - adjust these as needed
COLOR_SCALE_de1=0.001
COLOR_SCALE_de2=0.001
COLOR_SCALE_dT_T=0.005

# Bands and moments to process
BANDS=("u" "g" "r" "i" "z" "y" "ugrizy")
MOMENTS=("de1" "de2" "dT_T")

echo "Replotting SkyPlot_vs_secondMoment with custom color scales..."
echo "  de1: +/- ${COLOR_SCALE_de1}"
echo "  de2: +/- ${COLOR_SCALE_de2}"
echo "  dT_T: +/- ${COLOR_SCALE_dT_T}"
echo "=================================================="

for band in "${BANDS[@]}"; do
    for moment in "${MOMENTS[@]}"; do
        # Build pkl filename
        pkl_file="${PKL_DIR}/${moment}_sky_${band}_3600_0_coaddOnly.pkl"

        if [[ ! -f "$pkl_file" ]]; then
            echo "Skipping (not found): ${pkl_file}"
            continue
        fi

        # Get color scale for this moment
        color_scale_var="COLOR_SCALE_${moment}"
        COLOR_SCALE=${!color_scale_var}

        echo "Replotting: ${moment}_sky_${band} (colorScale=${COLOR_SCALE})"

        python ${SCRIPT_DIR}/SkyPlot_vs_secondMoment.py \
            --bands ${band} \
            --visitMappingFile "${VISIT_MAPPING_FILE}" \
            --pklInput "$pkl_file" \
            --key_second_moment ${moment} \
            --colorScale ${COLOR_SCALE} \
            --repOutPlot "${PKL_DIR}" \
            --bin_spacing 3600 \
            --coaddDetectorFile "${COADD_DETECTOR_FILE}" \
            --exclude_crowded \
            --galactic_b_min 20
    done
done

echo "=================================================="
echo "Done!"
