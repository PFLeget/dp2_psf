#!/bin/bash
#
# Replot SkyPlot_vs_secondMoment from pkl files with custom color scales
#

SCRIPT_DIR="/sdf/home/l/leget/rubin-user/lsst_dev/tickets/dp2_psf"
PKL_DIR="${SCRIPT_DIR}/plots"
VISIT_MAPPING_FILE="${SCRIPT_DIR}/data/visit_parquet_mapping_skycoord.pkl"

# Color scales - adjust these as needed
COLOR_SCALE_ELLIPTICITY=0.01   # for de1, de2
COLOR_SCALE_SIZE=0.005         # for dT_T

echo "Replotting SkyPlot_vs_secondMoment with custom color scales..."
echo "  Ellipticity (de1, de2): +/- ${COLOR_SCALE_ELLIPTICITY}"
echo "  Size (dT_T): +/- ${COLOR_SCALE_SIZE}"
echo "=================================================="

for pkl_file in ${PKL_DIR}/*.pkl; do
    if [[ ! -f "$pkl_file" ]]; then
        continue
    fi

    filename=$(basename "$pkl_file")

    # Skip coadd files
    if [[ "$filename" == coadd_* ]]; then
        continue
    fi

    # Extract band from filename (e.g., dT_T_sky_r_3600_0.pkl -> r)
    # Pattern: {moment}_sky_{band}_{binspacing}_{psfmax}.pkl
    band=$(echo "$filename" | sed -n 's/.*_sky_\([a-zA-Z]*\)_.*/\1/p')

    # Extract moment from filename (everything before _sky_)
    moment=$(echo "$filename" | sed -n 's/\(.*\)_sky_.*/\1/p')

    # Determine color scale based on moment type
    if [[ "$moment" == "de1" ]] || [[ "$moment" == "de2" ]]; then
        COLOR_SCALE=${COLOR_SCALE_ELLIPTICITY}
    elif [[ "$moment" == "dT_T" ]]; then
        COLOR_SCALE=${COLOR_SCALE_SIZE}
    else
        echo "Skipping unknown moment type: $filename (moment=$moment)"
        continue
    fi

    echo "Replotting: $filename (moment=${moment}, band=${band}, colorScale=${COLOR_SCALE})"

    python ${SCRIPT_DIR}/SkyPlot_vs_secondMoment.py \
        --bands ${band} \
        --visitMappingFile "${VISIT_MAPPING_FILE}" \
        --pklInput "$pkl_file" \
        --key_second_moment ${moment} \
        --colorScale ${COLOR_SCALE} \
        --repOutPlot "${PKL_DIR}"

done

echo "=================================================="
echo "Done!"
