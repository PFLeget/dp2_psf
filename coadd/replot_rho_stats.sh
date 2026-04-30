#!/bin/bash
#
# Replot rho statistics from pkl files with custom y-axis limits
#

SCRIPT_DIR="/sdf/home/l/leget/rubin-user/lsst_dev/tickets/dp2_psf/coadd"
PKL_DIR="${SCRIPT_DIR}/rho_stats"
DES_FILE="/sdf/home/l/leget/rubin-user/lsst_dev/tickets/dp2_psf/all_xip-xim+errs_dict_v2.pkl"

# Bands to process
BANDS=("r")  # Add more: ("u" "g" "r" "i" "z" "y")

# Modes and ellipticity types
MODES=("coadd" "single_visit")
ELLIPTICITY_TYPES=("distortion" "shear")

# Y-axis limits per rho stat - adjust these as needed
YLIM_RHO1=1e-5
YLIM_RHO2=1e-5
YLIM_RHO3=1e-7
YLIM_RHO4=1e-6
YLIM_RHO5=1e-6
YLIM_RHO3ALT_MIN=0
YLIM_RHO3ALT_MAX=2e-5

echo "Replotting rho statistics with custom y-axis limits..."
echo "  rho1: +/- ${YLIM_RHO1}"
echo "  rho2: +/- ${YLIM_RHO2}"
echo "  rho3: +/- ${YLIM_RHO3}"
echo "  rho4: +/- ${YLIM_RHO4}"
echo "  rho5: +/- ${YLIM_RHO5}"
echo "  rho3alt: [${YLIM_RHO3ALT_MIN}, ${YLIM_RHO3ALT_MAX}]"
echo "=================================================="

for band in "${BANDS[@]}"; do
    for mode in "${MODES[@]}"; do
        for etype in "${ELLIPTICITY_TYPES[@]}"; do
            echo ""
            echo "Processing: band=${band}, mode=${mode}, ellipticity=${etype}"
            echo "--------------------------------------------------"

            # Build pkl filename based on mode
            if [[ "$mode" == "coadd" ]]; then
                pkl_file="${PKL_DIR}/rho_stats_coadd_${band}_${etype}.pkl"
            else
                pkl_file="${PKL_DIR}/rho_stats_single_visit_${band}_${etype}_coaddOnly.pkl"
            fi

            if [[ -f "$pkl_file" ]]; then
                echo "  Replotting: $pkl_file"
                python ${SCRIPT_DIR}/comp_rho_stat.py --mode replot \
                    --pklInput "$pkl_file" \
                    --ylim_rho1 ${YLIM_RHO1} \
                    --ylim_rho2 ${YLIM_RHO2} \
                    --ylim_rho3 ${YLIM_RHO3} \
                    --ylim_rho4 ${YLIM_RHO4} \
                    --ylim_rho5 ${YLIM_RHO5} \
                    --ylim_rho3alt ${YLIM_RHO3ALT_MIN} ${YLIM_RHO3ALT_MAX} \
                    --desFile ${DES_FILE}
            else
                echo "  Skipping (not found): $pkl_file"
            fi

        done
    done
done

echo "=================================================="
echo "Done!"
