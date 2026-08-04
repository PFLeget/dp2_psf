#!/bin/bash
#
# SLURM job submission script for dcr_psf_residuals.py
# Reproduces the SITCOMTN-174 PSF residual batch (Figs 2-6) from finalized_src_table.
# One job per band; submits to S3DF torino partition.
#
# Usage:
#   ./submit_dcr_psf_residuals_jobs.sh <collection>
#   ./submit_dcr_psf_residuals_jobs.sh <collection> --frame radec
#

# Configuration
SCRIPT_DIR="/sdf/home/l/leget/rubin-user/lsst_dev/tickets/dp2_psf"
SCRIPT_NAME="dcr_psf_residuals.py"
LOG_DIR="${SCRIPT_DIR}/logs"
REP_OUT_PLOT="${SCRIPT_DIR}/plots"
REPO="dp2_prep"

# visit->parquet mapping (getURI on refit_psf_star) whose parquet carry the
# alt/az moment + fgcm_mag columns. Build/point this at your alt/az re-run.
VISIT_MAPPING_FILE="${SCRIPT_DIR}/data/visit_parquet_mapping.pkl"

# Default collection exposing visit_table (used only for DCR inputs; override as first positional arg)
COLLECTION="LSSTCam/runs/DRP/DP2/v30_0_0/DM-53881/stage2"

# First non-flag argument overrides the collection
EXTRA_ARGS=""
for arg in "$@"; do
    case $arg in
        --*)
            EXTRA_ARGS="${EXTRA_ARGS} ${arg}"
            ;;
        *)
            COLLECTION="${arg}"
            ;;
    esac
done

mkdir -p "${LOG_DIR}"

BANDS=("u" "g" "r" "i" "z" "y")

job_count=0
echo "Submitting dcr_psf_residuals.py jobs to S3DF..."
echo "Collection: ${COLLECTION}"
echo "=================================================="

for band in "${BANDS[@]}"; do

    JOB_NAME="dcr_res_${band}"

    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --account=rubin:developers
#SBATCH --partition=torino
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=${LOG_DIR}/${JOB_NAME}_%j.out
#SBATCH --error=${LOG_DIR}/${JOB_NAME}_%j.err

# Load LSST environment
source /sdf/group/rubin/sw/d_latest/loadLSST.bash
setup lsst_distrib -t d_latest

cd ${SCRIPT_DIR}

echo "Starting job: ${JOB_NAME}"
echo "Band: ${band}, Collection: ${COLLECTION}"
echo "Time: \$(date)"
echo "Node: \$(hostname)"
echo "=================================================="

python ${SCRIPT_NAME} \\
    --repo ${REPO} \\
    --collection "${COLLECTION}" \\
    --visitMappingFile "${VISIT_MAPPING_FILE}" \\
    --bands ${band} \\
    --repOutPlot "${REP_OUT_PLOT}"${EXTRA_ARGS}

echo "=================================================="
echo "Job completed at: \$(date)"
EOF

    job_count=$((job_count + 1))
    echo "Submitted: ${JOB_NAME}"

done

echo "=================================================="
echo "Total jobs submitted: ${job_count}"
echo "Log files will be in: ${LOG_DIR}"
