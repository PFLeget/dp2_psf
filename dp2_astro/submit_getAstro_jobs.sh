#!/bin/bash
#
# SLURM job submission script for getAstro_single.py
# Processes astrometric residuals per healpix region
# Batches multiple healpix per job to reduce overhead
#

# Configuration
SCRIPT_DIR="/sdf/home/l/leget/rubin-user/lsst_dev/tickets/dp2_psf/dp2_astro"
SCRIPT_NAME="getAstro_single.py"
OUTPUT_DIR="/sdf/scratch/users/c/csaunder/DP2_fp_resids"
LOG_DIR="${SCRIPT_DIR}/logs"
PHYSICAL_FILTER="i_39"

# Number of healpix per job
HEALPIX_PER_JOB=10

# Create directories if they don't exist
mkdir -p "${LOG_DIR}"
mkdir -p "${OUTPUT_DIR}"

# Get list of healpix to process (excluding already processed ones)
echo "Querying healpix regions to process..."
HEALPIX_LIST_FILE=$(mktemp)

python3 << EOF > "${HEALPIX_LIST_FILE}"
from lsst.daf.butler import Butler
import os

butler = Butler('dp2_prep', collections='LSSTCam/runs/DRP/DP2/v30_0_0/DM-53881/stage2')
fitStarsRefs = butler.query_datasets('gbdesHealpix3AstrometricFit_fitStars', physical_filter='${PHYSICAL_FILTER}')

for ref in fitStarsRefs:
    healpix = ref.dataId['healpix3']
    output_file = f'${OUTPUT_DIR}/healpix_{healpix}.parq'
    if not os.path.exists(output_file):
        print(healpix)
EOF

# Read healpix into array
mapfile -t ALL_HEALPIX < "${HEALPIX_LIST_FILE}"
rm "${HEALPIX_LIST_FILE}"

TOTAL_HEALPIX=${#ALL_HEALPIX[@]}
echo "Total healpix to process: ${TOTAL_HEALPIX}"
echo "Healpix per job: ${HEALPIX_PER_JOB}"

if [ ${TOTAL_HEALPIX} -eq 0 ]; then
    echo "No healpix to process. All done!"
    exit 0
fi

# Calculate number of jobs needed
NUM_JOBS=$(( (TOTAL_HEALPIX + HEALPIX_PER_JOB - 1) / HEALPIX_PER_JOB ))
echo "Number of jobs to submit: ${NUM_JOBS}"
echo "=================================================="

# Counter for submitted jobs
job_count=0

for ((job_idx=0; job_idx<NUM_JOBS; job_idx++)); do
    # Calculate start and end indices for this batch
    START_IDX=$((job_idx * HEALPIX_PER_JOB))
    END_IDX=$((START_IDX + HEALPIX_PER_JOB))
    if [ ${END_IDX} -gt ${TOTAL_HEALPIX} ]; then
        END_IDX=${TOTAL_HEALPIX}
    fi

    # Get healpix for this batch
    BATCH_HEALPIX="${ALL_HEALPIX[@]:${START_IDX}:${HEALPIX_PER_JOB}}"

    JOB_NAME="astro_hp_${job_idx}"

    # Submit the job
    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --account=rubin:developers
#SBATCH --partition=torino
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=${LOG_DIR}/${JOB_NAME}_%j.out
#SBATCH --error=${LOG_DIR}/${JOB_NAME}_%j.err

# Load LSST environment
source /sdf/group/rubin/sw/d_latest/loadLSST.bash
setup lsst_distrib -t d_latest

# Change to script directory
cd ${SCRIPT_DIR}

echo "Starting job: ${JOB_NAME}"
echo "Processing healpix batch ${job_idx}/${NUM_JOBS}"
echo "Healpix IDs: ${BATCH_HEALPIX}"
echo "Time: \$(date)"
echo "Node: \$(hostname)"
echo "=================================================="

python ${SCRIPT_NAME} \\
    --healpix ${BATCH_HEALPIX} \\
    --output_dir "${OUTPUT_DIR}" \\
    --physical_filter "${PHYSICAL_FILTER}"

echo "=================================================="
echo "Job completed at: \$(date)"
EOF

    job_count=$((job_count + 1))
    echo "Submitted: ${JOB_NAME} (healpix ${START_IDX}-$((END_IDX - 1)))"

done

echo "=================================================="
echo "Total jobs submitted: ${job_count}"
echo "Log files will be in: ${LOG_DIR}"
echo "Output files will be in: ${OUTPUT_DIR}"
