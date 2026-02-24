#!/bin/bash
#
# SLURM job submission script for SingleVisit_FocusGradient.py
# Processes visits in batches of 100 per job
#

# Configuration
SCRIPT_DIR="/sdf/home/l/leget/rubin-user/lsst_dev/tickets/dp2_psf"
SCRIPT_NAME="SingleVisit_FocusGradient.py"
VISIT_MAPPING_FILE="${SCRIPT_DIR}/data/visit_parquet_mapping.pkl"
ZERNIKE_CORNERS_FILE="${SCRIPT_DIR}/data/visit_zernike_corners.pkl"
HEIGHT_MAP_FILE="${SCRIPT_DIR}/data/LSST_FP_cold_b_measurement_4col_bysurface.fits"
LOG_DIR="${SCRIPT_DIR}/logs"
REP_OUT_PLOT="${SCRIPT_DIR}/plots/focus_gradient"

# Create output directories if they don't exist
mkdir -p "${LOG_DIR}"
mkdir -p "${REP_OUT_PLOT}"

# Number of visits per job
VISITS_PER_JOB=100

# Extract visit IDs from the mapping file
# This creates a temporary file with all visit IDs
VISIT_LIST_FILE=$(mktemp)
python3 << EOF > "${VISIT_LIST_FILE}"
import pickle
with open("${VISIT_MAPPING_FILE}", 'rb') as f:
    visit_mapping = pickle.load(f)
for visit in sorted(visit_mapping.keys()):
    print(visit)
EOF

# Read visits into array
mapfile -t ALL_VISITS < "${VISIT_LIST_FILE}"
rm "${VISIT_LIST_FILE}"

TOTAL_VISITS=${#ALL_VISITS[@]}
echo "Total visits: ${TOTAL_VISITS}"
echo "Visits per job: ${VISITS_PER_JOB}"

# Calculate number of jobs needed
NUM_JOBS=$(( (TOTAL_VISITS + VISITS_PER_JOB - 1) / VISITS_PER_JOB ))
echo "Number of jobs to submit: ${NUM_JOBS}"
echo "=================================================="

# Counter for submitted jobs
job_count=0

for ((job_idx=0; job_idx<NUM_JOBS; job_idx++)); do
    # Calculate start and end indices for this batch
    START_IDX=$((job_idx * VISITS_PER_JOB))
    END_IDX=$((START_IDX + VISITS_PER_JOB))
    if [ ${END_IDX} -gt ${TOTAL_VISITS} ]; then
        END_IDX=${TOTAL_VISITS}
    fi

    # Get visits for this batch
    BATCH_VISITS="${ALL_VISITS[@]:${START_IDX}:${VISITS_PER_JOB}}"

    JOB_NAME="focus_grad_${job_idx}"

    # Submit the job
    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --account=rubin:developers
#SBATCH --partition=torino
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=${LOG_DIR}/${JOB_NAME}_%j.out
#SBATCH --error=${LOG_DIR}/${JOB_NAME}_%j.err

# Load LSST environment
source /sdf/group/rubin/sw/d_latest/loadLSST.bash
setup lsst_distrib -t d_latest

# Change to script directory
cd ${SCRIPT_DIR}

echo "Starting job: ${JOB_NAME}"
echo "Processing visits ${START_IDX} to $((END_IDX - 1)) (batch ${job_idx}/${NUM_JOBS})"
echo "Time: \$(date)"
echo "Node: \$(hostname)"
echo "=================================================="

# Process each visit in this batch
VISITS=(${BATCH_VISITS})
for visit in \${VISITS[@]}; do
    echo "Processing visit: \${visit}"
    python ${SCRIPT_NAME} \\
        --visit \${visit} \\
        --visitMappingFile "${VISIT_MAPPING_FILE}" \\
        --fitHeightMap "${HEIGHT_MAP_FILE}" \\
        --zernikeCornersFile "${ZERNIKE_CORNERS_FILE}" \\
        --repOutPlot "${REP_OUT_PLOT}" \\
        || echo "Failed: \${visit}"
done

echo "=================================================="
echo "Job completed at: \$(date)"
EOF

    job_count=$((job_count + 1))
    echo "Submitted: ${JOB_NAME} (visits ${START_IDX}-$((END_IDX - 1)))"

done

echo "=================================================="
echo "Total jobs submitted: ${job_count}"
echo "Log files will be in: ${LOG_DIR}"
echo "Plots will be in: ${REP_OUT_PLOT}"
