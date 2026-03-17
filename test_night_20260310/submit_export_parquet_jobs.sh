#!/bin/bash
#
# SLURM job submission script for exporting visits to parquet files
# Splits the 628 visits into parallel jobs
#

# Configuration
SCRIPT_DIR="/sdf/home/l/leget/rubin-user/lsst_dev/tickets/dp2_psf/test_night_20260310"
LOG_DIR="${SCRIPT_DIR}/logs"
VISIT_FILE="${SCRIPT_DIR}/visitIds.txt"
OUTPUT_DIR="${SCRIPT_DIR}/parquet_cache"

# Create directories
mkdir -p "${LOG_DIR}"
mkdir -p "${OUTPUT_DIR}"

# Get total number of visits
TOTAL_VISITS=$(wc -l < "${VISIT_FILE}")
echo "Total visits: ${TOTAL_VISITS}"

# Number of visits per job
VISITS_PER_JOB=2
NUM_JOBS=$(( (TOTAL_VISITS + VISITS_PER_JOB - 1) / VISITS_PER_JOB ))

echo "Submitting ${NUM_JOBS} jobs (${VISITS_PER_JOB} visits each)..."
echo "=================================================="

job_count=0

for ((i=0; i<NUM_JOBS; i++)); do
    START_IDX=$((i * VISITS_PER_JOB))
    END_IDX=$(( (i + 1) * VISITS_PER_JOB ))

    JOB_NAME="export_parquet_${i}"

    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --account=rubin:developers
#SBATCH --partition=torino
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=${LOG_DIR}/${JOB_NAME}_%j.out
#SBATCH --error=${LOG_DIR}/${JOB_NAME}_%j.err

# Load LSST environment
source /sdf/group/rubin/sw/d_latest/loadLSST.bash
setup lsst_distrib -t d_latest

cd ${SCRIPT_DIR}

echo "Starting job: ${JOB_NAME}"
echo "Visit range: ${START_IDX} to ${END_IDX}"
echo "Time: \$(date)"
echo "Node: \$(hostname)"
echo "=================================================="

python export_visits_to_parquet.py \\
    --visitIds "${VISIT_FILE}" \\
    --output_dir "${OUTPUT_DIR}" \\
    --start_idx ${START_IDX} \\
    --end_idx ${END_IDX}

echo "=================================================="
echo "Job completed at: \$(date)"
EOF

    job_count=$((job_count + 1))
    echo "Submitted: ${JOB_NAME} (visits ${START_IDX}-${END_IDX})"
done

echo "=================================================="
echo "Total jobs submitted: ${job_count}"
echo "Log files will be in: ${LOG_DIR}"
echo "Parquet files will be in: ${OUTPUT_DIR}"
