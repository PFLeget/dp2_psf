#!/bin/bash
#
# SLURM job submission script for fit_optics.py
# Submits one job per visit for batoid optical fitting
#

# Usage
usage() {
    echo "Usage: $0 [-n MAX_JOBS] [-b BANDS]"
    echo "  -n MAX_JOBS  Maximum number of jobs to submit (default: all)"
    echo "  -b BANDS     Filter by bands, e.g. 'r' or 'rig' (default: r)"
    exit 1
}

# Default values
MAX_JOBS=""
BANDS="r"

# Parse arguments
while getopts "n:b:h" opt; do
    case ${opt} in
        n) MAX_JOBS=${OPTARG} ;;
        b) BANDS=${OPTARG} ;;
        h) usage ;;
        *) usage ;;
    esac
done

# Configuration
SCRIPT_DIR="/sdf/home/l/leget/rubin-user/lsst_dev/tickets/dp2_psf/rayTracingFocalPlane"
SCRIPT_NAME="fit_optics.py"
VISIT_MAPPING_FILE="/sdf/home/l/leget/rubin-user/lsst_dev/tickets/dp2_psf/data/visit_parquet_mapping.pkl"
LOG_DIR="${SCRIPT_DIR}/logs"
REP_OUT="${SCRIPT_DIR}/output"

# DOF parameters to fit (including M1M3 bending modes 0-6)
PARAMS="m2_dz m2_rx m2_ry cam_dz cam_rx cam_ry m1m3_bend_0 m1m3_bend_1 m1m3_bend_2 m1m3_bend_3 m1m3_bend_4 m1m3_bend_5 m1m3_bend_6"

# Create output directories if they don't exist
mkdir -p "${LOG_DIR}"
mkdir -p "${REP_OUT}"

# Extract visit IDs from the mapping file (filtered by band)
VISIT_LIST_FILE=$(mktemp)
python3 << EOF > "${VISIT_LIST_FILE}"
import pickle
with open("${VISIT_MAPPING_FILE}", 'rb') as f:
    visit_mapping = pickle.load(f)
for visit, info in sorted(visit_mapping.items()):
    if info['band'] in "${BANDS}":
        print(visit)
EOF

# Read visits into array
mapfile -t ALL_VISITS < "${VISIT_LIST_FILE}"
rm "${VISIT_LIST_FILE}"

TOTAL_VISITS=${#ALL_VISITS[@]}
echo "Total visits for bands '${BANDS}': ${TOTAL_VISITS}"

# Apply max jobs limit
if [ -n "${MAX_JOBS}" ] && [ ${MAX_JOBS} -lt ${TOTAL_VISITS} ]; then
    NUM_JOBS=${MAX_JOBS}
    echo "Limiting to ${NUM_JOBS} jobs"
else
    NUM_JOBS=${TOTAL_VISITS}
fi

echo "Number of jobs to submit: ${NUM_JOBS}"
echo "DOF parameters: ${PARAMS}"
echo "=================================================="

# Counter for submitted jobs
job_count=0

for ((i=0; i<NUM_JOBS; i++)); do
    VISIT=${ALL_VISITS[$i]}
    JOB_NAME="fit_optics_${VISIT}"

    # Submit the job
    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --account=rubin:developers
#SBATCH --partition=milano
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=${LOG_DIR}/${JOB_NAME}_%j.out
#SBATCH --error=${LOG_DIR}/${JOB_NAME}_%j.err

# Load LSST environment
source /sdf/group/rubin/sw/d_latest/loadLSST.bash
setup lsst_distrib -t d_latest

# Change to script directory
cd ${SCRIPT_DIR}

echo "Starting fit_optics for visit: ${VISIT}"
echo "Time: \$(date)"
echo "Node: \$(hostname)"
echo "=================================================="

python ${SCRIPT_NAME} \\
    --visitID ${VISIT} \\
    --visitMappingFile "${VISIT_MAPPING_FILE}" \\
    --repOut "${REP_OUT}" \\
    --params ${PARAMS}

echo "=================================================="
echo "Job completed at: \$(date)"
EOF

    job_count=$((job_count + 1))
    echo "Submitted: ${JOB_NAME}"

done

echo "=================================================="
echo "Total jobs submitted: ${job_count}"
echo "Log files will be in: ${LOG_DIR}"
echo "Output will be in: ${REP_OUT}"
