#!/bin/bash
#
# SLURM job submission script for SNR_vs_dT.py
# Loops over bands
#

# Configuration
SCRIPT_DIR="/sdf/home/l/leget/rubin-user/lsst_dev/tickets/dp2_psf"
SCRIPT_NAME="SNR_vs_dT.py"
VISIT_MAPPING_FILE="${SCRIPT_DIR}/data/visit_parquet_mapping.pkl"
LOG_DIR="${SCRIPT_DIR}/logs"
REP_OUT_PLOT="${SCRIPT_DIR}/plots"

# Create log directory if it doesn't exist
mkdir -p "${LOG_DIR}"

# Define bands
BANDS=("u" "g" "r" "i" "z" "y")

# Counter for submitted jobs
job_count=0

echo "Submitting SNR_vs_dT.py jobs to S3DF..."
echo "=================================================="

for band in "${BANDS[@]}"; do

    JOB_NAME="snr_dT_${band}"

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
#SBATCH --mem=16G
#SBATCH --output=${LOG_DIR}/${JOB_NAME}_%j.out
#SBATCH --error=${LOG_DIR}/${JOB_NAME}_%j.err

# Load LSST environment
source /sdf/group/rubin/sw/d_latest/loadLSST.bash
setup lsst_distrib -t d_latest

# Change to script directory
cd ${SCRIPT_DIR}

echo "Starting job: ${JOB_NAME}"
echo "Band: ${band}"
echo "Time: \$(date)"
echo "Node: \$(hostname)"
echo "=================================================="

python ${SCRIPT_NAME} \\
    --bands ${band} \\
    --visitMappingFile "${VISIT_MAPPING_FILE}" \\
    --repOutPlot "${REP_OUT_PLOT}" \\
    --snr_min 10 \\
    --snr_max 2000 \\
    --bin_spacing 20

echo "=================================================="
echo "Job completed at: \$(date)"
EOF

    job_count=$((job_count + 1))
    echo "Submitted: ${JOB_NAME}"

done

echo "=================================================="
echo "Total jobs submitted: ${job_count}"
echo "Log files will be in: ${LOG_DIR}"
