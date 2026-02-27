#!/bin/bash
#
# SLURM job submission script for SNR_vs_dT.py
# Loops over bands and optionally over x-axis modes (SNR or psf_max)
#
# Usage:
#   ./submit_SNR_vs_dT_jobs.sh           # Submit SNR mode jobs
#   ./submit_SNR_vs_dT_jobs.sh --psf_max # Submit psf_max mode jobs
#   ./submit_SNR_vs_dT_jobs.sh --both    # Submit both SNR and psf_max jobs
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

# Parse arguments
USE_PSF_MAX=false
USE_SNR=true
if [[ "$1" == "--psf_max" ]]; then
    USE_PSF_MAX=true
    USE_SNR=false
elif [[ "$1" == "--both" ]]; then
    USE_PSF_MAX=true
    USE_SNR=true
fi

# Counter for submitted jobs
job_count=0

echo "Submitting SNR_vs_dT.py jobs to S3DF..."
echo "=================================================="

# Submit SNR mode jobs
if [[ "$USE_SNR" == true ]]; then
    echo "Submitting SNR mode jobs..."
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
echo "Mode: SNR"
echo "Time: \$(date)"
echo "Node: \$(hostname)"
echo "=================================================="

python ${SCRIPT_NAME} \\
    --bands ${band} \\
    --visitMappingFile "${VISIT_MAPPING_FILE}" \\
    --repOutPlot "${REP_OUT_PLOT}"

echo "=================================================="
echo "Job completed at: \$(date)"
EOF

        job_count=$((job_count + 1))
        echo "Submitted: ${JOB_NAME}"

    done
fi

# Submit psf_max mode jobs
if [[ "$USE_PSF_MAX" == true ]]; then
    echo "Submitting psf_max mode jobs..."
    for band in "${BANDS[@]}"; do

        JOB_NAME="psfmax_dT_${band}"

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
echo "Mode: psf_max"
echo "Time: \$(date)"
echo "Node: \$(hostname)"
echo "=================================================="

python ${SCRIPT_NAME} \\
    --bands ${band} \\
    --visitMappingFile "${VISIT_MAPPING_FILE}" \\
    --repOutPlot "${REP_OUT_PLOT}" \\
    --use_psf_max

echo "=================================================="
echo "Job completed at: \$(date)"
EOF

        job_count=$((job_count + 1))
        echo "Submitted: ${JOB_NAME}"

    done
fi

echo "=================================================="
echo "Total jobs submitted: ${job_count}"
echo "Log files will be in: ${LOG_DIR}"
