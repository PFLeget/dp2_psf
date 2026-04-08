#!/bin/bash
#
# SLURM job submission script for avgPSF_vs_Zernike.py
# Parallelizes processing by submitting one job per bin
#

# Configuration
SCRIPT_DIR="/sdf/home/l/leget/rubin-user/lsst_dev/tickets/dp2_psf"
SCRIPT_NAME="avgPSF_vs_Zernike.py"
LOG_DIR="${SCRIPT_DIR}/logs/avgPSF"
REP_OUT_PLOT="${SCRIPT_DIR}/plots/avgPSF"
REP_OUT_FILE="${SCRIPT_DIR}/data"

# Parameters - adjust these as needed
BANDS="griz"
ZERNIKE_KEY="z4"
DETECTOR=94
N_BINS=40  # Set this to the number of bins (run script once without --bin_idx to see)

# Create output directories
mkdir -p "${LOG_DIR}"
mkdir -p "${REP_OUT_PLOT}"

echo "Submitting avgPSF_vs_Zernike.py jobs..."
echo "  Bands: ${BANDS}"
echo "  Zernike: ${ZERNIKE_KEY}"
echo "  Detector: ${DETECTOR}"
echo "  Number of bins: ${N_BINS}"
echo "=================================================="

job_count=0

for ((bin_idx=0; bin_idx<N_BINS; bin_idx++)); do

    JOB_NAME="avgPSF_${ZERNIKE_KEY}_bin${bin_idx}"

    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --account=rubin:developers
#SBATCH --partition=milano
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=${LOG_DIR}/${JOB_NAME}_%j.out
#SBATCH --error=${LOG_DIR}/${JOB_NAME}_%j.err

# Load LSST environment
source /sdf/group/rubin/sw/d_latest/loadLSST.bash
setup lsst_distrib -t d_latest

cd ${SCRIPT_DIR}

echo "Starting avgPSF_vs_Zernike: bin ${bin_idx}"
echo "Time: \$(date)"
echo "Node: \$(hostname)"
echo "=================================================="

python ${SCRIPT_NAME} \\
    --band ${BANDS} \\
    --zernikeKey ${ZERNIKE_KEY} \\
    --detector ${DETECTOR} \\
    --bin_idx ${bin_idx} \\
    --repOutPlot "${REP_OUT_PLOT}" \\
    --repOutFile "${REP_OUT_FILE}"

echo "=================================================="
echo "Job completed at: \$(date)"
EOF

    job_count=$((job_count + 1))
    echo "Submitted: ${JOB_NAME}"

done

echo "=================================================="
echo "Total jobs submitted: ${job_count}"
echo "Log files will be in: ${LOG_DIR}"
echo "Plots will be in: ${REP_OUT_PLOT}"
