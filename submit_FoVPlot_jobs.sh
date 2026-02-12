#!/bin/bash
#
# SLURM job submission script for FoVPlot_vs_heightMap.py
# Loops over filters, second moments, and Zernike coefficients
# Submits individual jobs to S3DF torino partition
#

# Configuration
SCRIPT_DIR="/sdf/home/l/leget/rubin-user/lsst_dev/tickets/dp2_psf"
SCRIPT_NAME="FoVPlot_vs_heightMap.py"
PSF_BASE_PATH="/sdf/home/l/leget/rubin-user/lsst_dev/tickets/dp2_psf/data"
LOG_DIR="${SCRIPT_DIR}/logs"

# Create log directory if it doesn't exist
mkdir -p "${LOG_DIR}"

# Define parameter arrays
FILTERS=("u" "g" "r" "i" "z" "y")
SECOND_MOMENTS=("dT" "de1" "de2")
ZERNIKES=("z4" "z5" "z6" "z7" "z8" "z9" "z10" "z11")

# Counter for submitted jobs
job_count=0

echo "Submitting FoVPlot_vs_heightMap.py jobs to S3DF..."
echo "=================================================="

for band in "${FILTERS[@]}"; do
    for moment in "${SECOND_MOMENTS[@]}"; do
        for zernike in "${ZERNIKES[@]}"; do

            JOB_NAME="fov_${band}_${moment}_${zernike}"
            PSF_PATH="${PSF_BASE_PATH}/${band}"

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
echo "Band: ${band}, Second Moment: ${moment}, Zernike: ${zernike}"
echo "Time: \$(date)"
echo "Node: \$(hostname)"
echo "=================================================="

python ${SCRIPT_NAME} \\
    --band ${band} \\
    --pathPSFFile "${PSF_PATH}" \\
    --secondMomentKey ${moment} \\
    --zernikeKey ${zernike}

echo "=================================================="
echo "Job completed at: \$(date)"
EOF

            job_count=$((job_count + 1))
            echo "Submitted: ${JOB_NAME}"

        done
    done
done

echo "=================================================="
echo "Total jobs submitted: ${job_count}"
echo "Log files will be in: ${LOG_DIR}"
