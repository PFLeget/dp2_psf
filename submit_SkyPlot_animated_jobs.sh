#!/bin/bash
#
# SLURM job submission script for SkyPlot_vs_secondMoment_animated.py
# Creates animated videos showing PSF residuals evolution over time
# Submits individual jobs to S3DF torino partition
#

# Configuration
SCRIPT_DIR="/sdf/home/l/leget/rubin-user/lsst_dev/tickets/dp2_psf"
SCRIPT_NAME="SkyPlot_vs_secondMoment_animated.py"
VISIT_MAPPING_FILE="/sdf/home/l/leget/rubin-user/lsst_dev/tickets/dp2_psf/data/visit_parquet_mapping.pkl"
LOG_DIR="${SCRIPT_DIR}/logs"
REP_OUT_PLOT="${SCRIPT_DIR}/plots"

# Create log directory if it doesn't exist
mkdir -p "${LOG_DIR}"

# Define parameter arrays
FILTERS=("u" "g", "r", "i", "z", "y")
SECOND_MOMENTS=("dT_T")

# Animation parameters
FPS=24
VISITS_PER_FRAME=1
BIN_SPACING=3600

# Counter for submitted jobs
job_count=0

echo "Submitting SkyPlot_vs_secondMoment_animated.py jobs to S3DF..."
echo "=================================================="

for band in "${FILTERS[@]}"; do
    for moment in "${SECOND_MOMENTS[@]}"; do

        JOB_NAME="sky_anim_${band}_${moment}"

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
#SBATCH --mem=32G
#SBATCH --output=${LOG_DIR}/${JOB_NAME}_%j.out
#SBATCH --error=${LOG_DIR}/${JOB_NAME}_%j.err

# Load LSST environment
source /sdf/group/rubin/sw/d_latest/loadLSST.bash
setup lsst_distrib -t d_latest

# Change to script directory
cd ${SCRIPT_DIR}

echo "Starting job: ${JOB_NAME}"
echo "Band: ${band}, Second Moment: ${moment}"
echo "FPS: ${FPS}, Visits per frame: ${VISITS_PER_FRAME}"
echo "Time: \$(date)"
echo "Node: \$(hostname)"
echo "=================================================="

python ${SCRIPT_NAME} \\
    --bands ${band} \\
    --visitMappingFile "${VISIT_MAPPING_FILE}" \\
    --key_second_moment ${moment} \\
    --bin_spacing ${BIN_SPACING} \\
    --repOutPlot "${REP_OUT_PLOT}" \\
    --fps ${FPS} \\
    --visits_per_frame ${VISITS_PER_FRAME}

echo "=================================================="
echo "Job completed at: \$(date)"
EOF

        job_count=$((job_count + 1))
        echo "Submitted: ${JOB_NAME}"

    done
done

echo "=================================================="
echo "Total jobs submitted: ${job_count}"
echo "Log files will be in: ${LOG_DIR}"
