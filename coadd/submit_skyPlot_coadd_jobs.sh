#!/bin/bash
#
# SLURM job submission script for skyPlot_coadd_secondMoment.py
# Loops over bands and second moments
# Submits individual jobs to S3DF milano partition
#

# Configuration
SCRIPT_DIR="/sdf/home/l/leget/rubin-user/lsst_dev/tickets/dp2_psf/coadd"
SCRIPT_NAME="skyPlot_coadd_secondMoment.py"
TRACT_MAPPING_FILE="/sdf/home/l/leget/rubin-user/lsst_dev/tickets/dp2_psf/data/tract_parquet_mapping.pkl"
LOG_DIR="${SCRIPT_DIR}/logs"
REP_OUT_PLOT="${SCRIPT_DIR}/plots"

# Create output directories if they don't exist
mkdir -p "${LOG_DIR}"
mkdir -p "${REP_OUT_PLOT}"

# Define parameter arrays
BANDS=("u" "g" "r" "i" "z" "y")
SECOND_MOMENTS=("T" "e1" "e2" "dT_T" "de1" "de2")

# Counter for submitted jobs
job_count=0

echo "Submitting skyPlot_coadd_secondMoment.py jobs to S3DF..."
echo "=================================================="

for band in "${BANDS[@]}"; do
    for moment in "${SECOND_MOMENTS[@]}"; do

        JOB_NAME="coadd_sky_${band}_${moment}"

        # Submit the job
        sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --account=rubin:developers
#SBATCH --partition=milano
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
echo "Band: ${band}, Second Moment: ${moment}"
echo "Time: \$(date)"
echo "Node: \$(hostname)"
echo "=================================================="

python ${SCRIPT_NAME} \\
    --band ${band} \\
    --tractMappingFile "${TRACT_MAPPING_FILE}" \\
    --key_second_moment ${moment} \\
    --bin_spacing 3600 \\
    --autoColorScale \\
    --repOutPlot "${REP_OUT_PLOT}"

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
echo "Plots will be in: ${REP_OUT_PLOT}"
