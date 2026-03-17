#!/bin/bash
#
# SLURM job submission script for creating the time series video
# Run this AFTER parquet export jobs are complete
#

SCRIPT_DIR="/sdf/home/l/leget/rubin-user/lsst_dev/tickets/dp2_psf/test_night_20260310"
LOG_DIR="${SCRIPT_DIR}/logs"

mkdir -p "${LOG_DIR}"

for MOMENT in "dT" "de1" "de2"; do
    JOB_NAME="timeseries_video_${MOMENT}"

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

cd ${SCRIPT_DIR}

echo "Starting job: ${JOB_NAME}"
echo "Second moment key: ${MOMENT}"
echo "Time: \$(date)"
echo "Node: \$(hostname)"
echo "=================================================="

python FoVPlot_vs_heightMap_timeSeries_parquet.py \\
    --secondMomentKey ${MOMENT} \\
    --fps 5

echo "=================================================="
echo "Job completed at: \$(date)"
EOF

    echo "Submitted: ${JOB_NAME}"
done

echo "=================================================="
echo "Submitted 3 video jobs (dT, de1, de2)"
echo "Log files will be in: ${LOG_DIR}"
