#!/bin/bash
#
# SLURM job submission script for comp_rho_stat.py
# Submits jobs for r-band coadd and single visit rho statistics
#

# Configuration
SCRIPT_DIR="/sdf/home/l/leget/rubin-user/lsst_dev/tickets/dp2_psf/coadd"
SCRIPT_NAME="comp_rho_stat.py"
TRACT_MAPPING_FILE="/sdf/home/l/leget/rubin-user/lsst_dev/tickets/dp2_psf/data/tract_parquet_mapping.pkl"
VISIT_MAPPING_FILE="/sdf/home/l/leget/rubin-user/lsst_dev/tickets/dp2_psf/data/visit_parquet_mapping_skycoord.pkl"
COADD_DETECTOR_FILE="/sdf/home/l/leget/rubin-user/lsst_dev/tickets/dp2_psf/data/coadd_detector_mapping.pkl"
LOG_DIR="${SCRIPT_DIR}/logs"
REP_OUT="${SCRIPT_DIR}/rho_stats"

# Create output directories
mkdir -p "${LOG_DIR}"
mkdir -p "${REP_OUT}"

BAND="r"

echo "Submitting Rho statistics jobs for ${BAND}-band..."
echo "=================================================="

# Job 1: Coadd
JOB_NAME="rho_coadd_${BAND}"
sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --account=rubin:developers
#SBATCH --partition=milano
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

cd ${SCRIPT_DIR}

echo "Starting Rho statistics: COADD ${BAND}-band"
echo "Time: \$(date)"
echo "Node: \$(hostname)"
echo "=================================================="

python ${SCRIPT_NAME} \\
    --mode coadd \\
    --band ${BAND} \\
    --tractMappingFile "${TRACT_MAPPING_FILE}" \\
    --repOut "${REP_OUT}" \\
    --galactic_b_min 25.

echo "=================================================="
echo "Job completed at: \$(date)"
EOF

echo "Submitted: ${JOB_NAME}"

# Job 2: Single Visit
JOB_NAME="rho_single_${BAND}"
sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --account=rubin:developers
#SBATCH --partition=milano
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

cd ${SCRIPT_DIR}

echo "Starting Rho statistics: SINGLE VISIT ${BAND}-band"
echo "Time: \$(date)"
echo "Node: \$(hostname)"
echo "=================================================="

python ${SCRIPT_NAME} \\
    --mode single_visit \\
    --band ${BAND} \\
    --visitMappingFile "${VISIT_MAPPING_FILE}" \\
    --repOut "${REP_OUT}" \\
    --galactic_b_min 25. \\
    --coaddDetectorFile "${COADD_DETECTOR_FILE}"

echo "=================================================="
echo "Job completed at: \$(date)"
EOF

echo "Submitted: ${JOB_NAME}"

echo "=================================================="
echo "Total jobs submitted: 2"
echo "Log files will be in: ${LOG_DIR}"
echo "Results will be in: ${REP_OUT}"
