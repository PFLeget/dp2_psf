#!/bin/bash
#
# SLURM job submission script for comp_rho_stat.py
# Submits jobs for riz bands (combined) coadd and single visit rho statistics
# with both distortion and shear ellipticity types
#

# Configuration
SCRIPT_DIR="/sdf/home/l/leget/rubin-user/lsst_dev/tickets/dp2_psf/coadd"
SCRIPT_NAME="comp_rho_stat.py"
TRACT_MAPPING_FILE="/sdf/home/l/leget/rubin-user/lsst_dev/tickets/dp2_psf/data/tract_parquet_mapping.pkl"
VISIT_MAPPING_FILE="/sdf/home/l/leget/rubin-user/lsst_dev/tickets/dp2_psf/data/visit_parquet_mapping_skycoord.pkl"
LOG_DIR="${SCRIPT_DIR}/logs"
REP_OUT="${SCRIPT_DIR}/rho_stats_riz"

# Parameters
BANDS="riz"
MIN_SEP=0.5
MAX_SEP=200
NBINS=25

# Ellipticity types to process
ELLIPTICITY_TYPES=("distortion" "shear")

# Create output directories
mkdir -p "${LOG_DIR}"
mkdir -p "${REP_OUT}"

echo "Submitting Rho statistics jobs for bands: ${BANDS}"
echo "Angular bins: ${NBINS} bins from ${MIN_SEP} to ${MAX_SEP} arcmin"
echo "Ellipticity types: ${ELLIPTICITY_TYPES[*]}"
echo "=================================================="

JOB_COUNT=0

for ETYPE in "${ELLIPTICITY_TYPES[@]}"; do
    echo ""
    echo "Ellipticity type: ${ETYPE}"
    echo "--------------------------------------------------"

    # Job: Coadd (16 cpus, 64GB)
    JOB_NAME="rho_coadd_${BANDS}_${ETYPE}"
    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --account=rubin:developers
#SBATCH --partition=torino
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=${LOG_DIR}/${JOB_NAME}_%j.out
#SBATCH --error=${LOG_DIR}/${JOB_NAME}_%j.err

# Load LSST environment
source /sdf/group/rubin/sw/d_latest/loadLSST.bash
setup lsst_distrib -t d_latest

cd ${SCRIPT_DIR}

echo "Starting Rho statistics: COADD ${BANDS}-band (${ETYPE})"
echo "Time: \$(date)"
echo "Node: \$(hostname)"
echo "=================================================="

python ${SCRIPT_NAME} \\
    --mode coadd \\
    --band ${BANDS} \\
    --tractMappingFile "${TRACT_MAPPING_FILE}" \\
    --repOut "${REP_OUT}" \\
    --galactic_b_min 25. \\
    --ellipticityType ${ETYPE} \\
    --min_sep ${MIN_SEP} \\
    --max_sep ${MAX_SEP} \\
    --nbins ${NBINS}

echo "=================================================="
echo "Job completed at: \$(date)"
EOF

    echo "  Submitted: ${JOB_NAME}"
    ((JOB_COUNT++))

    # Job: Single Visit (32 cpus, 128GB)
    JOB_NAME="rho_sv_${BANDS}_${ETYPE}"
    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --account=rubin:developers
#SBATCH --partition=torino
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --output=${LOG_DIR}/${JOB_NAME}_%j.out
#SBATCH --error=${LOG_DIR}/${JOB_NAME}_%j.err

# Load LSST environment
source /sdf/group/rubin/sw/d_latest/loadLSST.bash
setup lsst_distrib -t d_latest

cd ${SCRIPT_DIR}

echo "Starting Rho statistics: SINGLE VISIT ${BANDS}-band (${ETYPE})"
echo "Time: \$(date)"
echo "Node: \$(hostname)"
echo "=================================================="

python ${SCRIPT_NAME} \\
    --mode single_visit \\
    --band ${BANDS} \\
    --visitMappingFile "${VISIT_MAPPING_FILE}" \\
    --repOut "${REP_OUT}" \\
    --galactic_b_min 25. \\
    --ellipticityType ${ETYPE} \\
    --min_sep ${MIN_SEP} \\
    --max_sep ${MAX_SEP} \\
    --nbins ${NBINS}

echo "=================================================="
echo "Job completed at: \$(date)"
EOF

    echo "  Submitted: ${JOB_NAME}"
    ((JOB_COUNT++))

done

echo "=================================================="
echo "Total jobs submitted: ${JOB_COUNT}"
echo "  - 2 coadd jobs (16 cpus, 64GB each)"
echo "  - 2 single_visit jobs (32 cpus, 128GB each)"
echo "Log files will be in: ${LOG_DIR}"
echo "Results will be in: ${REP_OUT}"
