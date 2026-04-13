#!/bin/bash
#
# SLURM submission script for parallel training data preparation.
#
# Usage:
#   ./submit_training_data_jobs.sh annotations.csv 20
#

ANNOTATIONS=${1:-"annotations.csv"}
N_CHUNKS=${2:-20}

SCRIPT_DIR="/sdf/home/l/leget/rubin-user/lsst_dev/tickets/dp2_psf/miniDonutsClassifier"
LOG_DIR="${SCRIPT_DIR}/training_data_logs"
OUTPUT_DIR="${SCRIPT_DIR}/training_data_chunks"

mkdir -p "${LOG_DIR}"
mkdir -p "${OUTPUT_DIR}"

echo "Submitting ${N_CHUNKS} jobs for ${ANNOTATIONS}"
echo "Output directory: ${OUTPUT_DIR}"
echo "=================================================="

for ((chunk_id=0; chunk_id<N_CHUNKS; chunk_id++)); do
    JOB_NAME="train_data_chunk${chunk_id}"
    OUTPUT_FILE="${OUTPUT_DIR}/training_data_chunk_${chunk_id}.h5"

    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --account=rubin:developers
#SBATCH --partition=milano
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=${LOG_DIR}/${JOB_NAME}_%j.out
#SBATCH --error=${LOG_DIR}/${JOB_NAME}_%j.err

source /sdf/group/rubin/sw/d_latest/loadLSST.bash
setup lsst_distrib -t d_latest

cd ${SCRIPT_DIR}

echo "Starting chunk ${chunk_id}/${N_CHUNKS}"
echo "Time: \$(date)"
echo "=================================================="

python prepare_training_data.py \\
    --annotations "${ANNOTATIONS}" \\
    --output "${OUTPUT_FILE}" \\
    --chunk_id ${chunk_id} \\
    --n_chunks ${N_CHUNKS}

echo "=================================================="
echo "Completed at: \$(date)"
EOF

    echo "Submitted: ${JOB_NAME}"
done

echo "=================================================="
echo "Jobs submitted. After completion, run:"
echo "  python merge_training_data.py --input_dir ${OUTPUT_DIR} --output training_data.h5"
