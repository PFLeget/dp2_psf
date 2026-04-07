#!/bin/bash
#
# SLURM job submission script for generate_annotation_images.py
#
# Visit selection logic:
# - ALL visits with abs(z4+0.25) > 1.8 (likely_bad)
# - Random sample of the SAME NUMBER from visits with abs(z4+0.25) <= 1.8 (likely_good)
# - Process in batches of 10 visits per job
#

# Configuration
SCRIPT_DIR="/sdf/home/l/leget/rubin-user/lsst_dev/tickets/dp2_psf/miniDonutsClassifier"
SCRIPT_NAME="generate_annotation_images.py"
LOG_DIR="${SCRIPT_DIR}/logs"
REP_OUT="${SCRIPT_DIR}/plotForAnnotation"
DIC_ZERNIKE="/sdf/home/l/leget/rubin-user/lsst_dev/tickets/dp2_psf/data/visit_to_band_mapv2.pkl"

VISITS_PER_JOB=10

# Create output directories
mkdir -p "${LOG_DIR}"
mkdir -p "${REP_OUT}"

echo "Generating visit list with balanced sampling..."
echo "=================================================="

# Use Python to select visits with balanced sampling
VISIT_FILE=$(mktemp)

python << EOF > "${VISIT_FILE}"
import pickle
import numpy as np

# Load Zernike data
with open("${DIC_ZERNIKE}", 'rb') as f:
    zernike_table = pickle.load(f)

# Compute z4 median for each visit
visits = []
z4_values = []
for visit in zernike_table:
    z4_med = np.nanmedian(zernike_table[visit]['z4'])
    if np.isfinite(z4_med):
        visits.append(visit)
        z4_values.append(z4_med)

visits = np.array(visits)
z4_values = np.array(z4_values)

# Split by threshold
threshold = 1.8
likely_bad_mask = np.abs(z4_values + 0.25) > threshold
likely_good_mask = ~likely_bad_mask

likely_bad_visits = visits[likely_bad_mask]
likely_good_visits = visits[likely_good_mask]

print(f"# Found {len(likely_bad_visits)} likely_bad visits", file=__import__('sys').stderr)
print(f"# Found {len(likely_good_visits)} likely_good visits", file=__import__('sys').stderr)

# Take ALL likely_bad visits
selected_bad = likely_bad_visits

# Random sample of same size from likely_good
np.random.seed(42)  # For reproducibility
n_sample = len(selected_bad)
if n_sample <= len(likely_good_visits):
    selected_good = np.random.choice(likely_good_visits, size=n_sample, replace=False)
else:
    selected_good = likely_good_visits
    print(f"# Warning: only {len(likely_good_visits)} good visits available", file=__import__('sys').stderr)

print(f"# Selected {len(selected_bad)} likely_bad + {len(selected_good)} likely_good = {len(selected_bad) + len(selected_good)} total", file=__import__('sys').stderr)

# Output all selected visits
for v in selected_bad:
    print(v)
for v in selected_good:
    print(v)
EOF

# Read visits into array
mapfile -t VISITS < "${VISIT_FILE}"
rm "${VISIT_FILE}"

TOTAL_VISITS=${#VISITS[@]}
echo "Total visits to process: ${TOTAL_VISITS}"
echo "Visits per job: ${VISITS_PER_JOB}"

# Calculate number of jobs
N_JOBS=$(( (TOTAL_VISITS + VISITS_PER_JOB - 1) / VISITS_PER_JOB ))
echo "Number of jobs: ${N_JOBS}"
echo "=================================================="

job_count=0

for ((job_idx=0; job_idx<N_JOBS; job_idx++)); do

    # Get visits for this job
    START_IDX=$((job_idx * VISITS_PER_JOB))
    END_IDX=$((START_IDX + VISITS_PER_JOB))
    if [ $END_IDX -gt $TOTAL_VISITS ]; then
        END_IDX=$TOTAL_VISITS
    fi

    # Build visit list for this job
    JOB_VISITS=""
    for ((i=START_IDX; i<END_IDX; i++)); do
        JOB_VISITS="${JOB_VISITS} ${VISITS[$i]}"
    done

    JOB_NAME="annotation_batch${job_idx}"

    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --account=rubin:developers
#SBATCH --partition=torino
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

echo "Starting annotation batch ${job_idx}"
echo "Time: \$(date)"
echo "Node: \$(hostname)"
echo "Visits: ${JOB_VISITS}"
echo "=================================================="

for VISIT in ${JOB_VISITS}; do
    echo "Processing visit \${VISIT}..."
    python ${SCRIPT_NAME} \\
        --visit \${VISIT} \\
        --dicZernike "${DIC_ZERNIKE}" \\
        --repOut "${REP_OUT}"
done

echo "=================================================="
echo "Batch completed at: \$(date)"
EOF

    job_count=$((job_count + 1))
    echo "Submitted: ${JOB_NAME} (visits ${START_IDX}-$((END_IDX-1)))"

done

echo "=================================================="
echo "Total jobs submitted: ${job_count}"
echo "Log files will be in: ${LOG_DIR}"
echo "Output images will be in: ${REP_OUT}"
