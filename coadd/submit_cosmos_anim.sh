#!/bin/bash
#SBATCH --job-name=cosmos_rho_anim
#SBATCH --partition=torino
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/cosmos_rho_anim_%j.out
#SBATCH --error=logs/cosmos_rho_anim_%j.err

# COSMOS DDF Animated Rho Statistics
#
# Usage:
#   sbatch submit_cosmos_anim.sh [BAND]
#
# Example:
#   sbatch submit_cosmos_anim.sh r
#   sbatch submit_cosmos_anim.sh i

SCRIPT_DIR="/sdf/home/l/leget/rubin-user/lsst_dev/tickets/dp2_psf/coadd"
VISIT_MAPPING="/sdf/home/l/leget/rubin-user/lsst_dev/tickets/dp2_psf/data/visit_parquet_mapping_skycoord.pkl"
OUTPUT_DIR="${SCRIPT_DIR}/cosmos_rho_anim"

# Band from command line or default
BAND=${1:-r}

echo "=============================================="
echo "COSMOS DDF Animated Rho Statistics"
echo "=============================================="
echo "Band: ${BAND}"
echo "Output: ${OUTPUT_DIR}"
echo "Visit mapping: ${VISIT_MAPPING}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "CPUs: ${SLURM_CPUS_PER_TASK}"
echo "Memory: 128GB"
echo "=============================================="

# Create output directories
mkdir -p ${OUTPUT_DIR}
mkdir -p logs

# Setup LSST environment
source /sdf/group/rubin/sw/tag/w_2024_22/loadLSST.bash
setup lsst_distrib

# Run animation script
python ${SCRIPT_DIR}/anim_rho_stat_cosmos.py \
    --band ${BAND} \
    --visitMappingFile ${VISIT_MAPPING} \
    --repOut ${OUTPUT_DIR} \
    --ellipticityType distortion \
    --min_sep 0.01 \
    --max_sep 300 \
    --nbins 30 \
    --bin_spacing 240 \
    --frame_interval 10 \
    --ylim_rho1 5e-4 \
    --ylim_rho2 5e-4 \
    --ylim_rho3 5e-6 \
    --ylim_rho4 1e-5 \
    --ylim_rho5 5e-5 \
    --ylim_rho3alt_min -1e-3 \
    --ylim_rho3alt_max 1e-3 \
    --sky_scale_dT 0.02 \
    --sky_scale_de 0.01

echo "=============================================="
echo "Done! Creating animation..."
echo "=============================================="

# Create animation with ffmpeg if frames exist
FRAMES_DIR="${OUTPUT_DIR}/cosmos_frames_${BAND}_distortion"
if [ -d "${FRAMES_DIR}" ]; then
    ffmpeg -y -framerate 5 -i ${FRAMES_DIR}/frame_%04d.png \
        -c:v libx264 -pix_fmt yuv420p \
        ${OUTPUT_DIR}/cosmos_rho_anim_${BAND}.mp4
    echo "Animation saved: ${OUTPUT_DIR}/cosmos_rho_anim_${BAND}.mp4"
fi

echo "Job complete."
