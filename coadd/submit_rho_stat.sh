#!/bin/bash
#
# Submit rho statistics jobs for coadd and single visit
#
# Usage:
#   ./submit_rho_stat.sh
#
# Submits 4 jobs:
#   - coadd distortion (16 cpus, 64GB)
#   - coadd shear (16 cpus, 64GB)
#   - single_visit distortion (32 cpus, 128GB)
#   - single_visit shear (32 cpus, 128GB)

SCRIPT_DIR="/sdf/home/l/leget/rubin-user/lsst_dev/tickets/dp2_psf/coadd"
VISIT_MAPPING="/sdf/home/l/leget/rubin-user/lsst_dev/tickets/dp2_psf/data/visit_parquet_mapping_skycoord.pkl"
TRACT_MAPPING="/sdf/home/l/leget/rubin-user/lsst_dev/tickets/dp2_psf/data/tract_parquet_mapping.pkl"
OUTPUT_DIR="${SCRIPT_DIR}/rho_stats_riz/"

BANDS="riz"
MIN_SEP=0.5
MAX_SEP=200
NBINS=25

mkdir -p ${OUTPUT_DIR}
mkdir -p logs

# Coadd distortion
sbatch --job-name=rho_coadd_dist \
    --partition=torino \
    --account=rubin:developers \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=16 \
    --mem=64G \
    --time=12:00:00 \
    --output=logs/rho_coadd_dist_%j.out \
    --error=logs/rho_coadd_dist_%j.err \
    --wrap="source /sdf/group/rubin/sw/tag/w_2024_22/loadLSST.bash && setup lsst_distrib && \
python ${SCRIPT_DIR}/comp_rho_stat.py \
    --mode coadd \
    --band ${BANDS} \
    --tractMappingFile ${TRACT_MAPPING} \
    --repOut ${OUTPUT_DIR} \
    --ellipticityType distortion \
    --min_sep ${MIN_SEP} \
    --max_sep ${MAX_SEP} \
    --nbins ${NBINS}"

# Coadd shear
sbatch --job-name=rho_coadd_shear \
    --partition=torino \
    --account=rubin:developers \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=16 \
    --mem=64G \
    --time=12:00:00 \
    --output=logs/rho_coadd_shear_%j.out \
    --error=logs/rho_coadd_shear_%j.err \
    --wrap="source /sdf/group/rubin/sw/tag/w_2024_22/loadLSST.bash && setup lsst_distrib && \
python ${SCRIPT_DIR}/comp_rho_stat.py \
    --mode coadd \
    --band ${BANDS} \
    --tractMappingFile ${TRACT_MAPPING} \
    --repOut ${OUTPUT_DIR} \
    --ellipticityType shear \
    --min_sep ${MIN_SEP} \
    --max_sep ${MAX_SEP} \
    --nbins ${NBINS}"

# Single visit distortion
sbatch --job-name=rho_sv_dist \
    --partition=torino \
    --account=rubin:developers \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=32 \
    --mem=128G \
    --time=24:00:00 \
    --output=logs/rho_sv_dist_%j.out \
    --error=logs/rho_sv_dist_%j.err \
    --wrap="source /sdf/group/rubin/sw/tag/w_2024_22/loadLSST.bash && setup lsst_distrib && \
python ${SCRIPT_DIR}/comp_rho_stat.py \
    --mode single_visit \
    --band ${BANDS} \
    --visitMappingFile ${VISIT_MAPPING} \
    --repOut ${OUTPUT_DIR} \
    --ellipticityType distortion \
    --min_sep ${MIN_SEP} \
    --max_sep ${MAX_SEP} \
    --nbins ${NBINS}"

# Single visit shear
sbatch --job-name=rho_sv_shear \
    --partition=torino \
    --account=rubin:developers \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=32 \
    --mem=128G \
    --time=24:00:00 \
    --output=logs/rho_sv_shear_%j.out \
    --error=logs/rho_sv_shear_%j.err \
    --wrap="source /sdf/group/rubin/sw/tag/w_2024_22/loadLSST.bash && setup lsst_distrib && \
python ${SCRIPT_DIR}/comp_rho_stat.py \
    --mode single_visit \
    --band ${BANDS} \
    --visitMappingFile ${VISIT_MAPPING} \
    --repOut ${OUTPUT_DIR} \
    --ellipticityType shear \
    --min_sep ${MIN_SEP} \
    --max_sep ${MAX_SEP} \
    --nbins ${NBINS}"

echo "Submitted 4 jobs:"
echo "  - coadd distortion (16 cpus, 64GB)"
echo "  - coadd shear (16 cpus, 64GB)"
echo "  - single_visit distortion (32 cpus, 128GB)"
echo "  - single_visit shear (32 cpus, 128GB)"
echo ""
echo "Bands: ${BANDS}"
echo "Angular bins: ${NBINS} bins from ${MIN_SEP} to ${MAX_SEP} arcmin"
echo "Output: ${OUTPUT_DIR}"
