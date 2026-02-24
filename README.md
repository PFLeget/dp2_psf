# DP2 PSF Analysis

Analysis tools for PSF (Point Spread Function) residuals in Rubin/LSST Data Preview 2 (DP2).

## Overview

This repository contains scripts to analyze and visualize PSF shape residuals across the focal plane and sky coordinates. The analysis computes differences between measured star shapes and PSF model shapes:

- **dT/T**: Fractional size residual `(T_src - T_psf) / T_src`
- **de1, de2**: Ellipticity residuals `e1_src - e1_psf`, `e2_src - e2_psf`

where `T = Ixx + Iyy` and `e1 = (Ixx - Iyy) / T`, `e2 = 2*Ixy / T`.

## Data Access

The scripts read PSF star catalogs from the DP2 butler collection:
- **Repository**: `dp2_prep`
- **Collection**: `LSSTCam/runs/DRP/DP2/v30_0_0/DM-53881/stage2`
- **Dataset type**: `refit_psf_star`

Data is accessed via a mapping file (`visit_parquet_mapping.pkl`) that stores paths to parquet files, enabling fast direct reads with `polars` instead of slow butler queries.

## Scripts

### Data Preparation

| Script | Description |
|--------|-------------|
| `getData.py` | Creates `visit_parquet_mapping.pkl` mapping visit IDs to parquet file paths |
| `getZernike.ipynb` | Queries consdb for Zernike coefficients, creates `visit_to_band_mapv2.pkl` |
| `getZernike_withCorners.ipynb` | Same as above + creates `visit_zernike_corners.pkl` with corner positions |

### Analysis Scripts

| Script | Description |
|--------|-------------|
| `FoVPlot_vs_secondMoment.py` | Focal plane maps of PSF residuals averaged over all visits |
| `FoVPlot_vs_heightMap.py` | Correlation between PSF residuals and Zernike wavefront coefficients |
| `SkyPlot_vs_secondMoment.py` | Sky coordinate (HEALPix) maps of PSF residuals |
| `SkyPlot_vs_secondMoment_animated.py` | Animated sky maps showing temporal evolution of residuals |
| `SingleVisit_FocusGradient.py` | Per-visit focus gradient analysis: correlates star size with height map |

### SLURM Submission Scripts

| Script | Description |
|--------|-------------|
| `submit_FoVPlot_secondMoment_jobs.sh` | Submit FoVPlot_vs_secondMoment jobs |
| `submit_FoVPlot_jobs.sh` | Submit FoVPlot_vs_heightMap jobs |
| `submit_SkyPlot_secondMoment_jobs.sh` | Submit SkyPlot_vs_secondMoment jobs |
| `submit_SkyPlot_animated_jobs.sh` | Submit animated sky plot jobs |
| `submit_SingleVisit_FocusGradient_jobs.sh` | Submit single visit focus gradient jobs (100 visits/job) |

## Usage

### Step 1: Create Visit Mapping (run once)

```bash
python getData.py --repOut data/
```

This creates `data/visit_parquet_mapping.pkl` containing paths to all visit parquet files.

### Step 2: Run Analysis Scripts

#### Local Python Execution

**Focal plane second moment plot:**
```bash
python FoVPlot_vs_secondMoment.py \
    --bands g \
    --visitMappingFile data/visit_parquet_mapping.pkl \
    --key_second_moment dT_T \
    --bin_spacing 80 \
    --repOutPlot plots/
```

**Focal plane vs height map:**
```bash
python FoVPlot_vs_heightMap.py \
    --band g \
    --visitMappingFile data/visit_parquet_mapping.pkl \
    --secondMomentKey dT \
    --zernikeKey z4
```

**Sky coordinate plot:**
```bash
python SkyPlot_vs_secondMoment.py \
    --bands ugrizy \
    --visitMappingFile data/visit_parquet_mapping.pkl \
    --key_second_moment dT_T \
    --bin_spacing 3600 \
    --repOutPlot plots/
```

**Animated sky plot:**
```bash
python SkyPlot_vs_secondMoment_animated.py \
    --bands g \
    --visitMappingFile data/visit_parquet_mapping.pkl \
    --key_second_moment dT_T \
    --bin_spacing 3600 \
    --fps 24 \
    --visits_per_frame 1 \
    --repOutPlot plots/
```

**Single visit focus gradient analysis:**
```bash
python SingleVisit_FocusGradient.py \
    --visit 2024110800256 \
    --visitMappingFile data/visit_parquet_mapping.pkl \
    --fitHeightMap data/LSST_FP_cold_b_measurement_4col_bysurface.fits \
    --zernikeCornersFile data/visit_zernike_corners.pkl \
    --repOutPlot plots/focus_gradient/
```

This creates a 4-panel plot showing:
1. Height map from SLAC metrology
2. z4 wavefront at corner sensors (with gradient direction)
3. Star size residuals (T - <T>) per CCD
4. Correlation coefficient between height and star size per CCD (with gradient direction)

#### SLURM Execution (S3DF)

Before running, update the paths in the submission scripts:
- `SCRIPT_DIR`: Path to this repository on S3DF
- `VISIT_MAPPING_FILE`: Path to `visit_parquet_mapping.pkl`

Then submit jobs:
```bash
# Submit all FoVPlot_vs_secondMoment jobs (7 bands x 3 moments = 21 jobs)
bash submit_FoVPlot_secondMoment_jobs.sh

# Submit all FoVPlot_vs_heightMap jobs (6 bands x 3 moments x 8 zernikes = 144 jobs)
bash submit_FoVPlot_jobs.sh

# Submit all SkyPlot_vs_secondMoment jobs (7 bands x 3 moments = 21 jobs)
bash submit_SkyPlot_secondMoment_jobs.sh

# Submit animated sky plot jobs (6 bands x 1 moment = 6 jobs)
bash submit_SkyPlot_animated_jobs.sh

# Submit single visit focus gradient jobs (100 visits per job)
bash submit_SingleVisit_FocusGradient_jobs.sh
```

### SLURM Job Management

```bash
# Check job status
squeue -u $USER

# View job log in real-time
tail -f logs/job_name_JOBID.out

# Cancel all your jobs
scancel -u $USER

# Check failed jobs
sacct -u $USER --state=FAILED --starttime=today
```

## Output

- **Plots**: Saved to `plots/` directory (PNG format)
- **Videos**: Animated plots saved as MP4 (requires ffmpeg)
- **Logs**: SLURM logs saved to `logs/` directory

## Dependencies

- LSST Science Pipelines (`lsst_distrib`)
- `treegp` (for spatial averaging | need to work on a dev branch `dev/pleget/meanifyStream`)
- `polars` (for fast parquet reading)
- `skyproj` (for sky projections)
- `hpgeom` (for HEALPix operations)
- `ffmpeg` (for video creation)

On S3DF, load the environment with:
```bash
source /sdf/group/rubin/sw/d_latest/loadLSST.bash
setup lsst_distrib -t d_latest
```

## Directory Structure

```
dp2_psf/
├── data/
│   ├── visit_parquet_mapping.pkl      # Visit ID to parquet file mapping
│   ├── visit_to_band_mapv2.pkl        # Zernike coefficients per visit
│   ├── visit_zernike_corners.pkl      # Zernike at corner positions
│   └── LSST_FP_cold_b_measurement_4col_bysurface.fits  # SLAC height map
├── plots/                             # Output plots and videos
│   └── focus_gradient/                # Single visit focus gradient plots
├── logs/                              # SLURM job logs
├── getData.py                         # Create visit mapping
├── getZernike.ipynb                   # Query Zernike from consdb
├── getZernike_withCorners.ipynb       # Query Zernike with corner positions
├── FoVPlot_vs_secondMoment.py
├── FoVPlot_vs_heightMap.py
├── SkyPlot_vs_secondMoment.py
├── SkyPlot_vs_secondMoment_animated.py
├── SingleVisit_FocusGradient.py       # Per-visit focus gradient analysis
├── submit_*.sh                        # SLURM submission scripts
└── README.md
```
