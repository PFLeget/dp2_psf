# Flat Field A/B Testing

This directory contains scripts for A/B testing different flat field calibrations on PSF shape residuals.

## Overview

The goal is to compare PSF fitting results between:
- **A (control)**: Default flat from `LSSTCam/defaults`
- **B (test)**: New flat from `u/tguillem/test_LED_weights/flat_g_sky_w0381`

The test runs ISR + calibrateImage on detector 87 for all g-band visits in DP2, then compares the PSF shape residuals (dT/T, de1, de2) between the two flat field calibrations.

## Prerequisites

- Access to SLAC `/repo/main` Butler repository
- DP2 visit mapping file at `/sdf/home/l/leget/rubin-user/lsst_dev/tickets/dp2_psf/data/visit_parquet_mapping.pkl`
- BPS configuration at `/sdf/group/rubin/user/leget/batch/bps_generic_main.yaml`

## Step 1: Generate visit list

```bash
python query_visits.py
```

This reads the DP2 visit mapping and extracts all g-band visit IDs to `visitIds_flat_test.txt`.

## Step 2: Submit BPS jobs

Submit both A/B test jobs:

```bash
# Default flat (control)
bash bps_default.sh

# New flat (test)
bash bps_newflat.sh
```

Output collections:
- A: `u/leget/LSSTCam/testFlat/ABtest_default`
- B: `u/leget/LSSTCam/testFlat/ABtest_newflat`

## Step 3: Analyze results

After BPS jobs complete, run the analysis script:

```bash
python FoVPlot_ABtest.py \
    --detector 87 \
    --visit_file visitIds_flat_test.txt \
    --collection_A u/leget/LSSTCam/testFlat/ABtest_default \
    --collection_B u/leget/LSSTCam/testFlat/ABtest_newflat \
    --key_second_moment dT_T \
    --repOutPlot plots/
```

### Analysis options

| Option | Description |
|--------|-------------|
| `--key_second_moment` | PSF residual to analyze: `dT_T`, `de1`, `de2` |
| `--bin_spacing` | Spatial binning in pixels (default: 150) |
| `--colorScale` | Fixed color scale range (default: 0.005) |
| `--autoColorScale` | Compute color scale from data (A sets scale for both) |
| `--autoColorScaleCst` | Number of sigma for auto scale (default: 2.0) |
| `--statisticsMedian` | Use median instead of mean for binning |

### Output

The script produces:
- `dT_T_det87_default.png` - Spatial map for default flat
- `dT_T_det87_newflat.png` - Spatial map for new flat
- `dT_T_det87_comparison.png` - Side-by-side comparison with difference map
- Corresponding `.pkl` files with the binned data

## Files

| File | Description |
|------|-------------|
| `query_visits.py` | Extract g-band visit IDs from DP2 mapping |
| `bps_default.sh` | BPS submit script for default flat |
| `bps_newflat.sh` | BPS submit script for new flat |
| `FoVPlot_ABtest.py` | Analysis script for comparing PSF residuals |
| `visitIds_flat_test.txt` | Generated list of visit IDs |
