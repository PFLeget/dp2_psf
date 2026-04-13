#!/usr/bin/env python
"""
Prepare training data for PSF quality CNN classifier.

For each annotated detector:
- Select 9 random stars
- Generate 6 augmented versions per star:
  1. Original
  2. Flip LR
  3. Flip UD
  4. Original + random sub-pixel shift
  5. Flip LR + random sub-pixel shift
  6. Flip UD + random sub-pixel shift

Output: HDF5 file with X (N, 1, 41, 41) and y (N,) arrays for PyTorch.
"""

import numpy as np
import h5py
import pandas as pd
import argparse
import os
from tqdm import tqdm

os.environ["POLARS_MAX_THREADS"] = "1"
import polars

from lsst.daf.butler import Butler
import lsst.geom as geom
import galsim


# Configuration
STAMP_SIZE = 41
STAMP_SIZE_LARGE = 47  # Extra margin for shifting
MAX_SHIFT = 3.0
N_STARS_PER_DETECTOR = 9
SEED = 42

PARQUET_COLUMNS = ['slot_Centroid_x', 'slot_Centroid_y', 'detector']


def shift_image_galsim(image, dx, dy):
    """
    Shift image by sub-pixel amount using GalSim interpolation.

    Parameters
    ----------
    image : np.ndarray
        Input image (larger than final stamp)
    dx, dy : float
        Shift in pixels (positive = shift right/up in image coords)

    Returns
    -------
    np.ndarray
        Shifted and cropped image (STAMP_SIZE x STAMP_SIZE)
    """
    # Create GalSim InterpolatedImage
    gs_image = galsim.Image(image, scale=1.0)
    interp = galsim.InterpolatedImage(gs_image, x_interpolant='lanczos5')

    # Shift
    shifted = interp.shift(dx, dy)

    # Draw onto output stamp
    output = galsim.Image(STAMP_SIZE, STAMP_SIZE, scale=1.0)
    shifted.drawImage(output, method='no_pixel')

    return output.array


def extract_star_stamp(calexp, x, y, stamp_size):
    """Extract a stamp centered on (x, y) from calexp."""
    try:
        position = geom.Point2D(x, y)
        cutout = calexp.getCutout(position, geom.Extent2I(stamp_size, stamp_size))
        stamp = cutout.getMaskedImage().image.array.copy()

        # Normalize
        stamp = stamp / np.sum(stamp)
        return stamp
    except Exception:
        return None


def generate_augmented_stamps(stamp_large, rng):
    """
    Generate 6 augmented versions from a large stamp.

    Returns list of 6 stamps, each STAMP_SIZE x STAMP_SIZE.
    """
    # Center crop for non-shifted versions
    margin = (STAMP_SIZE_LARGE - STAMP_SIZE) // 2
    stamp_center = stamp_large[margin:margin+STAMP_SIZE, margin:margin+STAMP_SIZE]

    # Original
    original = stamp_center.copy()

    # Flips
    flip_lr = np.fliplr(stamp_center).copy()
    flip_ud = np.flipud(stamp_center).copy()

    # Random shifts for each
    dx1, dy1 = rng.uniform(-MAX_SHIFT, MAX_SHIFT, 2)
    dx2, dy2 = rng.uniform(-MAX_SHIFT, MAX_SHIFT, 2)
    dx3, dy3 = rng.uniform(-MAX_SHIFT, MAX_SHIFT, 2)

    # Shifted versions (apply shift then crop)
    original_shifted = shift_image_galsim(stamp_large, dx1, dy1)
    flip_lr_shifted = shift_image_galsim(np.fliplr(stamp_large), dx2, dy2)
    flip_ud_shifted = shift_image_galsim(np.flipud(stamp_large), dx3, dy3)

    return [original, flip_lr, flip_ud, original_shifted, flip_lr_shifted, flip_ud_shifted]


def process_detector(butler, visit, detector, annotation, rng):
    """
    Process one detector: extract 9 random stars, generate augmented stamps.

    Returns list of (stamp, label) tuples.
    """
    dataID = {"instrument": "LSSTCam", "visit": visit, "detector": detector}

    try:
        # Get star positions
        uri = butler.getURI("refit_psf_star", **dataID)
        parquet_path = uri.geturl()
        psf_table = polars.scan_parquet(parquet_path).select(PARQUET_COLUMNS).collect()
        psf_table = psf_table.filter(polars.col("detector") == detector)

        if len(psf_table) < N_STARS_PER_DETECTOR:
            return []

        # Get calexp
        calexp = butler.get("preliminary_visit_image", **dataID)

        # Random sample of stars
        indices = rng.choice(len(psf_table), size=N_STARS_PER_DETECTOR, replace=False)

        results = []
        for idx in indices:
            x = psf_table['slot_Centroid_x'][idx]
            y = psf_table['slot_Centroid_y'][idx]

            # Extract large stamp for shifting
            stamp_large = extract_star_stamp(calexp, x, y, STAMP_SIZE_LARGE)
            if stamp_large is None:
                continue

            # Check for NaNs or bad values
            if not np.isfinite(stamp_large).all():
                continue

            # Generate augmented versions
            augmented = generate_augmented_stamps(stamp_large, rng)

            for stamp in augmented:
                if np.isfinite(stamp).all():
                    results.append((stamp, annotation))

        return results

    except Exception as e:
        print(f"  Error processing visit={visit}, detector={detector}: {e}")
        return []


def display_augmented_samples(all_stamps, all_labels, n_detectors=3):
    """
    Display augmented samples for visual inspection.
    Shows 6 augmented versions for a few stars.
    """
    import matplotlib.pyplot as plt

    # Group by detector (every 9*6=54 stamps is one detector)
    stamps_per_detector = N_STARS_PER_DETECTOR * 6

    fig, axes = plt.subplots(n_detectors * N_STARS_PER_DETECTOR, 6,
                             figsize=(12, 2 * n_detectors * N_STARS_PER_DETECTOR))

    aug_names = ['Original', 'Flip LR', 'Flip UD', 'Orig+Shift', 'LR+Shift', 'UD+Shift']

    for det_idx in range(min(n_detectors, len(all_stamps) // stamps_per_detector)):
        start = det_idx * stamps_per_detector
        label = all_labels[start]
        label_str = {0: 'GOOD', 0.5: 'UNSURE', 1: 'BAD'}.get(label, str(label))

        for star_idx in range(N_STARS_PER_DETECTOR):
            row = det_idx * N_STARS_PER_DETECTOR + star_idx
            for aug_idx in range(6):
                idx = start + star_idx * 6 + aug_idx
                ax = axes[row, aug_idx] if n_detectors * N_STARS_PER_DETECTOR > 1 else axes[aug_idx]

                stamp = all_stamps[idx]
                ax.imshow(stamp, cmap='Greys_r', origin='lower')
                ax.set_xticks([])
                ax.set_yticks([])

                if star_idx == 0 and det_idx == 0:
                    ax.set_title(aug_names[aug_idx], fontsize=9)
                if aug_idx == 0:
                    ax.set_ylabel(f"Det{det_idx+1} S{star_idx+1}\n{label_str}", fontsize=8)

    plt.tight_layout()
    plt.savefig("augmented_samples.png", dpi=150)
    plt.show()
    print("Saved augmented_samples.png")


def main():
    parser = argparse.ArgumentParser(description="Prepare training data for PSF classifier")
    parser.add_argument("--annotations", type=str, required=True,
                       help="CSV file with annotations (visit, detector, annotation)")
    parser.add_argument("--output", type=str, default="training_data.h5",
                       help="Output HDF5 file")
    parser.add_argument("--repoButler", type=str, default="dp2_prep",
                       help="Butler repository")
    parser.add_argument("--collectionButler", type=str,
                       default="LSSTCam/runs/DRP/DP2/v30_0_0/DM-53881/stage2",
                       help="Butler collection")
    parser.add_argument("--seed", type=int, default=SEED,
                       help="Random seed")
    parser.add_argument("--n_detectors", type=int, default=None,
                       help="Process only N detectors (for testing)")
    parser.add_argument("--display", action="store_true",
                       help="Display augmented samples (requires matplotlib)")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # Load annotations
    print(f"Loading annotations from {args.annotations}")
    df = pd.read_csv(args.annotations)
    df["annotation"] = pd.to_numeric(df["annotation"], errors='coerce')

    # Filter to annotated only
    df = df[df["annotation"].notna()].copy()
    print(f"Found {len(df)} annotated detectors")

    # Count by class
    print(f"  Good (0): {(df['annotation'] == 0).sum()}")
    print(f"  Unsure (0.5): {(df['annotation'] == 0.5).sum()}")
    print(f"  Bad (1): {(df['annotation'] == 1).sum()}")

    # Limit to N detectors if requested
    if args.n_detectors is not None:
        df = df.head(args.n_detectors)
        print(f"  -> Limited to {len(df)} detectors for testing")

    # Initialize Butler
    print(f"Connecting to Butler: {args.repoButler}")
    butler = Butler(args.repoButler, collections=args.collectionButler)

    # Process all detectors
    all_stamps = []
    all_labels = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing detectors"):
        visit = int(row['visit'])
        detector = int(row['detector'])
        annotation = float(row['annotation'])

        results = process_detector(butler, visit, detector, annotation, rng)

        for stamp, label in results:
            all_stamps.append(stamp)
            all_labels.append(label)

    # Display augmented samples if requested
    if args.display and len(all_stamps) > 0:
        print("\nDisplaying augmented samples...")
        display_augmented_samples(all_stamps, all_labels, n_detectors=min(3, len(df)))

    # Convert to arrays
    X = np.array(all_stamps, dtype=np.float32)
    y = np.array(all_labels, dtype=np.float32)

    # Reshape for PyTorch: (N, C, H, W)
    X = X[:, np.newaxis, :, :]

    print(f"\nFinal dataset:")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  Labels: good={np.sum(y == 0)}, unsure={np.sum(y == 0.5)}, bad={np.sum(y == 1)}")

    # Save to HDF5
    print(f"\nSaving to {args.output}")
    with h5py.File(args.output, 'w') as f:
        f.create_dataset('X', data=X, compression='gzip', compression_opts=4)
        f.create_dataset('y', data=y, compression='gzip', compression_opts=4)

        # Store metadata
        f.attrs['stamp_size'] = STAMP_SIZE
        f.attrs['n_stars_per_detector'] = N_STARS_PER_DETECTOR
        f.attrs['n_augmentations'] = 6
        f.attrs['max_shift'] = MAX_SHIFT
        f.attrs['seed'] = args.seed
        f.attrs['n_detectors'] = len(df)

    # Print file size
    file_size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"File size: {file_size_mb:.1f} MB")
    print("Done!")


if __name__ == "__main__":
    main()
