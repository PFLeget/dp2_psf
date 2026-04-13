#!/usr/bin/env python
"""
Merge training data chunks into a single HDF5 file.

Usage:
    python merge_training_data.py --input_dir training_data_chunks --output training_data.h5
"""

import numpy as np
import h5py
import glob
import os
import argparse


def main():
    parser = argparse.ArgumentParser(description="Merge training data chunks")
    parser.add_argument("--input_dir", type=str, required=True,
                       help="Directory containing chunk HDF5 files")
    parser.add_argument("--output", type=str, default="training_data.h5",
                       help="Output merged HDF5 file")
    parser.add_argument("--shuffle", action="store_true", default=True,
                       help="Shuffle the merged data")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for shuffling")
    args = parser.parse_args()

    # Find all chunk files
    chunk_files = sorted(glob.glob(os.path.join(args.input_dir, "training_data_chunk_*.h5")))
    print(f"Found {len(chunk_files)} chunk files")

    if len(chunk_files) == 0:
        print("No chunk files found!")
        return

    # Load all chunks
    all_X = []
    all_y = []

    for chunk_file in chunk_files:
        print(f"  Loading {os.path.basename(chunk_file)}...", end=" ")
        with h5py.File(chunk_file, 'r') as f:
            X = f['X'][:]
            y = f['y'][:]
            all_X.append(X)
            all_y.append(y)
            print(f"{len(X)} samples")

    # Concatenate
    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    print(f"\nTotal: {len(X)} samples")

    # Shuffle if requested
    if args.shuffle:
        print("Shuffling...")
        rng = np.random.default_rng(args.seed)
        indices = rng.permutation(len(X))
        X = X[indices]
        y = y[indices]

    # Print class distribution
    print(f"\nClass distribution:")
    print(f"  Good (0):    {np.sum(y == 0)}")
    print(f"  Unsure (0.5): {np.sum(y == 0.5)}")
    print(f"  Bad (1):     {np.sum(y == 1)}")

    # Save merged file
    print(f"\nSaving to {args.output}")
    with h5py.File(args.output, 'w') as f:
        f.create_dataset('X', data=X, compression='gzip', compression_opts=4)
        f.create_dataset('y', data=y, compression='gzip', compression_opts=4)

        # Store metadata
        f.attrs['n_samples'] = len(X)
        f.attrs['stamp_size'] = X.shape[2]
        f.attrs['n_chunks_merged'] = len(chunk_files)

    file_size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"File size: {file_size_mb:.1f} MB")
    print("Done!")


if __name__ == "__main__":
    main()
