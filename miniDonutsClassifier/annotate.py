#!/usr/bin/env python
"""
Fast image annotation tool for PSF quality classification.

Keyboard shortcuts (displayed on screen):
    0 = good
    8 = not sure (0.5)
    9 = bad (donuts/weird shapes)
    s = skip (leave as None)
    b = go back to previous
    q = quit (auto-saves)

Output: annotations.csv with columns: visit, detector, annotation
"""

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path


class Annotator:
    def __init__(self, image_dir, output_csv="annotations.csv"):
        self.image_dir = image_dir
        self.output_csv = output_csv

        # Get all images
        self.images = sorted(glob.glob(os.path.join(image_dir, "*.png")))
        print(f"Found {len(self.images)} images")

        # Load or create annotations CSV
        if os.path.exists(output_csv):
            self.df = pd.read_csv(output_csv)
            # Convert annotation column to numeric (handles empty strings as NaN)
            self.df["annotation"] = pd.to_numeric(self.df["annotation"], errors='coerce')
            print(f"Loaded {len(self.df)} existing annotations")
        else:
            # Parse all image filenames to build initial dataframe
            records = []
            for img_path in self.images:
                visit, detector, flag = self._parse_filename(img_path)
                records.append({
                    "visit": visit,
                    "detector": detector,
                    "annotation": None,
                    "filename": os.path.basename(img_path)
                })
            self.df = pd.DataFrame(records)
            self.save()
            print(f"Created new annotations file with {len(self.df)} entries")

        # Find first unannotated
        self.current_idx = self._find_first_unannotated()

        # Setup matplotlib
        self.fig, self.ax = plt.subplots(figsize=(8, 9))
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

    def _parse_filename(self, filepath):
        """Parse visit_detector_flag.png filename."""
        basename = os.path.basename(filepath)
        name = basename.replace(".png", "")
        parts = name.split("_")
        visit = int(parts[0])
        detector = int(parts[1])
        return visit, detector, "_".join(parts[2:])

    def _find_first_unannotated(self):
        """Find index of first unannotated image."""
        unannotated = self.df[self.df["annotation"].isna()]
        if len(unannotated) == 0:
            return len(self.df) - 1  # All done, show last
        return unannotated.index[0]

    def save(self):
        """Save annotations to CSV with clean formatting."""
        df_out = self.df.copy()
        # Convert 0.0 -> 0, 1.0 -> 1, keep 0.5 as is, keep NaN as empty
        def format_annotation(x):
            if pd.isna(x):
                return ""
            elif x == 0:
                return "0"
            elif x == 1:
                return "1"
            else:
                return str(x)
        df_out["annotation"] = df_out["annotation"].apply(format_annotation)
        df_out.to_csv(self.output_csv, index=False)

    def _show_image(self):
        """Display current image with info and shortcuts."""
        self.ax.clear()

        if self.current_idx >= len(self.df):
            self.ax.text(0.5, 0.5, "All images annotated!",
                        ha='center', va='center', fontsize=20)
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            self.fig.canvas.draw()
            return

        row = self.df.iloc[self.current_idx]
        img_path = os.path.join(self.image_dir, row["filename"])

        if os.path.exists(img_path):
            img = mpimg.imread(img_path)
            self.ax.imshow(img)
        else:
            self.ax.text(0.5, 0.5, f"Image not found:\n{row['filename']}",
                        ha='center', va='center', fontsize=12)

        self.ax.set_xticks([])
        self.ax.set_yticks([])

        # Progress stats
        n_annotated = self.df["annotation"].notna().sum()
        n_total = len(self.df)
        n_bad = (self.df["annotation"] == 1).sum()
        n_unsure = (self.df["annotation"] == 0.5).sum()
        n_good = (self.df["annotation"] == 0).sum()

        # Title with progress
        current_ann = row["annotation"]
        if pd.isna(current_ann):
            ann_str = "None"
        elif current_ann == 1:
            ann_str = "BAD"
        elif current_ann == 0.5:
            ann_str = "UNSURE"
        else:
            ann_str = "GOOD"
        title = f"[{self.current_idx + 1}/{n_total}] Visit {row['visit']} | Det {row['detector']} | Current: {ann_str}\n"
        title += f"Progress: {n_annotated}/{n_total} ({100*n_annotated/n_total:.1f}%) | Good: {n_good} | Unsure: {n_unsure} | Bad: {n_bad}"
        self.ax.set_title(title, fontsize=10)

        # Keyboard shortcuts at bottom
        shortcuts = "Keys:  [0] Good    [8] Unsure    [9] Bad    [s] Skip    [b] Back    [q] Quit"
        self.fig.text(0.5, 0.02, shortcuts, ha='center', va='bottom', fontsize=11,
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        self.fig.canvas.draw()

    def _on_key(self, event):
        """Handle keyboard input."""
        if event.key == '9':
            # Bad
            self.df.loc[self.current_idx, "annotation"] = 1
            self.save()
            self.current_idx += 1
            self._show_image()

        elif event.key == '8':
            # Unsure
            self.df.loc[self.current_idx, "annotation"] = 0.5
            self.save()
            self.current_idx += 1
            self._show_image()

        elif event.key == '0':
            # Good
            self.df.loc[self.current_idx, "annotation"] = 0
            self.save()
            self.current_idx += 1
            self._show_image()

        elif event.key == 's':
            # Skip
            self.current_idx += 1
            self._show_image()

        elif event.key == 'b':
            # Back
            if self.current_idx > 0:
                self.current_idx -= 1
            self._show_image()

        elif event.key == 'q':
            # Quit
            self.save()
            plt.close(self.fig)
            print(f"\nSaved {self.df['annotation'].notna().sum()} annotations to {self.output_csv}")

    def run(self):
        """Start the annotation session."""
        n_annotated = self.df["annotation"].notna().sum()
        print(f"Starting at image {self.current_idx + 1}/{len(self.df)}")
        print(f"Already annotated: {n_annotated}")
        print("Close window or press 'q' to quit (progress auto-saved)")

        self._show_image()
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.08)
        plt.show()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Annotate PSF images")
    parser.add_argument("--image_dir", type=str, default="plotForAnnotation",
                       help="Directory containing images")
    parser.add_argument("--output", type=str, default="annotations.csv",
                       help="Output CSV file")
    args = parser.parse_args()

    annotator = Annotator(args.image_dir, args.output)
    annotator.run()


if __name__ == "__main__":
    main()
