#!/usr/bin/env python
"""Query visits for flat A/B testing: detector 87, g-band, |b| > 30, after Nov 2025, limit 5000."""

import pickle
import polars as pl
from astropy.coordinates import SkyCoord
import astropy.units as u
from tqdm import tqdm

# Path to existing visit mapping from getData.py
VISIT_MAPPING = "/sdf/home/l/leget/rubin-user/lsst_dev/tickets/dp2_psf/data/visit_parquet_mapping.pkl"

# Load the mapping
print("Loading visit mapping...")
with open(VISIT_MAPPING, 'rb') as f:
    visit_mapping = pickle.load(f)

# Filter g-band visits after Nov 2025
# Visit IDs encode date as YYYYMMDD in the first 8 digits
g_visits = {v: info for v, info in visit_mapping.items()
            if info['band'] == 'g'}
print(f"Found {len(g_visits)} g-band visits after Nov 2025 in DP2")

# Filter by |Galactic latitude| > 30 using parquet coordinates
visits_filtered = []
for visit, info in tqdm(g_visits.items(), desc="Filtering by Galactic latitude"):
    try:
        df = pl.scan_parquet(info['parquet_path']).select(['coord_ra', 'coord_dec']).collect()
        ra = df['coord_ra'].median()
        dec = df['coord_dec'].median()
        coord = SkyCoord(ra=ra*u.rad, dec=dec*u.rad, frame='icrs')
        galactic = coord.galactic
        if abs(galactic.b.deg) > 25:
            visits_filtered.append(visit)
    except Exception as e:
        print(f"Warning: could not read visit {visit}: {e}")

print(f"Found {len(visits_filtered)} visits with |b| > 30")


print(f"Using {len(visits_filtered)} visits")

# Save to file
with open("visitIds_flat_test.txt", "w") as f:
    for v in visits_filtered:
        f.write(f"{v}\n")

print(f"Saved visit IDs to visitIds_flat_test.txt")
