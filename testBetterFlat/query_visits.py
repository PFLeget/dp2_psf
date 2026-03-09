#!/usr/bin/env python
"""Query g-band visits from DP2 for flat field A/B testing."""

import pickle
from tqdm import tqdm

# Path to existing visit mapping from getData.py
VISIT_MAPPING = "/sdf/home/l/leget/rubin-user/lsst_dev/tickets/dp2_psf/data/visit_parquet_mapping.pkl"

# Load the mapping
print("Loading visit mapping...")
with open(VISIT_MAPPING, 'rb') as f:
    visit_mapping = pickle.load(f)

# Filter g-band visits
g_visits = [v for v, info in visit_mapping.items() if info['band'] == 'g']
print(f"Found {len(g_visits)} g-band visits in DP2")

# Save to file
with open("visitIds_flat_test.txt", "w") as f:
    for v in g_visits:
        f.write(f"{v}\n")

print(f"Saved {len(g_visits)} visit IDs to visitIds_flat_test.txt")
