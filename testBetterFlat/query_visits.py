#!/usr/bin/env python
"""Query visits for flat A/B testing: detector 87, g-band, |b| > 30, after Nov 2025, limit 5000."""

from lsst.daf.butler import Butler
from astropy.coordinates import SkyCoord
import astropy.units as u

butler = Butler('/repo/main')

# Query visits after Nov 2025 for g-band
print("Querying visits...")
refs = list(butler.registry.queryDimensionRecords(
    "visit",
    where="instrument='LSSTCam' AND visit.day_obs > 20251101 AND band='g'",
))
print(f"Found {len(refs)} g-band visits after Nov 2025")

# Filter by |Galactic latitude| > 30
visits_filtered = []
for ref in refs:
    ra = ref.ra
    dec = ref.dec
    coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')
    galactic = coord.galactic
    if abs(galactic.b.deg) > 30:
        visits_filtered.append(ref.id)

print(f"Found {len(visits_filtered)} visits with |b| > 30")

# Limit to 5000
visits_filtered = visits_filtered[:5000]
print(f"Using {len(visits_filtered)} visits (limit 5000)")

# Save to file
with open("visitIds_flat_test.txt", "w") as f:
    for v in visits_filtered:
        f.write(f"{v}\n")

print(f"Saved visit IDs to visitIds_flat_test.txt")
