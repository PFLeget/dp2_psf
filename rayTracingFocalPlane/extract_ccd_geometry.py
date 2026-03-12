#!/usr/bin/env python
"""
Extract CCD geometry (center positions) from LSST camera.
Run this at SLAC and copy the output file back.
"""

import numpy as np
from lsst.obs.lsst import LsstCam
import lsst.afw.cameraGeom as cameraGeom

camera = LsstCam.getCamera()

# Extract detector info
data = []
for det in camera:
    det_id = det.getId()
    name = det.getName()

    # Get detector center in focal plane coordinates (mm)
    # Using the center pixel transformed to FOCAL_PLANE
    bbox = det.getBBox()
    center_pix_x = (bbox.getMinX() + bbox.getMaxX()) / 2
    center_pix_y = (bbox.getMinY() + bbox.getMaxY()) / 2

    transform = det.getTransform(cameraGeom.PIXELS, cameraGeom.FOCAL_PLANE)
    center_fp = transform.applyForward([center_pix_x, center_pix_y])

    # Get corners to determine orientation/size
    corners_pix = [
        (bbox.getMinX(), bbox.getMinY()),
        (bbox.getMaxX(), bbox.getMinY()),
        (bbox.getMaxX(), bbox.getMaxY()),
        (bbox.getMinX(), bbox.getMaxY()),
    ]
    corners_fp = [transform.applyForward(c) for c in corners_pix]

    data.append({
        'detector': det_id,
        'name': name,
        'x_center': center_fp[0],
        'y_center': center_fp[1],
        'corner0_x': corners_fp[0][0],
        'corner0_y': corners_fp[0][1],
        'corner1_x': corners_fp[1][0],
        'corner1_y': corners_fp[1][1],
        'corner2_x': corners_fp[2][0],
        'corner2_y': corners_fp[2][1],
        'corner3_x': corners_fp[3][0],
        'corner3_y': corners_fp[3][1],
    })

# Save to CSV
import csv
with open('ccd_geometry.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=data[0].keys())
    writer.writeheader()
    writer.writerows(data)

print(f"Saved {len(data)} detectors to ccd_geometry.csv")
print("Copy this file back to rayTracingFocalPlane/data/")
