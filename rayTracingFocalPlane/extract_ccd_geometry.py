#!/usr/bin/env python
"""
Extract CCD geometry (center positions) from LSST camera.
Run this at SLAC and copy the output file back.
"""

import numpy as np
from lsst.obs.lsst import LsstCam
import lsst.afw.cameraGeom as cameraGeom
import lsst.geom as geom

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
    center_fp = transform.applyForward(geom.Point2D(center_pix_x, center_pix_y))

    # Get corners to determine orientation/size
    corners_pix = [
        geom.Point2D(bbox.getMinX(), bbox.getMinY()),
        geom.Point2D(bbox.getMaxX(), bbox.getMinY()),
        geom.Point2D(bbox.getMaxX(), bbox.getMaxY()),
        geom.Point2D(bbox.getMinX(), bbox.getMaxY()),
    ]
    corners_fp = [transform.applyForward(c) for c in corners_pix]

    data.append({
        'detector': det_id,
        'name': name,
        'x_center': center_fp.getX(),
        'y_center': center_fp.getY(),
        'corner0_x': corners_fp[0].getX(),
        'corner0_y': corners_fp[0].getY(),
        'corner1_x': corners_fp[1].getX(),
        'corner1_y': corners_fp[1].getY(),
        'corner2_x': corners_fp[2].getX(),
        'corner2_y': corners_fp[2].getY(),
        'corner3_x': corners_fp[3].getX(),
        'corner3_y': corners_fp[3].getY(),
    })

# Save to CSV
import csv
with open('ccd_geometry.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=data[0].keys())
    writer.writeheader()
    writer.writerows(data)

print(f"Saved {len(data)} detectors to ccd_geometry.csv")
print("Copy this file back to rayTracingFocalPlane/data/")
