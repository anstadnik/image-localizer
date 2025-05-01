# image_localizer/utils.py

from pathlib import Path
from typing import Optional, Tuple
from PIL import Image, ExifTags
from pycolmap import Reconstruction
import numpy as np

def get_latlon(img_path: Path) -> Optional[Tuple[float, float]]:
    """
    Read a JPEG/PNG’s EXIF GPS tags and return (lat, lon) in decimal degrees,
    or None if no GPS info is found.
    """
    img = Image.open(img_path)
    exif = img._getexif() or {}
    gps_info = {}
    # find the GPSInfo tag
    for tag_id, val in exif.items():
        tag = ExifTags.TAGS.get(tag_id)
        if tag == "GPSInfo":
            for key, v in val.items():
                gps_info[ExifTags.GPSTAGS.get(key)] = v

    if ("GPSLatitude" not in gps_info) or ("GPSLongitude" not in gps_info):
        return None

    def _conv(ratios):
        d, m, s = ratios
        return (d.numerator / d.denominator
                + m.numerator / m.denominator / 60
                + s.numerator / s.denominator / 3600)

    lat = _conv(gps_info["GPSLatitude"])
    if gps_info.get("GPSLatitudeRef", "N") == "S":
        lat = -lat

    lon = _conv(gps_info["GPSLongitude"])
    if gps_info.get("GPSLongitudeRef", "E") == "W":
        lon = -lon

    return lat, lon


def compute_bbox(
    points3D_ids: list,
    sfm_dir: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given a list of 3D‐point IDs and the COLMAP model directory,
    returns (mins, maxs) corner coordinates of the AABB.
    """
    model = Reconstruction(sfm_dir)
    pts = np.vstack([model.points3D[p].xyz for p in points3D_ids])
    return pts.min(axis=0), pts.max(axis=0)
