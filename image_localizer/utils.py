# image_localizer/utils.py

from pathlib import Path
from typing import Optional, Tuple, List, Dict
from PIL import Image, ExifTags
import numpy as np
import matplotlib.pyplot as plt
from pycolmap import Reconstruction

from .retrieval import MatchResult


def get_latlon(img_path: Path) -> Optional[Tuple[float, float]]:
    """
    Read a JPEG/PNG’s EXIF GPS tags and return (lat, lon) in decimal degrees,
    or None if no GPS info is found.
    """
    img = Image.open(img_path)
    exif = img._getexif() or {}
    gps = {}
    for tag_id, val in exif.items():
        tag = ExifTags.TAGS.get(tag_id)
        if tag == "GPSInfo":
            for t, v in val.items():
                gps[ExifTags.GPSTAGS.get(t)] = v

    if "GPSLatitude" not in gps or "GPSLongitude" not in gps:
        return None

    def conv(rats):
        d, m, s = rats
        return (d.numerator/d.denominator
                + m.numerator/m.denominator/60
                + s.numerator/s.denominator/3600)

    lat = conv(gps["GPSLatitude"])
    if gps.get("GPSLatitudeRef", "N") == "S":
        lat = -lat
    lon = conv(gps["GPSLongitude"])
    if gps.get("GPSLongitudeRef", "E") == "W":
        lon = -lon
    return lat, lon


def visualize_retrieval(
    query_image: Path,
    matches: List[MatchResult],
    image_map: Dict[Path, Path],
    top_k: int = 5,
):
    """
    Display the query and its top_k retrieved images side-by-side.
    
    - query_image: path to your query JPG/PNG
    - matches:     list of MatchResult from retrieve_image_matches()
    - image_map:   maps each `h5_path` → its image directory (Path)
    - top_k:       how many matches to show (will take first top_k of `matches`)
    """
    # Prepare a flat list of all possible image directories for fallback
    all_dirs = list({d for d in image_map.values()})

    # 1) create figure
    fig, axes = plt.subplots(1, top_k+1, figsize=(4*(top_k+1), 4))
    # 2) plot query
    axes[0].imshow(Image.open(query_image))
    axes[0].set_title("Query")
    axes[0].axis("off")

    # 3) plot each match
    for col, m in enumerate(matches[:top_k], start=1):
        # primary lookup
        img_dir = image_map.get(m.h5_path)
        img_path = img_dir and (img_dir / m.image_name)
        # fallback recursive search
        if not img_path or not img_path.exists():
            for base in all_dirs:
                candidates = list(base.rglob(f"{Path(m.image_name).stem}.*"))
                if candidates:
                    img_path = candidates[0]
                    break

        ax = axes[col]
        if img_path and img_path.exists():
            ax.imshow(Image.open(img_path))
            title = f"{m.image_name}\n{m.score:.3f}"
            if m.lat is not None and m.lon is not None:
                title += f"\n{m.lat:.6f}, {m.lon:.6f}"
            ax.set_title(title)
        else:
            ax.text(
                0.5, 0.5,
                f"Missing\n{m.image_name}",
                ha="center", va="center", fontsize=12
            )
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def compute_bbox(
    points3D_ids: list,
    sfm_dir: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given 3D‐point IDs and a COLMAP model dir, return (mins, maxs) of the AABB.
    """
    model = Reconstruction(sfm_dir)
    pts = np.vstack([model.points3D[p].xyz for p in points3D_ids])
    return pts.min(axis=0), pts.max(axis=0)
