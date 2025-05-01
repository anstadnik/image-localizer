# image_localizer/utils.py

from pathlib import Path
from typing import Optional, Tuple, List, Dict, TYPE_CHECKING
from PIL import Image, ExifTags
import numpy as np
import tempfile
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plyfile import PlyData
from pycolmap import Image as ColmapImage, Reconstruction, Camera
from hloc.utils.viz_3d import init_figure, plot_camera_colmap, plot_points
from .localization import PoseResult

# only for static type checking — no runtime import
if TYPE_CHECKING:
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
        return (d.numerator / d.denominator
                + m.numerator / m.denominator / 60
                + s.numerator / s.denominator / 3600)

    lat = conv(gps["GPSLatitude"])
    if gps.get("GPSLatitudeRef", "N") == "S":
        lat = -lat
    lon = conv(gps["GPSLongitude"])
    if gps.get("GPSLongitudeRef", "E") == "W":
        lon = -lon
    return lat, lon


def visualize_retrieval(
    query_image: Path,
    matches: List["MatchResult"],
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
    fig, axes = plt.subplots(1, top_k + 1, figsize=(4 * (top_k + 1), 4))

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


def visualize_localization(
    model: Reconstruction,
    camera: Camera,
    pose: PoseResult,
    work_dir: Path = None,
    max_pts: int = 200_000,
    show_bbox: bool = True,
    bbox_color: str = "yellow",
):
    """
    Export the sparse COLMAP model, plot it in 3D with Plotly,
    overlay your estimated camera frustum and the inlier points,
    and (optionally) draw an axis-aligned box around the inliers.

    Arguments:
      model      – a pycolmap.Reconstruction
      camera     – the pycolmap.Camera you used for localization
      pose       – the PoseResult from localize_image()
      work_dir   – where to write the temporary PLY (defaults to a temp dir)
      max_pts    – maximum number of map points to sample for speed
      show_bbox  – whether to draw an AABB around the inlier cloud
      bbox_color – line‐color for the bounding‐box edges
    """
    # 0) dump to PLY
    if work_dir is None:
        work_dir = Path(tempfile.gettempdir()) / "image_localizer_viz"
    work_dir.mkdir(exist_ok=True, parents=True)
    ply_path = work_dir / "model.ply"
    model.export_PLY(str(ply_path))

    # 1) read and sample the colored point cloud
    ply = PlyData.read(str(ply_path))
    v   = ply["vertex"]
    coords = np.vstack([v["x"], v["y"], v["z"]]).T
    colors = np.vstack([v["red"], v["green"], v["blue"]]).T
    n_show = min(len(coords), max_pts)
    idx    = np.random.choice(len(coords), n_show, replace=False)
    xyz    = coords[idx]
    rgb    = [f"rgb({r},{g},{b})" for r,g,b in colors[idx]]

    # 2) init figure and plot map
    fig = init_figure()
    fig.add_trace(go.Scatter3d(
        x=xyz[:,0], y=xyz[:,1], z=xyz[:,2],
        mode="markers",
        marker=dict(size=1, color=rgb, opacity=0.6),
        name="sparse map"
    ))

    # 3) estimated camera (world→cam)
    cam = ColmapImage(
        cam_from_world=pose.cam_from_world
    )
    plot_camera_colmap(
        fig, cam, camera,
        color="rgba(0,255,0,0.5)",
        name="estimated camera",
        fill=True
    )

    # 4) inlier 3D points
    pts3d = np.array([model.points3D[p].xyz
                      for p,k in zip(pose.points3D_ids, pose.inlier_mask) if k])
    plot_points(fig, pts3d, color="cyan", ps=5, name="inliers")

    # 5) optional bounding‐box
    if show_bbox and len(pts3d):
        mn, mx = pts3d.min(axis=0), pts3d.max(axis=0)
        corners = np.array([[x,y,z]
            for x in (mn[0],mx[0])
            for y in (mn[1],mx[1])
            for z in (mn[2],mx[2])])
        edges = [
            (0,1),(0,2),(1,3),(2,3),
            (4,5),(4,6),(5,7),(6,7),
            (0,4),(1,5),(2,6),(3,7),
        ]
        for i,j in edges:
            fig.add_trace(go.Scatter3d(
                x=[corners[i,0], corners[j,0]],
                y=[corners[i,1], corners[j,1]],
                z=[corners[i,2], corners[j,2]],
                mode="lines",
                line=dict(color=bbox_color, width=4),
                showlegend=False
            ))

    # 6) polish camera view
    fig.update_layout(scene=dict(
        bgcolor="white",
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
    ))
    fig.show()
