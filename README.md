# image-localizer

A Python package for image-based localization.  It implements a full pipeline:
[HLoc](https://github.com/cvg/Hierarchical-Localization)  

It combines:

1. **NetVLAD** global feature extraction  
2. **FAISS** retrieval (cosine similarity)  
3. **2D visualization** of query vs. retrieved images  
4. **SuperPoint + LightGlue** local feature extraction & matching  
5. **COLMAP RANSAC + PnP + refinement** for pose estimation  
6. **3D visualization** (Plotly) with inlier bounding box  

---

## Installation

1. Clone and install in editable mode:

   ```bash
   git clone https://github.com/dbakshinska/image-localizer.git
   cd image-localizer
   pip install -e .

---
## Camera Intrinsics

By default we assume a DJI Mavic Pro sensor (½.3″) with these pinhole parameters:

```python
# used in examples/localize, fx≈4.49/6.17·W, fy≈4.49/4.55·H
fx = 4.49/6.17 * W  
fy = 4.49/4.55 * H  
cx, cy = W/2, H/2  
```
W, H : image width & height in pixels

4.49 mm : horizontal focal length of the Mavic camera

6.17 mm, 4.55 mm : Mavic sensor size (w, h in mm)

---

## Notes on FAISS dependencies
On Linux, faiss-cpu is installed automatically.
```bash
pip install faiss-cpu
```
On macOS, prebuilt FAISS wheels are not always available. We recommend:
```bash
conda install -c conda-forge faiss-cpu
```
## Project Structure
``` graphql
image_localizer/
├── netvlad.py          # extract_netvlad()
├── retrieval.py        # build_faiss_index(), query_faiss(), retrieve_image_matches()
├── localization.py     # localize_image(), PoseResult
├── utils.py            # get_latlon(), visualize_retrieval(), compute_bbox()
└── __main__.py         # CLI entry point (`image-localizer`)
```

## Quickstart Example
Below is a self-contained Jupyter-style workflow. Adapt paths to your own data layout.

## 1) Extract NetVLAD Descriptors
```python
from pathlib import Path
from image_localizer.netvlad import extract_netvlad

images_dir = Path("/path/to/drone_images")
output_h5  = Path("/path/to/output/netvlad_descriptors.h5") #directory where you would like to save the descriptors to

extract_netvlad(
    image_dir  = images_dir,
    output_h5  = output_h5,
    image_list = None,    # None → process all images in `images_dir`
    overwrite  = True
)

print("Wrote:", output_h5, "Contains:", list(output_h5.open("rb").read()[:10]))
```
## 2) FAISS retrieval + visualize
``` python
from pathlib import Path
from image_localizer.retrieval import retrieve_image_matches
from image_localizer.utils     import visualize_retrieval

# 1) query lives in its own folder
query_jpg = Path("/path/to/query_folder/query.jpg")

# 2) NetVLAD database H5 + image root
db_h5   = Path("/path/to/drone_images_netvlad.h5")
img_dir = Path("/path/to/drone_images")
image_map = { db_h5: img_dir }

# 3) run retrieval
matches = retrieve_image_matches(
    query_image = query_jpg,
    db_h5_list  = [db_h5],
    image_map   = image_map,
    top_k       = 5,
)

# 4) print & visualize
for m in matches:
    print(f"{m.image_name:30s}  score={m.score:.4f}  lat={m.lat}  lon={m.lon}")

visualize_retrieval(
    query_image = query_jpg,
    matches     = matches,
    image_map   = image_map,
    top_k       = 5,
)
```
## 3) Localize with SuperPoint + LightGlue + COLMAP

``` python
from pathlib import Path
from image_localizer.localization import localize_image, PoseResult
from image_localizer.utils      import compute_bbox

sfm_dir         = Path("/path/to/colmap/model")   # contains cameras.bin, images.bin, points3D.bin
db_img_dir      = Path("/path/to/drone_images")
retrieved_names = [m.image_name for m in matches]

pose: PoseResult = localize_image(
    query_image     = query_jpg,
    sfm_dir         = sfm_dir,
    retrieved_names = retrieved_names,
    image_dirs      = [db_img_dir],
    local_ext       = "superpoint_max",
    match_method    = "disk+lightglue",
    ransac_err      = 200.0,
)

print("Camera pose (world→cam):")
print(pose.cam_from_world)
print(f"Inliers: {pose.inlier_mask.sum()} / {len(pose.inlier_mask)}")

# 4D) Compute 3D axis‐aligned bounding box of inliers
inlier_ids = [pid for pid, ok in zip(pose.points3D_ids, pose.inlier_mask) if ok]
mins, maxs = compute_bbox(inlier_ids, sfm_dir)
print("3D AABB min:", mins)
print("3D AABB max:", maxs)
```

## 4) 3D Visualization (Plotly)
``` python
import numpy as np
import plotly.graph_objects as go
from plyfile import PlyData
import pycolmap
from hloc.utils.viz_3d import init_figure, plot_camera_colmap, plot_points

# a) Export your COLMAP model to PLY
model   = pycolmap.Reconstruction(str(sfm_dir))
tmp_ply = "tmp_model.ply"
model.export_PLY(tmp_ply)

# b) Load & downsample
plydata = PlyData.read(tmp_ply)
v       = plydata['vertex']
coords  = np.vstack([v['x'], v['y'], v['z']]).T
colors  = np.vstack([v['red'], v['green'], v['blue']]).T
idx     = np.random.choice(len(coords), size=min(200_000, len(coords)), replace=False)
xyz_s   = coords[idx]
rgb_s   = [f"rgb({r},{g},{b})" for r,g,b in colors[idx]]

# c) Initialize figure & plot map
fig = init_figure()
fig.add_trace(go.Scatter3d(
    x=xyz_s[:,0], y=xyz_s[:,1], z=xyz_s[:,2],
    mode="markers",
    marker=dict(size=1, color=rgb_s, opacity=0.6),
    name="sparse map"
))

# d) Prepare camera & intrinsics
from PIL import Image
W, H = Image.open(query_jpg).size
fx, fy = 4.49/6.17 * W, 4.49/4.55 * H
cx, cy = W/2, H/2
cam = pycolmap.Image(cam_from_world=pose.cam_from_world)
colcam = pycolmap.Camera(model="PINHOLE", width=W, height=H, params=[fx, fy, cx, cy])

# e) Plot estimated camera
plot_camera_colmap(fig, cam, colcam, color="rgba(0,255,0,0.5)", name="camera", fill=True)

# f) Plot inlier points
inlier_pts = np.vstack([model.points3D[p].xyz for p in inlier_ids])
plot_points(fig, inlier_pts, color="cyan", ps=4, name="inliers")

# g) Draw AABB edges
corners = np.array([
    [mins[0], mins[1], mins[2]], [mins[0], mins[1], maxs[2]],
    [mins[0], maxs[1], mins[2]], [mins[0], maxs[1], maxs[2]],
    [maxs[0], mins[1], mins[2]], [maxs[0], mins[1], maxs[2]],
    [maxs[0], maxs[1], mins[2]], [maxs[0], maxs[1], maxs[2]],
])
edges = [(0,1),(0,2),(1,3),(2,3),(4,5),(4,6),(5,7),(6,7),(0,4),(1,5),(2,6),(3,7)]
for i,j in edges:
    fig.add_trace(go.Scatter3d(
        x=[corners[i,0], corners[j,0]],
        y=[corners[i,1], corners[j,1]],
        z=[corners[i,2], corners[j,2]],
        mode="lines", line=dict(color="yellow", width=4), showlegend=False
    ))

fig.update_layout(scene=dict(bgcolor="white", camera=dict(eye=dict(x=1.5,y=1.5,z=1.5))))
fig.show()
```

## License

This project is licensed under the MIT License – see the  
[LICENSE](LICENSE) file for details.
