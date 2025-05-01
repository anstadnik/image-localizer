# image_localizer/localization.py

from pathlib import Path
from typing import List, NamedTuple, Optional
import numpy as np
from PIL import Image

from pycolmap import Camera, Reconstruction
from hloc import extract_features, match_features
from hloc.localize_sfm import QueryLocalizer, pose_from_cluster


class PoseResult(NamedTuple):
    cam_from_world: np.ndarray      # 4×4 pose
    inlier_mask: np.ndarray         # boolean mask of inliers
    points3D_ids: List[int]         # all candidate 3D‐point IDs


def localize_image(
    query_image: Path,
    sfm_dir: Path,
    retrieved_names: List[str],
    image_dirs: List[Path],
    local_ext: str       = "superpoint_max",
    match_method: str    = "disk+lightglue",
    ransac_err: float    = 200.0,
    refine_focal: bool   = False,
    refine_extra: bool   = True,
    work_dir: Optional[Path] = None,
) -> PoseResult:
    """
    End-to-end localization:
      1) extract local features with SuperPoint
      2) match with LightGlue
      3) RANSAC + PnP + refinement via COLMAP

    Args:
      query_image:     Path to your query JPG/PNG
      sfm_dir:         folder with cameras.bin, images.bin, points3D.bin
      retrieved_names: top-K image filenames from retrieval.query_faiss()
      image_dirs:      list of root folders to look for those images
      local_ext:       e.g. "superpoint_max"
      match_method:    e.g. "disk+lightglue"
      work_dir:        scratch folder (will be created if needed)
    """

    # — 0) prepare a little working folder
    if work_dir is None:
        work_dir = Path(".hloc_tmp")
    work_dir.mkdir(exist_ok=True)

    # — 1) write the pairs file for HLoc matching
    pairs_file = work_dir / "pairs.txt"
    with open(pairs_file, "w") as f:
        for ref in retrieved_names:
            f.write(f"{query_image.name} {ref}\n")

    # — 2) EXTRACT local features (superpoint)
    feats_h5 = work_dir / "features.h5"

    # LightGlue expects the config key "superpoint", so alias if needed
    if local_ext.startswith("superpoint"):
        extract_features.confs["superpoint"] = extract_features.confs[local_ext]

    # 2a) query image
    extract_features.main(
        extract_features.confs[local_ext],
        query_image.parent,             # root dir for just the query
        image_list=[query_image.name],
        feature_path=feats_h5,
        overwrite=True
    )

    # 2b) each retrieved reference
    for ref in retrieved_names:
        # locate on disk
        img_path = None
        for d in image_dirs:
            cand = d / ref
            if cand.exists():
                img_path = cand
                break
        if img_path is None:
            # fallback glob by stem
            stem = Path(ref).stem
            for d in image_dirs:
                hits = list(d.rglob(f"{stem}.*"))
                if hits:
                    img_path = hits[0]
                    break
        if img_path is None:
            raise FileNotFoundError(f"Could not find reference image {ref}")

        extract_features.main(
            extract_features.confs[local_ext],
            img_path.parent,
            image_list=[img_path.name],
            feature_path=feats_h5,
            overwrite=False
        )

    # — 3) MATCH with LightGlue
    matches_h5 = work_dir / "matches.h5"
    match_conf = match_features.confs[match_method].copy()
    match_conf["model"]["features"] = "superpoint" if local_ext.startswith("superpoint") else local_ext

    match_features.main(
        match_conf,
        pairs_file,
        features=str(feats_h5),
        matches=str(matches_h5),
        overwrite=True
    )

    # — 4) LOAD your COLMAP SfM model
    model = Reconstruction(str(sfm_dir))

    # — 5) BUILD a Camera object from actual image size & your intrinsics
    W, H = Image.open(str(query_image)).size
    fx = 4.49/6.17 * W
    fy = 4.49/4.55 * H
    cx, cy = W/2, H/2

    camera = Camera(
        model="PINHOLE",
        width=W, height=H,
        params=[fx, fy, cx, cy]
    )

    # — 6) TURN filenames → COLMAP image_ids
    ref_ids = [
        model.find_image_with_name(name).image_id
        for name in retrieved_names
        if model.find_image_with_name(name) is not None
    ]

    # — 7) RANSAC + refine settings
    conf = {
        "estimation": {"ransac": {"max_error": ransac_err}},
        "refinement": {
            "refine_focal_length": refine_focal,
            "refine_extra_params": refine_extra
        }
    }
    localizer = QueryLocalizer(model, conf)

    # — 8) RUN the localization
    ret, log = pose_from_cluster(
        localizer,
        query_name = query_image.name,
        camera     = camera,
        ref_ids    = ref_ids,
        features_h5= str(feats_h5),
        matches_h5 = str(matches_h5),
    )

    return PoseResult(
        cam_from_world = ret["cam_from_world"].matrix(),  # 4×4
        inlier_mask    = np.array(ret["inlier_mask"], dtype=bool),
        points3D_ids   = log["points3D_ids"]
    )
