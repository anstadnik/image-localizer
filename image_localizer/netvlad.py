# image_localizer/netvlad.py

from pathlib import Path
from typing import Optional, List, Union
from hloc import extract_features

def extract_netvlad(
    image_dir: Path,
    output_h5: Path,
    image_list: Optional[List[Union[str, Path]]] = None,
    config: str = "netvlad",
    overwrite: bool = True,
) -> Path:
    """
    Extract NetVLAD global descriptors.

    - image_dir:    folder with images (will be auto-overridden
                    if you pass absolute paths in image_list)
    - output_h5:    where to write the .h5
    - image_list:   list of filenames or full paths; None → all images
    - config:       which HLoc config to use (default "netvlad")
    - overwrite:    whether to overwrite existing H5
    """
    # — normalize any Path-like entries in image_list
    if image_list:
        paths = [Path(p) for p in image_list]
        parents = {p.parent for p in paths}
        if len(parents) == 1:
            # if they all live in the same dir, use that as image_dir
            image_dir = parents.pop()
        # reduce to basenames for HLoc
        image_list = [p.name for p in paths]

    # ensure output folder exists
    output_h5.parent.mkdir(parents=True, exist_ok=True)

    # delegate to HLoc’s extractor
    conf = extract_features.confs[config]
    extract_features.main(
        conf,
        image_dir,
        image_list=image_list,
        feature_path=output_h5,
        overwrite=overwrite,
    )
    return output_h5

