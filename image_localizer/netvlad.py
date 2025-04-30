from pathlib import Path
from hloc import extract_features

def extract_netvlad(
    image_dir: Path,
    output_h5: Path,
    config: str = "netvlad",
    overwrite: bool = True,
) -> Path:
    """
    Extract NetVLAD global descriptors for all images in `image_dir`.
    Writes them into `output_h5` and returns its path.
    """
    conf = extract_features.confs[config]
    extract_features.main(
        conf,
        image_dir,
        image_list=None,       # all images in the folder
        feature_path=output_h5,
        overwrite=overwrite,
    )
    return output_h5
