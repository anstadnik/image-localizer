# image_localizer/retrieval.py

from pathlib import Path
from typing import List, Tuple, Dict, NamedTuple, Optional
import h5py
import numpy as np
import faiss

from .utils import get_latlon
from .netvlad import extract_netvlad

MatchResult = NamedTuple("MatchResult", [
    ("h5_path",    Path),
    ("image_name", str),
    ("score",      float),
    ("lat",        Optional[float]),
    ("lon",        Optional[float]),
])

def build_faiss_index(
    h5_list: List[Path]
) -> Tuple[faiss.IndexFlatIP, List[Tuple[Path,str]]]:
    """
    Given a list of NetVLAD .h5 files, read all descriptors and build
    an inner-product (cosine) FAISS index.
    Returns (index, keys) where keys[i] = (h5_path, image_name).
    """
    descs, keys = [], []
    for h5_path in h5_list:
        with h5py.File(h5_path, "r") as f:
            for img_name in f.keys():
                vec = f[img_name]["global_descriptor"][:].astype("float32")
                descs.append(vec)
                keys.append((h5_path, img_name))

    arr = np.vstack(descs)
    faiss.normalize_L2(arr)
    index = faiss.IndexFlatIP(arr.shape[1])
    index.add(arr)
    return index, keys

def query_faiss(
    index: faiss.IndexFlatIP,
    keys: List[Tuple[Path,str]],
    query_h5: Path,
    image_map: Dict[Path,Path],
    top_k: int = 5
) -> List[MatchResult]:
    """
    Given a FAISS index+keys and a query .h5, run a search and return up to
    top_k unique matches.  image_map maps each h5_path → its image folder.
    """
    # load & normalize the query descriptor
    with h5py.File(query_h5, "r") as f:
        query_name = next(iter(f.keys()))
        qvec       = f[query_name]["global_descriptor"][:].astype("float32")
    faiss.normalize_L2(qvec.reshape(1, -1))

    # overshoot for filename dedupe
    D, I = index.search(qvec.reshape(1, -1), top_k * 3)

    results, seen = [], set()
    for score, idx in zip(D[0], I[0]):
        h5_path, img_name = keys[idx]
        if img_name in seen or img_name == query_name:
            continue
        seen.add(img_name)

        # try to pull GPS from the actual image file
        lat = lon = None
        img_file = image_map[h5_path] / img_name
        ll = get_latlon(img_file)
        if ll:
            lat, lon = ll

        results.append(MatchResult(h5_path, img_name, float(score), lat, lon))
        if len(results) >= top_k:
            break

    return results

def retrieve_image_matches(
    query_image: Path,
    db_h5_list: List[Path],
    image_map: Dict[Path,Path],
    top_k: int = 5,
    tmp_dir: Optional[Path] = None,
) -> List[MatchResult]:
    """
    Full pipeline wrapper: 
      1) extract NetVLAD for `query_image` into a temp H5, 
      2) build FAISS index over `db_h5_list`, 
      3) query and return top_k MatchResult.
    """
    if tmp_dir is None:
        tmp_dir = query_image.parent
    tmp_h5 = tmp_dir / f"{query_image.stem}_netvlad.h5"

    # 1) extract just the query’s descriptor
    extract_netvlad(
        image_dir   = query_image.parent,
        output_h5   = tmp_h5,
        overwrite   = True,
        # note: extract_netvlad currently extracts all images; if you
        # only want the query, you can add an `image_list=[query_image.name]`
        # argument once you’ve updated extract_netvlad accordingly.
    )

    # 2) build & 3) query
    index, keys = build_faiss_index(db_h5_list)
    return query_faiss(index, keys, tmp_h5, image_map, top_k)
