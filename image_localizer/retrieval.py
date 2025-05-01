# image_localizer/retrieval.py

from pathlib import Path
from typing import List, Tuple, Dict, NamedTuple, Optional
import h5py
import numpy as np
import faiss
from .utils import get_latlon

MatchResult = NamedTuple("MatchResult", [
    ("h5_path", Path),
    ("image_name", str),
    ("score", float),
    ("lat", Optional[float]),
    ("lon", Optional[float]),
])

def build_faiss_index(h5_list: List[Path]) -> Tuple[faiss.Index, List[Tuple[Path,str]]]:
    """
    Given a list of NetVLAD .h5 files, read all descriptors and build
    an inner-product FAISS index (cosine similarity).
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
    index: faiss.Index,
    keys: List[Tuple[Path,str]],
    query_h5: Path,
    image_map: Dict[Path,Path],
    top_k: int=5
) -> List[MatchResult]:
    """
    Given a built FAISS index and its keys, plus a query .h5 file,
    run a search to return up to top_k unique matches.  image_map
    maps each h5_path -> corresponding image directory.
    """
    # load the query descriptor
    with h5py.File(query_h5, "r") as f:
        query_name = next(iter(f.keys()))
        qvec = f[query_name]["global_descriptor"][:].astype("float32")
    faiss.normalize_L2(qvec.reshape(1, -1))

    # search a bit more for deduplication
    D, I = index.search(qvec.reshape(1, -1), top_k * 3)

    results = []
    seen = set()
    for score, idx in zip(D[0], I[0]):
        h5_path, img_name = keys[idx]
        if img_name in seen or img_name == query_name:
            continue
        seen.add(img_name)

        # attempt GPS
        lat, lon = None, None
        img_file = image_map[h5_path] / img_name
        ll = get_latlon(img_file)
        if ll is not None:
            lat, lon = ll

        results.append(MatchResult(h5_path, img_name, float(score), lat, lon))
        if len(results) >= top_k:
            break

    return results
