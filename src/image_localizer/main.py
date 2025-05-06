#!/usr/bin/env python
import argparse
from pathlib import Path

from .netvlad import extract_netvlad
from .retrieval import retrieve_image_matches
from .localization import localize_image
from .utils import visualize_retrieval, compute_bbox


def main():
    p = argparse.ArgumentParser(
        prog="image-localizer",
        description="NetVLAD → FAISS retrieval → SuperPoint+LightGlue localization",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # netvlad subcommand
    t = sub.add_parser("extract", help="extract NetVLAD descriptors")
    t.add_argument("image_dir", type=Path)
    t.add_argument("output_h5", type=Path)
    t.add_argument("--config", default="netvlad")
    t.add_argument("--overwrite", action="store_true")

    # retrieve subcommand
    t = sub.add_parser("retrieve", help="run FAISS retrieval")
    t.add_argument("query_image", type=Path)
    t.add_argument("db_h5", type=Path)
    t.add_argument("img_dir", type=Path)
    t.add_argument("--top_k", type=int, default=5)

    # localize subcommand
    t = sub.add_parser("localize", help="localize a single query")
    t.add_argument("query_image", type=Path)
    t.add_argument("sfm_model_dir", type=Path)
    t.add_argument("db_h5", type=Path)
    t.add_argument("img_dir", type=Path)
    t.add_argument("--top_k", type=int, default=5)
    t.add_argument("--local_ext", default="superpoint_max")
    t.add_argument("--match_method", default="disk+lightglue")
    t.add_argument("--ransac_err", type=float, default=200.0)

    args = p.parse_args()

    if args.cmd == "extract":
        extract_netvlad(
            image_dir=args.image_dir,
            output_h5=args.output_h5,
            config=args.config,
            overwrite=args.overwrite,
        )
    elif args.cmd == "retrieve":
        matches = retrieve_image_matches(
            query_image=args.query_image,
            db_h5_list=[args.db_h5],
            image_map={args.db_h5: args.img_dir},
            top_k=args.top_k,
        )
        for m in matches:
            print(f"{m.image_name}\t{m.score:.4f}")
    elif args.cmd == "localize":
        # first do retrieval
        matches = retrieve_image_matches(
            query_image=args.query_image,
            db_h5_list=[args.db_h5],
            image_map={args.db_h5: args.img_dir},
            top_k=args.top_k,
        )
        retrieved_names = [m.image_name for m in matches]
        pose = localize_image(
            query_image=args.query_image,
            sfm_dir=args.sfm_model_dir,
            retrieved_names=retrieved_names,
            image_dirs=[args.img_dir],
            local_ext=args.local_ext,
            match_method=args.match_method,
            ransac_err=args.ransac_err,
        )
        print("Pose (world→cam):")
        print(pose.cam_from_world)
        print("Inliers:", pose.inlier_mask.sum(), "/", len(pose.inlier_mask))
    else:
        p.print_help()


if __name__ == "__main__":
    main()
