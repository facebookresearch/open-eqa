# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from pathlib import Path

import imageio.v2 as imageio
import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--frames-directory",
        type=Path,
        default="data/frames",
        help="input directory (default: data/frames)",
    )
    parser.add_argument(
        "--split",
        choices=["hm3d-v0", "scannet-v0"],
        default="scannet-v0",
        help="dataset split (default: scannet-v0)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="video fps (default: 30)",
    )
    parser.add_argument(
        "--videos-directory",
        type=Path,
        default="viewer/static/videos",
        help="output directory (default: viewer/static/videos)",
    )
    args = parser.parse_args()
    args.input_directory = args.frames_directory / args.split
    if not args.input_directory.exists():
        raise argparse.ArgumentError(
            "could not find input directory: {}".format(args.input_directory)
        )
    args.output_directory = args.videos_directory / args.split
    args.output_directory.mkdir(parents=True, exist_ok=True)
    return args


def get_folders(args: argparse.Namespace) -> list:
    folders = sorted(args.input_directory.glob("*"))
    print("found {:,} folders".format(len(folders)))
    return folders


def create_video(folder: Path, args: argparse.Namespace):
    output_path = args.output_directory / (folder.name + "-0.mp4")
    if output_path.exists():
        print("WARNING: skipping {}; file already exists".format(output_path))
        return

    files = sorted(folder.glob("*-rgb.png"))

    writer = imageio.get_writer(
        output_path,
        fps=args.fps,
        macro_block_size=8,
        input_params=["-probesize", "32M"],
    )
    for path in tqdm.tqdm(files):
        writer.append_data(imageio.imread(path))
    writer.close()


def main(args):
    folders = get_folders(args)
    for folder in folders:
        create_video(folder, args)


if __name__ == "__main__":
    main(parse_args())
