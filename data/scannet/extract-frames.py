# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
from pathlib import Path
from typing import Dict

import tqdm
from SensorData import SensorData


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default="data/open-eqa-v0.json",
        help="path to open-eqa dataset (default: data/open-eqa-v0.json)",
    )
    parser.add_argument(
        "--scannet-root",
        type=Path,
        default="data/raw/scannet",
        help="path to scannet data (default: data/raw/scannet)",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default="data/frames",
        help="path to output folder (default: data/frames)",
    )
    parser.add_argument(
        "--rgb-only",
        action="store_true",
        help="only extract rgb frames (default: false)",
    )
    parser.add_argument(
        "--max-num-frames",
        type=int,
        default=600,
        help="maximum frames to extract from a scene (default: 600)",
    )
    args = parser.parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    return args


def get_folder_to_scene(args: argparse.Namespace) -> Dict[str, str]:
    # get unique scannet folders
    scene_folders = set()
    dataset = json.load(args.dataset.open())
    for item in dataset:
        if "scannet" not in item["episode_history"]:
            continue
        scene_folders.add(item["episode_history"])

    # map folders to scene name
    scenes = {}
    for folder in sorted(scene_folders):
        scene_name = folder.split("/")[-1].split("-")[-1]
        scenes[folder] = scene_name

    return scenes


def get_scene_path(args: argparse.Namespace, scene: str) -> Path:
    for folder in ["scans", "scans_test"]:
        scene_path = args.scannet_root / folder / scene / (scene + ".sens")
        if scene_path.exists():
            return scene_path
    raise ValueError("scene ({}) not found in {}".format(scene, args.scannet_root))


def extract_frames(
    scene_path: str, output_folder: Path, args: argparse.Namespace
) -> None:
    output_folder.mkdir(exist_ok=True, parents=True)
    output_folder = str(output_folder)

    print("Extracting frames to: {}".format(output_folder))
    sd = SensorData(scene_path)
    if not args.rgb_only:
        sd.export_intrinsics(output_folder)
        sd.export_poses(output_folder, num_frames=args.max_num_frames)
        sd.export_depth_images(output_folder, num_frames=args.max_num_frames)
    sd.export_color_images(output_folder, num_frames=args.max_num_frames)
    print("Extracting frames to: {} done!".format(output_folder))


def main(args):
    folder_to_scene = get_folder_to_scene(args)
    for folder, scene in tqdm.tqdm(folder_to_scene.items()):
        output_folder = args.output_directory / folder
        scene_path = get_scene_path(args, scene)
        extract_frames(scene_path=scene_path, output_folder=output_folder, args=args)


if __name__ == "__main__":
    main(parse_args())
