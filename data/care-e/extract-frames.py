# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import pickle
from pathlib import Path
from typing import List
import cv2

import numpy as np
import tqdm
from PIL import Image

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--caree-root",
        type=Path,
        default="/share/care-e/testing_videos",
        help="path to care-e video data (default: /share/care-e/testing_videos)",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default="data/frames/caree-v0",
        help="output path (default: data/frames/caree-v0)",
    )
    args = parser.parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    return args

def load_video(path: Path) -> cv2.VideoCapture:
    vidcap = cv2.VideoCapture(str(path))
    return vidcap

def extract_frames(video_path: Path, args: argparse.Namespace) -> None:
    print("Extracting frames from: {}".format(video_path))
    folder_name = str(video_path).replace('.mp4', '').split('/')[-1]
    os.makedirs(f'{args.output_directory}/{folder_name}', exist_ok=True)

    vidcap = load_video(path=video_path)
    fps, n_frame = int(round(vidcap.get(cv2.CAP_PROP_FPS))), int(round(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)))
    print(f'Loading video from `{video_path}` with {fps} fps with {n_frame} frames')
    
    frame_idx = 0
    for frame_idx in tqdm.tqdm(range(n_frame)):
        success, image = vidcap.read()
        cv2.imwrite(f'{args.output_directory}/{folder_name}/{frame_idx:05d}-rgb.png', image) 
    
    vidcap.release()
    print(f"Extracting frames to: {args.output_directory/folder_name} done!")

def main(args):
    video_paths = sorted(args.caree_root.glob("*.mp4"))
    for video_path in tqdm.tqdm(video_paths):
        extract_frames(video_path, args)


if __name__ == "__main__":
    main(parse_args())
