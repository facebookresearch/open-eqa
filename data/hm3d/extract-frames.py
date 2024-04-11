# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import pickle
from pathlib import Path
from typing import List

import habitat_sim
import numpy as np
import tqdm
from config import make_cfg
from PIL import Image

os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hm3d-root",
        type=Path,
        default="data/scene_datasets/hm3d/val",
        help="path to hm3d scene data (default: data/scene_datasets/hm3d/val)",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default="data/frames/hm3d-v0",
        help="output path (default: data/frames/hm3d-v0)",
    )
    parser.add_argument(
        "--rgb-only",
        action="store_true",
        help="only extract rgb frames (default: false)",
    )
    args = parser.parse_args()
    return args


def get_config(
    scene_id: str, sensor_position: float
) -> habitat_sim.simulator.Configuration:
    settings = {
        "scene_id": scene_id,
        "sensor_hfov": 90,
        "sensor_width": 1920,
        "sensor_height": 1080,
        "sensor_position": sensor_position,  # height only
    }
    return make_cfg(settings)


def load_sim(path: Path) -> habitat_sim.Simulator:
    data = pickle.load(path.open("rb"))
    scene_id = data["scene_id"]
    agent_state = data["agent_state"]
    sensor_position = (
        agent_state.sensor_states["rgb"].position[1] - agent_state.position[1]
    )
    cfg = get_config(scene_id=scene_id, sensor_position=sensor_position)
    return habitat_sim.Simulator(cfg)


def save_intrinsics(path: Path) -> None:
    data = pickle.load(path.open("rb"))
    height, width = data["resolution"]
    hfov = np.deg2rad(data["hfov"])
    vfov = hfov * height / width
    K = np.array(
        [
            [width / np.tan(hfov / 2.0) / 2.0, 0.0, width / 2, 0.0],
            [0.0, height / np.tan(vfov / 2.0) / 2.0, height / 2, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    output_path = path.parent / "intrinsic_color.txt"
    np.savetxt(output_path, K, fmt="%.6f")
    output_path = path.parent / "intrinsic_depth.txt"
    np.savetxt(output_path, K, fmt="%.6f")


def save_pose(path: Path, sim: habitat_sim.Simulator) -> None:
    camera_pose_path = str(path).replace(".pkl", ".txt")
    camera_pose = sim._sensors["rgb"]._sensor_object.node.absolute_transformation()
    np.savetxt(camera_pose_path, camera_pose, fmt="%.6f")


def save_depth(path: Path, sim: habitat_sim.Simulator) -> None:
    obs = sim.get_sensor_observations()
    depth_path = str(path).replace(".pkl", "-depth.png")
    depth = (obs["depth"] / 10 * 65535).astype(np.uint16)
    Image.fromarray(depth).save(depth_path)


def save_color(path: Path, sim: habitat_sim.Simulator) -> None:
    obs = sim.get_sensor_observations()
    rgb_path = str(path).replace(".pkl", "-rgb.png")
    Image.fromarray(obs["rgb"]).convert("RGB").save(rgb_path)


def extract_frames(folder: Path, args: argparse.Namespace) -> None:
    print("Extracting frames to: {}".format(folder))
    files = sorted(folder.glob("*.pkl"))
    sim = load_sim(path=files[0])

    print("Processing {} agent positions...".format(len(files)))
    for idx, path in enumerate(files):
        # set agent state
        data = pickle.load(path.open("rb"))
        agent = sim.get_agent(0)
        agent.set_state(data["agent_state"])

        # save data
        if not args.rgb_only:
            if idx == 0:
                save_intrinsics(path)
            save_pose(path, sim)
            save_depth(path, sim)
        save_color(path, sim)

    sim.close()
    print("Extracting frames to: {} done!".format(folder))


def main(args):
    folders = sorted(args.output_directory.glob("*"))
    for folder in tqdm.tqdm(folders):
        extract_frames(folder, args)


if __name__ == "__main__":
    main(parse_args())
