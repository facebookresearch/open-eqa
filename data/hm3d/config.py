# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict

import numpy as np
from habitat_sim import (
    ActionSpec,
    ActuationSpec,
    CameraSensorSpec,
    Configuration,
    SensorType,
    SimulatorConfiguration,
)
from habitat_sim.agent import AgentConfiguration


def _create_sensor_spec(
    uuid: str, type, hfov, height, width, sensor_position, sensor_pitch
):
    spec = CameraSensorSpec()
    spec.uuid = uuid
    spec.hfov = hfov
    spec.sensor_type = type
    spec.resolution = [height, width]
    spec.position = [0.0, sensor_position, 0.0]
    spec.orientation = [sensor_pitch, 0.0, 0.0]
    return spec


def _add_move_actions(action_space, amount):
    for key in [
        "move_backward",
        "move_forward",
        "move_left",
        "move_right",
        "move_down",
        "move_up",
    ]:
        action_space[key] = ActionSpec(key, ActuationSpec(amount=amount))


def _add_turn_actions(action_space, amount):
    for key in ["turn_left", "turn_right"]:
        action_space[key] = ActionSpec(key, ActuationSpec(amount=amount))


def _add_look_actions(action_space, amount):
    for key in ["look_up", "look_down"]:
        action_space[key] = ActionSpec(key, ActuationSpec(amount=amount))


default_settings = {
    "random_seed": 42,
    "scene_id": None,
    "sensor_hfov": 90.0,
    "sensor_width": 1920,
    "sensor_height": 1080,
    "sensor_position": 1.0,  # height only
    "sensor_pitch": np.deg2rad(0),
    "agent_height": 1.5,
    "agent_radius": 0.1,
}


def make_cfg(settings: Dict) -> SimulatorConfiguration:
    s = default_settings | settings
    assert s["scene_id"] is not None

    sim_cfg = SimulatorConfiguration()
    sim_cfg.scene_id = s["scene_id"]
    sim_cfg.random_seed = s["random_seed"]

    agent_cfg = AgentConfiguration()
    agent_cfg.height = s["agent_height"]
    agent_cfg.radius = s["agent_radius"]

    # sensors
    agent_cfg.sensor_specifications = []
    agent_cfg.sensor_specifications.append(
        _create_sensor_spec(
            "rgb",
            SensorType.COLOR,
            s["sensor_hfov"],
            s["sensor_height"],
            s["sensor_width"],
            s["sensor_position"],
            s["sensor_pitch"],
        )
    )
    agent_cfg.sensor_specifications.append(
        _create_sensor_spec(
            "depth",
            SensorType.DEPTH,
            s["sensor_hfov"],
            s["sensor_height"],
            s["sensor_width"],
            s["sensor_position"],
            s["sensor_pitch"],
        )
    )
    agent_cfg.sensor_specifications.append(
        _create_sensor_spec(
            "semantic",
            SensorType.SEMANTIC,
            s["sensor_hfov"],
            s["sensor_height"],
            s["sensor_width"],
            s["sensor_position"],
            s["sensor_pitch"],
        )
    )

    # actions
    agent_cfg.action_space = {}
    _add_move_actions(agent_cfg.action_space, amount=0.25)
    _add_turn_actions(agent_cfg.action_space, amount=15.0)
    _add_look_actions(agent_cfg.action_space, amount=15.0)

    return Configuration(sim_cfg, [agent_cfg])
