# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import random
from pathlib import Path

from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

# load dataset
DATASET_PATH = Path("../data/open-eqa-v0.json")
with DATASET_PATH.open("r") as f:
    DATASET = json.load(f)

# add index to dataset
for idx in range(len(DATASET)):
    DATASET[idx]["index"] = idx

# add video paths to dataset
for idx in range(len(DATASET)):
    episode_history = DATASET[idx]["episode_history"]
    DATASET[idx]["video_path"] = episode_history + "-0.mp4"

# remove items if video is missing
DATASET = [
    item for item in DATASET if Path("static/videos/" + item["video_path"]).exists()
]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/get_video", methods=["POST"])
def get_video():
    index = request.json["index"] % len(DATASET)
    return jsonify(DATASET[index])


@app.route("/get_random_index", methods=["GET"])
def get_random_index():
    index = random.randrange(len(DATASET))
    return jsonify({"index": index})


if __name__ == "__main__":
    app.run(debug=True)
