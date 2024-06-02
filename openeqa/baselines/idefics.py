# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Optional, List

import tqdm

from openeqa.utils.idefics_utils import IdeficsRunner, enable_full_determinism, prepare_idefics_vision_messages
from openeqa.utils.prompt_utils import load_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default="data/open-eqa-v0.json",
        help="path to EQA dataset (default: data/open-eqa-v0.json)",
    )
    parser.add_argument(
        "-m",
        "--model-path",
        type=Path,
        required=True,
        help="path to weights in huggingface format",
    )
    parser.add_argument(
        "--frames-directory",
        type=Path,
        default="data/frames/",
        help="path image frames (default: data/frames/)",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=10,
        help="num frames in gpt4v (default: 50)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="gpt seed (default: 1234)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="gpt temperature (default: 0.2)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="gpt maximum tokens (default: 128)",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default="data/results",
        help="output directory (default: data/results)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="continue running on API errors (default: false)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="only process the first 5 questions",
    )
    args = parser.parse_args()
    enable_full_determinism(args.seed)
    args.output_directory.mkdir(parents=True, exist_ok=True)
    args.output_path = args.output_directory / (
        args.dataset.stem + "-{}-{}.json".format(str(args.model_path).replace('/', '-'), args.seed)
    )
    return args


def parse_output(output: str) -> str:
    output_split = output[0].split("Assistant:")
    if len(output_split)==1:
        raise ValueError("Invalid output string: {}".format(output[0]))
    return output[0].split("Assistant:")[-1].strip()


def ask_question(
    model, question: str, image_paths: List[str], max_tokens: int = 128, temperature: float = 0.2
) -> Optional[str]:
    prompt = load_prompt("idefics")
    input = prompt.format(question=question)
    prefix, suffix = prompt.split("User Query:")
    suffix = "User Query:" + suffix.format(question=question)

    input = prepare_idefics_vision_messages(prefix, suffix, image_paths)
    output = model(input, image_paths=image_paths, max_new_tokens=max_tokens, temperature=temperature)
    return parse_output(output)


def main(args: argparse.Namespace):
    # load dataset
    dataset = json.load(args.dataset.open("r"))
    print("found {:,} questions".format(len(dataset)))

    # load model
    model = IdeficsRunner(args.model_path)

    # load results
    results = []
    if args.output_path.exists():
        results = json.load(args.output_path.open())
        print("found {:,} existing results".format(len(results)))
    completed = [item["question_id"] for item in results]

    # process data
    for idx, item in enumerate(tqdm.tqdm(dataset)):
        if args.dry_run and idx >= 5:
            break

        # skip completed questions
        question_id = item["question_id"]
        if question_id in completed:
            continue  # skip existing

        # extract scene paths
        folder = args.frames_directory / item["episode_history"]
        frames = sorted(folder.glob("*-rgb.png"))
        indices = np.round(np.linspace(0, len(frames) - 1, args.num_frames)).astype(int)
        paths = [str(frames[i]) for i in indices]

        # generate answer
        question = item["question"]
        answer = ask_question(
            model=model,
            question=question,
            image_paths=paths,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )

        # store results
        results.append({"question_id": question_id, "answer": answer})
        json.dump(results, args.output_path.open("w"), indent=2)

    # save at end (redundant)
    json.dump(results, args.output_path.open("w"), indent=2)
    print("saving {:,} answers".format(len(results)))


if __name__ == "__main__":
    main(parse_args())
