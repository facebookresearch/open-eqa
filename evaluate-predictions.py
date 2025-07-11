# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from openeqa.evaluation.llm_match import get_llm_match_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "results",
        type=Path,
        help="path to a results file",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default="data/open-eqa-v0.json",
        help="path to dataset (default: data/open-eqa-v0.json)",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default="data/metrics",
        help="path to an output directory (default: data/metrics)",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="evaluate results even if responses are missing (default: false)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="print verbose outputs (default: false)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="only evaluate the first 5 questions",
    )
    args = parser.parse_args()
    assert args.results.exists()
    assert args.dataset.exists()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    args.output_path = args.output_directory / (args.results.stem + "-metrics.json")
    if args.verbose:
        print("output path: {}".format(args.output_path))
    return args


def main(args: argparse.Namespace):
    # load dataset
    dataset = json.load(args.dataset.open("r"))
    dataset_question_ids = [item["question_id"] for item in dataset]
    question_id_to_item = {item["question_id"]: item for item in dataset}
    print("found {:,} questions".format(len(dataset)))

    # load results
    results = json.load(args.results.open("r"))
    results_question_ids = [item["question_id"] for item in results]
    question_id_to_result = {result["question_id"]: result for result in results}
    print("found {:,} results".format(len(results)))

    # check that results and dataset match
    if not args.force:
        assert len(dataset_question_ids) == len(results_question_ids)
        assert set(dataset_question_ids) == set(results_question_ids)

    # load scores
    all_scores = []
    if args.output_path.exists():
        all_scores = json.load(args.output_path.open("r"))
        print("found {:,} existing scores".format(len(all_scores)))
    complete_question_id = [item["question_id"] for item in all_scores]

    # evaluate predictions
    for idx, item in enumerate(tqdm(results)):
        if args.dry_run and idx >= 5:
            break

        question_id = item["question_id"]
        if question_id in complete_question_id:
            continue

        item = question_id_to_item[question_id]
        result = question_id_to_result[question_id]
        extra_answers = item["extra_answers"] if "extra_answers" in item else None

        # pre-process answers
        if result["answer"]:
            # remove anything after the last period
            end_idx = result["answer"].rfind(".")
            if end_idx >= 0 and end_idx + 1 < len(result["answer"]):
                result["answer"] = result["answer"][: end_idx + 1]

        score = get_llm_match_score(
            question=item["question"],
            answer=item["answer"],
            prediction=result["answer"],
            extra_answers=extra_answers,
        )

        all_scores[question_id] = score
        json.dump(all_scores, args.output_path.open("w"), indent=2)

    all_scores_converted = {item["question_id"]: item["score"] for item in all_scores}

    # calculate final score
    scores = np.array(list(all_scores_converted.values()))
    scores = 100.0 * (np.clip(scores, 1, 5) - 1) / 4
    print("final score: {:.1f}".format(np.mean(scores)))


if __name__ == "__main__":
    main(parse_args())
