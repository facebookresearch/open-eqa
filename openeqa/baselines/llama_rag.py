# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import argparse
import sys
import os
import json
from pathlib import Path
from typing import Optional
import time, datetime

import pickle
import numpy as np
from numpy.linalg import norm
import tqdm

sys.path.insert(0, './')
sys.path.insert(0, './openeqa')
from sentence_transformers import SentenceTransformer
from openeqa.utils.llama_utils import LLaMARunner, enable_full_determinism
from openeqa.utils.prompt_utils import load_prompt

log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default="data/open-eqa-v0.json",
        help="path to EQA dataset (default: data/open-eqa-v0.json)",
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="scannet or hm3d",
    )
    parser.add_argument(
        "-m",
        "--model-path",
        type=Path,
        required=True,
        help="path to weights in huggingface format",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        help="model name (defaults to model path folder name)",
    )
    parser.add_argument(
        "--load-in-8bit",
        action="store_true",
        help="load model in 8bit mode (default: false)",
    )
    parser.add_argument(
        "--use-fast-kernels",
        action="store_true",
        help="use fast kernels (default: false)",
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
        default=7000,
        help="gpt maximum tokens (default: 128)",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default="data/results",
        help="output directory (default: data/results)",
    )
    parser.add_argument(
        "--frames-directory",
        type=Path,
        default="data/frames/",
        help="path image frames (default: data/frames/)",
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
    parser.add_argument(
        "--ic-example-num",
        type=int,
        default=3,
        help="using rag in-context example number",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--captioning-model",
        type=str,
        required=True,
    )
    args = parser.parse_args()
    enable_full_determinism(args.seed)
    if args.model_name is None:
        args.model_name = args.model_path.name.lower()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    args.output_path = args.output_directory / (
        args.dataset.stem + "-{}-{}-{}-rag.json".format(args.model_name, args.source, args.prompt)
    )
    return args

def parse_output(output: str) -> str:
    #start_idx = output.find("A:")
    end_idx = output.find("Q")
    print(f'end_idx: {end_idx}')
    # if end_idx == -1:
    #     return output[start_idx:].replace("A:", "").strip()
    answer_text = output[:end_idx].strip()
    print(f'answer_text: {answer_text}')
    return answer_text

def ask_question(args, model, question: str, ic_ex_prompt: list, 
                        max_tokens: int = 200, temperature: float = 0.2) -> Optional[str]:
    prompt = load_prompt(args.prompt)

    input = prompt.format(question=question, img_1=ic_ex_prompt[0], img_2=ic_ex_prompt[1], img_3=ic_ex_prompt[2])  #top-3

    output = model(input, max_new_tokens=max_tokens, temperature=temperature)

    return parse_output(output)

def cosine_similarity(embedding1, embedding2):
    """Calculate cosine similarity between two embeddings."""
    return np.dot(embedding1, embedding2) / (norm(embedding1) * norm(embedding2))

def retrieval(paths, question, ic_example_num):
    print('retrieval')
    embedding_model='all-MiniLM-L6-v2'
    sbert = SentenceTransformer(embedding_model)

    em_question = sbert.encode(question)

    ic_ex_encode_list = []

    for text_traj in paths:
        with open(text_traj, 'rb') as file:
            ic_ex_encoding = pickle.load(file)

        similarity = cosine_similarity(ic_ex_encoding['embedding'], em_question)
        ic_ex_encoding['similarity'] = similarity
        ic_ex_encode_list.append(ic_ex_encoding)

    sorted_ic_ex_encode_list = sorted(ic_ex_encode_list, key=lambda x: x['similarity'], reverse=True)

    ic_ex_files = []
    for idx, encoding in enumerate(sorted_ic_ex_encode_list):
        if idx < ic_example_num:
            ic_ex_files.append(encoding['text_traj_path'])
        else:
            pass

    return ic_ex_files

def read_txt_file(file_path):
    with open(file_path) as file:
        text_description = file.read()
    return text_description

def main(args: argparse.Namespace):
    # load dataset
    dataset = json.load(args.dataset.open("r"))
    print("found {:,} questions".format(len(dataset)))

    # load model
    model = LLaMARunner(
        args.model_path,
        load_in_8bit=args.load_in_8bit,
        use_fast_kernels=args.use_fast_kernels,
    )

    # load results
    results = []
    if args.output_path.exists():
        results = json.load(args.output_path.open())
        print("found {:,} existing results".format(len(results)))
    completed = [item["question_id"] for item in results]
    dataset_name = [item["episode_history"] for item in dataset if args.source in item["episode_history"]]

    start = time.time()

    # process data
    for idx, item in enumerate(tqdm.tqdm(dataset)):
        if args.dry_run and idx >= 5:
            break

        # skip completed questions
        question_id = item["question_id"]
        if question_id in completed:
            continue  # skip existing
        
        # Use this for experiments that require dataset splitting. For experiments on the full dataset, remove the if-statement.
        if 'hm3d' in item["episode_history"]:
            pass
        elif args.source in item["episode_history"]:
            folder = args.frames_directory / item["episode_history"]
            if 'llava' in args.captioning_model:
                frames = sorted(folder.glob("*-llava.pkl"))
            elif 'qwen' in args.captioning_model:
                frames = sorted(folder.glob("*-qwen.pkl"))
            else:
                frames = sorted(folder.glob("*-rgb.pkl"))
            paths = [str(frames[i]) for i in range(len(frames))]

            # generate answer
            question = item["question"]
            ic_ex_files = retrieval(paths, question, args.ic_example_num)
            
            ic_examples = [read_txt_file(ic_ex_file) for ic_ex_file in ic_ex_files]
            answer = ask_question(args, model=model, question=question, ic_ex_prompt=ic_examples)

            # store results
            results.append({"question_id": question_id, "category": item['category'], "question": question, "answer": answer, "GT answer": item["answer"], "ic_ex_files": ic_ex_files, "ic_examples": ic_examples, "time":str(datetime.timedelta(seconds=(time.time() - start)))})
            json.dump(results, args.output_path.open("w"), indent=2)

            print(f'{idx+1}/{len(dataset_name)}')
        else:
            break

    # save at end (redundant)
    json.dump(results, args.output_path.open("w"), indent=2)
    print("saving {:,} answers".format(len(results)))

if __name__ == "__main__":
    main(parse_args())
