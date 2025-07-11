from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

import argparse
import json
import os
import traceback
from pathlib import Path
from typing import List, Optional
from PIL import Image
import numpy as np
import tqdm
import time, datetime

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_path',
        type=Path,
        default='data/open-eqa-v0.json',
    )
    parser.add_argument(
        '--model_base',
    )
    parser.add_argument(
        '--image_path',
        type=Path,
        default='data/refcoco/train2014',
    )
    parser.add_argument(
        '--data_path',
        type=Path,
        default='data/annotations/finetune_refcoco_testA.json',
    )
    parser.add_argument(
        '--answers_file',
        type=Path,
        default='refexp_result/refcoco_testA',
    )
    parser.add_argument(
        '--conv_mode',
        type=str,
        default='llava_v1',
    )
    parser.add_argument(
        '--num_chunks',
        type=int,
        default='1',
    )
    parser.add_argument(
        '--chunk_idx',
        type=int,
        default='0',
    )
    parser.add_argument(
        '--image_w',
        type=int,
        default='336',
    )
    parser.add_argument(
        '--image_h',
        type=int,
        default='336',
    )
    parser.add_argument(
        '--add_region_feature',
        type=str,
        default='True',
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default='1',
    )
    parser.add_argument(
        '--top_p',
    )
    parser.add_argument(
        '--num_beams',
        type=int,
        default='1',
    )
    parser.add_argument(
        '--dataset',
        type=Path,
        default='data/open-eqa-v0.json',
    )
    parser.add_argument(
        '--output_directory',
        type=Path,
        default='data/results',
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="only process the first 5 questions",
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
        default=4,
        help="num frames in gpt4v (default: 50)",
    )
    parser.add_argument(
        "--single-image",
        action="store_true",
    )
    args = parser.parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    args.output_path = args.output_directory / (args.dataset.stem + "-qwen.json")

    return args


def ask_question(args, frame, model, processor):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": f"{frame}",
                },
                {"type": "text", "text": "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."},
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    image_inputs, _ = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    )

    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=1024)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output_text


def main(args: argparse.Namespace):
    # load dataset
    dataset = json.load(args.dataset.open("r"))
    print("found {:,} questions".format(len(dataset)))

    # default: Load the model on the available device(s)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto")

    # default processer
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

    # load results
    results = []
    if args.output_path.exists():
        results = json.load(args.output_path.open())
        print("found {:,} existing results".format(len(results)))

    start = time.time()
    time.sleep(1)

    # process data
    for idx, item in enumerate(tqdm.tqdm(dataset)):
        if args.dry_run and idx >= 5:
            break
        
        question = item['question']

        # extract scene paths
        folder = args.frames_directory / item["episode_history"]
        frames = sorted(folder.glob("*-rgb.png"))
        paths = [str(frames[i]) for i in range(len(frames))]


        for img in tqdm.tqdm(paths):
            file_path = img.split('/')[-1]
            file_path = file_path.split('.')[0]
            save_file = str(folder) + '/' + file_path + '-qwen.txt'
            #image = Image.open(img)
            
            if os.path.exists(save_file):
                print(f'{save_file} is existing result')
                pass
            else:                
                answer = ask_question(args, img, model, processor)
                with open(save_file , 'w') as file:
                        if type(answer) is list:
                            answer = ''.join(answer)
                            file.write(answer)
                        else:
                            file.write(answer)

if __name__ == "__main__":
    main(parse_args())