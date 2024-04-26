# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from pathlib import Path
from typing import List, Union, Optional

import torch
import transformers
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image


def enable_full_determinism(seed: int):
    transformers.enable_full_determinism(seed)


def prepare_idefics_vision_messages(
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
    image_paths: Optional[List[str]] = None,
):
    if image_paths is None:
        image_paths = []

    content = []

    if prefix:
        content.append({"text": prefix, "type": "text"})
    for _ in image_paths:
        content.append({"type": "image"})
    if suffix:
        content.append({"text": suffix, "type": "text"})

    return [{"role": "user", "content": content}]


class IdeficsRunner:
    def __init__(
        self,
        model: Union[str, Path],
    ):
        self.processor = AutoProcessor.from_pretrained(model)
        self.model = AutoModelForVision2Seq.from_pretrained(model, device_map="auto")
        self.model.eval()

    def __call__(
        self,
        input: str,
        image_paths: List[str],
        max_new_tokens: int = 128,
        do_sample: bool = True,
        top_p: float = 1.0,
        top_k: int = 50,
        temperature: float = 1.0,
        repetition_penalty: float = 1.0,
        length_penalty: int = 1,
        use_cache: bool = True,
    ) -> str:

        images = [load_image(i) for i in image_paths]
        prompt = self.processor.apply_chat_template(input, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=images, return_tensors="pt")

        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                top_p=top_p,
                top_k=top_k,
                temperature=temperature,
                use_cache=use_cache,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
            )
        generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        return generated_texts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        type=Path,
        help="path to idefics weights (in huggingface format)",
        required=True,
    )
    args = parser.parse_args()
    idefics = IdeficsRunner(args.model)

    input = "What is behind the plant?"
    image_paths = ["data/frames/hm3d-v0/000-hm3d-BFRyYbPCCPE/00000-rgb.png", 
        "data/frames/hm3d-v0/000-hm3d-BFRyYbPCCPE/00001-rgb.png",
        "data/frames/hm3d-v0/000-hm3d-BFRyYbPCCPE/00002-rgb.png"]
    
    prompt = prepare_idefics_vision_messages(prefix=None, suffix=input, image_paths=image_paths)
    output = idefics(prompt, image_paths, do_sample=False)
    print("Q: {}".format(input.strip()))
    print("A: {}".format(output.strip()))
