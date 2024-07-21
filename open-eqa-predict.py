import os
import json
import glob
import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

# os.environ['OPENAI_API_KEY'] = '<OPENAI_API_KEY>'
# os.environ['OPENAI_AZURE_DEPLOYMENT'] = '1'

if os.environ.get('OPENAI_AZURE_DEPLOYMENT') == '1':
    from openeqa.baselines.gpt4_azure import ask_question as ask_blind_gpt4
    from openeqa.baselines.gpt4o_azure import ask_question as ask_gpt4o
else:
    from openeqa.baselines.gpt4 import ask_question as ask_blind_gpt4
    from openeqa.baselines.gpt4v import ask_question as ask_gpt4o

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--method",
        type=str,
        default='blind-gpt4',
        help="the method to be used for inference",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=Path,
        default="data/open-eqa-v0.json",
        help="path to dataset (default: data/open-eqa-v0.json)",
    )
    parser.add_argument(
        "-o",
        "--output-directory",
        type=Path,
        default="data/predictions",
        help="path to an output directory (default: data/predictions)",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="perform inference even if responses are missing (default: false)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="print verbose outputs (default: false)",
    )
    parser.add_argument(
        "-dr",
        "--dry-run",
        action="store_true",
        help="only evaluate the first 5 questions",
    )
    args = parser.parse_args()
    
    assert args.dataset.exists()    
    args.output_directory.mkdir(parents=True, exist_ok=True)
    args.output_path = args.output_directory / (args.method + ("-dry_run" if args.dry_run else "") + "-predictions.json")
    if args.verbose:
        print("output path: {}".format(args.output_path))
    return args
    
def main(args: argparse.Namespace):
    # Ensure that OpenAI API key is set
    assert "OPENAI_API_KEY" in os.environ
    
    # Load Dataset
    eqa_data = json.load(open("data/open-eqa-v0.json"))
    if args.dry_run:
        eqa_data = eqa_data[:5]
        
    print(f'#EQA : {len(eqa_data)} instances')
    print('\nTop-5 Samples:')
    print(json.dumps(eqa_data[:5], indent=2))

    for i, item in tqdm(enumerate(eqa_data)):
        q = item["question"]
        g = item["answer"]

        if args.method == 'blind-gpt4':
            a = ask_blind_gpt4(
                question=q,
                openai_model="gpt-4o",
            )
        elif args.method == 'gpt4o':
            image_paths = sorted(glob.glob(f"data/frames/{item['episode_history']}/*.png"))
            filt_image_paths = []
            for depth_img, rgb_img in zip(image_paths[::16], image_paths[1::16]):
                filt_image_paths.append(depth_img)
                filt_image_paths.append(rgb_img)
            
            a = ask_gpt4o(
                question=q,
                image_paths=filt_image_paths,
                openai_key=os.environ["OPENAI_API_KEY"],
                openai_model="gpt-4o",
            )
        elif args.method == 'video-lm':
            # TODO: Video LM
            pass
        elif args.method == 'concept-graph':
            # TODO: Concept Graph
            pass
        else:
            raise ValueError(f'method `{method}` is not supported')

        # Set answer
        item["answer"] = a
        
        if args.verbose:
            # print the question and the model's answer
            print("Q{}: {} | G: {} | P: {}\n".format(i, q, g, a))

    # Dump Results
    json.dump(eqa_data, open(args.output_path, 'w'), indent=2)

if __name__ == "__main__":
    main(parse_args())