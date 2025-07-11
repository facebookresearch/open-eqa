from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from pathlib import Path
from tqdm import tqdm
import os
import re
import json
import pickle
import argparse

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
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
        "--frames-directory",
        type=Path,
        default="data/frames/",
        help="path image frames (default: data/frames/)",
    )

    args = parser.parse_args()

    return args

def extract_emb(sber, tokenizer, path, save_dir):
    """extract_sentence_embedding from txt trajectory files """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    with open(path) as file:
        text_traj = file.read()

    parsing_result = parsing_text_traj(text_traj)
    task_goal_text = parsing_result['task_goal']
    goal_embedding = sbert.encode(task_goal_text.split('Your task is to: ')[1])

    tokens = tokenizer(text_traj)['input_ids']
    token_count = len(tokens)
    encode_name = path.split('/')[-1].replace('.txt', '.pkl')

    encoding = {'text_trajectory': text_traj,
            'embedding': goal_embedding,
            'text_traj_path': path,
            'token_count': token_count}
    
    em_encod_path = os.path.join(save_dir, encode_name)


    with open(em_encod_path, 'wb') as pickle_file:
        pickle.dump(encoding, pickle_file)

def main(args: argparse.Namespace):
    embedding_model='all-MiniLM-L6-v2'
    sbert = SentenceTransformer(embedding_model)
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B')

    dataset = json.load(args.dataset.open("r"))

    for idx, item in enumerate(tqdm(dataset)):
        # extact scene paths
        if 'hm3d' in item["episode_history"]:
        #     pass
        # else:
            folder = args.frames_directory / item["episode_history"]
            frames = sorted(folder.glob("*qwen.txt"))
            paths = [str(frames[i]) for i in range(len(frames))]

            for text_path in tqdm(paths):
                with open(text_path) as file:
                    text_traj = file.read()
                embedding = sbert.encode(text_traj)

                tokens = tokenizer(text_traj)
                token_count = len(tokens)
                encode_name = text_path.split('/')[-1].replace('.txt', '.pkl')

                encoding = {'embedding' : embedding,
                            'text_traj_path' : text_path,
                            'token_count' : token_count}

                save_dir = os.path.join(folder, encode_name)

                if os.path.exists(save_dir):
                    print(f'{save_dir} is existing file')
                    pass
                else:
                    with open(save_dir, 'wb') as pickle_file:
                        pickle.dump(encoding, pickle_file)
                        print(f'saved: {save_dir}')


if __name__=="__main__":
    main(parse_args())