"""
Usage:
# 7B
python3 -m ferret.model.apply_delta \
    --base ./model/vicuna-7b-v1-3 \
    --target ./model/ferret-7b-v1-3 \
    --delta ./checkpoints/ferret_ft_clipL336_vicunaV1-3-7b_3Ep_dataV16_RSamplerV2/ferret-7b-delta

# 13B
python3 -m ferret.model.apply_delta \
    --base ./model/vicuna-13b-v1-3 \
    --target ./model/ferret-13b-v1-3 \
    --delta ./checkpoints/ferret_ft_clipL336_vicunaV1-3-13b_3Ep_dataV16_RSamplerV2/ferret-13b-delta
"""
import argparse

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from openeqa.baselines.ferret_transform import FERRETLlamaForCausalLM


exclude_name_lists = ['model.mm_projector.weight', 'model.mm_projector.bias', 
                    'model.region_geo_sampler.agg_projector_list.0.net.0.bias', 'model.region_geo_sampler.agg_projector_list.0.net.0.weight', 
                    'model.region_geo_sampler.agg_projector_list.0.norm.bias', 'model.region_geo_sampler.agg_projector_list.0.norm.weight', 
                    'model.region_geo_sampler.agg_projector_list.1.net.0.bias', 'model.region_geo_sampler.agg_projector_list.1.net.0.weight', 
                    'model.region_geo_sampler.agg_projector_list.1.norm.bias', 'model.region_geo_sampler.agg_projector_list.1.norm.weight', 
                    'model.region_geo_sampler.diff_projector_list.0.bias', 'model.region_geo_sampler.diff_projector_list.0.weight', 
                    'model.region_geo_sampler.diff_projector_list.1.bias', 'model.region_geo_sampler.diff_projector_list.1.weight', 
                    'model.region_geo_sampler.dim_projector.bias', 'model.region_geo_sampler.dim_projector.weight', 
                    'model.region_geo_sampler.flatten_projector.bias', 'model.region_geo_sampler.flatten_projector.weight'
                    ]


def apply_delta(base_model_path, target_model_path, delta_path):
    print("Loading base model")
    base = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True)

    print("Loading delta")
    delta = FERRETLlamaForCausalLM.from_pretrained(delta_path, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    delta_tokenizer = AutoTokenizer.from_pretrained(delta_path)

    print("Applying delta")
    for name, param in tqdm(delta.state_dict().items(), desc="Applying delta"):
        if name not in base.state_dict():
            assert name in exclude_name_lists, f'{name} not in base model'
            continue
        if param.data.shape == base.state_dict()[name].shape:
            param.data += base.state_dict()[name]
        else:
            assert name in ['model.embed_tokens.weight', 'lm_head.weight'], \
                f'{name} dimension mismatch: {param.data.shape} vs {base.state_dict()[name].shape}'
            bparam = base.state_dict()[name]
            param.data[:bparam.shape[0], :bparam.shape[1]] += bparam

    print("Saving target model")
    delta.save_pretrained(target_model_path)
    delta_tokenizer.save_pretrained(target_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", type=str, required=True)
    parser.add_argument("--target-model-path", type=str, required=True)
    parser.add_argument("--delta-path", type=str, required=True)

    args = parser.parse_args()

    apply_delta(args.base_model_path, args.target_model_path, args.delta_path)
