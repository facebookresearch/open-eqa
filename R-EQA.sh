## ========================= HM3D =========================
#### RAG
CUDA_VISIBLE_DEVICES=6,7,0,1,2,3 python openeqa/baselines/llama_rag.py --source hm3d -m meta-llama/Llama-3.1-70B --prompt vlm_rag --captioning-model qwen

### unifrom sampling
 CUDA_VISIBLE_DEVICES=6,7,0,1,2,3 python openeqa/baselines/llama_uniform_sampling.py --source hm3d -m meta-llama/Llama-3.1-70B --prompt vlm_uniform_sampling --captioning-model qwen

## ========================= scannet =========================
#### RAG
python openeqa/baselines/llama_rag.py --source scannet -m meta-llama/Llama-3.1-70B --prompt ferret_rag --captioning-model ferret

### uniform sampling
python openeqa/baselines/llama_uniform_sampling.py --source scannet -m meta-llama/Llama-3.1-70B --prompt ferret_uniform_sampling --captioning-model ferret