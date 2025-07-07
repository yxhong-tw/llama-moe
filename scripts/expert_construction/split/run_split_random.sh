#!/usr/bin/bash

llama_size="Llama-2-7b-hf"

num_experts=4

data_path=/dataspace/P76124215/yxhong-mt-lfs/llama-moe-outputs
model_path=/workspace/models/${llama_size}
save_path=${data_path}/split
python3 -m smoe.entrypoint.expert_construction.llama_split_random \
  --model_path ${model_path} \
  --save_path ${save_path} \
  --template layers.{}.mlp.gate_proj.weight \
  --num_experts ${num_experts}
