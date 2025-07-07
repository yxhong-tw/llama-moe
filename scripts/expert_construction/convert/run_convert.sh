#!/usr/bin/bash

llama_size="Llama-2-7b-hf"

num_experts=4
num_selects=2
split_type=Random
proj_type=gate_proj
gate_type="mlp"
use_softmax="False"
multiply_gate_scores="False"

score_scale_factor=1.0
score_scale_factor_file_path=""

convert_type=LlamaMoEForCausalLM

data_path=/dataspace/P76124215/yxhong-mt-lfs/llama-moe-outputs
model_path=/workspace/models/${llama_size}
split_file_path=${data_path}/split/${llama_size}-${num_experts}Expert-Split-${split_type}

select_file_path=""
save_path=${data_path}/models/${convert_type}/${split_type}/${llama_size}-${num_experts}Select${num_selects}-${proj_type}-Scale${score_scale_factor}
python3 -m smoe.entrypoint.expert_construction.llama_convert \
  --model_path ${model_path} \
  --split_file_path ${split_file_path} \
  --select_file_path "${select_file_path}" \
  --save_path ${save_path} \
  --template layers.{}.mlp.${proj_type}.weight \
  --num_experts ${num_experts} \
  --num_selects ${num_selects} \
  --use_random_gate True \
  --gate_type ${gate_type} \
  --use_softmax ${use_softmax} \
  --multiply_gate_scores ${multiply_gate_scores} \
  --score_scale_factor ${score_scale_factor} \
  --score_scale_factor_file_path "${score_scale_factor_file_path}" \
  --convert_type ${convert_type}
