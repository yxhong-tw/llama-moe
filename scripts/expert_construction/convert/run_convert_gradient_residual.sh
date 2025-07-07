#!/usr/bin/bash

llama_size="Llama-2-7b-hf"

# True, False
share_neurons=True
num_experts=3
num_experts_residual=1
num_selects=1

# original intermediate_size = 11008
expert_size=2752

score_scale_factor_residual=1.0
score_scale_factor=1.0

convert_type=LlamaMoEResidualForCausalLM

kernel=l1_norm
criterion=max                  #  min  max
accumulate_level=sample        #  sample  total
importance_type=feature_change #  feature_grad  feature_change
proj_type=gate_proj            #  gate_proj  up_proj

data_path=/dataspace/P76124215/yxhong-mt-lfs/llama-moe-outputs
model_path=/workspace/models/${llama_size}
split_file_path=${data_path}/split/${llama_size}-Split-Gradient-${criterion}-${kernel}-${accumulate_level}-${importance_type}/${num_experts}Experts-${num_experts_residual}Residuals-${expert_size}Neurons
save_path=${data_path}/models/${convert_type}/Gradient-${criterion}-${kernel}-${accumulate_level}-${importance_type}/${llama_size}-${num_experts}Select${num_selects}-${num_experts_residual}Residuals-${expert_size}Neurons
if [ ${share_neurons} = "True" ]; then
  split_file_path=${split_file_path}-Share
  save_path=${save_path}-Share
fi

python3 -m smoe.entrypoint.expert_construction.llama_convert_neuron_index_residual \
  --model_path ${model_path} \
  --split_file_path ${split_file_path} \
  --select_file_path "" \
  --save_path ${save_path} \
  --template layers.{}.mlp.${proj_type}.weight \
  --num_experts ${num_experts} \
  --num_experts_residual ${num_experts_residual} \
  --num_selects ${num_selects} \
  --score_scale_factor ${score_scale_factor} \
  --score_scale_factor_residual ${score_scale_factor_residual} \
  --convert_type ${convert_type} \
  --use_random_gate True
