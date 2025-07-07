#!/usr/bin/bash

llama_size="Llama-2-7b-hf"

# True, False
share_neurons=True
expert_num_moe=3
expert_num_residual=1
total_expert_num=$((${expert_num_moe} + ${expert_num_residual}))

# original intermediate_size = 11008
expert_size=2752

echo ${total_expert_num}\(${expert_num_moe}+${expert_num_residual}\) ${expert_size} ${share_neurons}

kernel=l1_norm
accumulate_level=sample        #  sample  total
importance_type=feature_change #  feature_grad  feature_change
criterion=max                  #  min  max
proj_type=gate_proj

data_path=/dataspace/P76124215/yxhong-mt-lfs/llama-moe-outputs
model_path=/workspace/models/${llama_size}
score_file_path=${data_path}/split/Gradients${total_expert_num}/${llama_size}-Gradients-${kernel}-${accumulate_level}-${importance_type}
save_path=${data_path}/split
python3 -m smoe.entrypoint.expert_construction.llama_split_gradient_residual \
  --model_path ${model_path} \
  --score_file_path ${score_file_path} \
  --save_path ${save_path} \
  --expert_num_moe ${expert_num_moe} \
  --expert_num_residual ${expert_num_residual} \
  --expert_size ${expert_size} \
  --template layers.{}.mlp.${proj_type}.weight \
  --kernel ${kernel} \
  --accumulate_level ${accumulate_level} \
  --importance_type ${importance_type} \
  --criterion ${criterion} \
  --share_neurons ${share_neurons}
