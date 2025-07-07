#!/usr/bin/bash

set -e

num_nodes=1        # should match with --nodes
num_gpu_per_node=4 # should match with --gres

export OMP_NUM_THREADS=32
export LOGLEVEL=INFO

{
  llama_size="Llama-2-7b-hf"

  accumulate_level=sample        #  sample  total
  kernel=l1_norm                 #  plain  l1_norm  l2_norm
  importance_type=feature_change #  feature_grad  feature_change

  data_use_range_begin=0.0
  data_use_range_end=1.0

  data_path=/dataspace/P76124215/yxhong-mt-lfs/llama-moe-outputs
  save_path=${data_path}/split
  pretrained_model=/workspace/models/${llama_size}
  tokenizer_path=/workspace/models/${llama_size}

  per_device_train_batch_size=4
  block_size=2048
  total_clusters=4
  dataset_dir=/workspace/P76124215/llama-moe/SlimPajama-6B-train-0.01-processed-tokenized
  dataset_name=()
  for ((i = 0; i < ${total_clusters}; i++)); do
    dataset_name+=("${i}.jsonl")
  done

  output_dir=${data_path}/train-moe/outputs/${llama_size}
  echo "output_dir: $output_dir"

  deepspeed_config_file=/workspace/P76124215/llama-moe/conf/deepspeed/bf16.json

  for name in "${dataset_name[@]}"; do
    echo "@@@@@@@@@@@@@@@@@@@@@@" ${llama_size} ${name} "@@@@@@@@@@@@@@@@@@@@@@"
    CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
      --nnodes ${num_nodes} \
      --nproc_per_node ${num_gpu_per_node} \
      smoe/entrypoint/expert_construction/llama_split_gradient_get_grads.py \
      --deepspeed ${deepspeed_config_file} \
      --model_name_or_path ${pretrained_model} \
      --tokenizer_name_or_path ${tokenizer_path} \
      --dataset_dir ${dataset_dir}/${name} \
      --validation_split_percentage 0 \
      --per_device_train_batch_size ${per_device_train_batch_size} \
      --do_train \
      --seed 48763 \
      --bf16 \
      --num_train_epochs 1 \
      --final_lr_portion 0.1 \
      --optim sgd \
      --learning_rate 0 \
      --weight_decay 0 \
      --max_grad_norm 1.0 \
      --warmup_steps 1000 \
      --logging_strategy steps \
      --logging_steps 10 \
      --save_strategy no \
      --dataloader_num_workers 8 \
      --block_size ${block_size} \
      --output_dir ${output_dir} \
      --overwrite_output_dir \
      --ddp_timeout 30000 \
      --logging_first_step True \
      --torch_dtype bfloat16 \
      --ddp_find_unused_parameters False \
      --report_to tensorboard \
      --gradient_checkpointing \
      --log_level info \
      --save_path ${save_path} \
      --total_clusters ${total_clusters} \
      --accumulate_level ${accumulate_level} \
      --kernel ${kernel} \
      --importance_type ${importance_type} \
      --data_use_range_begin ${data_use_range_begin} \
      --data_use_range_end ${data_use_range_end}
  done
}
