#!/usr/bin/bash

set -vx

tokenizer_dir=/workspace/models/Llama-2-7b-hf
data_dir=/workspace/P76124215/llama-moe/SlimPajama-6B-train-0.01-processed
out_dir=/workspace/P76124215/llama-moe/SlimPajama-6B-train-0.01-processed-tokenized

mkdir -p $out_dir

# for loop in: en_arxiv, en_book, en_c4, en_cc, en_stack, en_wikipedia, github
for data_type in $(ls $data_dir)
do
    python3 -m smoe.utils.tokenize \
        -f jsonl \
        -t $tokenizer_dir \
        -i $data_dir/$data_type \
        -o $out_dir/$data_type
done
