import argparse
import os

from smoe.utils.expert_construction.convert_llama_moe_neuron_index import (
    convert_llama_model_for_causal_lm_neuron_index,
    convert_llama_model_for_sequence_classification_neuron_index,
    convert_llama_model_neuron_index,
)
from smoe.utils.operations.operation_string import str2bool

# fmt: off
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="/home/data/models/llama-transformers/7B")
    parser.add_argument('--split_file_path', type=str, default="/home/dongdz/workspace/moefication/llama_moe_temp_files/llama_7B-8Expert-Split-Clustering")
    parser.add_argument('--save_path', type=str, default="/home/data/models/llama-moe-transformers/7B/")
    parser.add_argument('--template', type=str, default='layers.{}.mlp.gate_proj.weight')

    parser.add_argument('--num_experts', type=int, default=8, help='number of experts')
    parser.add_argument('--num_selects', type=int, default=2, help='number of selected experts')
    parser.add_argument('--score_scale_factor', type=float, default=1.0, help='scale factor for experts in all layers')
    parser.add_argument('--score_scale_factor_file_path', type=str, default=None, help='file storing the layer-wise scale factors, this will override the argument "score_scale_factor"')

    parser.add_argument('--convert_type', type=str, default="LlamaMoEForCausalLM", choices=("LlamaMoEModel", "LlamaMoEForCausalLM", "LlamaMoEForSequenceClassification"))

    args = parser.parse_args()
    print(args, "\n")

    if args.score_scale_factor_file_path is not None and args.score_scale_factor_file_path != "":
        with open(os.path.join(args.score_scale_factor_file_path, "score_scale_factors.txt"), "r") as file:
            layer_wise_score_scale_factor_str = file.readlines()[0]
            layer_wise_score_scale_factor = eval(layer_wise_score_scale_factor_str)
            args.score_scale_factor = layer_wise_score_scale_factor

    print(args.score_scale_factor, flush=True)

    if args.convert_type == "LlamaMoEModel":
        convert_llama_model_neuron_index(
            args.model_path,
            args.split_file_path,
            args.save_path,
            args.template,
            args.num_experts,
            args.num_selects,
            score_scale_factor=args.score_scale_factor,
        )
    elif args.convert_type == "LlamaMoEForCausalLM":
        convert_llama_model_for_causal_lm_neuron_index(
            args.model_path,
            args.split_file_path,
            args.save_path,
            args.template,
            args.num_experts,
            args.num_selects,
            score_scale_factor=args.score_scale_factor,
        )
    elif args.convert_type == "LlamaMoEForSequenceClassification":
        convert_llama_model_for_sequence_classification_neuron_index(
            args.model_path,
            args.split_file_path,
            args.save_path,
            args.template,
            args.num_experts,
            args.num_selects,
            score_scale_factor=args.score_scale_factor,
        )
    else:
        raise ValueError

    print("Done.")
