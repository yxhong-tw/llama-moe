"""Convert a vanilla llama to llama-moe"""
import os
import shutil
from collections import Counter

import torch
from tqdm import tqdm
from transformers import LlamaForCausalLM, LlamaForSequenceClassification, LlamaModel

from smoe.models.llama_moe import (
    LlamaMoEConfig,
    LlamaMoEForCausalLM,
    LlamaMoEForSequenceClassification,
    LlamaMoEModel,
)
from smoe.utils.io import torch_load_template_file


def convert_llama_model(
    llama_model_path,
    split_index_path,
    save_path,
    template,
    num_experts,
    num_selects,
    score_scale_factor=None,
    gate_type="mlp",  # "linear"
    use_softmax=True,
    multiply_gate_scores=True,
):
    """
    LlamaMoEModel
    """

    moe_indices = []
    size_experts = []

    """load model"""
    print("Loading llama model...")
    model_llama = LlamaModel.from_pretrained(llama_model_path)
    model_llama.to("cpu")
    model_llama_state_dict = model_llama.state_dict()

    """load indices and gate weights"""
    hidden_size = model_llama.config.hidden_size
    num_layers = model_llama.config.num_hidden_layers

    for i in tqdm(range(num_layers), desc="loading indices and gate weights"):
        this_layer_index = torch_load_template_file(split_index_path, template, i)
        moe_indices.append(torch.tensor(this_layer_index, dtype=torch.int))

        this_layer_size_expert = Counter(this_layer_index)
        this_layer_size_expert = [this_layer_size_expert[j] for j in range(num_experts)]
        size_experts.append(this_layer_size_expert)

    """build config"""
    print("Buiding llama-moe config...")
    config_llama_moe = LlamaMoEConfig.from_pretrained(llama_model_path)
    config_llama_moe.num_experts = num_experts
    config_llama_moe.num_selects = num_selects
    config_llama_moe.size_experts = size_experts
    config_llama_moe.gates = gate_type
    config_llama_moe.gate_use_softmax = use_softmax
    config_llama_moe.score_scale_factor = (
        1.0 if score_scale_factor is None else score_scale_factor
    )
    config_llama_moe.multiply_gate_scores = multiply_gate_scores

    """initialize moe model"""
    print("Initializing llama-moe model...")
    model_llama_moe = LlamaMoEModel(config_llama_moe)
    model_llama_moe.to("cpu")
    model_llama_moe_state_dict = model_llama_moe.state_dict().copy()

    # fmt: off
    """conversion"""
    print("Locating state dict values...")
    for key in model_llama_state_dict.keys():
        if "mlp" not in key:
            model_llama_moe_state_dict[key] = model_llama_state_dict[key].cpu().half()
        else:
            layer_index = int(key.split(".")[1])
            for expert_index in range(num_experts):
                if "gate" in key:
                    model_llama_moe_state_dict["layers.{}.mlp.calculator.experts.weight_gate.{}".format(layer_index, expert_index)] = model_llama_state_dict[key][moe_indices[layer_index] == expert_index].cpu().half()
                elif "up" in key:
                    model_llama_moe_state_dict["layers.{}.mlp.calculator.experts.weight_up.{}".format(layer_index, expert_index)] = model_llama_state_dict[key][moe_indices[layer_index] == expert_index].cpu().half()
                elif "down" in key:
                    model_llama_moe_state_dict["layers.{}.mlp.calculator.experts.weight_down.{}".format(layer_index, expert_index)] = model_llama_state_dict[key].transpose(0, 1)[moe_indices[layer_index] == expert_index].transpose(0, 1).cpu().half()

    for layer_index in range(num_layers):
        model_llama_moe_state_dict["layers.{}.mlp.gate.weight_noise.weight".format(layer_index)] = torch.zeros((num_experts, hidden_size), requires_grad=True)
    # fmt: on

    print("Converting...")
    model_llama_moe.load_state_dict(model_llama_moe_state_dict)
    model_llama_moe = model_llama_moe.half()

    """save to file"""
    if os.path.exists(save_path):
        print(f'Removed existed files in "{save_path}"')
        shutil.rmtree(save_path)
    os.makedirs(save_path)

    print("Saving converted model...")
    config_llama_moe.save_pretrained(save_path)
    model_llama_moe.save_pretrained(save_path)
    print(f'Converted LlamaMoEModel saved to "{save_path}".')


def convert_llama_model_for_causal_lm(
    llama_model_path,
    split_index_path,
    save_path,
    template,
    num_experts,
    num_selects,
    score_scale_factor=None,
    gate_type="mlp",  # "linear"
    use_softmax=True,
    multiply_gate_scores=True,
):
    """
    LlamaMoEForCausalLM
    """

    moe_indices = []
    size_experts = []

    """load model"""
    print("Loading llama model...")
    model_llama = LlamaForCausalLM.from_pretrained(llama_model_path)
    model_llama.to("cpu")
    model_llama_state_dict = model_llama.state_dict()

    """load indices and gate weights"""
    hidden_size = model_llama.config.hidden_size
    num_layers = model_llama.config.num_hidden_layers

    for i in tqdm(range(num_layers), desc="loading indices and gate weights"):
        this_layer_index = torch_load_template_file(split_index_path, template, i)
        moe_indices.append(torch.tensor(this_layer_index, dtype=torch.int))

        this_layer_size_expert = Counter(this_layer_index)
        this_layer_size_expert = [this_layer_size_expert[j] for j in range(num_experts)]
        size_experts.append(this_layer_size_expert)

    """build config"""
    print("Buiding llama-moe config...")
    config_llama_moe = LlamaMoEConfig.from_pretrained(llama_model_path)
    config_llama_moe.num_experts = num_experts
    config_llama_moe.num_selects = num_selects
    config_llama_moe.size_experts = size_experts
    config_llama_moe.gates = gate_type
    config_llama_moe.gate_use_softmax = use_softmax
    config_llama_moe.score_scale_factor = (
        1.0 if score_scale_factor is None else score_scale_factor
    )
    config_llama_moe.multiply_gate_scores = multiply_gate_scores

    """initialize moe model"""
    print("Initializing llama-moe model...")
    model_llama_moe = LlamaMoEForCausalLM(config_llama_moe)
    model_llama_moe.to("cpu")
    model_llama_moe_state_dict = model_llama_moe.state_dict().copy()

    # fmt: off
    """conversion"""
    print("Locating state dict values...")
    for key in model_llama_state_dict.keys():
        if "mlp" not in key:
            model_llama_moe_state_dict[key] = model_llama_state_dict[key].cpu().half()
        else:
            layer_index = int(key.split(".")[2])
            for expert_index in range(num_experts):
                if "gate" in key:
                    model_llama_moe_state_dict["model.layers.{}.mlp.calculator.experts.weight_gate.{}".format(layer_index, expert_index)] = model_llama_state_dict[key][moe_indices[layer_index] == expert_index].cpu().half()
                elif "up" in key:
                    model_llama_moe_state_dict["model.layers.{}.mlp.calculator.experts.weight_up.{}".format(layer_index, expert_index)] = model_llama_state_dict[key][moe_indices[layer_index] == expert_index].cpu().half()
                elif "down" in key:
                    model_llama_moe_state_dict["model.layers.{}.mlp.calculator.experts.weight_down.{}".format(layer_index, expert_index)] = model_llama_state_dict[key].transpose(0, 1)[moe_indices[layer_index] == expert_index].transpose(0, 1).cpu().half()

    for layer_index in range(num_layers):
        model_llama_moe_state_dict["model.layers.{}.mlp.gate.weight_noise.weight".format(layer_index)] = torch.zeros((num_experts, hidden_size), requires_grad=True)
    # fmt: on

    print("Converting...")
    model_llama_moe.load_state_dict(model_llama_moe_state_dict)
    model_llama_moe = model_llama_moe.half()

    """save to file"""
    if os.path.exists(save_path):
        print(f'Removed existed files in "{save_path}"')
        shutil.rmtree(save_path)
    os.makedirs(save_path)

    print("Saving converted model...")
    config_llama_moe.save_pretrained(save_path)
    model_llama_moe.save_pretrained(save_path)
    print(f'Converted LlamaMoEForCausalLM saved to "{save_path}".')


def convert_llama_model_for_sequence_classification(
    llama_model_path,
    split_index_path,
    save_path,
    template,
    num_experts,
    num_selects,
    score_scale_factor=None,
    gate_type="mlp",  # "linear"
    use_softmax=True,
    multiply_gate_scores=True,
):
    """
    LlamaMoEForSequenceClassification
    """

    moe_indices = []
    size_experts = []

    """load model"""
    print("Loading llama model...")
    model_llama = LlamaForSequenceClassification.from_pretrained(llama_model_path)
    model_llama.to("cpu")
    model_llama_state_dict = model_llama.state_dict()

    """load indices and gate weights"""
    hidden_size = model_llama.config.hidden_size
    num_layers = model_llama.config.num_hidden_layers

    for i in tqdm(range(num_layers), desc="loading indices and gate weights"):
        this_layer_index = torch_load_template_file(split_index_path, template, i)
        moe_indices.append(torch.tensor(this_layer_index, dtype=torch.int))

        this_layer_size_expert = Counter(this_layer_index)
        this_layer_size_expert = [this_layer_size_expert[j] for j in range(num_experts)]
        size_experts.append(this_layer_size_expert)

    """build config"""
    print("Buiding llama-moe config...")
    config_llama_moe = LlamaMoEConfig.from_pretrained(llama_model_path)
    config_llama_moe.num_experts = num_experts
    config_llama_moe.num_selects = num_selects
    config_llama_moe.size_experts = size_experts
    config_llama_moe.gates = gate_type
    config_llama_moe.gate_use_softmax = use_softmax
    config_llama_moe.score_scale_factor = (
        1.0 if score_scale_factor is None else score_scale_factor
    )
    config_llama_moe.multiply_gate_scores = multiply_gate_scores

    """initialize moe model"""
    print("Initializing llama-moe model...")
    model_llama_moe = LlamaMoEForSequenceClassification(config_llama_moe)
    model_llama_moe.to("cpu")
    model_llama_moe_state_dict = model_llama_moe.state_dict().copy()

    # fmt: off
    """conversion"""
    print("Locating state dict values...")
    for key in model_llama_state_dict.keys():
        if "mlp" not in key:
            model_llama_moe_state_dict[key] = model_llama_state_dict[key].cpu().half()
        else:
            layer_index = int(key.split(".")[2])
            for expert_index in range(num_experts):
                if "gate" in key:
                    model_llama_moe_state_dict["model.layers.{}.mlp.calculator.experts.weight_gate.{}".format(layer_index, expert_index)] = model_llama_state_dict[key][moe_indices[layer_index] == expert_index].cpu().half()
                elif "up" in key:
                    model_llama_moe_state_dict["model.layers.{}.mlp.calculator.experts.weight_up.{}".format(layer_index, expert_index)] = model_llama_state_dict[key][moe_indices[layer_index] == expert_index].cpu().half()
                elif "down" in key:
                    model_llama_moe_state_dict["model.layers.{}.mlp.calculator.experts.weight_down.{}".format(layer_index, expert_index)] = model_llama_state_dict[key].transpose(0, 1)[moe_indices[layer_index] == expert_index].transpose(0, 1).cpu().half()

    for layer_index in range(num_layers):
        model_llama_moe_state_dict["model.layers.{}.mlp.gate.weight_noise.weight".format(layer_index)] = torch.zeros((num_experts, hidden_size), requires_grad=True)
    # fmt: on

    print("Converting...")
    model_llama_moe.load_state_dict(model_llama_moe_state_dict)
    model_llama_moe = model_llama_moe.half()

    """save to file"""
    if os.path.exists(save_path):
        print(f'Removed existed files in "{save_path}"')
        shutil.rmtree(save_path)
    os.makedirs(save_path)

    print("Saving converted model...")
    config_llama_moe.save_pretrained(save_path)
    model_llama_moe.save_pretrained(save_path)
    print(f'Converted LlamaMoEForSequenceClassification saved to "{save_path}".')


if __name__ == "__main__":
    llama_model_path = "/home/data/models/llama-transformers/7B/"
    split_index_path = "/home/dongdz/workspace/moefication/llama_moe_temp_files/llama_7B-8Expert-Split-Clustering/"  # split
    save_path = "/home/data/models/llama-moe-transformers/7B/"
    template = "layers.{}.mlp.gate_proj.weight"
    num_experts = 8
    num_selects = 2
    score_scale_factor = 8.0
    use_random_gate = False

    convert_llama_model(
        llama_model_path,
        split_index_path,
        save_path,
        template,
        num_experts,
        num_selects,
        score_scale_factor=score_scale_factor,
    )

    # load test
    model_llama_moe = LlamaMoEForCausalLM.from_pretrained(save_path)
    print(model_llama_moe)
