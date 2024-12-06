# Expert Construction of LLaMA-MoE

This documentation provides the procedures to convert a LLaMA model to LLaMA-MoE.



## Procedures

The conversion from LLaMA to LLaMA-MoE consists of two steps:

1. **Split.** Create indices sets $S_1,S_2,\dots,S_n$ (Eq. 5 in the technical report) for the each FFN layer in LLaMA. The indices sets indicate the intermediate neurons that should be assigned to experts. Save the indices sets to disk.
2. **Convert.** Create a LLaMA-MoE model from an existing LLaMA checkpoint. Reinitialize the LLaMA-MoE experts by selecting the corressponding neurons in the indices sets. Save the initialized LLaMA-MoE model to disk.



### Split

#### Random Split (Neuron-Independent)

To randomly split the intermediate neurons in FFNs, you can run:

```shell
bash ./scripts/expert_construction/split/run_split_random.sh
```

Remember to change the following variables:

```shell
num_experts="" # number of experts in each MoE layer

model_path="" # path to the LLaMA checkpoint
save_path="" # path to save the indices sets
```



#### Clustering Split (Neuron-Independent)

To split the intermediate neurons in FFNs by k-means clustering, you can run:

```shell
bash ./scripts/expert_construction/split/run_split_clustering.sh
```

Remember to change the following variables:

```shell
num_experts="" # number of experts in each MoE layer

model_path="" # path to the LLaMA checkpoint
save_path="" # path to save the indices sets

metric="" # metric for clustering, choices: `l2` `cos`
proj_type="" # weights to perform clustering, choices: `up_proj` `gate_proj`
```



#### Co-activation Graph Split (Neuron-Independent)

> This part is not included in our technical report.

We also implenmented the co-activation graph based method in [MoEfication](https://arxiv.org/abs/2110.01786) here.

You need to install [METIS](http://glaros.dtc.umn.edu/gkhome/metis/metis/download) first. Then you can run to following script to perform splitting:

```shell
bash ./scripts/expert_construction/split/run_split_graph.sh
```

Remember to change the following variables:

```shell
num_experts="" # number of experts in each MoE layer

model_path="" # path to the LLaMA checkpoint
save_path="" # path to save the indices sets

metric="" # metric to measure the sparsity, choices: `l1_norm` `l2_norm` `plain`
proj_type="" # weights to perform clustering, choices: `up_proj` `gate_proj`
```



#### Gradient Split

Before performing gradient-based splitting (Eq. 8 in the technical report), you need to prepare a bunch of pretraining data and group them into different clusters by running:

```shell
python smoe/entrypoint/expert_construction/text_clustering.py
```

Then, you need to run the following script to get the importance vector $v$ for the intermediate neurons in each layer:

```shell
bash scripts/expert_construction/split/run_split_gradient_get_grads.sh
```

Remember to change the following variables:

```shell
dataset_dir="" # path to clustered data
pretrained_model="" # path to the LLaMA checkpoint
tokenizer_path="" # path to the LLaMA tokenizer
save_path="" # path to save the indices sets

accumulate_level="" # should be set to `sample`
kernel="" # should be set to `l1_norm`
importance_type="" # should be set to `feature_change`
```



##### Neuron Independent

> This part is not included in our technical report.

You can also split the intermediate neurons in a neuron-independent manner by treating the expert split as a task assignment problem. To perform the split, you can run:

```shell
bash ./scripts/expert_construction/split/run_split_gradient.sh
```

Remember to change the following variables:

```shell
expert_num="" # number of experts in each MoE layer
expert_size="" # intermediate neurons in each expert
share_neurons="False" ######### SET AS FLASE TO BE NEURON-INDEPENDENT #########

model_path="" # path to the LLaMA checkpoint
score_file_path="" # path to the score files generated above
save_path="" # path to save the indices sets
visualization_path="" # path to save the visualization results

criterion="" # criterion to judge the importance of neurons, should be set to `max`
proj_type="" # weights to perform clustering, choices: `up_proj` `gate_proj`
```



##### Inner-Sharing

Here we use the same entrance as the **Neuron Independent** strategy above for gradient split.

```shell
bash ./scripts/expert_construction/split/run_split_gradient.sh
```

Remember to change the following variables:

```shell
expert_num="" # number of experts in each MoE layer
expert_size="" # intermediate neurons in each expert
share_neurons="True" ######### SET AS TRUE TO BE INNER-SHARING #########

model_path="" # path to the LLaMA checkpoint
score_file_path="" # path to the score files generated above
save_path="" # path to save the indices sets
visualization_path="" # path to save the visualization results

criterion="" # criterion to judge the importance of neurons, should be set to `max`
proj_type="" # weights to perform clustering, choices: `up_proj` `gate_proj`
```



##### Inter-Sharing (Residual MoE)

You can run the following script to perform inter-sharing split:

```shell
bash ./scripts/expert_construction/split/run_split_gradient_residual.sh
```

Remember to change the following variables:

```shell
expert_num_moe="" # number of non-residual experts
expert_num_residual="" # number of residual experts
expert_size="" # intermediate neurons in each expert
share_neurons="" # Whether to share neurons in non-residual experts

model_path="" # path to the LLaMA checkpoint
score_file_path="" # path to the score files generated above
save_path="" # path to save the indices sets
visualization_path="" # path to save the visualization results

criterion="" # criterion to judge the importance of neurons, should be set to `max`
proj_type="" # weights to perform clustering, choices: `up_proj` `gate_proj`
```



### Convert

#### Convert LLaMA-MoE from Neuron-Independent Methods

Run the following script:

```shell
bash ./scripts/expert_construction/convert/run_convert.sh
```



#### Convert LLaMA-MoE from Inner-Sharing Methods

Run the following script:

```shell
bash ./scripts/expert_construction/convert/run_convert_gradient.sh
```



#### Convert LLaMA-MoE from Inter-Sharing Methods (Residual MoE)

Run the following script:

```shell
bash ./scripts/expert_construction/convert/run_convert_gradient_residual.sh
```



## File Structure

```
--smoe
	-- scripts
        -- expert_construction
            -- convert
            -- split
    -- smoe
        -- entrypoint
            -- expert_construction
```
