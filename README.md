# Cellmates: Understanding Cell-Cell Communication Using Transformer
This repository contains the code base for the Cellmates project.
This project was done by Jonathan Somer and Itay Itzhak as part of the Transformers course at the Technion.

## Introduction
Cellmates is a project that aims to understand cell-cell communication using Transformer models. In this project, we explore how Transformer models can be applied to analyze and interpret the interactions between cells in biological systems.

## Installation
To use Cellmates, you need to have the following dependencies installed:
- Python 3.11
- PyTorch
- Install the dataset repository from https://github.com/JonathanSomer/tumor-dynamical-modeling

You can install the remaining required dependencies by running the following command:
```bash
pip install -r requirements.txt
```
## Training
To train a new Cellmates models use the pipeline.py script file with required properties, e.g. -

```bash
python pipeline.py \
--batch_size 32 \
--n_epochs 30 \
--D 512 \
--H 32 \
--F 1024 \
--M 512 \
--num_encoder_layers 4 \
--learning_rate 0.0005 \
--experiment_name Cellmate_training \
--save_checkpoint \
--log_every_n_steps 100 \
```

## Evaluation
To evalute the Cellmates model run the cross validatoin k-fold script, e.g. -

```bash
python kfold_cv.py \
--batch_size 32 \
--n_epochs 30 \
--D 512 \
--H 32 \
--F 1024 \
--M 512 \
--num_encoder_layers 4 \
--learning_rate 0.0005 \
--experiment_name Cellmate_kfold_cv \
--save_checkpoint \
--log_every_n_steps 100 \
--n_splits 10 \
```

To recreate our analysis figures you can use the functions avaialbe in the 04-insights.ipynb notebook.


## License

Cellmates is not free to use and intended for the project course only.