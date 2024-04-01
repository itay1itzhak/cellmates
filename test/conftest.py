import pytest
import torch
from cellmates.data.sample import Sample
from cellmates.utils import N_CELL_TYPES


@pytest.fixture
def single_sample():
    return Sample(
        cell_types=[1, 2, 3, 4, 5],
        distances=torch.tensor(
            [
                [0.0, 1.0, 2.0, 3.0, 4.0],
                [1.0, 0.0, 1.0, 2.0, 3.0],
                [2.0, 1.0, 0.0, 1.0, 2.0],
                [3.0, 2.0, 1.0, 0.0, 1.0],
                [4.0, 3.0, 2.0, 1.0, 0.0],
            ]
        ),
        responder_cell_type=1,
        is_dividing=True,
    )


@pytest.fixture
def model_config():

    # TODO add - if torch.cuda.is_available(): device = "cuda:0"
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    model_config = {
        "D": 512,
        "H": 16,
        "K": 32,
        "F": 1024,
        "M": 512,
        "n_cell_types": N_CELL_TYPES,
        "num_encoder_layers": 4,
        "dropout_p": 0.1,
        "activation": torch.relu,
        "layer_norm_eps": 1e-5,
        "batch_first": True,
        "norm_first": True,
        "bias": True,
        "device": device,
        "use_wandb": True,
    }
    return model_config
