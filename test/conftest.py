import pytest
import torch
from cellmates.data.sample import Sample


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
    model_config = {
        "D": 512,
        "H": 16,
        "K": 32,
        "F": 2048,
        "M": 512,
        "n_cell_types": 6,
        "num_encoder_layers": 8,
        "dropout_p": 0.1,
        "activation": torch.relu,
        "layer_norm_eps": 1e-5,
        "batch_first": True,
        "norm_first": True,
        "bias": True,
    }
    return model_config
