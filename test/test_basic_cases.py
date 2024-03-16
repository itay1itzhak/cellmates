import pytest
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from cellmates.data.stubs import (
    generate_dataset_for_distances,
    generate_dataset_for_cell_type,
    generate_dataset_for_n_cells_test,
)
from cellmates.train import train_model
from cellmates.data.dataset import collate_fn


def test_distances_influence(model_config):
    """
    Tests that the model correctly learns the relationship between cell distances and division.
    """
    test_dataset = generate_dataset_for_distances()

    trained_model = train_model(
        train_dataset=test_dataset,
        valid_dataset=test_dataset,
        num_epochs=20,
        batch_size=1,  # len(test_dataset),
        **model_config,
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=len(test_dataset), collate_fn=collate_fn
    )
    cell_types_BL, distances_BLL, target = next(iter(test_dataloader))

    output_B1 = trained_model(cell_types_BL, distances_BLL)

    # print the output with context
    print(f"test_distances_influence predictions:\n{torch.sigmoid(output_B1)}")
    print(f"test_distances_influence target:\n{target}")

    # Output should match targets closely
    assert torch.all(torch.isclose(torch.sigmoid(output_B1), target.float(), atol=0.05))


def test_cell_type_influence(model_config):
    """
    Tests that the model correctly learns the relationship between cell types and division.
    """
    test_dataset = generate_dataset_for_cell_type()

    trained_model = train_model(
        train_dataset=test_dataset,
        valid_dataset=test_dataset,
        num_epochs=10,
        batch_size=1,  # len(test_dataset),
        **model_config,
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=len(test_dataset), collate_fn=collate_fn
    )
    cell_types_BL, distances_BLL, target = next(iter(test_dataloader))

    output_B1 = trained_model(cell_types_BL, distances_BLL)

    # print the output with context
    print(f"test_cell_type_influence predictions:\n{torch.sigmoid(output_B1)}")
    print(f"test_cell_type_influence target:\n{target}")

    # Output should match targets closely
    assert torch.all(torch.isclose(torch.sigmoid(output_B1), target.float(), atol=0.05))


def test_n_cells_influence(model_config):
    """
    Tests that the model correctly learns the relationship between the number of cells and division.
    """
    test_dataset = generate_dataset_for_n_cells_test(5)

    trained_model = train_model(
        train_dataset=test_dataset,
        valid_dataset=test_dataset,
        num_epochs=40,  # 20,
        batch_size=1,  # len(test_dataset),
        **model_config,
    )
    test_dataloader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn)
    # cell_types_BL, distances_BLL, target = next(iter(test_dataloader))
    all_outputs = []
    all_targets = []
    for cell_types_BL, distances_BLL, target in test_dataloader:
        output_B1 = trained_model(cell_types_BL, distances_BLL)
        all_outputs.append(output_B1)
        all_targets.append(target)

    output_B1 = torch.cat(all_outputs, dim=0)
    target = torch.cat(all_targets, dim=0)

    # print the output with context
    print(f"test_n_cells_influence predictions:\n{torch.sigmoid(output_B1)}")
    print(f"test_n_cells_influence target:\n{target}")

    # Output should match targets closely
    assert torch.all(torch.isclose(torch.sigmoid(output_B1), target.float(), atol=0.05))


# run the tests without pytest for debugging
# model_config = {
#     "D": 512,
#     "H": 16,
#     "K": 32,
#     "F": 2048,
#     "M": 512,
#     "n_cell_types": 6,
#     "num_encoder_layers": 8,
#     "dropout_p": 0.1,
#     "activation": torch.relu,
#     "layer_norm_eps": 1e-5,
#     "batch_first": True,
#     "norm_first": True,
#     "bias": True,
#     "wandb": True,
#     "experiment_name": "test_distances_influence",
# }
# test_distances_influence(model_config)
# model_config["experiment_name"] = "test_cell_type_influence"
# test_cell_type_influence(model_config)
# model_config["experiment_name"] = "test_n_cells_influence"
# test_n_cells_influence(model_config)
