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

import pandas as pd
import pytest


@pytest.mark.parametrize(
    "load_dataset_func, n_steps, learning_rate, num_encoder_layers",
    [
        # (generate_dataset_for_distances, 20, 1e-3, 4),
        # (generate_dataset_for_cell_type, 20, 1e-3, 4),
        (generate_dataset_for_n_cells_test, 100, 1e-3, 0),
    ],
)
def test_toy_dataset(
    model_config, load_dataset_func, n_steps, learning_rate, num_encoder_layers
):
    """
    Tests that the model correctly learns the relationship between cell distances and division.
    """
    test_dataset = load_dataset_func()

    model_config["num_encoder_layers"] = num_encoder_layers

    trained_model = train_model(
        train_dataset=test_dataset,
        valid_dataset=test_dataset,
        num_epochs=n_steps,
        batch_size=len(test_dataset),
        **model_config,
        learning_rate=learning_rate
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=len(test_dataset), collate_fn=collate_fn
    )

    # move test batch to model device:
    device = model_config["device"]
    batch = next(iter(test_dataloader))
    batch = {k: v.to(device) for k, v in batch.items()}

    # fetch batch components:
    cell_types_BL = batch["cell_types_BL"]
    distances_BLL = batch["distances_BLL"]
    target = batch["is_dividing_B"]
    padding_mask_BL = batch["padding_mask_BL"]

    # compute final output:
    trained_model.to(device).eval()  # disables dropout
    output_B1 = trained_model(
        cell_types_BL=cell_types_BL,
        distances_BLL=distances_BLL,
        padding_mask_BL=padding_mask_BL,
    )

    model_probs = torch.sigmoid(output_B1).squeeze()

    print(
        pd.DataFrame(
            {
                "model_probs": model_probs.cpu().detach().numpy().round(3),
                "true_label": target.cpu().numpy(),
            }
        )
    )

    # Output should match targets closely
    assert torch.all(torch.isclose(model_probs, target.float(), atol=0.01))
