import pytest
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from cellmates.train import train_model
from cellmates.data.dataset import CellMatesDataset
from cellmates.data.dataset import collate_fn


def test_single_instance_overfit(single_sample, model_config):
    # Wrap the sample instance in a dataset
    single_instance_dataset = CellMatesDataset([single_sample])

    # We don't have a separate validation set, so we're using the same dataset for validation
    trained_model = train_model(
        train_dataset=single_instance_dataset,
        valid_dataset=single_instance_dataset,
        num_epochs=5,
        batch_size=1,
        **model_config,
    )

    # Get the batch from the DataLoader (a batch will include only one sample here)
    single_instance_loader = DataLoader(
        single_instance_dataset, batch_size=1, collate_fn=collate_fn
    )

    batch = next(iter(single_instance_loader))

    # fetch batch components:
    cell_types_BL = batch["cell_types_BL"]
    distances_BLL = batch["distances_BLL"]
    target = batch["is_dividing_B"]
    padding_mask_BL = batch["padding_mask_BL"]

    # Test that we are reasonably close to 100% accuracy
    trained_model.eval()
    output_B1 = trained_model(
        cell_types_BL=cell_types_BL,
        distances_BLL=distances_BLL,
        padding_mask_BL=padding_mask_BL,
    )

    # print the output with context
    print(f"test_single_instance_overfit accuracy: {torch.sigmoid(output_B1).item()}")

    assert torch.sigmoid(output_B1) > 0.95


def test_batch_overfit(single_sample, model_config):
    # Create a batch of similar instances
    batch_samples = [single_sample for _ in range(4)]

    # Wrap the batch of instances in a dataset
    batch_dataset = CellMatesDataset(batch_samples)

    # We don't have a separate validation set, so we're using the same dataset for validation
    trained_model = train_model(
        **model_config,
        train_dataset=batch_dataset,
        valid_dataset=batch_dataset,
        num_epochs=5,
        batch_size=len(batch_samples),
    )

    # Get the batch from the DataLoader
    batch_loader = DataLoader(
        batch_dataset, batch_size=len(batch_samples), collate_fn=collate_fn
    )
    cell_types_BL, distances_BLL, target = next(iter(batch_loader))

    # Test that we are reasonably close to 100% accuracy
    output_B1 = trained_model(cell_types_BL, distances_BLL)

    # print the output with context
    print(f"test_batch_overfit accuracy: {torch.sigmoid(output_B1)}")

    assert torch.sigmoid(output_B1).mean() > 0.95


# run the tests without pytest for debugging
# (need also to comment out the pytest.fixture decorators)
#   test_single_instance_overfit(single_sample(), model_config())
# test_batch_overfit(single_sample(), model_config())
