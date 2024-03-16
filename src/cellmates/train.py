import os
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from fire import Fire
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger, CSVLogger

from cellmates.data.sample import Sample
from cellmates.data.dataset import CellMatesDataset, collate_fn
from cellmates.model.LightningCellMates import LightningCellMates


def train_model(
    train_dataset: CellMatesDataset = None,
    valid_dataset: CellMatesDataset = None,
    train_dataset_path: str = None,
    valid_dataset_path: str = None,
    batch_size: int = 32,
    num_epochs: int = 100,
    D: int = 512,
    H: int = 16,
    K: int = 512,
    F: int = 2048,
    M: int = 512,
    n_cell_types: int = 6,
    num_encoder_layers: int = 8,
    dropout_p: float = 0.1,
    activation: str = "relu",
    layer_norm_eps: float = 1e-5,
    batch_first: bool = True,
    norm_first: bool = False,
    bias: bool = True,
    checkpoint_path: str = None,
    wandb: bool = False,
    save_checkpoint: bool = False,
    experiment_name: str = "cellmates",
):

    # Load datasets if not provided
    if train_dataset is None and train_dataset_path is None:
        raise ValueError("train_dataset or train_dataset_path must be provided")
    if valid_dataset is None and valid_dataset_path is None:
        raise ValueError("valid_dataset or valid_dataset_path must be provided")
    if train_dataset is None:  # load dataset from path
        train_dataset = CellMatesDataset.load(train_dataset_path)
    if valid_dataset is None:  # load dataset from path
        valid_dataset = CellMatesDataset.load(valid_dataset_path)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, collate_fn=collate_fn
    )

    # Initialize the Model
    model = LightningCellMates(
        model_config={
            "D": D,
            "H": H,
            "K": K,
            "F": F,
            "M": M,
            "n_cell_types": n_cell_types,
            "num_encoder_layers": num_encoder_layers,
            "dropout_p": dropout_p,
            "activation": torch.relu if activation == "relu" else torch.tanh,
            "layer_norm_eps": layer_norm_eps,
            "batch_first": batch_first,
            "norm_first": norm_first,
            "bias": bias,
        }
    )

    # Load from checkpoint if provided
    if checkpoint_path is not None and os.path.isfile(checkpoint_path):
        model.load_from_checkpoint(checkpoint_path)

    # Create callback for model checkpointing
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="checkpoints",
        filename="cellmates-{experiment_name}" + "-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        mode="min",
    )

    # Create loggers (both Weights&Biases and local CSV file)
    wandb_logger = WandbLogger(
        name=f"cellmates_train-{experiment_name}", project="cellmates"
    )

    # log train loss
    wandb_logger.log_hyperparams(
        {
            "D": D,
            "H": H,
            "K": K,
            "F": F,
            "M": M,
            "n_cell_types": n_cell_types,
            "num_encoder_layers": num_encoder_layers,
            "dropout_p": dropout_p,
            "activation": activation,
            "layer_norm_eps": layer_norm_eps,
            "batch_first": batch_first,
            "norm_first": norm_first,
            "bias": bias,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
        }
    )
    # Save the model checkpoint
    if save_checkpoint:
        callbacks = [checkpoint_callback]
    else:
        callbacks = None

    # Define PyTorch Lightning trainer
    if wandb:
        trainer = pl.Trainer(
            max_epochs=num_epochs,
            logger=[wandb_logger],
            callbacks=callbacks,
        )
    else:
        trainer = pl.Trainer(
            max_epochs=num_epochs,
            callbacks=callbacks,
        )

    # Train the model
    trainer.fit(model, train_loader, valid_loader)

    return model


if __name__ == "__main__":
    Fire(train_model)
