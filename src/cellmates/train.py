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
    train_dataset: CellMatesDataset | str,
    valid_dataset: CellMatesDataset | str,
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
    use_wandb: bool = False,
    save_checkpoint: bool = False,
    experiment_name: str = "cellmates",
    device: str | None = None,
    learning_rate: float = 1e-3,
):
    """Train a CellMatesTransformer model.

    Args:
        train_dataset (CellMatesDataset | str): An instance of CellMatesDataset or a path to a saved CellMatesDataset.
        valid_dataset (CellMatesDataset | str): _description_
        batch_size (int, optional): _description_. Defaults to 32.
        num_epochs (int, optional): _description_. Defaults to 100.
        D (int, optional): model dimension (cell-type embedding dimension). Defaults to 512.
        H (int, optional): number of attention heads in a layer. Defaults to 16.
        K (int, optional): size of each attention key or value (also dimension of distance embeddings). Defaults to 512.
        F (int, optional): feedfoward subnetwork hidden size. Defaults to 2048.
        M (int, optional): mlp hidden size. Defaults to 512.
        n_cell_types (int, optional): _description_. Defaults to 6.
        num_encoder_layers (int, optional): _description_. Defaults to 8.
        dropout_p (float, optional): _description_. Defaults to 0.1.
        activation (str, optional): _description_. Defaults to "relu".
        layer_norm_eps (float, optional): _description_. Defaults to 1e-5.
        batch_first (bool, optional): _description_. Defaults to True.
        norm_first (bool, optional): _description_. Defaults to False.
        bias (bool, optional): _description_. Defaults to True.
        checkpoint_path (str, optional): _description_. Defaults to None.
        wandb (bool, optional): _description_. Defaults to False.
        save_checkpoint (bool, optional): _description_. Defaults to False.
        experiment_name (str, optional): _description_. Defaults to "cellmates".
        device (str | None, optional): _description_. Defaults to None.
        learning_rate (float, optional): _description_. Defaults to 1e-3.

    Returns:
        _type_: _description_
    """
    # load datasets if paths were provided
    if type(train_dataset) is str:
        train_dataset = CellMatesDataset.load(train_dataset)
    if type(valid_dataset) is str:
        valid_dataset = CellMatesDataset.load(valid_dataset)

    # init dataloaders:
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
            "device": device,
        },
        learning_rate=learning_rate,
    )

    # Load from checkpoint if provided
    if checkpoint_path:
        model.load_from_checkpoint(checkpoint_path)

    # init callbacks:
    if save_checkpoint:
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            dirpath="checkpoints",
            filename="cellmates-{experiment_name}" + "-{epoch:02d}-{val_loss:.2f}",
            save_top_k=3,
            mode="min",
        )
        callbacks = [checkpoint_callback]
    else:
        callbacks = None

    # Define PyTorch Lightning trainer
    if use_wandb:
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
