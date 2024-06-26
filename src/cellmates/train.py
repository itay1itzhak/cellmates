import os
import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader
from fire import Fire
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger, CSVLogger

from cellmates.data.sample import Sample
from cellmates.data.dataset import CellMatesDataset, collate_fn
from cellmates.model.LightningCellMates import LightningCellMates
from cellmates.utils import N_CELL_TYPES


def train_model(
    train_ds: CellMatesDataset | str,
    val_ds: CellMatesDataset | str,
    test_ds: CellMatesDataset | None = None,
    batch_size: int = 32,
    n_epochs: int = 100,
    D: int = 512,
    H: int = 16,
    K: int = 512,
    F: int = 2048,
    M: int = 512,
    n_cell_types: int = N_CELL_TYPES,
    num_encoder_layers: int = 8,
    dropout_p: float = 0.28,  # 0.1,
    activation: str = "relu",
    layer_norm_eps: float = 0.000294,  # 1e-5,
    batch_first: bool = True,
    norm_first: bool = True,
    bias: bool = True,
    checkpoint_path: str = None,
    use_wandb: bool = True,
    save_checkpoint: bool = False,
    experiment_name: str = "cellmates",
    device: str | None = None,
    learning_rate: float = 1e-3,
    num_workers: int = 4,
    log_every_n_steps: int = 1,
    num_sanity_val_steps: int = 2,
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
    pl.seed_everything(seed=42)

    # load datasets if paths were provided
    if type(train_ds) is str:
        train_ds = CellMatesDataset.load(train_ds)
    if type(val_ds) is str:
        val_ds = CellMatesDataset.load(val_ds)

    # init dataloaders:
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, collate_fn=collate_fn, num_workers=num_workers
    )

    if test_ds is not None:
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            collate_fn=collate_fn,
            num_workers=num_workers,
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
            # dirpath="checkpoints",
            # filename=f"cellmates-{experiment_name}" + "-{epoch:02d}-{val_loss:.2f}",
            dirpath=f"checkpoints/{experiment_name}",
            filename="-{epoch:02d}-{val_loss:.2f}",
            save_top_k=1,
            mode="min",
            save_last=True,  # checkpoint the last epoch to be able to resume training named as "last.ckpt"
            enable_version_counter=False,  # do not append version number to checkpoint filename to enable resuming training
        )
        callbacks = [checkpoint_callback]
        # if last checkpoint exists, load it
        ckpt_path = os.path.join(f"checkpoints/{experiment_name}", "last.ckpt")
        # check if exists
        if not os.path.exists(ckpt_path):
            ckpt_path = None
        else:
            print(f"Resuming training from checkpoint: {ckpt_path}")
    else:
        callbacks = None
        ckpt_path = None

    # Define PyTorch Lightning trainer
    logger = None
    if use_wandb:
        # Create loggers (both Weights&Biases and local CSV file)
        logger = WandbLogger(
            name=f"cellmates_train-{experiment_name}", project="cellmates"
        )

        # log train loss
        logger.log_hyperparams(
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
                "num_epochs": n_epochs,
                "learning_rate": learning_rate,
            }
        )

    # https://github.com/Lightning-AI/pytorch-lightning/discussions/5796
    accumulate_grad_batches = 1024 // (
        batch_size * max(1, torch.cuda.device_count())
    )  # support cpu and gpu training

    trainer = pl.Trainer(
        max_epochs=n_epochs,
        logger=logger,
        callbacks=callbacks,
        accumulate_grad_batches=accumulate_grad_batches,
        log_every_n_steps=log_every_n_steps,
        # overfit_batches=3, # For debugging
        # devices=1
        default_root_dir=os.path.join("checkpoints", experiment_name),
        num_sanity_val_steps=num_sanity_val_steps,
    )

    # print training details in one line
    print(
        f"Training model with D={D}, H={H}, K={K}, F={F}, M={M}, num_encoder_layers={num_encoder_layers}, dropout_p={dropout_p}, activation={activation}, layer_norm_eps={layer_norm_eps}, batch_first={batch_first}, norm_first={norm_first}, bias={bias}, batch_size={batch_size}, num_epochs={n_epochs}, learning_rate={learning_rate}"
    )

    # Train the model
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        # add checkpoint if exists, else start from scratch
        ckpt_path=ckpt_path,
    )

    if test_ds is not None:
        # fetches best checkpoint and computes loss:
        test_loss = trainer.test(dataloaders=test_loader)[0]["test_loss"]
        return model, test_loss
    else:
        return model


if __name__ == "__main__":
    Fire(train_model)
