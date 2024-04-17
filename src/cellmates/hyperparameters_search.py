# import lightning as L
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

# import lightning.pytorch as L
from lightning.pytorch.loggers import WandbLogger
import torch
from torch.utils.data import DataLoader
import os
from fire import Fire
from cellmates.utils import N_CELL_TYPES

# import lightning.pytorch as pl
import lightning.pytorch as pl
from cellmates.data.dataset import CellMatesDataset, collate_fn
from cellmates.model.LightningCellMates import LightningCellMates
from tabulate import tabulate

from tdm.cell_types import FIBROBLAST
from cellmates.pipeline import get_datasets_split


BATCHSIZE = 32
EPOCHS = 40
DIR = os.getcwd()
num_workers = 8
pl.seed_everything(42)
RESPONDER_CELL_TYPE = None
USE_TOY_SIZE = None


def objective(trial: optuna.trial.Trial) -> float:
    # We optimize for every parameter in the model
    num_encoder_layers = trial.suggest_categorical("num_encoder_layers", [1, 2, 4])
    D = trial.suggest_categorical("D", [128, 256, 512])
    H = trial.suggest_categorical("H", [8, 16, 32])
    K = D // H
    F = trial.suggest_categorical("F", [512, 1024])
    M = trial.suggest_categorical("M", [512, 1024, 2048])
    n_cell_types = N_CELL_TYPES
    dropout_p = trial.suggest_float("dropout_p", 0.1, 0.3)
    # activation = trial.suggest_categorical("activation", ["relu", "gelu"])
    layer_norm_eps = trial.suggest_float("layer_norm_eps", 1e-5, 1e-3)
    batch_first = True
    norm_first = True
    bias = True
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3)

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
            "layer_norm_eps": layer_norm_eps,
            "batch_first": batch_first,
            # "activation": activation,
            "norm_first": norm_first,
            "bias": bias,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        },
        learning_rate=learning_rate,
    )
    train_ds, val_ds, _ = get_datasets_split(
        RESPONDER_CELL_TYPE, use_toy_size=USE_TOY_SIZE
    )
    # init dataloaders:
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCHSIZE,
        collate_fn=collate_fn,
        num_workers=num_workers,
        shuffle=True,
    )
    valid_loader = DataLoader(
        val_ds, batch_size=BATCHSIZE, collate_fn=collate_fn, num_workers=num_workers
    )

    logger = WandbLogger(name=f"cellmates_sweep-", project="cellmates")

    # stop training if val_loss does not improve for 3 epochs
    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=0.0005, patience=3, verbose=False, mode="min"
    )

    trainer = pl.Trainer(
        logger=logger,
        enable_checkpointing=False,
        max_epochs=EPOCHS,
        accelerator="auto",
        callbacks=[
            PyTorchLightningPruningCallback(trial, monitor="val_loss"),
            early_stop_callback,
        ],
        accumulate_grad_batches=1024 // BATCHSIZE,
    )
    hyperparameters = dict(
        num_encoder_layers=num_encoder_layers,
        D=D,
        H=H,
        K=K,
        F=F,
        M=M,
        n_cell_types=n_cell_types,
        dropout_p=dropout_p,
        # activation=activation,
        layer_norm_eps=layer_norm_eps,
        batch_first=batch_first,
        norm_first=norm_first,
        bias=bias,
        learning_rate=learning_rate,
    )
    trainer.logger.log_hyperparams(hyperparameters)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)

    return trainer.callback_metrics["val_loss"].item()


def main(
    responder_cell_type: str = FIBROBLAST, n_trials=35, pruning=True, use_toy_size=False
):
    # change global var
    global USE_TOY_SIZE
    USE_TOY_SIZE = use_toy_size
    global RESPONDER_CELL_TYPE
    RESPONDER_CELL_TYPE = responder_cell_type
    print(f"Optimizing hyperparameters for {responder_cell_type}")

    pruner = optuna.pruners.MedianPruner() if pruning else optuna.pruners.NopPruner()
    study = optuna.create_study(
        direction="minimize", pruner=pruner, study_name="cellmates", load_if_exists=True
    )
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=600,
        gc_after_trial=True,
    )

    print("Number of finished trials: {}".format(len(study.trials)))

    # print table of all trials and their values
    df = study.trials_dataframe(multi_index=True)
    # print df keys
    print("Trials data frame keys: ", df.keys())
    # drop keys 'datetime_start', 'datetime_complete'
    df = df.drop(columns=["datetime_start", "datetime_complete"])
    print(tabulate(df, headers="keys", tablefmt="pretty", floatfmt=".5f"))

    # output to csv
    df.to_csv("cellmates_hyperparameters.csv")

    print("Best trial:")
    trial = study.best_trial

    # print value with the best trial
    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


if __name__ == "__main__":
    Fire(main)
