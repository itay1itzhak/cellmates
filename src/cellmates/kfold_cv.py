import numpy as np
import torch
from torch.utils.data import DataLoader, random_split, Subset
import lightning.pytorch as pl

from cellmates.data.breast import generate_one_tissue_dataset
from cellmates.train import train_model
from cellmates.data.dataset import collate_fn

from cellmates.data.breast import get_datasets
from cellmates.utils import MAX_EFFECTIVE_DISTANCE
from tdm.cell_types import FIBROBLAST

from sklearn.model_selection import KFold

from fire import Fire


def main(
    n_splits: int = 10,
    responder_cell_type: str = FIBROBLAST,
    batch_size: int = 32,
    n_epochs: int = 1,
    learning_rate: float = 1e-4,
    D: int = 256,
    H: int = 16,
    F: int = 512,
    M: int = 1024,
    num_encoder_layers: int = 2,
    experiment_name: str = "",
):
    """
    Tests that the model correctly learns the relationship between cell distances and division.
    """
    pl.seed_everything(42)

    # args -> model configuration:
    K = D // H
    model_config = {
        "D": D,
        "H": H,
        "K": K,
        "F": F,
        "M": M,
        "num_encoder_layers": num_encoder_layers,
    }

    print("fetching datasets")

    ds = get_datasets(
        responder_cell_type=responder_cell_type,
        effective_distance=MAX_EFFECTIVE_DISTANCE,
        concatenated=True,
    )
    print("done")

    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    losses = []

    for fold, (train_ids, test_ids) in enumerate(kfold.split(ds)):

        # split train_ids into train and validation:
        n = len(train_ids)
        n_train = int(n * (7 / 9))
        n_val = n - n_train
        train_ids, val_ids = random_split(train_ids, [n_train, n_val])

        # subset datasets:
        train_ds = Subset(ds, train_ids)
        val_ds = Subset(ds, val_ids)
        test_ds = Subset(ds, test_ids)

        # fit - saves top checkpoint for every fold:
        trained_model, test_loss = train_model(
            train_ds=train_ds,
            val_ds=val_ds,
            test_ds=test_ds,
            n_epochs=n_epochs,
            batch_size=batch_size,
            **model_config,
            learning_rate=learning_rate,
            device="cuda" if torch.cuda.is_available() else "cpu",
            experiment_name=experiment_name + f"_fold_{fold}",
        )

        losses.append(test_loss)

    print(f"{n_splits}-fold losses: {losses}")
    np.savetxt(
        f"./kfold/{experiment_name}_{n_splits}-fold_losses.csv",
        np.array(losses),
        delimiter=",",
    )


if __name__ == "__main__":
    Fire(main)

"""
python src/cellmates/kfold_cv.py --batch_size 16 --n_epochs 1 --D 128 --H 8 --F 128 --M 128 --num_encoder_layers 1 --learning_rate 1e-3 --experiment_name 1_layers_lr_1e-3_epoch_1_kfolds_10 --n_splits 3
"""
