import os
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split, Subset
import lightning.pytorch as pl

from cellmates.data.breast import generate_one_tissue_dataset
from cellmates.train import train_model
from cellmates.data.dataset import collate_fn

from cellmates.data.breast import get_datasets
from cellmates.utils import MAX_EFFECTIVE_DISTANCE, PROJECT_ROOT_PATH
from tdm.cell_types import FIBROBLAST, TUMOR, B_CELL

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
    dropout_p: float = 0.1,
    layer_norm_eps: float = 1e-3,
    use_toy_size: bool = False,
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
        "dropout_p": dropout_p,
        "layer_norm_eps": layer_norm_eps,
    }

    print("fetching datasets")

    ds = get_datasets(
        responder_cell_type=responder_cell_type,
        effective_distance=MAX_EFFECTIVE_DISTANCE,
        concatenated=True,
    )
    print("done")

    if use_toy_size:
        print("=" * 30)
        print("Using toy dataset")
        print("=" * 30)
        # use only first 50 examples from ds
        ds = Subset(ds, range(50))

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

        print(
            f"n train, val, test samples = {len(train_ds), len(val_ds), len(test_ids)}"
        )

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
            save_checkpoint=True,
        )

        losses.append(test_loss)

    # TODO - error due to DDP, each process computes loss using a subset of the data
    print(f"{n_splits}-fold losses: {losses}")

    save_path = (
        str(PROJECT_ROOT_PATH)
        + f"/kfold_cv/{experiment_name}_{n_splits}-fold_losses.csv"
    )
    # create a directory if it does not exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print(f"saving losses to {save_path}")
    np.savetxt(
        save_path,
        np.array(losses),
        delimiter=",",
    )


if __name__ == "__main__":
    Fire(main)

"""
python src/cellmates/kfold_cv.py --batch_size 2 --n_epochs 40 --num_encoder_layers 4 --D 512 --H 32 --F 1024 --M 512 --dropout_p 0.28330882325349427 --layer_norm_eps 0.0002947682439783808 --learning_rate 0.000525684163595057 --experiment_name best_params_10fold_cv --n_splits 10 


params:
    num_encoder_layers: 4
    D: 512
    H: 32
    F: 1024
    M: 512
    dropout_p: 0.28330882325349427
    layer_norm_eps: 0.0002947682439783808
    learning_rate: 0.000525684163595057

results:
[
    0.04596949741244316, 
    0.051153168082237244, 
    0.06382358819246292, 
    0.05529382452368736, 
    0.06056812405586243,
    0.05057327821850777, 
    0.10690890997648239,
    0.07534198462963104,
    0.09805398434400558,
    0.06615480035543442
]


python src/cellmates/kfold_cv.py --batch_size 2 --n_epochs 40 --num_encoder_layers 4 --D 512 --H 32 --F 1024 --M 512 --dropout_p 0.28330882325349427 --layer_norm_eps 0.0002947682439783808 --learning_rate 0.000525684163595057 --experiment_name TUMOR_CELLS_10fold_cv --n_splits 10 responder_cell_type=Tu
"""
