import numpy as np
import torch
from torch.utils.data import DataLoader, random_split, Subset
import pytorch_lightning as pl

from cellmates.model.LightningCellMates import LightningCellMates
from cellmates.data.breast import generate_one_tissue_dataset
from cellmates.train import train_model
from cellmates.data.dataset import collate_fn

from cellmates.data.breast import get_datasets
from cellmates.utils import MAX_EFFECTIVE_DISTANCE, PROJECT_ROOT_PATH
from tdm.cell_types import FIBROBLAST

from sklearn.model_selection import KFold

from fire import Fire


def main():
    """
    """
    pl.seed_everything(42)


    """
    Parameters:    
    """
    fold_checkpoints = [
        '/checkpoints/cellmates-best_params_10fold_cv_fold_0-epoch=15-val_loss=0.07.ckpt',
        '/checkpoints/cellmates-best_params_10fold_cv_fold_1-epoch=17-val_loss=0.07.ckpt',
        '/checkpoints/cellmates-best_params_10fold_cv_fold_2-epoch=23-val_loss=0.08.ckpt',
        '/checkpoints/cellmates-best_params_10fold_cv_fold_3-epoch=25-val_loss=0.08.ckpt',
        '/checkpoints/cellmates-best_params_10fold_cv_fold_4-epoch=20-val_loss=0.07.ckpt',
        '/checkpoints/cellmates-best_params_10fold_cv_fold_5-epoch=22-val_loss=0.07.ckpt',
        '/checkpoints/cellmates-best_params_10fold_cv_fold_6-epoch=19-val_loss=0.07.ckpt',
        '/checkpoints/cellmates-best_params_10fold_cv_fold_7-epoch=29-val_loss=0.07.ckpt',
        '/checkpoints/cellmates-best_params_10fold_cv_fold_8-epoch=20-val_loss=0.08.ckpt',
        '/checkpoints/cellmates-best_params_10fold_cv_fold_9-epoch=26-val_loss=0.07.ckpt'
    ]
    fold_checkpoints = [ '~/cellmates' + p for p in fold_checkpoints]
    n_splits = 10
    responder_cell_type = FIBROBLAST

    """
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    """

    print('fetching datasets')

    ds = get_datasets(
        responder_cell_type=responder_cell_type,
        effective_distance=MAX_EFFECTIVE_DISTANCE,
        concatenated=True,
    )
    print('done')

    kfold = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42
    )

    losses = []

    for fold, (train_ids, test_ids) in enumerate(kfold.split(ds)):

        test_ds = Subset(ds, test_ids)
        test_dl = DataLoader(test_ds, collate_fn=collate_fn, batch_size=8)

        model = LightningCellMates.load_from_checkpoint(fold_checkpoints[fold])
        model.eval()

        trainer = pl.Trainer(devices=[4])
        test_loss = trainer.test(model=model, dataloaders=test_dl)

        print(fold, test_loss)

        losses.append(test_loss)

    print(losses)

if __name__ == "__main__":
    Fire(main)


"""
[
0.06534115225076675,
0.07408205419778824,
0.07024448364973068,
0.07264914363622665,
0.07080429792404175,
0.06821686774492264,
0.08247258514165878,
]
"""