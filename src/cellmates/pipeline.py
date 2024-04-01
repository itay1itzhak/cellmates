import torch
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl

from cellmates.data.breast import generate_one_tissue_dataset
from cellmates.train import train_model
from cellmates.data.dataset import collate_fn

from cellmates.data.breast import get_datasets
from cellmates.utils import MAX_EFFECTIVE_DISTANCE
from tdm.cell_types import FIBROBLAST

from fire import Fire


def main(
    responder_cell_type: str = FIBROBLAST,
    batch_size: int = 32,
    n_epochs: int = 20,
    learning_rate: float = 1e-4
):
    """
    Tests that the model correctly learns the relationship between cell distances and division.
    """
    pl.seed_everything(42)

    ds = get_datasets(
        responder_cell_type=responder_cell_type, 
        effective_distance=MAX_EFFECTIVE_DISTANCE, 
        concatenated=True
    )

    n = len(ds)
    m = n//10
    train_ds, val_ds, test_ds = random_split(ds, [7*m, 2*m, n - (9*m)])


    D = 256
    H = 16
    K = D//H
    model_config = {
        "D": D,
        "H": H,
        "K": K,
        "F": 1024,
        "M": 512,
        "num_encoder_layers": 2,
    }

    trained_model = train_model(
        train_ds=train_ds,
        val_ds=val_ds,
        n_epochs=n_epochs,
        batch_size=batch_size,
        **model_config,
        learning_rate=learning_rate,
        device='cuda'
    )


if __name__ == "__main__":
    Fire(main)
