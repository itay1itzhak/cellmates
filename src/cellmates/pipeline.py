import torch
from torch.utils.data import DataLoader, random_split
import lightning.pytorch as pl

from cellmates.data.breast import generate_one_tissue_dataset
from cellmates.train import train_model
from cellmates.data.dataset import collate_fn

from cellmates.data.breast import get_datasets
from cellmates.utils import MAX_EFFECTIVE_DISTANCE
from tdm.cell_types import FIBROBLAST, TUMOR

from fire import Fire


def get_datasets_split(
    responder_cell_type, mask_third_party_cells_distances=False, use_toy_size=False
):
    print("fetching datasets")
    ds = get_datasets(
        responder_cell_type=responder_cell_type,
        effective_distance=MAX_EFFECTIVE_DISTANCE,
        concatenated=True,
        mask_third_party_cells_distances=mask_third_party_cells_distances,
    )
    print("done")

    # split the dataset into train, validation, and test sets
    n = len(ds)
    m = n // 10
    train_ds, val_ds, test_ds = random_split(ds, [7 * m, 2 * m, n - (9 * m)])

    if (
        use_toy_size
    ):  # use only 20 train smaples for testing using random_split set all to tests
        train_ds, val_ds, test_ds = random_split(ds, [10, 20, n - 30])
        print("=" * 30)
        print("Using toy dataset")
        print("=" * 30)

    return train_ds, val_ds, test_ds


def main(
    responder_cell_type: str = FIBROBLAST,
    batch_size: int = 32,
    n_epochs: int = 20,
    learning_rate: float = 1e-4,
    D: int = 256,
    H: int = 16,
    F: int = 512,
    M: int = 1024,
    num_encoder_layers: int = 2,
    experiment_name: str = "",
    use_wandb: bool = True,
    save_checkpoint: bool = True,
    log_every_n_steps : int = 1,
    use_toy_size: bool = False,
    mask_third_party_cells_distances: bool = False,
):
    """
    Tests that the model correctly learns the relationship between cell distances and division.
    """
    pl.seed_everything(42)

    train_ds, val_ds, test_ds = get_datasets_split(
        responder_cell_type,
        mask_third_party_cells_distances=mask_third_party_cells_distances,
        use_toy_size=use_toy_size,
    )

    K = D // H
    model_config = {
        "D": D,
        "H": H,
        "K": K,
        "F": F,
        "M": M,
        "num_encoder_layers": num_encoder_layers,
    }

    trained_model = train_model(
        train_ds=train_ds,
        val_ds=val_ds,
        n_epochs=n_epochs,
        batch_size=batch_size,
        **model_config,
        learning_rate=learning_rate,
        device="cuda" if torch.cuda.is_available() else "cpu",
        experiment_name=experiment_name,
        save_checkpoint=save_checkpoint,
        use_wandb=use_wandb,
        log_every_n_steps=log_every_n_steps,
    )


if __name__ == "__main__":
    Fire(main)
