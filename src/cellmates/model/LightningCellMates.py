import pytorch_lightning as pl
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
import torch
from torch import Tensor
from cellmates.model.transformer import CellMatesTransformer


class LightningCellMates(pl.LightningModule):
    def __init__(self, model_config: dict, learning_rate: float):
        super().__init__()
        self.model = CellMatesTransformer(**model_config)
        self.loss_fn = BCEWithLogitsLoss()
        self.learning_rate = learning_rate

    def forward(
        self, cell_types_BL: Tensor, distances_BLL: Tensor, padding_mask_BL: Tensor
    ):
        return self.model(cell_types_BL, distances_BLL, padding_mask_BL)

    def training_step(self, batch, batch_nb):
        # fetch batch components:
        cell_types_BL = batch["cell_types_BL"]
        distances_BLL = batch["distances_BLL"]
        padding_mask_BL = batch["padding_mask_BL"]
        target = batch["is_dividing_B"]

        output_B1 = self(cell_types_BL, distances_BLL, padding_mask_BL).squeeze(-1)
        loss = self.loss_fn(output_B1, target)

        logs = {"train_loss": loss}
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

        return {"loss": loss, "log": logs}

    def validation_step(self, batch, batch_nb):
        # fetch batch components:
        cell_types_BL = batch["cell_types_BL"]
        distances_BLL = batch["distances_BLL"]
        padding_mask_BL = batch["padding_mask_BL"]
        target = batch["is_dividing_B"]

        output_B1 = self(cell_types_BL, distances_BLL, padding_mask_BL).squeeze(-1)
        val_loss = self.loss_fn(output_B1, target)

        self.log("val_loss", val_loss)

        return val_loss

    def configure_optimizers(self):
        return Adam(self.model.parameters(), lr=self.learning_rate)
