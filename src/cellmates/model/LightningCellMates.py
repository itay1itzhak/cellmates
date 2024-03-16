import pytorch_lightning as pl
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
import torch
from cellmates.model.transformer import CellMatesTransformer


class LightningCellMates(pl.LightningModule):
    def __init__(self, model_config):
        super().__init__()
        self.model = CellMatesTransformer(**model_config)
        self.loss_fn = BCEWithLogitsLoss()

    def forward(self, cell_types_BL, distances_BLL):
        return self.model(cell_types_BL, distances_BLL)

    def training_step(self, batch, batch_nb):
        cell_types_BL, distances_BLL, target = batch

        output_B1 = self(cell_types_BL, distances_BLL)
        loss = self.loss_fn(output_B1, target)

        logs = {"train_loss": loss}
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

        return {"loss": loss, "log": logs}

    def validation_step(self, batch, batch_nb):
        cell_types_BL, distances_BLL, target = batch

        output_B1 = self(cell_types_BL, distances_BLL)
        val_loss = self.loss_fn(output_B1, target)

        self.log("val_loss", val_loss)

        return val_loss

    def configure_optimizers(self):
        return Adam(self.model.parameters(), lr=1e-3)
