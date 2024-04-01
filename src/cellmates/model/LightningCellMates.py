import pytorch_lightning as pl
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
import torch
from torch import Tensor
from torch.nn.functional import sigmoid
from cellmates.model.transformer import CellMatesTransformer
from cellmates.metrics import plot_calibration
import wandb


class LightningCellMates(pl.LightningModule):
    def __init__(self, model_config: dict, learning_rate: float):
        super().__init__()
        self.model = CellMatesTransformer(**model_config)
        self.loss_fn = BCEWithLogitsLoss()
        self.learning_rate = learning_rate

        self.validation_preds = []
        self.validation_labels = []


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

        self.validation_preds.append(sigmoid(output_B1).detach())
        self.validation_labels.append(target.detach())

        return val_loss

    def on_validation_epoch_end(self):
        all_predicted_probs = torch.stack(self.validation_preds).cpu().numpy().flatten()
        all_true_labels = torch.stack(self.validation_labels).cpu().numpy().flatten()

        fig = plot_calibration(
            predicted_probs=all_predicted_probs,
            true_labels=all_true_labels,
            n_cells_per_bin=10,
        )

        image = wandb.Image(fig, caption="Calibration Plot")
        wandb.log({"calibration_plot": image})

        self.validation_preds.clear()
        self.validation_labels.clear()


    def configure_optimizers(self):
        return Adam(self.model.parameters(), lr=self.learning_rate)
