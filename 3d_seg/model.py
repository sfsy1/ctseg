import lightning as L
import segmentation_models_pytorch as smp
import torch
from torchmetrics.regression import MeanSquaredError as MSE
from torchmetrics.segmentation import DiceScore, MeanIoU


class Seg3dNet(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.dice = DiceScore(num_classes=1)
        self.iou = MeanIoU(num_classes=1)
        self.mse = MSE()

    def forward(self, x):
        return self.modal(x)

    def _parse_batch(self, batch):
        x = batch["img"].float()
        y = batch["seg"].float()
        return x, y

    def training_step(self, batch, batch_idx):
        x, y = self._parse_batch(batch)

        y_pred = self.model(x)
        mse_loss = self.mse(y_pred, y)
        loss = mse_loss

        # metrics
        y_pred_mask = y_pred > 0.5
        dice_score = self.dice(y_pred_mask, y)

        self.log('tr_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('tr_dice', dice_score, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = self._parse_batch(batch)

        y_pred = self.model(x)
        mse_loss = self.mse(y_pred, y)
        loss = mse_loss

        # metrics
        y_pred_mask = y_pred > 0.5
        dice_score = self.dice(y_pred_mask, y)

        pred_vols = y_pred_mask.sum(axis=(2, 3))
        y_vols = y.sum(axis=(2, 3))

        mean_vol_sim = 1 - ((pred_vols - y_vols).abs() / y_vols).mean()
        mean_vol_sim = mean_vol_sim.clip(0, 1)

        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_dice', dice_score, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_vols', mean_vol_sim, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=2e-3)
        return optimizer