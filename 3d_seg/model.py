import lightning as L
import torch
from segmentation_models_pytorch.losses import DiceLoss
from torch.nn import BCEWithLogitsLoss
from torch.nn import functional as F
from torchmetrics.regression import MeanSquaredError as MSE
from torchmetrics.segmentation import DiceScore, MeanIoU


class Seg3dNet(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.dice_score = DiceScore(num_classes=1)
        self.iou = MeanIoU(num_classes=1)
        self.mse = MSE()
        self.bce = BCEWithLogitsLoss(pos_weight=torch.tensor([1e3]))
        self.dice = DiceLoss("binary")

    def forward(self, x):
        return self.model(x)

    def _parse_batch(self, batch):
        x = batch["img"].float()
        y = batch["seg"].float()
        return x, y

    def _downsample_multiples(self, x, n):
        """For downsampling a tensor by n times exactly"""
        if n == 1:
            return x
        # ignore batch and channel dimensions
        slices = [slice(None)] * 2 + [slice(None, None, n)] * (x.ndim - 2)
        return x[slices].clone()

    def training_step(self, batch, batch_idx):
        x, y = self._parse_batch(batch)
        y_preds = self.model(x)

        # calculate loss for output tensors at every scale
        # first tensor is original scale, 2nd is 1/2, then 1/4, then 1/8
        bce_losses = []
        dice_losses = []
        n = len(y_preds)
        for i, y_pred in enumerate(y_preds):
            y_down = self._downsample_multiples(y, 2 ** i)
            # increase weight on loss on larger res (64, 27, 8, 1)
            weight = 0.1 # (n - i) ** 3 / (n ** 3)
            bce_losses.append(weight * self.bce(y_pred, y_down))
            dice_losses.append(self.dice(y_pred, y_down))
            break

        print("bce",  [round(x.item(), 3) for x in bce_losses])
        print("dice", [round(x.item(), 3) for x in dice_losses])

        # # dice loss for the final output map only
        # dice_loss = self.dice(y_preds[0], y)
        loss = sum(bce_losses)  + sum(dice_losses)

        # metrics
        y_pred_mask = F.sigmoid(y_preds[0]) > 0.5
        dice_score = self.dice_score(y_pred_mask, y)

        # self.log('train_bce', bce_loss)
        # self.log('train_dice', dice_loss)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_dice', dice_score, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = self._parse_batch(batch)
        y_pred = self.model(x)  # for val, only the final heatmap is returned

        bce_loss = self.bce(y_pred, y)
        dice_loss = self.dice(y_pred, y)
        loss = bce_loss  + dice_loss

        # metrics
        y_pred_mask = F.sigmoid(y_pred) > 0.5
        dice_score = self.dice_score(y_pred_mask, y)

        # sum on the spatial channels of an N, C, D, H, W tensor
        pred_vols = y_pred_mask.sum(axis=(2, 3, 4))
        y_vols = y.sum(axis=(2, 3, 4))

        mean_vol_sim = 1 - ((pred_vols - y_vols).abs() / y_vols).mean()
        mean_vol_sim = mean_vol_sim.clip(0, 1)

        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_dice', dice_score, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_vols', mean_vol_sim, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        return optimizer