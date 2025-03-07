import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import lightning as L

import segmentation_models_pytorch as smp
from torchmetrics.segmentation import DiceScore, MeanIoU
from torchmetrics.regression import MeanSquaredError as MSE
from evaluation.eval_utils import volume_similarity


class UNet(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = smp.Unet(
            encoder_name="resnet18", # resnext50_32x4d",
            encoder_weights="imagenet",
            in_channels=2,
            classes=1,
            activation='sigmoid',
        )
        self.dice = DiceScore(num_classes=1)
        self.iou = MeanIoU(num_classes=1)
        self.mse = MSE()

    def forward(self, x):
        return self.modal(x)


    def training_step(self, batch, batch_idx):
        x = batch["input"]
        y = batch["mask"]
        y_pred = self.model(x)

        # loss
        mse_loss = self.mse(y_pred, y)
        loss = mse_loss

        # metrics
        y_pred_mask = y_pred > 0.5
        dice_score = self.dice(y_pred_mask, y)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_dice', dice_score, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["input"]
        y = batch["mask"]
        y_pred = self.model(x)

        # loss
        mse_loss = self.mse(y_pred, y)
        loss = mse_loss

        # metrics
        y_pred_mask = y_pred > 0.5
        vol_sim = 1 - abs(y_pred_mask.sum() - y.sum()) / y.sum()
        dice_score = self.dice(y_pred_mask, y)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_dice', dice_score, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_vols', vol_sim, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-2)
        return optimizer