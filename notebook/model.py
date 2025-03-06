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

class UNet(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = smp.Unet(
            encoder_name="resnet18",
            encoder_weights="imagenet",
            in_channels=2,
            classes=1,
            activation='sigmoid',
        )
        self.dice = DiceScore(num_classes=1)
        self.iou = MeanIoU(num_classes=1)

    def forward(self, x):
        return self.modal(x)


    def training_step(self, batch, batch_idx):
        x = batch["input"]
        y = batch["mask"]
        y_pred = self.model(x)
        dice_score = self.dice(y_pred, y)
        loss = 1 - dice_score
        # iou_loss = self.iou(y_pred, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # self.log('train_iou', iou, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["input"]
        y = batch["mask"]
        y_pred = self.model(x)
        dice_score = self.dice(y_pred, y)
        loss = 1 - dice_score

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-2)
        return optimizer