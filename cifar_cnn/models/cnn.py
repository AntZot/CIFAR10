from typing import Tuple
from torch import nn
from cifar_cnn.models.ResSKBlock import ResSKBlock
import torch
import lightning as L
from torchmetrics.functional import accuracy, auroc
from torch.nn import functional as F

BATCH_SIZE = 256 if torch.cuda.is_available() else 64

class CIFAR10Model(L.LightningModule):

    def __init__(self,num_classes,lr):
        super().__init__()
        self.learning_rate = lr
        self.num_classes = num_classes
        self.cnn_relu_seq = nn.Sequential(
            nn.Conv2d(3,16,5),
            nn.ReLU(),
            ResSKBlock(in_channels=16,out_channels=16*2,groups = 4),
            nn.ReLU(),
        )

        self.lin_layer_seq = nn.Sequential(
            nn.Linear(32*28*28,12544),
            nn.ReLU(),
            nn.Linear(12544,6272),
            nn.ReLU(),
            nn.Linear(6272,784),
            nn.ReLU(),
            nn.Linear(784,10)
        )

    def forward(self,x):
        conv_res = self.cnn_relu_seq(x)
        flattened = conv_res.view(conv_res.size(0), -1)
        lin_res = self.lin_layer_seq(flattened)
        return F.log_softmax(lin_res,dim=1)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_nb: int) -> torch.Tensor:
        x,y = batch
        loss = F.cross_entropy(self(x), y)
        preds = self(x)

        """
        metrics
        """
        rocauc = auroc(preds, y, task="multiclass",num_classes=self.num_classes)
        self.log("train_rocauc", rocauc, prog_bar=True)
        acc = accuracy(preds, y, task="multiclass",num_classes=self.num_classes)
        self.log("train_accuracy", acc, prog_bar=True)
        return {'loss': loss, 'prediction': preds}

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        """
        metrics
        """
        rocauc = auroc(logits,y,task="multiclass",num_classes=self.num_classes)
        acc = accuracy(logits, y, task="multiclass",num_classes=self.num_classes)
        self.log("val_accuracy", acc, prog_bar=True)
        self.log("val_rocauc",rocauc,prog_bar=True)
        self.log("val_loss", loss, prog_bar=True)

    def test_step(self,batch: Tuple[torch.Tensor, torch.Tensor], batch_idx):
        x, y = batch
        logits = self(x)

        test_loss = F.cross_entropy(logits, y)
        """
        metrics
        """
        rocauc = auroc(logits,y,task="multiclass",num_classes=self.num_classes)
        acc = accuracy(logits, y, task="multiclass",num_classes=self.num_classes)
        self.log("test_accuracy", acc, prog_bar=True)
        self.log("test_rocauc",rocauc,prog_bar=True)
        self.log("test_loss", test_loss, prog_bar=True)
        
    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), self.learning_rate,momentum=0.9)
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                optimizer=optimizer,
                max_lr=1e-2,
                epochs=self.trainer.max_epochs,
                steps_per_epoch = 50000 // BATCH_SIZE),
            "interval": "step"
        }
        return {"optimizer": optimizer, "lr_scheduler":  scheduler}