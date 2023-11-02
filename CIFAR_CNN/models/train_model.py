from typing import List, Union
from CIFAR_CNN.data.data_loader import CIFAR10DataModule
from CIFAR_CNN.models.cnn import CIFAR10Model
import torch
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor
import wandb

ACCELERATOR = "gpu" if torch.cuda.is_available() else "cpu"
LOGGER = WandbLogger(log_model=True)


class ImageCallback(L.Callback):
    def __init__(self) -> None:
        super().__init__()
        self.outputs = None
        self.x = None
        self.y = None


    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx == trainer.num_training_batches-1:
            self.x, self.y = batch
            self.outputs = torch.argmax(outputs["prediction"],dim=1)


    def on_train_epoch_end(self, trainer, pl_module):
        n = 10
        x, y = self.x, self.y

        images = [img for img in x[:n]]
        captions = [f'Target: {y_i} - Prediction: {y_pred}' 
            for y_i, y_pred in zip(y[:n], self.outputs[:n])]

        trainer.logger.log_image(
                key='sample_images', 
                images=images, 
                caption=captions)




data_module = CIFAR10DataModule("/CIFAR10/CIFAR_CNN/datasets/raw")
model_module = CIFAR10Model(num_classes=data_module.num_classes,lr=2e-3)

callbacks =[
    LearningRateMonitor(logging_interval='step'),
    ImageCallback()  
]

   
def train(epoch: int = 45,
          device: str = "auto") -> None:
    try:
        trainer = L.Trainer(
            accelerator=ACCELERATOR,
            devices=device,
            max_epochs=epoch,
            logger=LOGGER,
            callbacks= callbacks
        )
        wandb.run

        trainer.fit(model_module, datamodule = data_module)           
        trainer.test(model_module,datamodule = data_module)

        wandb.finish(0)
    except RuntimeError:
        wandb.finish(1)


if __name__ == "__main__":
    train()
        