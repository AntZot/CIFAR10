import os
import torch
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, STL10
import lightning as L


PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
BATCH_SIZE = 64 if torch.cuda.is_available() else 16


class CIFAR10DataModule(L.LightningModule):
    def __init__(self, data_dir: str = PATH_DATASETS):
        super().__init__()
        self.size = 50000
        self.batch_size = BATCH_SIZE
        self.data_dir = data_dir
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(96),
                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)) 
            ]
        )
        self.dims = (3,96,396)
        self.num_classes = 10


    def prepare_data(self):
        # download
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            cifar10_full = CIFAR10(self.data_dir, train=True, transform=self.transform)
            self.cifar10_train, self.cifar10_val = random_split(cifar10_full, [45000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.cifar10_test = CIFAR10(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.cifar10_train, batch_size=BATCH_SIZE)

    def val_dataloader(self):
        return DataLoader(self.cifar10_val, batch_size=BATCH_SIZE)

    def test_dataloader(self):
        return DataLoader(self.cifar10_test, batch_size=BATCH_SIZE)
    

class STL10DataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = PATH_DATASETS) -> None:
        super().__init__()
        self.size = 5000
        self.batch_size = BATCH_SIZE
        self.data_dir = data_dir
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)) 
            ]
        )
        self.dims = (3,96,96)
        self.num_classes = 10


    def prepare_data(self):
        # download
        STL10(self.data_dir,split="train",download=True)
        STL10(self.data_dir,split="test",download=True)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            stl10_full = STL10(self.data_dir, split="train", transform=self.transform)
            #stl10_full.data = stl10_full.data.transpose(0,2,3,1)
            self.stl10_train, self.stl10_val = random_split(stl10_full, [4500, 500])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.stl10_test = STL10(self.data_dir, split="test", transform=self.transform)
            #self.stl10_test.data = self.stl10_test.data.transpose(0,2,3,1)


    def train_dataloader(self):
        return DataLoader(self.stl10_train, batch_size=BATCH_SIZE)

    def val_dataloader(self):
        return DataLoader(self.stl10_val, batch_size=BATCH_SIZE)

    def test_dataloader(self):
        return DataLoader(self.stl10_test, batch_size=BATCH_SIZE)