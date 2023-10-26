import torch
from torch import nn

class CIFAR10_Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cnn_relu_seq = nn.Sequential(
            nn.Conv2d(3,6,3),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(6,12,3),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        self.lin_layer_seq = nn.Sequential(
            nn.Linear(12*6*6,216),
            nn.ReLU(),
            nn.Linear(216,108),
            nn.ReLU(),
            nn.Linear(108,10)
        )


    def forward(self, x) -> torch.Tensor:
        x = self.cnn_relu_seq(x)
        x = torch.flatten(x,1)
        x = self.lin_layer_seq(x)
        return x