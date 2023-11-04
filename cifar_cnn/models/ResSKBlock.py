from torch import nn
from cifar_cnn.models.SKConv import SKConv
import torch.functional as F
from typing import List, Optional

class ResSKBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 hidden_channels: Optional[int] = None,
                 kernels: List[int] = [3, 5],
                 reduction: int = 16,
                 L: int = 32,
                 groups: int = 32) -> None:
        super(ResSKBlock,self).__init__()

        if hidden_channels is None:
            hidden_channels = out_channels
        self.resconv = nn.Sequential(
            nn.Conv2d(in_channels= in_channels,
                      out_channels= hidden_channels,
                      kernel_size=1),
            nn.BatchNorm2d(hidden_channels),
            SKConv(in_channels=hidden_channels,
                   kernels=kernels,
                   reduction=reduction,
                   L=L,
                   groups=groups),
            nn.BatchNorm2d(hidden_channels),
            nn.Conv2d(in_channels= hidden_channels,
                      out_channels= out_channels,
                      kernel_size=1),
            nn.BatchNorm2d(out_channels)
        )

        if in_channels == out_channels: # when dim not change, in could be added diectly to out
            self.shortcut = nn.Sequential()
        else: # when dim not change, in should also change dim to be added to out
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=1),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self,x):
        res = self.resconv(x)
        return res + self.shortcut(x)