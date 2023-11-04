import torch
from torch import nn
from typing import List, Optional

class SKConv(nn.Module):
    """
    Implementation of the Selective Kernel (SK) Convolution proposed in [1].
    Parameters
    ----------
    in_channels : int
        Number of channels in the input tensor.
    out_channels : int
        Number of channels produced by the convolution.
    kernels : List[int], optional, default=[3, 5]
        List of kernel sizes for each branch.
    reduction : int, optional, default=16
        Reduction ratio to control the dimension of "compact feature" ``z`` (see eq.4).
    L : int, optional, default=32
        Minimal value of the dimension of "compact feature" ``z`` (see eq.4).
    groups : int, optional, default=32
        Hyperparameter for ``torch.nn.Conv2d``.
    References
    ----------
    1. "`Selective Kernel Networks. <https://arxiv.org/abs/1903.06586>`_" Xiang Li, et al. CVPR 2019.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        kernels: List[int] = [3, 5],
        reduction: int = 16,
        L: int = 32,
        groups: int = 32
    ) -> None:
        super(SKConv, self).__init__()

        if out_channels is None:
            out_channels = in_channels
        self.out_channels = out_channels

        d = max(in_channels//reduction,L) # eq.4

        self.M = len(kernels)

        #Split
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels=in_channels,
                          out_channels=self.out_channels,
                          padding=2,
                          kernel_size=k,
                          dilation=1 if max(kernels)==k else max(kernels)-k,
                          groups = groups),
                nn.BatchNorm2d(self.out_channels),
                nn.ReLU()
            )
            for k in kernels
        ])
        #fuse
        self.pool = nn.AdaptiveAvgPool2d((1,1))

        self.fc_z = nn.Sequential(
            nn.Linear(in_features=self.out_channels,
                      out_features=d*1),
            nn.BatchNorm1d(d),
            nn.ReLU()
        )
        self.fc_attn = nn.Linear(in_features=d*1,
                                 out_features=self.M * self.out_channels,
                                 bias=False)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor (batch_size, in_channels, height, width)
            Input tensor.
        Returns
        -------
        out : torch.Tensor (batch_size, out_channels, height, width)
            Output of the SK convolution layer.
        """
        #Conv2d , AvgPoll, softmax, ReLU, BatchNorm, Linear

        # ----- split -----
        # x: [b, c, h, w]
        feats = torch.stack([module(x) for module in self.convs],dim=1) # [b, M, c, h, w]
        # ----- fuse -----
        # eq.1
        U = torch.sum(feats,dim=1)
        # channel-wise statistics, eq.2
        s = self.pool(U).squeeze(dim=(2,3)) #s: [b, c]
        # compact feature, eq.3
        z = self.fc_z(s) # z [b, d]

        # ----- select -----
        batch_size, out_channels = s.shape

        # attention map, eq.5
        score = self.fc_attn(z) # (batch_size, M * out_channels)
        score = score.reshape((batch_size,self.M,out_channels)).unsqueeze(3).unsqueeze(4)  # (batch_size, M, out_channels, 1, 1)
        att = self.softmax(score)


        # fuse multiple branches, eq.6
        out = torch.sum(feats*att,dim=1) # (batch_size, out_channels, height, width)
        return out