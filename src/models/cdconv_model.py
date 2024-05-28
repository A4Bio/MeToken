import os.path as osp
from functools import partial
from typing import Type, Any, Callable, Union, List, Optional

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.nn import MLP, fps, global_max_pool, global_mean_pool, radius
from sklearn.preprocessing import normalize
from src.modules.cdconv_module import *

class Linear(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 dropout: float = 0.0,
                 bias: bool = False,
                 leakyrelu_negative_slope: float = 0.1) -> nn.Module:
        super(Linear, self).__init__()

        module = []
        module.append(nn.LeakyReLU(leakyrelu_negative_slope))
        module.append(nn.Dropout(dropout))
        module.append(nn.Linear(in_channels, out_channels, bias = bias))
        self.module = nn.Sequential(*module)

    def forward(self, x):
        return self.module(x)

class MLP(nn.Module):
    def __init__(self,
                 in_channels: int,
                 mid_channels: int,
                 out_channels: int,
                 dropout: float = 0.0,
                 bias: bool = True,
                 leakyrelu_negative_slope: float = 0.2) -> nn.Module:
        super(MLP, self).__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels

        module = []
        module.append(nn.LeakyReLU(leakyrelu_negative_slope))
        module.append(nn.Dropout(dropout))
        if mid_channels is None:
            module.append(nn.Linear(in_channels, out_channels, bias = bias))
        else:
            module.append(nn.Linear(in_channels, mid_channels, bias = bias))
        module.append(nn.LeakyReLU(leakyrelu_negative_slope))
        if mid_channels is None:
            module.append(nn.Dropout(dropout))
        else:
            module.append(nn.Linear(mid_channels, out_channels, bias = bias))

        self.module = nn.Sequential(*module)

    def forward(self, input):
        return self.module(input)

class BasicBlock(nn.Module):
    def __init__(self,
                 r: float,
                 l: float,
                 kernel_channels: "list[int]",
                 in_channels: int,
                 out_channels: int,
                 base_width: float = 16.0,
                 dropout: float = 0.0,
                 bias: bool = False,
                 leakyrelu_negative_slope: float = 0.1) -> nn.Module:

        super(BasicBlock, self).__init__()

        if in_channels != out_channels:
            self.identity = Linear(in_channels=in_channels,
                                  out_channels=out_channels,
                                  dropout=dropout,
                                  bias=bias,
                                  leakyrelu_negative_slope=leakyrelu_negative_slope)
        else:
            self.identity = nn.Sequential()

        width = int(out_channels * (base_width / 64.)) # 32*(16/64)=8
        self.input = MLP(in_channels=in_channels,
                         mid_channels=None,
                         out_channels=width,
                         dropout=dropout,
                         bias=bias,
                         leakyrelu_negative_slope=leakyrelu_negative_slope)
        self.conv = CDConv(r=r, l=l, kernel_channels=kernel_channels, in_channels=width, out_channels=width)
        self.output = Linear(in_channels=width,
                             out_channels=out_channels,
                             dropout=dropout,
                             bias=bias,
                             leakyrelu_negative_slope=leakyrelu_negative_slope)

    def forward(self, x, pos, seq, ori, batch):
        identity = self.identity(x) # [16,500,32]->[16,500,32]
        x = self.input(x)
        x = self.conv(x, pos, seq, ori, batch)
        out = self.output(x).reshape(identity.shape[0], identity.shape[1], -1) + identity
        return out

class CDConv_Model(nn.Module):
    def __init__(self,args,
                 base_width: float = 16.0,
                 embedding_dim: int = 128,
                 dropout: float = 0.2,
                 bias: bool = False,
                 num_classes: int = 26) -> nn.Module:
        geometric_radii=args.geometric_radii
        sequential_kernel_size=args.sequential_kernel_size
        kernel_channels=args.kernel_channels
        channels=args.channels
        num_classes=args.num_classes

        super().__init__()

        assert (len(geometric_radii) == len(channels)), "Model: 'geometric_radii' and 'channels' should have the same number of elements!"

        self.embedding = torch.nn.Embedding(num_embeddings=24, embedding_dim=embedding_dim)
        self.local_mean_pool = AvgPooling()

        layers = []
        in_channels = embedding_dim
        for i, radius in enumerate(geometric_radii):
            layers.append(BasicBlock(r = radius,
                                     l = sequential_kernel_size,
                                     kernel_channels = kernel_channels,
                                     in_channels = in_channels,
                                     out_channels = channels[i],
                                     base_width = base_width,
                                     dropout = dropout,
                                     bias = bias))
            layers.append(BasicBlock(r = radius,
                                     l = sequential_kernel_size,
                                     kernel_channels = kernel_channels,
                                     in_channels = channels[i],
                                     out_channels = channels[i],
                                     base_width = base_width,
                                     dropout = dropout,
                                     bias = bias))
            in_channels = channels[i]

        self.layers = nn.Sequential(*layers)

        self.classifier = MLP(in_channels=channels[-1],
                              mid_channels=max(channels[-1], num_classes),
                              out_channels=num_classes,
                              dropout=dropout)
    def orientation(self,pos):
        u = normalize(X=pos[1:,:] - pos[:-1,:], norm='l2', axis=1)
        u1 = u[1:,:]
        u2 = u[:-1, :]
        b = normalize(X=u2 - u1, norm='l2', axis=1)
        n = normalize(X=np.cross(u2, u1), norm='l2', axis=1)
        o = normalize(X=np.cross(b, n), norm='l2', axis=1)
        ori = np.stack([b, n, o], axis=1)
        return np.concatenate([np.expand_dims(ori[0], 0), ori, np.expand_dims(ori[-1], 0)], axis=0)
    
    def forward(self, data):
        seq = data["S"]
        pos = data["X"][:, :, 1].unsqueeze(-2)
        batch = None

        x=self.embedding(seq)
        center=pos.sum(axis=0,keepdims=True)
        pos = pos - center
        ori = torch.tensor(self.orientation(pos.view(-1, 3).cpu()), dtype=torch.float, device=pos.device)
        for i, layer in enumerate(self.layers):
            x = layer(x, pos, seq, ori, batch)

        x = torch.masked_select(x, (data['mask'] == 1).unsqueeze(-1)).reshape(-1, x.shape[-1])
        log_probs = F.log_softmax(self.classifier(x))

        data.update({'Q': torch.masked_select(data['Q'], data['mask'] == 1), 'mask': torch.masked_select(data['mask'], data['mask'] == 1)})
        return {'log_probs': log_probs}