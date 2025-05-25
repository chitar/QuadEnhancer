#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

# SAW
import os
saw_method = os.environ["SAW_METHOD"]
saw_k = int(os.environ["SAW_K"])
# print(f'SAW method {saw_method}, k {saw_k}')


import math
import argparse
from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from cvnets.layers.base_layer import BaseLayer
from utils import logger


# SAW
from torch.utils.checkpoint import checkpoint

def roll_opt_channel_first(result,theta):
    shapes = [1] * result.dim()
    shapes[1] = theta.shape[0]
    theta = theta.reshape(*shapes)
    x = 1 + theta * torch.roll(result, shifts= 1, dims=1)
    result = x * result
    return result

def roll_opt_channel_last(result,theta):
    shapes = [1] * result.dim()
    shapes[-1] = theta.shape[0]
    theta = theta.reshape(*shapes)
    x = 1 + theta * torch.roll(result, shifts=1, dims=-1)
    result = x * result
    return result

import torch
class SAWLayer(nn.Module):
    def __init__(self,out_features,channel_first):
        super().__init__()
        self.saw_method = saw_method
        self.saw_k = saw_k
        self.out_features = out_features
        self.channel_first = channel_first

        if saw_method=='roll' and saw_k>0:
            for i in range(self.saw_k):
                self.__setattr__(f'saw_theta_{i}',
                                 nn.Parameter(torch.zeros(out_features)))
        elif self.saw_method == 'lrn' and self.saw_k > 0:
            self.saw_P = nn.Linear(self.out_features, saw_k, bias=False)
            self.saw_Q = nn.Linear(saw_k, self.out_features, bias=False)
            nn.init.kaiming_uniform_(self.saw_P.weight, a=math.sqrt(5))
            nn.init.zeros_(self.saw_Q.weight)


    def forward(self, result: Tensor) -> Tensor:

        if self.saw_method=='roll' and self.saw_k>0:
            #if self.saw_k>1:
            #    raise NotImplementedError
            # if self.channel_first:
            #     result = checkpoint(roll_opt_channel_first,result, self.saw_theta_0)
            # else:
            #     result = checkpoint(roll_opt_channel_last,result,self.saw_theta_0)

            shapes = [1] * result.dim()
            ind = 1 if self.channel_first else -1
            shapes[ind] = self.out_features
            x = 1
            for i in range(0, self.saw_k):
                theta = self.__getattr__(f'saw_theta_{i}').reshape(*shapes)
                x +=  theta * torch.roll(result, shifts=i + 1, dims=1 if self.channel_first else -1)
            result = x * result


        elif self.saw_method=='lrn' and self.saw_k>0:
            x = self.saw_Q(self.saw_P(result))
            x = x / (1 + x.norm(dim=-1, keepdim=True))
            result = (x+1) * result
        # print(f'After | mean: {result.mean().item():.4f} max abs: {result.abs().max().item():.4f}')
        return result



class LinearLayer(BaseLayer):
    """
    Applies a linear transformation to the input data

    Args:
        in_features (int): number of features in the input tensor
        out_features (int): number of features in the output tensor
        bias  (Optional[bool]): use bias or not
        channel_first (Optional[bool]): Channels are first or last dimension. If first, then use Conv2d

    Shape:
        - Input: :math:`(N, *, C_{in})` if not channel_first else :math:`(N, C_{in}, *)` where :math:`*` means any number of dimensions.
        - Output: :math:`(N, *, C_{out})` if not channel_first else :math:`(N, C_{out}, *)`

    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: Optional[bool] = True,
        channel_first: Optional[bool] = False,
        *args,
        **kwargs
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features)) if bias else None

        # SAW
        self.saw_method = saw_method
        self.saw_k = saw_k
        if saw_method != 'linear' and saw_k > 0:
            self.saw_layer = SAWLayer(out_features,channel_first)


        self.in_features = in_features
        self.out_features = out_features
        self.channel_first = channel_first

        self.reset_params()

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        parser.add_argument(
            "--model.layer.linear-init",
            type=str,
            default="xavier_uniform",
            help="Init type for linear layers",
        )
        parser.add_argument(
            "--model.layer.linear-init-std-dev",
            type=float,
            default=0.01,
            help="Std deviation for Linear layers",
        )
        return parser

    def reset_params(self):
        if self.weight is not None:
            torch.nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            torch.nn.init.constant_(self.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        if self.channel_first:
            if not self.training:
                logger.error("Channel-first mode is only supported during inference")
            if x.dim() != 4:
                logger.error("Input should be 4D, i.e., (B, C, H, W) format")
            # only run during conversion
            with torch.no_grad():
                return F.conv2d(
                    input=x,
                    weight=self.weight.clone()
                    .detach()
                    .reshape(self.out_features, self.in_features, 1, 1),
                    bias=None,
                )
        else:
            x = F.linear(x, weight=self.weight, bias=None)

        # SAW
        if saw_method != 'linear' and saw_k > 0:
            x = self.saw_layer(x)
        x = x if self.bias is None else x + self.bias

        return x

    def __repr__(self):
        repr_str = (
            "{}(in_features={}, out_features={}, bias={}, channel_first={}, saw_method={}, saw_k={})".format(
                self.__class__.__name__,
                self.in_features,
                self.out_features,
                True if self.bias is not None else False,
                self.channel_first,
                self.saw_method,
                self.saw_k
            )
        )
        return repr_str


class GroupLinear(BaseLayer):
    """
    Applies a GroupLinear transformation layer, as defined `here <https://arxiv.org/abs/1808.09029>`_,
    `here <https://arxiv.org/abs/1911.12385>`_ and `here <https://arxiv.org/abs/2008.00623>`_

    Args:
        in_features (int): number of features in the input tensor
        out_features (int): number of features in the output tensor
        n_groups (int): number of groups
        bias (Optional[bool]): use bias or not
        feature_shuffle (Optional[bool]): Shuffle features between groups

    Shape:
        - Input: :math:`(N, *, C_{in})`
        - Output: :math:`(N, *, C_{out})`

    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_groups: int,
        bias: Optional[bool] = True,
        feature_shuffle: Optional[bool] = False,
        *args,
        **kwargs
    ) -> None:
        if in_features % n_groups != 0:
            logger.error(
                "Input dimensions ({}) must be divisible by n_groups ({})".format(
                    in_features, n_groups
                )
            )
        if out_features % n_groups != 0:
            logger.error(
                "Output dimensions ({}) must be divisible by n_groups ({})".format(
                    out_features, n_groups
                )
            )

        in_groups = in_features // n_groups
        out_groups = out_features // n_groups

        super().__init__()

        self.weight = nn.Parameter(torch.Tensor(n_groups, in_groups, out_groups))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(n_groups, 1, out_groups))
        else:
            self.bias = None

        # SAW
        if saw_method != 'linear' and saw_k > 0:
            self.saw_layer = SAWLayer(out_groups,channel_first=False)

        self.out_features = out_features
        self.in_features = in_features
        self.n_groups = n_groups
        self.feature_shuffle = feature_shuffle

        self.reset_params()

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        parser.add_argument(
            "--model.layer.group-linear-init",
            type=str,
            default="xavier_uniform",
            help="Init type for group linear layers",
        )
        parser.add_argument(
            "--model.layer.group-linear-init-std-dev",
            type=float,
            default=0.01,
            help="Std deviation for group linear layers",
        )
        return parser

    def reset_params(self):
        if self.weight is not None:
            torch.nn.init.xavier_uniform_(self.weight.data)
        if self.bias is not None:
            torch.nn.init.constant_(self.bias.data, 0)

    def _forward(self, x: Tensor) -> Tensor:
        bsz = x.shape[0]
        # [B, N] -->  [B, g, N/g]
        x = x.reshape(bsz, self.n_groups, -1)

        # [B, g, N/g] --> [g, B, N/g]
        x = x.transpose(0, 1)
        # [g, B, N/g] x [g, N/g, M/g] --> [g, B, M/g]
        x = torch.bmm(x, self.weight)

        # SAW
        if saw_method != 'linear' and saw_k > 0:
            x = self.saw_layer(x)

        if self.bias is not None:
            x = torch.add(x, self.bias)

        if self.feature_shuffle:
            # [g, B, M/g] --> [B, M/g, g]
            x = x.permute(1, 2, 0)
            # [B, M/g, g] --> [B, g, M/g]
            x = x.reshape(bsz, self.n_groups, -1)
        else:
            # [g, B, M/g] --> [B, g, M/g]
            x = x.transpose(0, 1)

        return x.reshape(bsz, -1)

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() == 2:
            x = self._forward(x)
            return x
        else:
            in_dims = x.shape[:-1]
            n_elements = x.numel() // self.in_features
            x = x.reshape(n_elements, -1)
            x = self._forward(x)
            x = x.reshape(*in_dims, -1)
            return x

    def __repr__(self):
        repr_str = "{}(in_features={}, out_features={}, groups={}, bias={}, shuffle={})".format(
            self.__class__.__name__,
            self.in_features,
            self.out_features,
            self.n_groups,
            True if self.bias is not None else False,
            self.feature_shuffle,
        )
        return repr_str
