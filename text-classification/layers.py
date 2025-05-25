import torch
import math
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F, init
from torch.nn.parameter import Parameter, UninitializedParameter
from torch.nn import Module
import os
saw_k=0
saw_method='linear'


class SAWLinear(Module):
    r"""Applies an affine linear transformation to the incoming data: :math:`y = xA^T + b`.

    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.

    On certain ROCm devices, when using float16 inputs this module will use :ref:`different precision<fp16_on_mi200>` for backward.

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(*, H_\text{in})` where :math:`*` means any number of
          dimensions including none and :math:`H_\text{in} = \text{in\_features}`.
        - Output: :math:`(*, H_\text{out})` where all but the last dimension
          are the same shape as the input and :math:`H_\text{out} = \text{out\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`

    Examples::

        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """

    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        if saw_method=='roll' and saw_k>0:
            for i in range(saw_k):
                self.__setattr__(f'saw_theta_{i}',
                                 torch.nn.Parameter(torch.zeros(out_features)))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)


    def forward(self, result: Tensor) -> Tensor:
        result = F.linear(result, self.weight, None)
        shapes = [1] * result.dim()
        shapes[-1] = self.out_features
        x = 1
        # print(f'result: {result.abs().max()} | bias:{self.bias.abs().max()}')
        for i in range(0, saw_k):
            theta = self.__getattr__(f'saw_theta_{i}').reshape(*shapes)
            x += theta * torch.roll(result, shifts=i + 1, dims=-1)
        bias = 0 if self.bias is None else self.bias
        return x * result + bias

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, saw_k={saw_k}, bias={self.bias is not None}"




class SAWConv1D(nn.Module):
    """
    1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).

    Basically works like a linear layer but the weights are transposed.

    Args:
        nf (`int`): The number of output features.
        nx (`int`): The number of input features.
    """

    def __init__(self, nf, nx):
        super().__init__()
        self.nf = nf
        self.nx = nx
        self.weight = nn.Parameter(torch.empty(nx, nf))
        self.bias = nn.Parameter(torch.zeros(nf))
        nn.init.normal_(self.weight, std=0.02)
        if saw_method=='roll' and saw_k>0:
            for i in range(saw_k):
                self.__setattr__(f'saw_theta_{i}',
                                 torch.nn.Parameter(torch.zeros(nf)))
    def __repr__(self) -> str:
        return "Conv1D(nf={nf}, nx={nx}, saw_k={saw_k})".format(**self.__dict__)

    def forward(self, result):
        size_out = result.size()[:-1] + (self.nf,)
        result = torch.mm(result.view(-1, result.size(-1)), self.weight)

        x = 1
        for i in range(0, saw_k):
            theta = self.__getattr__(f'saw_theta_{i}')[None,:]
            x += theta * torch.roll(result, shifts=i + 1, dims=-1)

        bias = 0 if self.bias is None else self.bias
        result = x * result + bias
        result = result.view(size_out)

        return result