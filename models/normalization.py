import torch
import torch.nn as nn
import torch.nn.functional as F


class ForecastModel(nn.Module):
    def __init__(self, backbone, num_features, seq_len, process_method='RevIN', **kwargs):
        super().__init__()
        self.backbone = backbone
        if process_method.lower() == 'revin':
            self.processor = RevIN(num_features=num_features, **kwargs)
        elif process_method.lower() == 'dishts':
            self.processor = DishTS(num_features=num_features, seq_len=seq_len, **kwargs)
        else:
            raise NotImplementedError

    def forward(self, x, *args, process=True, **kwargs):
        if process:
            x = self.processor(x, mode='forward')
        outputs = self.backbone(x, *args, **kwargs)
        if not process:
            return outputs
        if isinstance(outputs, tuple):
            pred = self.processor(outputs[0], mode='inverse')
            return [pred] + [o for o in outputs[1:]]
        else:
            return self.processor(outputs, mode='inverse')


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, **kwargs):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str = 'forward'):
        if mode == 'forward':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'inverse':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x


class DishTS(RevIN):
    def __init__(self, num_features: int, eps=1e-8, affine=True, seq_len=None, init='standard', **kwargs):
        super().__init__(num_features, eps, affine, **kwargs)
        if init == 'standard':
            self.reduce_mlayer = nn.Parameter(torch.rand(num_features, seq_len, 2) / seq_len)
        elif init == 'avg':
            self.reduce_mlayer = nn.Parameter(torch.ones(num_features, seq_len, 2) / seq_len)
        elif init == 'uniform':
            self.reduce_mlayer = nn.Parameter(
                torch.ones(num_features, seq_len, 2) / seq_len + torch.rand(num_features, seq_len, 2) / seq_len)

    def _get_statistics(self, x):
        x_transpose = x.permute(2, 0, 1)
        theta = torch.bmm(x_transpose, self.reduce_mlayer).permute(1, 2, 0)
        # theta = F.gelu(theta)
        self.phil, self.phih = theta[:, :1, :], theta[:, 1:, :]
        self.xil = torch.sqrt(torch.sum(torch.pow(x - self.phil, 2), axis=1, keepdim=True) / (x.shape[1] - 1) + self.eps)
        self.xih = torch.sqrt(torch.sum(torch.pow(x - self.phih, 2), axis=1, keepdim=True) / (x.shape[1] - 1) + self.eps)

    def _normalize(self, x):
        x = (x - self.phil) / self.xil
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps)
        x = x * self.xih
        x = x + self.phih
        return x

