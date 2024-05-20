import math

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .DLinear import series_decomp
from util.lead_estimate import shifted_leader_seq


def instance_norm(ts, dim):
    mu = ts.mean(dim, keepdims=True)
    ts = ts - mu
    std = ((ts ** 2).mean(dim, keepdims=True) + 1e-8) ** 0.5
    return ts / std, mu, std


class Model(nn.Module):

    def __init__(self, backbone, configs):
        super(Model, self).__init__()
        self.backbone = backbone
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in * configs.in_dim
        self.K = min(configs.leader_num, self.channels)
        self.lead_refiner = LeadRefiner(self.seq_len, self.pred_len, self.channels, self.K,
                                        configs.state_num, temperature=configs.temperature,
                                        distributed=configs.local_rank != -1)

    def forward(self, x, leader_ids=None, shift=None, r=None, y_hat=None):
        if y_hat is None:
            y_hat = self.backbone(x)
        else:
            y_hat = y_hat.to(x.device)
        _x = x.permute(0, 2, 1)         # [B, C, L]
        _y_hat = y_hat.permute(0,2,1)   # [B, C, H]
        y_hat_new = self.lead_refiner(_x, _y_hat, leader_ids, shift, r)
        return y_hat_new.permute(0, 2, 1)


class LeadRefiner(nn.Module):
    def __init__(self, seq_len, pred_len, C, K, state_num=8, temperature=1.0, distributed=False):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.C = C
        self.K = K
        self.temperature = nn.Parameter(torch.ones(1) * temperature)
        if distributed:
            self.mix_layer = ComplexLinear((self.pred_len // 2 + 1) * 3, self.pred_len // 2 + 1)
        else:
            self.mix_layer = nn.Linear((self.pred_len // 2 + 1) * 3, self.pred_len // 2 + 1, dtype=torch.complex64)
        self.factory = FilterFactory(self.K, self.seq_len, self.C, (self.pred_len // 2 + 1) * (self.K * 2 + 1), state_num)
        self.leaders = None

        """ reduce time cost """
        self.register_buffer('const_indices', torch.arange(seq_len, seq_len + pred_len, dtype=torch.int)
                                          .unsqueeze(0).unsqueeze(0), persistent=False)
        self.register_buffer('const_ones', torch.ones(1, C, 1), persistent=False)

    def forward(self, x, y_hat, leader_ids=None, shift=None, r=None):
        B, C, L = x.shape

        _x, mu, std = instance_norm(x, -1)
        y_hat = (y_hat - mu) / std

        with torch.no_grad():
            seq_shifted, r = shifted_leader_seq(_x, y_hat, self.K, leader_ids, shift, r, const_indices=self.const_indices)
        r = torch.softmax(torch.cat([self.const_ones.expand(B, -1, -1), r.abs()], -1) / self.temperature, -1)
        filters, p = self.factory(x, r[..., 1:])
        filters = filters.view(B, C, -1, seq_shifted.shape[-1] // 2 + 1)
        _y_hat = y_hat
        y_hat_f = torch.fft.rfft(_y_hat)
        seq_shifted_f = torch.fft.rfft(seq_shifted) * filters[:, :, :self.K]
        seq_diff_f = (seq_shifted_f - y_hat_f.unsqueeze(2)) * filters[:, :, self.K:-1]
        y_hat_f = y_hat_f * filters[:, :, -1]
        y_hat = y_hat + torch.fft.irfft(self.mix_layer(torch.cat([seq_shifted_f.sum(2), seq_diff_f.sum(2), y_hat_f], -1)),
                                        n=self.pred_len)
        y_hat = y_hat * std + mu
        return y_hat


class FilterFactory(nn.Module):
    def __init__(self, feat_dim, seq_len, num_channel, out_dim, num_state=1, need_classifier=True):
        super().__init__()
        self.num_state = num_state
        self.need_classifier = need_classifier
        self.feat_dim = feat_dim
        self.seq_len = seq_len
        self.out_dim = out_dim
        if num_state > 1 and need_classifier:
            self.classifier = nn.Linear(seq_len, num_state, bias=False)
            self.basic_state = nn.Parameter(torch.empty(num_channel, num_state))
            self.bias = nn.Parameter(torch.empty(num_state))
            bound = 1 / math.sqrt(seq_len)
            nn.init.uniform_(self.bias, -bound, bound)
            bound = 1 / math.sqrt(num_state)
            nn.init.uniform_(self.basic_state, -bound, bound)

        if self.num_state == 1:
            self.mix_head = nn.Linear(feat_dim, self.out_dim)
        else:
            tmp_linear = nn.Linear(feat_dim, self.out_dim * num_state)
            self.mix_head_w = nn.Parameter(tmp_linear.weight.clone().view(self.out_dim, num_state, feat_dim).permute(1, 2, 0).reshape(num_state, -1))
            self.mix_head_b = nn.Parameter(tmp_linear.bias.clone().view(self.out_dim, num_state).transpose(0, 1))

    def forward(self, lookback_seq, corr_feat, p=None):
        if self.num_state == 1:
            return self.mix_head(corr_feat), None
        B, C = corr_feat.shape[:-1]
        if self.need_classifier:
            p = self.bias + self.basic_state + self.classifier(lookback_seq)
            p = torch.softmax(p, -1).unsqueeze(-2) # [B, C, 1, S]
        weight = (p @ self.mix_head_w).view(B, C, self.feat_dim, self.out_dim)
        bias = (p @ self.mix_head_b).view(B, C, self.out_dim)
        r = (corr_feat.unsqueeze(-2) @ weight).squeeze(-2) + bias
        return r, p


class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        weight_complex = torch.empty((out_features, in_features), dtype=torch.complex64)
        nn.init.kaiming_uniform_(weight_complex, a=math.sqrt(5))
        self.weight_real = nn.Parameter(weight_complex.real)
        self.weight_imag = nn.Parameter(weight_complex.imag)
        self.bias = bias
        if bias:
            bias_complex = torch.empty(out_features, dtype=torch.complex64)
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight_complex)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(bias_complex, -bound, bound)
            self.bias_real = nn.Parameter(bias_complex.real)
            self.bias_imag = nn.Parameter(bias_complex.imag)
        else:
            self.register_parameter('bias_real', None)
            self.register_parameter('bias_imag', None)

    def forward(self, x):
        return F.linear(x, torch.view_as_complex(torch.stack([self.weight_real, self.weight_imag], -1)),
                        torch.view_as_complex(torch.stack([self.bias_real, self.bias_imag], -1)) if self.bias else None)

