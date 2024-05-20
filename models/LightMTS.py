
import torch
import torch.nn as nn
from util.lead_estimate import shifted_leader_seq
from models import LIFT
from models.LIFT import instance_norm


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.backbone = nn.Linear(self.seq_len, self.pred_len)
        self.channels = configs.enc_in * configs.in_dim
        self.K = min(configs.leader_num, self.channels)
        self.lead_refiner = LeadRefiner(self.seq_len, self.pred_len, self.channels, self.K,
                                        configs.state_num, temperature=configs.temperature,
                                        distributed=configs.local_rank != -1)

    def forward(self, x, leader_ids=None, shift=None, r=None):
        x = x.permute(0, 2, 1)         # [B, C, L]
        _x, mu, norm = instance_norm(x, -1)
        y_hat = self.backbone(_x)

        y_hat_new = self.lead_refiner(_x, x, y_hat, leader_ids, shift, r)

        y_hat_new = y_hat_new * norm + mu
        return y_hat_new.permute(0, 2, 1)


class LeadRefiner(LIFT.LeadRefiner):
    def forward(self, x, raw_x, y_hat, leader_ids=None, shift=None, r=None):
        B, C, L = x.shape
        with torch.no_grad():
            seq_shifted, r = shifted_leader_seq(x, y_hat, self.K, leader_ids, shift, r, const_indices=self.const_indices)
        r = torch.softmax(torch.cat([self.const_ones.expand(B, -1, -1), r.abs()], -1) / self.temperature, -1)
        filters, p = self.factory(x, r[..., 1:])
        filters = filters.view(B, C, -1, seq_shifted.shape[-1] // 2 + 1)
        _y_hat = y_hat
        y_hat_f = torch.fft.rfft(_y_hat)
        seq_shifted_f = torch.fft.rfft(seq_shifted) * filters[:, :, :self.K]
        seq_diff_f = (seq_shifted_f - y_hat_f.unsqueeze(2)) * filters[:, :, self.K:-1]
        y_hat_f = y_hat_f * filters[:, :, -1]
        y_hat = y_hat + torch.fft.irfft(
            self.mix_layer(torch.cat([seq_shifted_f.sum(2), seq_diff_f.sum(2), y_hat_f], -1)),
            n=self.pred_len
        )

        return y_hat
