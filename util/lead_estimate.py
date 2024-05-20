import math

import torch


def accurate_indicator(x, j, K, local_max=True):
    C, L = x.shape[1:]
    B = x.shape[0] - L

    # for j in range(C):
    target = x[L:, [j]]
    cross_corr = torch.empty(B, C, L + 1, device=x.device)
    for lag in range(0, L + 1):
        cross_corr[..., lag] = (target * x[L-lag: (-lag if lag > 0 else x.shape[0] + 1)]).mean(-1)

    corr_abs = cross_corr.abs()
    if local_max:
        mask = (corr_abs[..., 1:-1] >= corr_abs[..., :-2]) & (corr_abs[..., 1:-1] >= corr_abs[..., 2:])
        cross_corr = cross_corr[..., 1:-1] * mask
        corr_abs = cross_corr.abs()

    corr_abs_max, shift = corr_abs.max(-1)  # [B, C]

    if not local_max:
        corr_abs_max *= (shift > 0)
    _, leader_ids = corr_abs_max.topk(K, dim=-1)  # [B, K]

    corr_max = cross_corr.gather(-1, shift.unsqueeze(-1)).squeeze(-1) # [B, C]
    r = corr_max.gather(-1, leader_ids)  # [B, K]
    shift = shift.gather(-1, leader_ids)  # [B, K]

    if local_max:
        shift = shift + 1

    return leader_ids, shift, r


def cross_corr_coef(x, variable_batch_size=32, predefined_leaders=None, local_max=True):
    B, C, L = x.shape

    rfft = torch.fft.rfft(x, dim=-1)  # [B, C, F]
    rfft_conj = torch.conj(rfft)
    if predefined_leaders is None:
        cross_corr = torch.cat([
            torch.fft.irfft(rfft.unsqueeze(2) * rfft_conj[:, i: i + variable_batch_size].unsqueeze(1),
                            dim=-1, n=L)
            for i in range(0, C, variable_batch_size)],
            2)  # [B, C, C, L]
    else:
        cross_corr = torch.fft.irfft(
            rfft.unsqueeze(2) * rfft_conj[:, predefined_leaders.view(-1)].view(B, C, -1, rfft.shape[-1]),
            dim=-1, n=L)

    if local_max:
        corr_abs = cross_corr.abs()
        mask = (corr_abs[..., 1:-1] >= corr_abs[..., :-2]) & (corr_abs[..., 1:-1] >= corr_abs[..., 2:])
        cross_corr = cross_corr[..., 1:-1] * mask

    # cross_corr[..., 0] = cross_corr[..., 0] * (1 - torch.eye(cross_corr.shape[1], device=cross_corr.device))

    return cross_corr / L


def estimate_indicator(x, K, variable_batch_size=32, predefined_leaders=None, local_max=True):
    cross_corr = cross_corr_coef(x, variable_batch_size, predefined_leaders)
    corr_abs = cross_corr.abs()     # [B, C, C, L]
    corr_abs_max, shift = corr_abs.max(-1)  # [B, C, C]
    if not local_max:
        corr_abs_max = corr_abs_max * (shift > 0)
    _, leader_ids = corr_abs_max.topk(K, dim=-1)  # [B, C, K]
    lead_corr = cross_corr.gather(2,
                                  leader_ids.unsqueeze(-1).expand(-1, -1, -1, cross_corr.shape[-1]))  # [B, C, K, L]
    shift = shift.gather(2, leader_ids)  # [B, C, K]
    r = lead_corr.gather(3, shift.unsqueeze(-1)).squeeze(-1)  # [B, C, K]
    if local_max:
        shift = shift + 1
    if predefined_leaders is not None:
        leader_ids = predefined_leaders.unsqueeze(0).expand(len(x), -1, -1).gather(-1, leader_ids)
    return leader_ids, shift, r


def shifted_leader_seq(x, y_hat, leader_num, leader_ids=None, shift=None, r=None, const_indices=None,
                       variable_batch_size=32, predefined_leaders=None):
    B, C, L = x.shape
    H = y_hat.shape[-1]
    
    if const_indices is None:
        const_indices = torch.arange(L, L + H, dtype=torch.int, device=x.device).unsqueeze(0).unsqueeze(0)

    if leader_ids is None:
        leader_ids, shift, r = estimate_indicator(x, leader_num,
                                                  variable_batch_size=variable_batch_size,
                                                  predefined_leaders=predefined_leaders)
    indices = const_indices - shift.view(B, -1, 1)  # [B, C*K, H]

    seq = torch.cat([x, y_hat], -1)  # [B, C, L+H]
    seq = seq.gather(1, leader_ids.view(B, -1, 1).expand(-1, -1, L + H))  # [B, C*K, L+H]
    seq_shifted = seq.gather(-1, indices)
    seq_shifted = seq_shifted.view(B, C, -1, indices.shape[-1])  # [B, C, K, H]

    r = r.view(B, C, -1)  # [B, C, K]
    seq_shifted = seq_shifted * torch.sign(r).unsqueeze(-1)

    return seq_shifted, r.abs()


def accurate_strict_indicator_coef(x, j):
    C, L = x.shape[1:]
    B = x.shape[0] - L

    # for j in range(C):
    target = x[L:, [j]]
    cross_corr = torch.empty(B, C, L + 1, device=x.device)
    for lag in range(0, L + 1):
        cross_corr[..., lag] = (target * x[L-lag: (-lag if lag > 0 else x.shape[0] + 1)]).mean(-1)

    corr_abs = cross_corr.abs()
    mask = (corr_abs[..., 1:-1] >= corr_abs[..., :-2]) & (corr_abs[..., 1:-1] >= corr_abs[..., 2:])
    cross_corr = cross_corr[..., 1:-1] * mask
    return cross_corr.abs()
    corr_abs_max, shift = corr_abs.max(-1)  # [B, C]
    corr_abs_max = corr_abs_max * (shift > 0)
    corr_abs_max, leader_ids = corr_abs_max.max(-1)  # [B, K]
    return corr_abs_max


def estimate_strict_indicator_coef(x, K, num_lead_step=1, variable_batch_size=32, predefined_leaders=None):
    B, C, L = x.shape
    rfft = torch.fft.rfft(x, dim=-1)  # [B, C, F]
    rfft_conj = torch.conj(rfft)
    if predefined_leaders is None:
        cross_corr = torch.cat([
            torch.fft.irfft(rfft.unsqueeze(2) * rfft_conj[:, i: i + variable_batch_size].unsqueeze(1),
                            dim=-1, n=L)
            for i in range(0, C, variable_batch_size)],
            2)  # [B, C, C, L]
    else:
        cross_corr = torch.fft.irfft(
            rfft.unsqueeze(2) * rfft_conj[:, predefined_leaders.view(-1)].view(B, C, -1, rfft.shape[-1]),
            dim=-1, n=L)
    corr_abs = cross_corr.abs()
    mask = (corr_abs[..., 1:-1] >= corr_abs[..., :-2]) & (corr_abs[..., 1:-1] >= corr_abs[..., 2:])
    cross_corr = cross_corr[..., 1:-1] * mask
    return cross_corr.abs()
    corr_abs_max, shift = corr_abs.max(-1)  # [B, C, C]
    # corr_abs_max = corr_abs_max * (shift > 0)
    return corr_abs_max.max(-1)[0] / L

