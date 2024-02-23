import math

import numpy as np
import pandas as pd


def update_metrics(pred, label, statistics, target_variate=None):
    if isinstance(pred, tuple):
        pred = pred[0]
    if target_variate is not None:
        pred = pred[:, :, target_variate]
        if label.dim() == 3:
            label = label[:, :, target_variate]

    balance = pred - label
    # statistics['all_preds'].append(pred)
    statistics['y_sum'] += label.abs().sum().item()
    statistics['total'] += len(label.view(-1))
    statistics['MAE'] += balance.abs().sum().item()
    statistics['MSE'] += (balance ** 2).sum().item()
    # RRSE += (balance ** 2).sum()
    # x2_sum += (target_batch ** 2).sum()
    # x_sum += target_batch.sum()


def calculate_metrics(statistics):
    MSE, MAE, total, y_sum = statistics['MSE'], statistics['MAE'], statistics['total'], statistics['y_sum']
    metrics = {'MSE': MSE / total, 'MAE': MAE / total}
    # metrics['NMAE'] = MAE / y_sum
    # metrics['NRMSE'] = math.sqrt((MSE / total)) / (y_sum / total)
    # var = x2_sum / total - (x_sum / total) ** 2
    # RRSE = math.sqrt(RRSE.item() / total) / var.item()
    return metrics

def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    d += 1e-12
    return 0.01*(u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    rse = RSE(pred, true)
    corr = CORR(pred, true)

    return mae, mse, rmse, mape, mspe, rse, corr


def calc_ic(pred=None, label=None, index=None, df=None, return_type='all', reduction='sum'):
    if df is None:
        if isinstance(pred, tuple):
            pred = pred[0]
        df = pd.DataFrame({'pred': pred, 'label': label}, index=index)
    if index is None:
        res = []
        if return_type != 'ric':
            res.append(df['pred'].corr(df['label']))
        if return_type != 'ic':
            res.append(df['pred'].corr(df['label'], method='spearman'))
        return res
    else:
        groups = df.groupby('datetime')
        res = []
        if return_type != 'ric':
            res.append(groups.apply(lambda df: df["pred"].corr(df["label"], method="pearson")))
        if return_type != 'ic':
            res.append(groups.apply(lambda df: df["pred"].corr(df["label"], method="spearman")))
        if reduction == 'sum':
            return [r.sum() for r in res] + [len(groups)]
        elif reduction == 'mean':
            return [r.mean() for r in res]
        else:
            return [r.to_numpy().tolist() for r in res]
