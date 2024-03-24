import os
import re

need_x_y_mark = ['Autoformer', 'Transformer', 'Informer']
need_x_mark = ['TCN', 'FSNet', 'OneNet']

data_settings = {
    'wind': {'data': 'wind.csv', 'T':'UK', 'M':[28,28], 'prefetch_batch_size': 64},
    'ECL':{'data':'electricity.csv','T':'OT','M':[321,321],'S':[1,1],'MS':[321,1], 'prefetch_batch_size': 10},
    'ETTh1':{'data':'ETTh1.csv','T':'OT','M':[7,7],'S':[1,1],'MS':[7,1], 'prefetch_batch_size': 128},
    'ETTh2':{'data':'ETTh2.csv','T':'OT','M':[7,7],'S':[1,1],'MS':[7,1], 'prefetch_batch_size': 128},
    'ETTm1':{'data':'ETTm1.csv','T':'OT','M':[7,7],'S':[1,1],'MS':[7,1], 'prefetch_batch_size': 128},
    'ETTm2':{'data':'ETTm2.csv','T':'OT','M':[7,7],'S':[1,1],'MS':[7,1], 'prefetch_batch_size': 128},
    'Solar':{'data':'solar_AL.txt','T': 136,'M':[137,137],'S':[1,1],'MS':[137,1], 'prefetch_batch_size': 32},
    'Weather':{'data':'weather.csv','T':'OT','M':[21,21],'S':[1,1],'MS':[21,1], 'prefetch_batch_size': 64},
    'Traffic': {'data': 'traffic.csv', 'T':'OT', 'M':[862,862], 'prefetch_batch_size': 2},
    'PeMSD8': {'data':'PeMSD8/PeMSD8.npz','T': 0,'M':[510,510],'S':[1,1],'MS':[510,1], 'prefetch_batch_size': 6, 'feat_dim': 3},
    'Exchange': {'data': 'exchange_rate.csv', 'T':'OT', 'M':[8,8], 'prefetch_batch_size': 128},
    'exchange_rate': {'data': 'exchange_rate.csv', 'T':'OT', 'M':[8,8], 'prefetch_batch_size': 128},
    'Illness': {'data': 'illness.csv', 'T':'OT', 'M':[7,7], 'prefetch_batch_size': 128},
}


hyperparams = {
    'PatchTST': {'e_layers': 3, 'patience': 5},
    'MTGNN': {},
    'Crossformer': {'lradj': 'Crossformer', 'e_layers': 3, 'seg_len': 24, 'd_ff': 512, 'd_model': 256, 'n_heads': 4, 'dropout': 0.2},
    'DLinear': {},
    'GPT4TS': {'e_layers': 3, 'd_model': 768, 'n_heads': 4, 'd_ff': 768, 'dropout': 0.3, 'train_epochs': 10}
}


def get_hyperparams(data, model, args):
    hyperparam: dict = hyperparams[model]

    if data in 'ECL|PeMSD4|PeMSD8|PEMS_BAY'.split('|'):
        hyperparam['temperature'] = 0.1
    # else:
    #     hyperparam['temperature'] = 1.0

    if model == 'PatchTST':
        hyperparam['patience'] = max(hyperparam['patience'], args.patience)
        # if data in ['ECL']:
        #     hyperparam['patience'] = 10

        if data in ['ETTh1', 'ETTh2', 'Weather', 'ETTm1', 'ETTm2', 'Exchange']:
            hyperparam['batch_size'] = 128
        elif data in ['Illness']:
            hyperparam['batch_size'] = 16

        if args.lradj != 'type3':
            if data in ['ETTh1', 'ETTh2', 'Weather', 'Exchange', 'wind']:
                hyperparam['lradj'] = 'type3'
            elif data in ['Illness']:
                hyperparam['lradj'] = 'constant'
            else:
                hyperparam['lradj'] = 'TST'

        if data in ['ETTh1', 'ETTh2', 'Illness']:
            hyperparam.update(**{'dropout': 0.3, 'fc_dropout': 0.3, 'n_heads': 4, 'd_model': 16, 'd_ff': 128})
        elif data in ['ETTm1', 'ETTm2', 'Weather', 'ECL', 'Traffic']:
            hyperparam.update(**{'dropout': 0.2, 'fc_dropout': 0.2, 'n_heads': 16, 'd_model': 128, 'd_ff': 256})
        else:
            hyperparam.update(**{'dropout': 0.2, 'fc_dropout': 0.2, 'n_heads': 16, 'd_model': 64, 'd_ff': 128})

    elif model in ['MTGNN']:
        if data in ['Traffic'] and args.pred_len >= 720:
            hyperparam['batch_size'] = 24

        if data in ['Exchange', 'Weather', 'wind']:
            hyperparam['subgraph_size'] = 8
        elif data in ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Illness']:
            hyperparam['subgraph_size'] = 4

    elif model == 'Crossformer':
        if data == 'ECL' or args.lradj == 'fixed':
            hyperparam['lradj'] = 'fixed'

        if data in ['Traffic', 'PeMSD4'] and args.pred_len >= 720:
            hyperparam['batch_size'] = 24
        if data in ['PeMSD8'] and args.pred_len >= 720:
            hyperparam['batch_size'] = 16

        if data in ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Weather', 'Illness', 'wind', 'Exchange']:
            hyperparam['d_model'] = 256
            hyperparam['n_heads'] = 4
        else:
            hyperparam['d_model'] = 64
            hyperparam['n_heads'] = 2

        if data in ['Traffic', 'ECL']:
            hyperparam['d_ff'] = 128

        if data in ['Illness']:
            hyperparam['e_layers'] = 2

    elif model == 'GPT4TS':
        if data == 'ETTh1':
            hyperparam['lradj'] = 'typy4'
            hyperparam['tmax'] = 20
        elif data == 'ETTh2':
            hyperparam['dropout'] = 1
            hyperparam['tmax'] = 20
        elif data == 'Traffic':
            hyperparam['dropout'] = 0.3
        elif data == 'ECL':
            hyperparam['tmax'] = 10
        elif data == 'Illness':
            hyperparam['patch_size'] = 24
            hyperparam['batch_size'] = 16

        if data in ['ETTm1', 'ETTm2', 'ECL', 'Traffic', 'Weather']:
            hyperparam['seq_len'] = 512

        if data.startswith('ETTm'):
            hyperparam['stride'] = 16
        elif args.seq_len == 104:
            hyperparam['stride'] = 2

    return hyperparam


def pretrain_lr(model, dataset, H, lr):
    if model == 'MTGNN':
        if dataset in 'Weather|ETTh1|ETTm1'.split('|'):
            return 0.0001
        elif dataset in 'ETTm2'.split('|'):
            return 0.0005
        elif dataset in 'ETTh2'.split('|'):
            return 0.001
        elif dataset in 'Solar'.split('|'):
            return 0.001
        elif dataset in ['ECL']:
            return 0.0005 if H == 720 else 0.001
        return 0.001
    if 'PatchTST' in model:
        if dataset in ['PeMSD8', 'Solar']:
            return 0.001
        return 0.0001
    if model == 'Crossformer':
        if dataset in ['ECL']:
            return 0.005
        elif dataset in ['wind']:
            if H <= 96:
                return 0.0001
            else:
                return 0.00005
        elif dataset in ['Weather']:
            if H >= 192:
                return 0.00001
            else:
                return 0.00005
        elif dataset in 'Solar'.split('|'):
            if H >= 192:
                return 0.0005
            else:
                return 0.001
        elif dataset in 'ETTh1|ETTh2'.split('|'):
            if H >= 168:
                return "0.00001"
            else:
                return 0.0001
        elif dataset in 'ETTm1'.split('|'):
            if H in [192, 336]:
                return "0.00001"
            else:
                return 0.0001
        if dataset in 'ETTm2'.split('|'):
            if H >= 288:
                return "0.00001"
            else:
                return 0.0001
        if dataset in ['Traffic']:
            if H in [720]:
                return 0.0005
            else:
                return 0.001
    return lr
