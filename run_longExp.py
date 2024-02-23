import argparse
import copy
import datetime
import os
import time
from data_provider import data_loader

ds = time.strftime("%Y%m%d", time.localtime())
dh = time.strftime("%Y%m%d%H", time.localtime())
cur_sec = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print(cur_sec)

from pprint import pprint
import torch
import settings
from settings import data_settings
from exp.exp_main import Exp_Main
from exp.exp_lead import Exp_Lead
import random
import numpy as np


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')


parser = argparse.ArgumentParser(description='Time Series Forecasting')

# basic config
parser.add_argument('--is_training', type=int, default=1, help='status')
parser.add_argument('--train_only', action='store_true', default=False, help='perform training on full input dataset without validation and testing')
parser.add_argument('--wo_test', action='store_true', default=False, help='only valid, not test')
parser.add_argument('--only_test', action='store_true', default=False)
parser.add_argument('--model', type=str, required=True, default='Autoformer',
                    help='model name, options: [Autoformer, Informer, Transformer]')
parser.add_argument('--override_hyper', action='store_true', default=True, help='Override hyperparams by setting.py')
parser.add_argument('--compile', action='store_true', default=False, help='Compile the model by Pytorch 2.0')
parser.add_argument('--reduce_bs', type=str_to_bool, default=False, help='Override batch_size in hyperparams by setting.py')
parser.add_argument('--normalization', type=str, default=None)
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# data loader
parser.add_argument('--root_path', type=str, default='./dataset/', help='root path of the data file')
parser.add_argument('--dataset', type=str, default='ETTh1', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--wrap_data_class', type=list, default=[])

# forecasting task
parser.add_argument('--seq_len', type=int, default=336, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=720, help='prediction sequence length')

# LIFT
parser.add_argument('--leader_num', type=int, default=4, help='# of leaders')
parser.add_argument('--state_num', type=int, default=8, help='# of variate states')
parser.add_argument('--prefetch_path', type=str, default='./prefetch/', help='location of prefetch files that records lead-lag relationships')
parser.add_argument('--tag', type=str, default='_max')
parser.add_argument('--prefetch_batch_size', type=int, default=16, help='prefetch_batch_size')
parser.add_argument('--variable_batch_size', type=int, default=32, help='variable_batch_size')
parser.add_argument('--max_leader_num', type=int, default=16, help='max # of leaders')
parser.add_argument('--masked_corr', action='store_true', default=False)
parser.add_argument('--efficient', type=str_to_bool, default=True)
parser.add_argument('--pin_gpu', type=str_to_bool, default=True)
parser.add_argument('--pretrain', action='store_true', default=False)
parser.add_argument('--freeze', action='store_true', default=False)
parser.add_argument('--lift', action='store_true', default=False)
parser.add_argument('--temperature', type=float, default=1.0, help='softmax temperature')

# DLinear
parser.add_argument('--individual', action='store_true', default=False, help='DLinear: a linear layer for each variate(channel) individually')

# PatchTST
parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
parser.add_argument('--patch_len', type=int, default=16, help='patch length')
parser.add_argument('--stride', type=int, default=8, help='stride')
parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')

# Formers
parser.add_argument('--embed_type', type=int, default=0, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=3, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
parser.add_argument('--output_enc', action='store_true', help='whether to output embedding from encoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

# Crossformer
parser.add_argument('--seg_len', type=int, default=24, help='segment length (L_seg)')
parser.add_argument('--win_size', type=int, default=2, help='window size for segment merge')
parser.add_argument('--num_routers', type=int, default=10, help='num of routers in Cross-Dimension Stage of TSA (c)')

# MTGNN
parser.add_argument('--subgraph_size',type=int,default=20,help='k')
parser.add_argument('--in_dim',type=int,default=1)

# GPT4TS
parser.add_argument('--gpt_layers', type=int, default=6)
parser.add_argument('--tmax', type=int, default=10)
parser.add_argument('--patch_size', type=int, default=16)

# optimization
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--itr', type=int, default=5, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
parser.add_argument('--begin_valid_epoch', type=int, default=0)
parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
parser.add_argument('--optim', type=str, default='Adam')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
parser.add_argument('--warmup_epochs', type=int, default=5)

# GPU
parser.add_argument('--use_gpu', type=str_to_bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')
parser.add_argument("--local-rank", default=os.getenv('LOCAL_RANK', -1), type=int)

args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

import platform
if platform.system() != 'Windows':
    args.num_workers = 0
else:
    torch.cuda.set_per_process_memory_fraction(48/61, 0)

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

args.enc_in, args.c_out = data_settings[args.dataset][args.features]
args.data_path = data_settings[args.dataset]['data']
args.dec_in = args.enc_in

if args.tag and args.tag[0] != '_':
    args.tag = '_' + args.tag

args.data = args.data_path[:5] if args.data_path.startswith('ETT') else 'custom'
if args.model.startswith('GPT4TS'):
    args.data += '_CI'

FLAG_LIFT = args.model == 'LightMTS' or args.lift
if FLAG_LIFT:
    Exp = Exp_Lead
    args.wrap_data_class.append(data_loader.Dataset_Lead_Pretrain if args.freeze else data_loader.Dataset_Lead)
    if args.dataset.startswith('ETT'):
        args.efficient = False
else:
    Exp = Exp_Main

args.model_id = f'{args.dataset}_{args.seq_len}_{args.pred_len}_{args.model}'
if args.normalization is not None:
    args.model_id += '_' + args.normalization

if args.override_hyper and args.model in settings.hyperparams:
    if 'prefetch_batch_size' in data_settings[args.dataset]:
        args.__setattr__('prefetch_batch_size', data_settings[args.dataset]['prefetch_batch_size'])
    for k, v in settings.get_hyperparams(args.dataset, args.model, args).items():
        args.__setattr__(k, v)

if args.local_rank != -1:
    torch.cuda.set_device(args.local_rank)
    args.gpu = args.local_rank
    torch.distributed.init_process_group(backend="nccl", init_method='env://')
    args.num_gpus = torch.cuda.device_count()
    args.batch_size = args.batch_size // args.num_gpus

if FLAG_LIFT and args.pretrain and args.freeze:
    args.lradj = 'type3'

if args.model in ['MTGNN']:
    if 'feat_dim' in data_settings[args.dataset]:
        args.in_dim = data_settings[args.dataset]['feat_dim']
        args.enc_in = int(args.enc_in / args.in_dim)
        if args.features == 'M':
            args.c_out = int(args.c_out / args.in_dim)

K_tag = f'_K{args.leader_num}' if args.leader_num > 8 and args.enc_in > 8 else ''
prefetch_path = os.path.join(args.prefetch_path, f'{args.dataset}_L{args.seq_len}{K_tag}{args.tag}')
if not os.path.exists(prefetch_path + '_train.npz'):
    K_tag = f'_K16' if args.leader_num > 8 and args.enc_in > 8 else ''
    prefetch_path = os.path.join(args.prefetch_path, f'{args.dataset}_L{args.seq_len}{K_tag}{args.tag}')
args.prefetch_path = prefetch_path

if args.lift and 'Linear' in args.model or args.model == 'LightMTS':
    args.patience = max(args.patience, 5)

args.find_unused_parameters = args.model in ['MTGNN']

print('Args in experiment:')
print(args)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.benchmark=False
    # torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    train_data, train_loader, vali_data, vali_loader = None, None, None, None
    test_data, test_loader = None, None

    if args.is_training:
        all_results = {'mse': [], 'mae': []}
        for ii in range(args.itr):
            if args.model == 'PatchTST' and args.dataset in ['ECL', 'Traffic', 'Illness', 'Weather']:
                fix_seed = 2021 + ii
            else:
                fix_seed = 2023 + ii
            setup_seed(fix_seed)
            print('Seed:', fix_seed)

            setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_lr{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.model_id,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.learning_rate,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des, ii)

            if args.pretrain:
                pretrain_setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_lr{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                    args.model_id,
                    args.border_type if args.border_type else args.data,
                    args.features,
                    args.seq_len,
                    args.label_len,
                    args.pred_len,
                    settings.pretrain_lr(args.model, args.dataset, args.pred_len, args.learning_rate),
                    args.d_model,
                    args.n_heads,
                    args.e_layers,
                    args.d_layers,
                    args.d_ff,
                    args.factor,
                    args.embed,
                    args.distil,
                    args.des, ii)
                args.pred_path = os.path.join('./results/', pretrain_setting, 'real_prediction.npy')
                args.load_path = os.path.join('./checkpoints/', pretrain_setting, 'checkpoint.pth')
                if FLAG_LIFT and args.freeze:
                    if not os.path.exists(args.pred_path) and args.local_rank <= 0:
                        _args = copy.deepcopy(args)
                        _args.freeze = False
                        _args.wrap_data_class = []
                        exp = Exp_Main(_args)
                        print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(pretrain_setting))
                        exp.predict(pretrain_setting, True)
                        torch.cuda.empty_cache()

            if args.lift:
                setting += '_lift'

            exp = Exp(args)  # set experiments

            path = os.path.join("checkpoints", setting, 'checkpoint.pth')
            print('Checkpoints in', path)
            if not args.only_test or not os.path.exists(path):
                print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
                _, train_data, train_loader, vali_data, vali_loader = exp.train(setting, train_data, train_loader, vali_data, vali_loader)
                torch.cuda.empty_cache()
            else:
                print('Loading', path)
                exp.load_checkpoint(path)
                print('Learning rate of model_optim is', exp.model_optim.param_groups[0]['lr'])

            if not args.wo_test and not args.train_only and args.local_rank <= 0:
                print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                mse, mae, test_data, test_loader = exp.test(setting, test_data, test_loader)
                all_results['mse'].append(mse)
                all_results['mae'].append(mae)

            if args.do_predict:
                print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.predict(setting, True)

            torch.cuda.empty_cache()
        if not args.wo_test and not args.train_only and args.local_rank <= 0:
            for k in all_results.keys():
                all_results[k] = np.array(all_results[k])
                all_results[k] = [all_results[k].mean(), all_results[k].std()]
            pprint(all_results)
    else:
        ii = 0
        setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_lr{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.model_id,
            args.border_type if args.border_type else args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.learning_rate,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.des, ii)
        args.load_path = os.path.join(args.checkpoints, setting, 'checkpoint.pth')
        # args.pretrain = True
        if args.lift:
            setting += '_lift'

        exp = Exp(args)  # set experiments

        if args.do_predict:
            print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.predict(setting, True)
        else:
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting, test=1)
        torch.cuda.empty_cache()
