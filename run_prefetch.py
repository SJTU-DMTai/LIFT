import argparse
import collections
import datetime
import os
import time
import torch
from data_provider.data_factory import data_provider
from data_provider.data_loader import Dataset_Lead, Dataset_Lead_Stat
import random
import numpy as np
from settings import data_settings

ds = time.strftime("%Y%m%d", time.localtime())
dh = time.strftime("%Y%m%d%H", time.localtime())
cur_sec = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print(cur_sec)

def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')

parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')

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
parser.add_argument('--prefetch_path', type=str, default='./prefetch/', help='location of prefetch files')
parser.add_argument('--tag', type=str, default='_max')
parser.add_argument('--prefetch_batch_size', type=int, default=16, help='prefetch_batch_size')
parser.add_argument('--variable_batch_size', type=int, default=32, help='variable_batch_size')
parser.add_argument('--max_leader_num', type=int, default=16, help='max # of leaders')
parser.add_argument('--masked_corr', action='store_true', default=False)
parser.add_argument('--efficient', type=str_to_bool, default=True)
parser.add_argument('--local_max', action='store_true', default=False)
parser.add_argument('--lift', action='store_true', default=True)
parser.add_argument('--temperature', type=float, default=1.0, help='softmax temperature')

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
parser.add_argument("--local-rank", default=os.getenv('LOCAL_RANK', -1), type=int)

args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
args.pin_gpu = True

import platform

if platform.system() != 'Windows':
    args.num_workers = 2
    torch.set_num_threads(2)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.benchmark=False
    # torch.backends.cudnn.deterministic = True

setup_seed(2023)
if args.tag and args.tag[0] != '_':
    args.tag = '_' + args.tag

changes = collections.defaultdict(dict)
corrs = collections.defaultdict(dict)
for data_name in ['Weather', 'ECL', 'Traffic', 'PeMSD8', 'wind', 'Solar', 'ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Exchange', 'Illness']:
    if data_name.startswith('ETT') or data_name.lower() in ['weather', 'wind', 'exchange', 'illness', 'ill']:
        args.efficient = False
        args.tag = '_acc'
    else:
        args.efficient = True
        args.tag = '_max'

    args.threshold = 0.9 if data_name in ['Weather', 'PeMSD8'] else 0

    v = data_settings[data_name]
    print(data_name)
    seq_len = args.seq_len

    if data_name == 'Illness':
        args.seq_len = 104
    else:
        args.seq_len = seq_len
    args.dataset = data_name
    args.enc_in, args.c_out = data_settings[args.dataset][args.features]
    args.data_path = data_settings[args.dataset]['data']
    args.dec_in = args.enc_in
    args.model_id = f'{args.dataset}_{args.seq_len}_{args.pred_len}'

    args.data = args.data_path[:5] if args.data_path.startswith('ETT') else 'custom'

    prefetch_batch_size = args.prefetch_batch_size if 'prefetch_batch_size' not in v else v['prefetch_batch_size']

    for flag in ['train', 'val', 'test']:
        K_tag = f'_K{args.max_leader_num}' if args.max_leader_num > 8 and args.enc_in > 8 else ''
        prefetch_path = os.path.join(args.prefetch_path,
                                     f'{args.dataset}_L{args.seq_len}{K_tag}{args.tag}_{flag}.npz')
        data_set, data_loader = data_provider(args, flag, wrap_class=Dataset_Lead_Stat,
                                              leaders=None, prefetch_path=prefetch_path,
                                              leader_num=args.max_leader_num,
                                              leader_select_num=1,
                                              prefetch_batch_size=prefetch_batch_size, device='cuda',
                                              trunc_tail=0,
                                              variable_batch_size=args.variable_batch_size,
                                              efficient=args.efficient,
                                              threshold=args.threshold, local_max=args.local_max
                                              )
