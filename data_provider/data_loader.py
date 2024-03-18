
import traceback

import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from util.functional import instance_norm
from util.lead_estimate import estimate_indicator, accurate_indicator, shifted_leader_seq
from util.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')


def get_alldata(filename='electricity.csv', root_path='./'):
    path = os.path.join(root_path, filename)
    if filename.endswith('.csv'):
        df = pd.read_csv(path)
        if filename.startswith('wind'):
            df['date'] = pd.date_range(start='2000-01-01', periods=len(df), freq='H')
    else:
        if filename.startswith('nyc'):
            import h5py
            x = h5py.File(path, 'r')
            data = list()
            for key in x.keys():
                data.append(x[key][:])
            ts = np.stack(data, axis=1)
            ts = ts.reshape((ts.shape[0], ts.shape[1] * ts.shape[2]))
            df = pd.DataFrame(ts)
            df['date'] = pd.date_range(start='2007-04-01', periods=len(df), freq='30T')
        elif filename.endswith('.npz'):
            ts = np.load(path)['data']
            ts = ts.reshape((ts.shape[0], ts.shape[1] * ts.shape[2]))
            df = pd.DataFrame(ts)
            if filename == 'PeMSD4':
                df['date'] = pd.date_range(start='2017-07-01', periods=len(df), freq='5T')
            else:
                df['date'] = pd.date_range(start='2012-03-01', periods=len(df), freq='5T')
        elif filename.endswith('.h5'):
            df = pd.read_hdf(path)
            df['date'] = df.index.values
        elif filename.endswith('.txt'):
            df = pd.read_csv(path, header=None)
            df['date'] = pd.date_range(start='1/1/2007', periods=len(df), freq='10T')
        df = df[[df.columns[-1]] + list(df.columns[:-1])]
    return df


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', train_only=False, border=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.border = border

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        if self.border is None:
            border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
            border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        else:
            border1s, border2s = self.border
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, self.timeenc, freq=self.freq)

        self.data = data[:border2s[-1]]
        self.border = (border1, border2)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
        self.data_x = torch.tensor(self.data_x).float()
        self.data_y = torch.tensor(self.data_y).float()
        self.data_stamp = torch.tensor(self.data_stamp).float()

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[s_end:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t', train_only=False, border=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.border = border

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        if self.border is None:
            border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
            border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        else:
            border1s, border2s = self.border
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, self.timeenc, freq=self.freq)

        self.data = data[:border2s[-1]]
        self.border = (border1, border2)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
        self.data_x = torch.tensor(self.data_x).float()
        self.data_y = torch.tensor(self.data_y).float()
        self.data_stamp = torch.tensor(self.data_stamp).float()

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[s_end:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', train_only=False, border=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.train_only = train_only

        self.split_ratio = (0.7, 0.2) if border is None else border
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = get_alldata(self.data_path, self.root_path)

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        if self.features == 'S':
            cols.remove(self.target)
        cols.remove('date')
        # print(cols)
        num_train = int(len(df_raw) * (self.split_ratio[0] if not self.train_only else 1))
        num_test = int(len(df_raw) * self.split_ratio[1])
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            df_raw = df_raw[['date'] + cols]
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_raw = df_raw[['date'] + cols + [self.target]]
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            # print(self.scaler.mean_)
            # exit()
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, self.timeenc, freq=self.freq)

        self.data = data
        self.border = (border1, border2)

        self.data_x = data[border1:border2]
        if self.features == 'MS':
            self.data_y = data[:, -1][border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
        self.data_x = torch.tensor(self.data_x).float()
        self.data_y = torch.tensor(self.data_y).float()
        self.data_stamp = torch.tensor(self.data_stamp).float()

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[s_end:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_hour_CI(Dataset_ETT_hour):
    def __init__(self, root_path, *args, **kwargs):
        super().__init__(root_path, *args, **kwargs)
        self.enc_in = self.data_x.shape[-1]
        self.tot_len = super().__len__()

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id + 1]
        seq_y = self.data_y[s_end:r_end, feat_id:feat_id + 1]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return super().__len__() * self.enc_in


class Dataset_ETT_minute_CI(Dataset_ETT_minute):
    def __init__(self, root_path, *args, **kwargs):
        super().__init__(root_path, *args, **kwargs)
        self.enc_in = self.data_x.shape[-1]
        self.tot_len = super().__len__()

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id + 1]
        seq_y = self.data_y[s_end:r_end, feat_id:feat_id + 1]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return super().__len__() * self.enc_in


class Dataset_Custom_CI(Dataset_Custom):
    def __init__(self, root_path, *args, **kwargs):
        super().__init__(root_path, *args, **kwargs)
        self.enc_in = self.data_x.shape[-1]
        self.tot_len = super().__len__()

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id + 1]
        seq_y = self.data_y[s_end:r_end, feat_id:feat_id + 1]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return super().__len__() * self.enc_in


class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv', target='OT',
                 scale=True, inverse=False, timeenc=0, freq='15min', cols=None, train_only=False, border=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = get_alldata(self.data_path, self.root_path)
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols = self.cols.copy()
        else:
            cols = list(df_raw.columns)
            self.cols = cols.copy()
            cols.remove('date')
        if self.features == 'S':
            cols.remove(self.target)

        num_train = int(len(df_raw) * (0.7 if not self.train_only else 1))
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len - self.meta_len, len(df_raw) - num_test - self.seq_len - self.meta_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = 0
        border2 = len(df_raw)

        if self.features == 'M' or self.features == 'MS':
            df_raw = df_raw[['date'] + cols]
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_raw = df_raw[['date'] + cols + [self.target]]
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, self.timeenc, freq=self.freq)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
        self.data_x = torch.tensor(self.data_x).float()
        self.data_y = torch.tensor(self.data_y).float()
        self.data_stamp = torch.tensor(self.data_stamp).float()

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[s_end:r_end]
        else:
            seq_y = self.data_y[s_end:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Lead(Dataset):
    def __init__(self, dataset, prefetch_path=None,
                 leader_num=4, local_max=True,
                 prefetch_batch_size=32, device='cuda', pin_gpu=False,
                 variable_batch_size=32, efficient=True, **kwargs):
        self.moving_average_kernel = 25
        self.dataset = dataset
        self.seq_len = dataset.seq_len
        self.pred_len = dataset.pred_len
        self.cache_size = len(dataset)
        self.C = dataset.data_x.shape[-1]
        self.K = min(self.C, leader_num)
        self.prefetch_batch_size = prefetch_batch_size
        self.device = torch.device(device)
        self.pin_gpu = pin_gpu
        self.variable_batch_size = variable_batch_size
        self.efficient = efficient
        self.local_max = local_max
        self.cache = {}
        self._load_prefetch_files(prefetch_path, lambda x: instance_norm(x.permute(0, 2, 1), -1))

    def _load_prefetch_files(self, prefetch_path,
                             process_x_func,
                             suffix=''):
        try:
            print('Loading prefetch files from', prefetch_path)
            assert prefetch_path and os.path.exists(prefetch_path)
            prefetch = np.load(prefetch_path)
            assert prefetch['leader_ids' + suffix].shape[0] == len(self.dataset) + self.pred_len
            assert prefetch['leader_ids' + suffix].shape[-1] >= self.K
            self.cache['leader_ids' + suffix] = torch.tensor(prefetch['leader_ids' + suffix][:self.cache_size, :, :self.K])
            if self.pin_gpu:
                self.cache['leader_ids' + suffix] = self.cache['leader_ids' + suffix].to(self.device)
            for k in ['shift', 'corr']:
                assert prefetch[k + suffix].shape[-2] >= self.K
                self.cache[k + suffix] = torch.tensor(prefetch[k + suffix][:self.cache_size, :, :self.K])
                if self.pin_gpu:
                    self.cache[k + suffix] = self.cache[k + suffix].to(self.device)
        except Exception as e:
            traceback.print_exc()
            print('Fail to load prefetch files')
            if not os.path.exists(os.path.dirname(prefetch_path)):
                os.mkdir(os.path.dirname(prefetch_path))
            self._generate_prefetch_files(process_x_func=process_x_func, suffix=suffix, K=self.K)
            print('Generate new prefetch files to', prefetch_path)
            np.savez(prefetch_path,
                     **{k + suffix: self.cache[k + suffix].cpu().numpy() for k in ['leader_ids', 'shift', 'corr']})

    def _generate_prefetch_files(self, process_x_func, K=16, suffix=''):
        K = min(self.C, K)
        device = self.device if self.pin_gpu else torch.device('cpu')
        self.cache.update(**{
            'leader_ids' + suffix: torch.empty((self.cache_size + self.pred_len, self.C, K), dtype=torch.long, device=device),
            'corr' + suffix: torch.empty((self.cache_size + self.pred_len, self.C, K), dtype=torch.float32, device=device),
            'shift' + suffix: torch.empty((self.cache_size + self.pred_len, self.C, K), dtype=torch.long, device=device),
        })
        self.dataset.data_x = self.dataset.data_x.to(self.device)

        indices = torch.arange(self.dataset.seq_len).unsqueeze(0) + torch.arange(self.prefetch_batch_size).unsqueeze(-1)

        if self.efficient:
            estimate_num = self.cache_size + self.pred_len
        else:
            estimate_num = self.seq_len - self.dataset.border[0]

        if self.efficient or estimate_num > 0:
            for i in tqdm(range(0, estimate_num, self.prefetch_batch_size)):
                _idx = (indices + i)[:min(estimate_num - i, self.prefetch_batch_size)]
                x = self.dataset.data_x[_idx].to(self.device)
                res = estimate_indicator(process_x_func(x), K, local_max=self.local_max)
                for ei, (k, v) in enumerate(zip(['leader_ids', 'shift', 'corr'], res)):
                    self.cache[k + suffix][i: i+len(_idx)] = v if self.pin_gpu else v.cpu()

        if not self.efficient:
            data_x = self.dataset.data[max(0, self.dataset.border[0]-self.seq_len): self.dataset.border[1]]
            _idx = torch.arange(self.seq_len).unsqueeze(0) + torch.arange(len(data_x) - self.seq_len + 1).unsqueeze(-1)
            x = torch.tensor(data_x[_idx]).float().to(self.device)
            x = process_x_func(x)
            for j in tqdm(range(self.C)):
                res = accurate_indicator(x, j, K, local_max=self.local_max)
                for ei, (k, v) in enumerate(zip(['leader_ids', 'shift', 'corr'], res)):
                    self.cache[k + suffix][max(0, estimate_num):, j] = v if self.pin_gpu else v.cpu()

    def __getitem__(self, index):
        res = [self.cache[k][index] for k in ['leader_ids', 'shift', 'corr']]
        return self.dataset[index] + tuple(res)

    def __len__(self):
        return len(self.dataset)


class Dataset_Lead_Pretrain(Dataset_Lead):
    def __init__(self, dataset, pred_path=None, **kwargs):
        super().__init__(dataset, **kwargs)
        pred = np.load(pred_path)
        if self.dataset.border[1] == len(self.dataset.data):
            pred = pred[-len(self.dataset):]
        else:
            begin_index = max(0, self.dataset.border[0] - self.pred_len + 1)
            pred = pred[begin_index: begin_index + len(dataset)]
        self.pred = torch.tensor(pred)
        if self.pin_gpu:
            self.pred = self.pred.to(self.device)

    def __getitem__(self, index):
        return super().__getitem__(index) + (self.pred[index], )


class Dataset_Lead_Stat(Dataset_Lead):
    def __init__(self, dataset, threshold=0, **kwargs):
        self.threshold = threshold
        super().__init__(dataset, **kwargs)

    def _load_prefetch_files(self, prefetch_path,
                             process_x_func,
                             suffix=''):
        super()._load_prefetch_files(prefetch_path, process_x_func, suffix='')
        self.evaluate_prefetch_files(prefetch_path,
                                     process_x_func=process_x_func,
                                     suffix=suffix, K=self.K)

    def evaluate_prefetch_files(self, prefetch_path, process_x_func, K=16, suffix=''):
        print('mean corr', self.cache['corr'].mean())
        delta, future_rs = [], []

        def _instance_norm(ts, dim):
            mu = ts.mean(dim, keepdims=True)
            ts = ts - mu
            std = ((ts ** 2).mean(dim, keepdims=True) + 1e-8) ** 0.5
            return ts / std, mu, std

        const_indices = torch.arange(self.seq_len, self.seq_len + self.pred_len, dtype=torch.int, device=self.device).unsqueeze(0).unsqueeze(0)
        self.prefetch_batch_size = self.prefetch_batch_size // 2
        for i in tqdm(range(0, len(self.dataset), self.prefetch_batch_size)):
            batch = [[], [], [], [], []]
            for _i in range(i, min(i+self.prefetch_batch_size, len(self.dataset))):
                _item = self.__getitem__(_i)
                for j, _x in enumerate(_item[:2] + _item[4:]):
                    batch[j].append(_x)
            x, y, leader_ids, shift, r = [torch.stack(_data, 0).to(self.device) for _data in batch]

            x, y = x.permute(0, 2, 1), y.permute(0, 2, 1) # [B, C, H]
            x, mu, std = _instance_norm(x, -1)
            y = (y - mu) / std

            seq_shifted, r_abs = shifted_leader_seq(x, y, self.K, leader_ids, shift, r,
                                                const_indices) # [B, C, K, H]
            future_r = (instance_norm(seq_shifted, -1) @ instance_norm(y, -1).unsqueeze(-1)).squeeze(-1) / self.pred_len

            delta.append((future_r - r_abs)[r_abs > self.threshold].view(-1).cpu())
            future_rs.append(future_r[r_abs > self.threshold].view(-1).cpu())
        delta = torch.cat(delta, 0)
        future_rs = torch.cat(future_rs, 0).numpy()
        delta = np.sort(delta.numpy())

        print(delta.mean())
        print('75%', np.quantile(delta, 0.25))
        print('50%', np.median(delta))
        print(future_rs.mean())
