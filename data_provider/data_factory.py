
from data_provider.data_loader import *
from torch.utils.data import DataLoader, DistributedSampler

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'ETTh1_CI': Dataset_ETT_hour_CI,
    'ETTh2_CI': Dataset_ETT_hour_CI,
    'ETTm1_CI': Dataset_ETT_minute_CI,
    'ETTm2_CI': Dataset_ETT_minute_CI,
    'custom_CI': Dataset_Custom_CI,
    'pred': Dataset_Pred,
}


def get_dataset(args, flag, device='cpu', wrap_class=None, **kwargs):
    if not hasattr(args, 'timeenc'):
        args.timeenc = 0 if not hasattr(args, 'embed') or args.embed != 'timeF' else 1
    data_set = data_dict[args.data](
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=args.timeenc,
        freq=args.freq,
        train_only=args.train_only,
    )
    if args.pin_gpu and hasattr(data_set, 'data_x'):
        data_set.data_x = data_set.data_x.to(device)
        data_set.data_y = data_set.data_y.to(device)
        from settings import need_x_mark, need_x_y_mark
        if args.model in need_x_mark or args.model in need_x_y_mark:
            data_set.data_stamp = data_set.data_stamp.to(device)
    print(flag, len(data_set))

    if wrap_class is not None:
        if not isinstance(wrap_class, list):
            wrap_class = [wrap_class]
        for cls in wrap_class:
            data_set = cls(data_set, **kwargs)
    return data_set


def get_dataloader(data_set, args, flag, sampler=None):
    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag and args.local_rank == -1,
        num_workers=args.num_workers,
        drop_last=drop_last,
        pin_memory=False,
        sampler=sampler if args.local_rank == -1 or flag == 'test' else DistributedSampler(data_set))
    return data_loader


def data_provider(args, flag, device='cpu', wrap_class=None, sampler=None, **kwargs):
    data_set = get_dataset(args, flag, device, wrap_class=wrap_class, **kwargs)
    data_loader = get_dataloader(data_set, args, flag, sampler)
    return data_set, data_loader
