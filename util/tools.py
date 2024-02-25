import collections

import numpy as np
import torch
import time
from copy import deepcopy


def load_model_compile(model, model_pth, device, strict=False):
    model = model.to(device)
    origin_dict = torch.load(model_pth, map_location=device)
    state_dict = collections.OrderedDict()
    torch2_model_prefix = '_orig_mod.'
    offset2 = len(torch2_model_prefix)
    if list(origin_dict.keys())[0].startswith(torch2_model_prefix) and \
            not list(model.state_dict().keys())[0].startswith(torch2_model_prefix):
        for key, value in origin_dict.items():
            state_dict[key[offset2: len(key)]] = value
        model.load_state_dict(state_dict, strict=strict)
    else:
        model.load_state_dict(origin_dict, strict=strict)
    return model


def remove_state_key_prefix(origin_dict, model, prefix='_orig_mod.'):
    if isinstance(prefix, list):
        for p in prefix:
            origin_dict = remove_state_key_prefix(origin_dict, model, prefix=p)
        return origin_dict

    if list(origin_dict.keys())[0].startswith(prefix) and \
            not list(model.state_dict().keys())[0].startswith(prefix):
        state_dict = collections.OrderedDict()
        offset2 = len(prefix)
        for key, value in origin_dict.items():
            state_dict[key[offset2: len(key)]] = value
        return state_dict
    else:
        return origin_dict


def instance_norm(ts, dim):
    mu = ts.mean(dim, keepdims=True)
    ts = ts - mu
    var = ((ts ** 2).mean(dim, keepdims=True) + 1e-8) ** 0.5
    return ts / var


def adjust_learning_rate(optimizer, scheduler, epoch, args, printout=True):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == 'type4':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch) // 1))}
    elif args.lradj == 'type5':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.95 ** ((epoch - 3) // 1))}
    elif args.lradj == 'every5':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 5))}
    elif args.lradj == 'warmup':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((min(epoch, args.warmup_epochs) - 1) // 1))}
    elif args.lradj == 'constant':
        lr_adjust = {epoch: args.learning_rate}
    elif args.lradj == '3':
        lr_adjust = {epoch: args.learning_rate if epoch < 10 else args.learning_rate*0.1}
    elif args.lradj == '4':
        lr_adjust = {epoch: args.learning_rate if epoch < 15 else args.learning_rate*0.1}
    elif args.lradj == '5':
        lr_adjust = {epoch: args.learning_rate if epoch < 25 else args.learning_rate*0.1}
    elif args.lradj == '6':
        lr_adjust = {epoch: args.learning_rate if epoch < 5 else args.learning_rate*0.1}  
    elif args.lradj == 'TST':
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}
    elif args.lradj == 'Crossformer':
        lr_adjust = {2: args.learning_rate * 0.5 ** 1, 4: args.learning_rate * 0.5 ** 2,
                     6: args.learning_rate * 0.5 ** 3, 8: args.learning_rate * 0.5 ** 4,
                     10: args.learning_rate * 0.5 ** 5}
    else:
        lr_adjust = {}
    
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout: print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.best_checkpoint = None

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.best_checkpoint = deepcopy(model.state_dict())
            # self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_checkpoint = deepcopy(model.state_dict())
            # self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def test_params_flop(model, x_shape, x=None):
    """
    If you want to test former's flop, you need to give default value to inputs in model.forward(), the following code can only pass one argument to forward()
    """
    model_params = sum([param.nelement() if param.requires_grad else 0 for param in model.parameters()])
    print('INFO: Trainable parameter count: {}'.format(model_params))
    from ptflops import get_model_complexity_info    
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(model.cuda(), x_shape[1:], as_strings=True, print_per_layer_stat=False)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    import torchinfo
    torchinfo.summary(model, x_shape, depth=1)

    # from deepspeed.profiling.flops_profiler import get_model_profile
    # get_model_profile(model=model, input_shape=x_shape,
    #                   detailed=False, module_depth=0, top_modules=0)



