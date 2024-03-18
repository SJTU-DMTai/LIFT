import os
import warnings

import torch
import numpy as np
import typing
from collections import OrderedDict
from data_provider.data_factory import data_provider
from torch import optim
import torch.nn as nn

from util.tools import remove_state_key_prefix


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.label_position = 1
        self.device = self._acquire_device()
        self.wrap_data_kwargs = {}
        self.model_optim = None
        model = self._build_model()
        if model is not None:
            self.model = model.to(self.device)
            self.model_optim = self._select_optimizer()

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _build_model(self, model=None, framework_class=None):
        raise NotImplementedError

    def _get_data(self, flag, **kwargs):
        data_set, data_loader = data_provider(args=self.args, flag=flag, device=self.device,
                                              wrap_class=self.args.wrap_data_class, **self.wrap_data_kwargs, **kwargs)
        return data_set, data_loader

    def _select_optimizer(self, filter_frozen=True, return_self=True, model=None):
        if return_self and self.model_optim is not None:
            return self.model_optim
        else:
            # Need to instantiate a new one
            params = self.model.parameters() if model is None else model.parameters()
            if filter_frozen:
                params = filter(lambda p: p.requires_grad, params)
            if not hasattr(self.args, 'optim'):
                self.args.optim = 'Adam'
            model_optim = getattr(optim, self.args.optim)(params, lr=self.args.learning_rate)
            if return_self:
                self.model_optim = model_optim
            return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def _process_batch(self, batch):
        return batch

    def forward(self, batch):
        if not self.args.pin_gpu:
            batch = [batch[i].to(self.device) if isinstance(batch[i], torch.Tensor) and i != self.label_position
                     else batch[i] for i in range(len(batch))]
        inp = self._process_batch(batch)
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                outputs = self.model(*inp)
        else:
            outputs = self.model(*inp)
        return outputs

    def train_loss(self, criterion, batch, outputs):
        batch_y = batch[1]
        if not self.args.pin_gpu:
            batch_y = batch_y.to(self.device)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        loss = criterion(outputs, batch_y)
        return loss

    def _update(self, batch, criterion, optimizer, scaler=None):
        if not isinstance(optimizer, tuple):
            optimizer = (optimizer, )
        for optim in optimizer:
            optim.zero_grad()
        outputs = self.forward(batch)
        loss = self.train_loss(criterion, batch, outputs)
        if self.args.use_amp:
            scaler.scale(loss).backward()
            for optim in optimizer:
                scaler.step(optim)
            scaler.update()
        else:
            loss.backward()
            for optim in optimizer:
                optim.step()
        return loss, outputs

    def state_dict(self, destination: typing.OrderedDict[str, torch.Tensor]=None,
                   prefix='', keep_vars=False, local_rank=-1) -> typing.OrderedDict[str, torch.Tensor]:
        r"""Returns a dictionary containing a whole state of the module and the state of the optimizer.

        Returns:
            dict:
                a dictionary containing a whole state of the module and the state of the optimizer.
        """
        if hasattr(self.args, 'save_opt') and self.args.save_opt:
            if destination is None:
                destination = OrderedDict()
                destination._metadata = OrderedDict()
            destination['model'] = self.model.state_dict() if local_rank == -1 else self.model.module.state_dict()
            if hasattr(self.args, 'freeze') and self.args.freeze:
                for k, v in self.model.named_parameters():
                    if not v.requires_grad:
                        destination['model'].pop(k)
            destination['model_optim'] = self.model_optim.state_dict()
            return destination
        else:
            return self.model.state_dict() if local_rank == -1 else self.model.module.state_dict()

    def load_state_dict(self, state_dict: typing.OrderedDict[str, torch.Tensor], model=None, strict=False) -> nn.Module:
        r"""Copies parameters and buffers from :attr:`state_dict` into
        this module and the optimizer.

        Args:
            dict:
                a dict containing parameters and persistent buffers.
        """
        if model is None:
            model = self.model
        if 'model_optim' not in state_dict:
            model.load_state_dict(remove_state_key_prefix(state_dict, model), strict=strict)
        else:
            for k, v in state_dict.items():
                if k == 'model':
                    model.load_state_dict(remove_state_key_prefix(v, model), strict=strict)
                elif hasattr(self, k) and getattr(self, k) is not None:
                    if isinstance(getattr(self, k), optim.Optimizer):
                        assert len(getattr(self, k).param_groups) == len(v['param_groups'])
                        try:
                            getattr(self, k).load_state_dict(v)
                        except ValueError:
                            warnings.warn(f'{k} has different state dict from the checkpoint. '
                                          f'Trying to save all states of frozen parameters...')
                            assert k == 'model_optim'
                            self.model_optim = self._select_optimizer(filter_frozen=False, return_self=False)
                            self.model_optim.load_state_dict(v)
                    else:
                        getattr(self, k).load_state_dict(v, strict=strict)

        return model

    def load_checkpoint(self, load_path=None, model=None, strict=False):
        return self.load_state_dict(torch.load(load_path, map_location=self.device), model, strict=strict)

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
