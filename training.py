""" module providing basic training utilities"""
import os
from os.path import join
from time import time
from datetime import timedelta
from itertools import starmap

from cytoolz import curry, reduce

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import ReduceLROnPlateau
import tensorboardX


def get_basic_grad_fn(net, clip_grad, max_grad=1e2):
    def f():
        grad_norm = clip_grad_norm_(
            [p for p in net.parameters() if p.requires_grad], clip_grad)
        #print(type(grad_norm))
        #grad_norm = grad_norm
        if max_grad is not None and grad_norm >= max_grad:
            print('WARNING: Exploding Gradients {:.2f}'.format(grad_norm))
            grad_norm = max_grad
        grad_log = {}
        grad_log['grad_norm'] = grad_norm
        return grad_log
    return f

@curry
def compute_loss(net, criterion, fw_args, loss_args):

    loss0 = criterion(*((net(*fw_args)[0][0],) + loss_args))
    loss1 = criterion(*((net(*fw_args)[1][0],) + loss_args))
    return loss0+loss1

@curry
def val_step(loss_step, fw_args, loss_args):
    loss = loss_step(fw_args, loss_args)
    return loss.size(0), loss.sum().item()

@curry
def basic_validate(net, criterion, val_batches):
    print('running validation ... ', end='')
    net.eval()
    start = time()
    with torch.no_grad():
        validate_fn = val_step(compute_loss(net, criterion))
        n_data, tot_loss = reduce(
            lambda a, b: (a[0]+b[0], a[1]+b[1]),
            starmap(validate_fn, val_batches),
            (0, 0)
        )
    val_loss = tot_loss / n_data
    print(
        'validation finished in {}                                    '.format(
            timedelta(seconds=int(time()-start)))
    )
    print('validation loss: {:.4f} ... '.format(val_loss))
    return {'loss': val_loss}


class BasicPipeline(object):
    #this is use for training abstractor and extrctor
    def __init__(self, name, net,
                 train_batcher, val_batcher, batch_size,
                 val_fn, criterion, optim, grad_fn=None):
        self.name = name
        self._net = net
        self._train_batcher = train_batcher
        self._val_batcher = val_batcher
        self._criterion = criterion
        self._opt = optim
        # grad_fn is calleble without input args that modifyies gradient
        # it should return a dictionary of logging values
        self._grad_fn = grad_fn
        self._val_fn = val_fn
        self._n_epoch = 0  # epoch not very useful?
        self._batch_size = batch_size
        self._batches = self.batches()

        hops = net.hops
        self.I = torch.zeros(batch_size, hops, hops).cuda()
        for i in range(batch_size):
            for j in range(hops):
                self.I.data[i][j][j] = 1

    def Frobenius(self, mat):
        size = mat.size()
        if len(size) == 3:  # batched matrix
            ret = (torch.sum(torch.sum((mat ** 2), 1, keepdim=True), 2,keepdim=True).squeeze() + 1e-10) ** 0.5
            return torch.sum(ret) / size[0]
        else:
            raise Exception('matrix for computing Frobenius norm should be with 3 dims')

    def batches(self):
        while True:
            for fw_args, bw_args in self._train_batcher(self._batch_size):
                yield fw_args, bw_args
            self._n_epoch += 1

    def get_loss_args(self, net_out, bw_args):
        if isinstance(net_out, tuple):
            loss_args = net_out + bw_args
        else:
            loss_args = (net_out, ) + bw_args
        return loss_args

    def train_step(self):
        # forward pass of model
        self._net.train()
        fw_args, bw_args = next(self._batches)
        net_out, net_out_abs = self._net(*fw_args)

        # get logs and output for logging, backward
        log_dict = {}
        loss_args = self.get_loss_args(net_out[0], bw_args)
        loss_args_abs = self.get_loss_args(net_out_abs[0], bw_args)
        # backward and update ( and optional gradient monitoring )
        loss_art = self._criterion(*loss_args).mean()
        loss_abs = self._criterion(*loss_args_abs).mean()
        # add penalization term
        diff = torch.div(net_out[2]+1e-3, net_out_abs[2]+1e-3)-1
        feature_loss = self.Frobenius(diff)

        #covverage loss
        mask = loss_args[1] != 0
        loss_cov_abs = torch.masked_select(torch.sum(torch.min(net_out_abs[3], net_out_abs[4]), -1), mask).mean()
        loss_cov_art = torch.masked_select(torch.sum(torch.min(net_out[3], net_out[4]), -1), mask).mean()


        #artT = torch.transpose(net_out[1], 1, 2).contiguous()
        #extra_art_loss = self.Frobenius(torch.bmm(net_out[1], artT) - self.I[:net_out[1].size(0)])
        #absT = torch.transpose(net_out_abs[1], 1, 2).contiguous()
        #extra_abs_loss = self.Frobenius(torch.bmm(net_out_abs[1], absT) - self.I[:net_out_abs[1].size(0)])

        loss = loss_art + loss_abs + loss_cov_abs + loss_cov_art
        #0.3 * extra_abs_loss + 0.3 * extra_art_loss +
        loss.backward()
        log_dict['loss'] = loss.item()
        log_dict['loss_art'] = loss_art.item()
        log_dict["loss_abs"] = loss_abs.item()
        #log_dict["extra_abs_loss"] = extra_abs_loss.item()
        #log_dict["extra_art_loss"] = extra_art_loss.item()
        log_dict["loss_cov_abs"] = loss_cov_abs.item()
        log_dict["loss_cov_art"] = loss_cov_art.item()
        log_dict["feature_loss"] = feature_loss.item()
        if self._grad_fn is not None:
            log_dict.update(self._grad_fn())
        self._opt.step()
        self._net.zero_grad()

        return log_dict

    def validate(self):
        return self._val_fn(self._val_batcher(self._batch_size))

    def checkpoint(self, save_path, step, val_metric=None):
        save_dict = {}
        if val_metric is not None:
            name = 'ckpt-{:6f}-{}'.format(val_metric, step)
            save_dict['val_metric'] = val_metric
        else:
            name = 'ckpt-{}'.format(step)

        save_dict['state_dict'] = self._net.state_dict()
        save_dict['optimizer'] = self._opt.state_dict()
        torch.save(save_dict, join(save_path, name))

    def terminate(self):
        self._train_batcher.terminate()
        self._val_batcher.terminate()


class BasicTrainer(object):
    #this is used for training abstracot and extractor and full model
    """ Basic trainer with minimal function and early stopping"""
    def __init__(self, pipeline, save_dir, ckpt_freq, patience,
                 scheduler=None, val_mode='loss'):
        assert isinstance(pipeline, BasicPipeline)
        assert val_mode in ['loss', 'score']
        self._pipeline = pipeline
        self._save_dir = save_dir
        self._logger = tensorboardX.SummaryWriter(join(save_dir, 'log'))
        if not os.path.exists(join(save_dir, 'ckpt')):
            os.makedirs(join(save_dir, 'ckpt'))

        self._ckpt_freq = ckpt_freq
        self._patience = patience
        self._sched = scheduler
        self._val_mode = val_mode

        self._step = 0
        self._running_loss = None
        # state vars for early stopping
        self._current_p = 0
        self._best_val = None

    def log(self, log_dict):
        loss = log_dict['loss'] if 'loss' in log_dict else log_dict['reward']
        if self._running_loss is not None:
            self._running_loss = 0.99*self._running_loss + 0.01*loss
        else:
            self._running_loss = loss
        print('train step: {}, {}: {:.4f}\r'.format(
            self._step,
            'loss' if 'loss' in log_dict else 'reward',
            self._running_loss), end='')

        # if "loss_art" in log_dict:
        #     for key, value in log_dict:
        #         print("{} : {:.4f}\r".format(key, value), end='')
        for key, value in log_dict.items():
            self._logger.add_scalar(
                '{}_{}'.format(key, self._pipeline.name), value, self._step)

    def validate(self):
        print()
        val_log = self._pipeline.validate()
        for key, value in val_log.items():
            self._logger.add_scalar(
                'val_{}_{}'.format(key, self._pipeline.name),
                value, self._step
            )
        if 'reward' in val_log:
            val_metric = val_log['reward']
        else:
            val_metric = (val_log['loss'] if self._val_mode == 'loss'
                          else val_log['score'])
        return val_metric

    def checkpoint(self):
        val_metric = self.validate()
        self._pipeline.checkpoint(
            join(self._save_dir, 'ckpt'), self._step, val_metric)
        if isinstance(self._sched, ReduceLROnPlateau):
            self._sched.step(val_metric)
        else:
            self._sched.step()
        stop = self.check_stop(val_metric)
        return stop

    def check_stop(self, val_metric):
        if self._best_val is None:
            self._best_val = val_metric
        elif ((val_metric < self._best_val and self._val_mode == 'loss')
              or (val_metric > self._best_val and self._val_mode == 'score')):
            self._current_p = 0
            self._best_val = val_metric
        else:
            self._current_p += 1
        return self._current_p >= self._patience

    def train(self):
        try:
            start = time()
            print('Start training')
            while True:
                log_dict = self._pipeline.train_step()
                self._step += 1
                self.log(log_dict)

                if self._step % self._ckpt_freq == 0:
                    stop = self.checkpoint()
                    if stop:
                        break
            print('Training finised in ', timedelta(seconds=time()-start))
        finally:
            self._pipeline.terminate()
