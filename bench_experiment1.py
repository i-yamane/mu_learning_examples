from typing import Dict, Tuple, Any, Optional, Callable, NamedTuple, List, Union
import numpy as np
import pandas as pd  # type: ignore
import argparse

import os

import torch
from torch.utils.data import DataLoader, random_split, TensorDataset
import torchvision.datasets as torchdata  # type: ignore
import torchvision.transforms as transforms  # type: ignore

import torchvision.models as torchmodels  # type: ignore
from models import weights_init_normal
from PyTorch_GAN.implementations.pix2pix.models import UNet32x32
import models
from data import DataWithInfo
from data import MakeFromTorchData

import mlflow
import datetime

from mediated_uncoupled_learning.mu_learning.ubmin import NN
from mediated_uncoupled_learning.mu_learning.utils import mse_x2y_y, mse_x2y_u2y, mse_u2y_y
from mediated_uncoupled_learning.mu_learning.utils import acc_x2y_y, acc_x2y_u2y, acc_u2y_y
from mediated_uncoupled_learning.mu_learning.multi_stage.utils import Combined
from mediated_uncoupled_learning.mu_learning.utils import force2d

import warnings
warnings.simplefilter('default')


def main() -> None:
    parser: Any = argparse.ArgumentParser(
        prog='bench_experiment',
        usage='python toy_experiment.py [OPTIONS]',
        description='Experiments with benchmark data.',
        epilog='end',
        add_help=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--n0', help='Training sample size for (x, u).',
                        default=5000, type=int)
    parser.add_argument('--n1', help='Training sample size for (u, y).',
                        default=5000, type=int)
    parser.add_argument('--n_te', help='Test sample size.',
                        default=10000, type=int)
    parser.add_argument('--xdim', help='Dimensionality of x.',
                        default=5, type=int)
    parser.add_argument('--udim', help='Dimensionality of u.',
                        default=5, type=int)
    parser.add_argument('--epochs', help='Number of epochs.', type=int,
                        default=200)
    parser.add_argument('--n_channels', help='Number of channels of images (for image datasets).',
                        type=int, default=3)
    parser.add_argument('--n_filters', help='Base number of image filter blocks.',
                        type=int, default=8)
    parser.add_argument('--batch_size', help='Mini-batch size.', type=int,
                        default=512)
    parser.add_argument('--learning_rate', help='Learning rate.', type=float,
                        default=1E-4)
    parser.add_argument('--grad_clip', help='Value at which to clip gradients.', type=float,
                        default=1)
    parser.add_argument('--n_update_g', help='How many times g is updated for each update of f.',
                        type=int, default=3)
    add_bool_arg(parser, name='warm_start', default=False,
                 kw_options_on={'help': 'Turn on warm start.'},
                 kw_options_off={'help': 'Turn off warm start.'})
    parser.add_argument('--exp_type', help='Experiment type.', type=str,
                        choices=['bench_assump_violated'], default='bench_assump_violated')
    parser.add_argument('--seed', help='Random seed.',
                        default=0, type=int)
    parser.add_argument('--exp_id', help='Experiment ID.',
                        default='', type=str)
    parser.add_argument('--run_name', help='Name of run for MLflow.',
                        default='', type=str)
    parser.add_argument('--optimizer', help='Optimizer name.',
                        type=str, default='Adam')
    parser.add_argument('--gpu_id', help='GPU ID.', type=int, default='0')
    parser.add_argument('--mlflow_uri', help='MLflow tracking URI.',
                        type=str, default='.')
    parser.add_argument('--base_path', help='Base path to the data directory.',
                        type=str, default='./data')
    parser.add_argument('--dataname', help='Name of the dataset.',
                        type=str, default='CIFAR10',
                        choices=['CIFAR10',
                                 'CIFAR100',
                                 'FashionMNIST',
                                 'MNIST'])
    parser.add_argument('--transform', help='Transform applied to x or u to make u or x.',
                        type=str, default='cropping')
    parser.add_argument('--cropping_ratio', help='Cropping ratio.', type=float, default=0.8)
    parser.add_argument('--downsampling_kernel', help='Kernel size in down-sampling. Assumes a square kernel.', type=int, default=1)
    parser.add_argument('--downsampling_stride', help='Kernel size in down-sampling.'
                        'Assumes a common value for the vertical and the horizontal strides.',
                        type=int, default=1)
    parser.add_argument('--model_name_h', help='Name of the model for h in 2StepRR and JointRR.', type=str, default='resnet20',
                        choices=['resnet20', 'resnet32', 'resnet44',
                                 'resnet20_prob_square', 'resnet32_prob_square', 'resnet44_prob_square',
                                 ])
    parser.add_argument('--model_name_f', help='Name of the model for f in 2StepRR and JointRR.', type=str, default='resnet20',
                        choices=['resnet20', 'resnet32', 'resnet44',
                                 'resnet20_prob_square', 'resnet32_prob_square', 'resnet44_prob_square',
                                 ])
    parser.add_argument('--model_name_g', help='Name of the model for g in 2StepRR and JointRR.', type=str, default='resnet20',
                        choices=['resnet20', 'resnet32', 'resnet44',
                                 'resnet20_prob_square', 'resnet32_prob_square', 'resnet44_prob_square',
                                 ])
    parser.add_argument('--model_name_u2y', help='Name of the model for x2u in Naive.', type=str, default='resnet20',
                        choices=['resnet20', 'resnet32', 'resnet44',
                                 'resnet20_prob_square', 'resnet32_prob_square', 'resnet44_prob_square',
                                 ])
    parser.add_argument('--weight_decay', help='Weight decay.', type=float,
                        default=1E-5)
    args = parser.parse_args()

    if torch.cuda.is_available():  # type:ignore
        global _device
        _device = torch.device('cuda:{}'.format(args.gpu_id))
    else:
        _device = torch.device('cpu')


    torch.manual_seed(args.seed + 1)  # type: ignore

    args_dict = vars(args)
    tracking_uri = args.mlflow_uri if args.mlflow_uri else None
    mlflow.set_tracking_uri(tracking_uri)
    args_dict.pop('mlflow_uri')

    exp_id = args.exp_id if args.exp_id else None
    args_dict.pop('exp_id')
    run_name = args.run_name if args.run_name else None
    args_dict.pop('run_name')

    with mlflow.start_run(experiment_id=exp_id, run_name=run_name):
        mlflow.log_params(args_dict)
        bench_experiment1(args, gen_data=MakeFromTorchData(args))


def add_bool_arg(parser, name, default=False, kw_options_on={}, kw_options_off={}):
    # See https://stackoverflow.com/a/31347222
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--' + name, dest=name, action='store_true', **kw_options_on)
    group.add_argument('--no-' + name, dest=name, action='store_false', **kw_options_off)
    parser.set_defaults(**{name:default})


def bench_experiment1(args: Any, gen_data: Callable[[Any, bool], DataWithInfo]) -> Dict[str, Any]:
    data_tr = gen_data(args, train=True)  # type: ignore
    data_te = gen_data(args, train=False)  # type: ignore

    num_classes = {
        'CIFAR10': 10,
        'CIFAR100': 100,
        'FashionMNIST': 10,
        'MNIST': 10,
    }

    res = {}

    u_size : Tuple[int, int]

    u_size = data_tr.shape_u[1:]  # type:ignore
    model_x2u_Naive = UNet32x32(
        in_channels=args.n_channels, out_channels=args.n_channels,
        in_size=(32, 32), out_size=u_size,
        base_num_filters=args.n_filters
    ).to(_device)

    weights_init_normal(model_x2u_Naive)
    model_u2y = models.__dict__[args.model_name_u2y](
        num_classes=num_classes[args.dataname],
        num_channels=args.n_channels
    ).to(_device)

    model_JointRR = NN(
        model_f=models.__dict__[args.model_name_f](
                num_classes=num_classes[args.dataname],
                num_channels=args.n_channels
            ).to(_device),
        model_h=models.__dict__[args.model_name_h](
                num_classes=num_classes[args.dataname],
                num_channels=args.n_channels
            ).to(_device),
        model_g=models.__dict__[args.model_name_g](
                num_classes=num_classes[args.dataname],
                num_channels=args.n_channels
            ).to(_device),
        weight_decay_f=1E-5,
        weight_decay_h=1E-5,
        n_epochs=args.epochs//2 if args.warm_start else args.epochs,
        w_init=.5,
        optimizer=args.optimizer,
        batch_size=args.batch_size,
        lr_f=args.learning_rate,
        lr_h=args.learning_rate,
        grad_clip=args.grad_clip,
        warm_start=args.warm_start,
        correct_bias=False,
        optimize_w=False,
        analytic_w=False,
        batch_norm=False,
        record_loss=True,
        log_metric_label='JointRR',
        device=_device
    )
    model_JointRR.fit_indirect(data_tr.loader_xu_and_size[0], data_tr.loader_uy_and_size[0])
    mse_JointRR = mse_x2y_y(
        predict_y_from_x=model_JointRR.predict_y_from_x,
        loader_xy=data_te.loader_xy_and_size[0],
        device=_device)
    mlflow.log_metric('MSE_JointRR', mse_JointRR)
    mlflow.log_metric('w_JointRR', model_JointRR.w_.item())
    res['MSE_JointRR'] = mse_JointRR
    print('MSE of JointRR: {}'.format(mse_JointRR))
    acc_JointRR = acc_x2y_y(
        predict_y_from_x=model_JointRR.predict_y_from_x,
        loader_xy=data_te.loader_xy_and_size[0],
        device=_device)
    mlflow.log_metric('ACC_JointRR', acc_JointRR)
    mlflow.log_metric('w_JointRR', model_JointRR.w_.item())
    res['ACC_JointRR'] = acc_JointRR
    print('ACC of JointRR: {}'.format(acc_JointRR))

    model_2StepRR = NN(
        model_f=models.__dict__[args.model_name_f](
                num_classes=num_classes[args.dataname],
                num_channels=args.n_channels
            ).to(_device),
        model_h=models.__dict__[args.model_name_h](
                num_classes=num_classes[args.dataname],
                num_channels=args.n_channels
            ).to(_device),
        model_g=models.__dict__[args.model_name_g](
                num_classes=num_classes[args.dataname],
                num_channels=args.n_channels
            ).to(_device),
        weight_decay_f=args.weight_decay,
        weight_decay_h=args.weight_decay,
        n_epochs=args.epochs,
        optimizer=args.optimizer,
        batch_size=args.batch_size,
        lr_f=args.learning_rate,
        lr_h=args.learning_rate,
        grad_clip=args.grad_clip,
        two_step=True,
        batch_norm=False,
        record_loss=True,
        log_metric_label='2StepRR',
        device=_device
    )
    model_2StepRR.fit_indirect(data_tr.loader_xu_and_size[0], data_tr.loader_uy_and_size[0])
    mse_2StepRR = mse_x2y_y(
        predict_y_from_x=model_2StepRR.predict_y_from_x,
        loader_xy=data_te.loader_xy_and_size[0],
        device=_device)
    mlflow.log_metric('MSE_2StepRR', mse_2StepRR)
    res['MSE_2StepRR'] = mse_2StepRR
    print('MSE of 2StepRR: {}'.format(mse_2StepRR))
    acc_2StepRR = acc_x2y_y(
        predict_y_from_x=model_2StepRR.predict_y_from_x,
        loader_xy=data_te.loader_xy_and_size[0],
        device=_device)
    mlflow.log_metric('ACC_2StepRR', acc_2StepRR)
    res['ACC_2StepRR'] = acc_2StepRR
    print('ACC of 2StepRR: {}'.format(acc_2StepRR))

    model_Naive = Combined(
        model_x2u=model_x2u_Naive,
        model_u2y=model_u2y,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        n_epochs=args.epochs,
        optimizer=args.optimizer,
        lr_x2u=args.learning_rate,
        lr_u2y=args.learning_rate,
        grad_clip=args.grad_clip,
        record_loss=True,
        log_metric_label='Naive',
        device=_device,
    )
    model_Naive.fit_indirect(data_tr.loader_xu_and_size[0], data_tr.loader_uy_and_size[0])
    mse_Naive = mse_x2y_y(
        predict_y_from_x=model_Naive.predict_y_from_x,
        loader_xy=data_te.loader_xy_and_size[0],
        device=_device)
    mlflow.log_metric('MSE_Naive', mse_Naive)
    res['MSE_Naive'] = mse_Naive
    print('MSE of Naive: {}'.format(mse_Naive))
    acc_Naive = acc_x2y_y(
        predict_y_from_x=model_Naive.predict_y_from_x,
        loader_xy=data_te.loader_xy_and_size[0],
        device=_device)
    mlflow.log_metric('ACC_Naive', acc_Naive)
    res['ACC_Naive'] = acc_Naive
    print('ACC of Naive: {}'.format(acc_Naive))

    return res


if __name__ == '__main__':
    main()
