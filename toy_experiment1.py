from typing import Dict, Tuple, Sequence, Any, Optional, Callable
from typing_extensions import Protocol

import matplotlib.pyplot as plt
import matplotlib
import argparse

import torch
from torch.utils.data import TensorDataset, DataLoader

import mlflow
import datetime

from mediated_uncoupled_learning.mu_learning.ubmin import NN
from mediated_uncoupled_learning.mu_learning.utils import mse_x2y_y, mse_x2y_u2y, mse_u2y_y
from mediated_uncoupled_learning.mu_learning.multi_stage.utils import Combined
from mediated_uncoupled_learning.mu_learning.utils import force2d
from mediated_uncoupled_learning.mu_learning.utils._adapter import Predict, PredictYFromX, PredictYFromU
from mediated_uncoupled_learning.mu_learning.utils.models import MLP3


if torch.cuda.is_available():
    _device = torch.device('cuda:0')
else:
    _device = torch.device('cpu')


def main() -> None:
    parser: Any = argparse.ArgumentParser(
        prog='toy_experiment',
        usage='python toy_experiment.py [OPTIONS]',
        description='Experiments with toy data.',
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
    parser.add_argument('--noise_level_u', help='Level of noise in u.',
                        default=.5, type=float)
    parser.add_argument('--epochs', help='Number of epochs.', type=int,
                        default=200)
    parser.add_argument('--batch_size', help='Mini-batch size.', type=int,
                        default=512)
    parser.add_argument('--learning_rate', help='Learning rate.', type=float,
                        default=1E-3)
    parser.add_argument('--exp_type', help='Experiment type.', type=str,
                        choices=[
                                 'toy_assump_satisfied_ctl_noise',
                                 'toy_assump_violated_ctl_noise'
                        ],
                        default='True')
    parser.add_argument('--seed', help='Random seed.',
                        default=0, type=int)
    parser.add_argument('--exp_id', help='Experiment ID.',
                        default='', type=str)
    parser.add_argument('--run_name', help='Name of run for MLflow.',
                        default='', type=str)
    parser.add_argument('--optimizer', help='Optimizer name.',
                        type=str, default='Adam')
    parser.add_argument('--mlflow_uri', help='MLflow tracking URI.',
                        type=str, default='file:.')
    args = parser.parse_args()

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

        if args.exp_type == 'toy_assump_satisfied_ctl_noise':
            experiment1(args, gen_data=gen_data_assump_satisfied_ctl_noise)
        elif args.exp_type == 'toy_assump_violated_ctl_noise':
            experiment1(args, gen_data=gen_data_assump_violated_ctl_noise)
        else:
            assert False


def gen_data_assump_satisfied_ctl_noise(
    xdim: int,
    noise_level_u: float = 1,
    n: int = 1000,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    def gen_x(size: Tuple[int, int]) -> torch.Tensor:
        return force2d(2 * torch.rand(size) - 1).float()

    def gen_u(x_: torch.Tensor) -> torch.Tensor:
        noise = 2 * noise_level_u * (torch.rand_like(x_) - 1)
        return force2d(x_**3 + noise).float()

    def gen_y(u_: torch.Tensor) -> torch.Tensor:
        u_norm = torch.norm(u_, dim=1)
        return force2d(u_norm**2 + .1 * torch.randn_like(u_norm)).float()

    x = gen_x(size=(n, xdim))
    u = gen_u(x_=x)
    y = gen_y(u_=u)

    return x, u, y


def gen_data_assump_violated_ctl_noise(
    xdim: int,
    noise_level_u: float = 1,
    n: int = 1000,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    def gen_x(size: Tuple[int, int]) -> torch.Tensor:
        return force2d(2 * torch.rand(size) - 1).float()

    def gen_u(x_: torch.Tensor) -> torch.Tensor:
        noise = 2 * noise_level_u * (torch.rand_like(x_) - 1)
        return force2d(x_**3 + noise).float()

    def gen_y(x_: torch.Tensor) -> torch.Tensor:
        x_norm = torch.norm(x_, dim=1)
        return force2d(x_norm**2 + .1 * torch.randn_like(x_norm)).float()

    x = gen_x(size=(n, xdim))
    u = gen_u(x_=x)
    y = gen_y(x_=x)

    return x, u, y


def experiment1(
            args: Any,
            gen_data: Callable[
                ...,
                Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            ]
        ) -> Any:
    xall: torch.Tensor
    uall: torch.Tensor
    yall: torch.Tensor
    if args.exp_type in [
        'toy_assump_satisfied_ctl_noise',
        'toy_assump_violated_ctl_noise',
    ]:
        xall, uall, yall = gen_data(
            n=args.n0+args.n1+args.n_te,
            xdim=args.xdim,
            noise_level_u=args.noise_level_u
        )
    else:
        assert False

    x0, x1, x_te = torch.split(xall, [args.n0, args.n1, args.n_te])  # type: ignore
    u0, u1, u_te = torch.split(uall, [args.n0, args.n1, args.n_te])  # type: ignore
    y0, y1, y_te = torch.split(yall, [args.n0, args.n1, args.n_te])  # type: ignore


    loader_xu = DataLoader(
        TensorDataset(x0.float(), u0.float()),
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True
    )
    loader_uy = DataLoader(
        TensorDataset(u1.float(), y1.float()),
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True
    )
    loader_xy_te = DataLoader(
        TensorDataset(x_te.float(), y_te.float()),
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True
    )

    res = {}

    model_cmbnn = Combined(
        model_x2u=MLP3(
            dim_in=x0.shape[1],
            dim_hid=20,
            dim_out=u0.shape[1]),
        model_u2y=MLP3(
            dim_in=u0.shape[1],
            dim_hid=20,
            dim_out=y0.shape[1]),
        batch_size=args.batch_size,
        n_epochs=args.epochs,
        optimizer=args.optimizer,
        weight_decay=1E-5,
        lr_u2y=args.learning_rate,
        lr_x2u=args.learning_rate,
        record_loss=True,
        log_metric_label='Naive',
        device=_device
    )
    model_cmbnn.fit_indirect(loader_xu, loader_uy)
    mse_cmbnn = mse_x2y_y(
        predict_y_from_x=model_cmbnn.predict_y_from_x,
        loader_xy=loader_xy_te,
        device=_device)
    mlflow.log_metric('MSE_Naive', mse_cmbnn)
    res['MSE_Naive'] = mse_cmbnn
    print('MSE of Naive: {}'.format(mse_cmbnn))

    model_2StepRR = NN(
        model_f=MLP3(
            dim_in=x0.shape[1],
            dim_hid=20,
            dim_out=y0.shape[1]),
        model_h=MLP3(
            dim_in=u0.shape[1],
            dim_hid=20,
            dim_out=y0.shape[1]),
        batch_norm=False,
        two_step=True,
        batch_size=args.batch_size,
        n_epochs=args.epochs,
        optimizer=args.optimizer,
        weight_decay_f=1E-5,
        weight_decay_h=1E-5,
        lr_f=args.learning_rate,
        lr_h=args.learning_rate,
        record_loss=True,
        log_metric_label='2StepRR',
    )
    model_2StepRR.fit_indirect(loader_xu, loader_uy)
    mse_2StepRR = mse_x2y_y(
        predict_y_from_x=model_2StepRR.predict_y_from_x,
        loader_xy=loader_xy_te,
        device=_device)
    mlflow.log_metric('MSE_2StepRR', mse_2StepRR)
    res['MSE_2StepRR'] = mse_2StepRR
    print('MSE of 2StepRR: {}'.format(mse_2StepRR))

    model_JointRR = NN(
        model_f=MLP3(
            dim_in=x0.shape[1],
            dim_hid=20,
            dim_out=y0.shape[1]),
        model_h=MLP3(
            dim_in=u0.shape[1],
            dim_hid=20,
            dim_out=y0.shape[1]),
        batch_norm=False,
        two_step=True,
        batch_size=args.batch_size,
        n_epochs=args.epochs,
        optimizer=args.optimizer,
        weight_decay_f=1E-5,
        weight_decay_h=1E-5,
        lr_f=args.learning_rate,
        lr_h=args.learning_rate,
        w_init=.5,
        analytic_w=False,
        correct_bias=False,
        warm_start=False,
        record_loss=True,
        log_metric_label='UB_JointRR'
    )
    model_JointRR_nn.fit_indirect(loader_xu, loader_uy)
    mse_JointRR = mse_x2y_y(
        predict_y_from_x=model_JointRR_nn.predict_y_from_x,
        loader_xy=loader_xy_te,
        device=_device)
    mlflow.log_metric('MSE_UB_JointRR', mse_JointRR)
    mlflow.log_metric('w_UB_JointRR', model_JointRR.w())
    res['MSE_UB_JointRR'] = mse_JointRR
    print('MSE of UB_JointRR: {}'.format(mse_JointRR))

    return res


if __name__ == '__main__':
    main()

