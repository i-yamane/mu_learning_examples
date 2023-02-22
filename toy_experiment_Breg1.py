from typing import Dict, Tuple, Any, Optional, Callable, NamedTuple, List, Union
import argparse

import torch
from torch.utils.data import DataLoader, random_split, TensorDataset

import models
from data import DataWithInfo
from data import MakeFromTorchData

import mlflow
import datetime

from mediated_uncoupled_learning.mu_learning.ubmin import NN
from mediated_uncoupled_learning.mu_learning.utils import mse_x2y_y, mse_x2y_u2y, mse_u2y_y
from mediated_uncoupled_learning.mu_learning.utils._helpers import l2_CSUB_fullbatch, l2_SumUB_fullbatch
from mediated_uncoupled_learning.mu_learning.multi_stage.utils import Combined
from mediated_uncoupled_learning.mu_learning.utils import force2d
from mediated_uncoupled_learning.mu_learning.utils.models import MLP3

import warnings
warnings.simplefilter('default')


def main() -> None:
    parser: Any = argparse.ArgumentParser(
        prog='toy_experiment_BregMU',
        usage='python toy_experiment_BregMU.py [OPTIONS]',
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
    parser.add_argument('--noise_level_u', help='Level of noise in u.',
                        default=.5, type=float)
    parser.add_argument('--noise_level_y', help='Level of noise in y.',
                        default=.5, type=float)
    parser.add_argument('--xdim', help='Dimensionality of x.',
                        default=5, type=int)
    parser.add_argument('--epochs', help='Number of epochs.', type=int,
                        default=200)
    parser.add_argument('--d_hidden', help='Number of the hidden units of the multi-layer perceptron.',
                        type=int, default=10)
    parser.add_argument('--batch_size', help='Mini-batch size.', type=int,
                        default=512)
    parser.add_argument('--learning_rate', help='Learning rate.', type=float,
                        default=1E-4)
    parser.add_argument('--grad_clip', help='Value at which to clip gradients.', type=float,
                        default=1)
    add_bool_arg(parser, name='warm_start', default=False,
                 kw_options_on={'help': 'Turn on warm start.'},
                 kw_options_off={'help': 'Turn off warm start.'})
    parser.add_argument('--loss_type', help='Loss type.', type=str,
                        choices=[
                            'l2',
                            'cross_entropy',
                            'cross_entropy_CSUB',
                            'cross_entropy_heuristic',
                            'l2_CSUB',
                            'l2_SumUB',
                        ],
                        default='square')
    parser.add_argument('--exp_type', help='Experiment type.', type=str,
                        choices=[
                                 'toy_regression1',
                                 'toy_regression_quartic',
                                 'toy_regression_linear',
                        ],
                        default='toy_regression1')
    parser.add_argument('--seed', help='Random seed.',
                        default=0, type=int)
    parser.add_argument('--exp_id', help='Experiment ID.',
                        default='', type=str)
    parser.add_argument('--run_name', help='Name of run for MLflow.',
                        default='', type=str)
    parser.add_argument('--tags', help='Experiment tags. Format: '
                                       '"key1:value1[+key2:value2[+...]]"',
                        type=str, default='NO_TAGS:True')
    parser.add_argument('--labels',
                        help='Experiment labels that will be the keys of'
                             ' experiment tags with value=True.'
                             ' Format: "label1[+label2[+...]]"',
                        type=str, default='NO_LABELS')
    parser.add_argument('--optimizer', help='Optimizer name.',
                        type=str, default='Adam')
    parser.add_argument('--gpu_id', help='GPU ID.', type=int, default='0')
    add_bool_arg(parser, name='show_plot', default=False,
                 kw_options_on={'help': 'Turn on plotting for results.'},
                 kw_options_off={'help': 'Turn off plotting for results.'})
    parser.add_argument('--mlflow_uri', help='MLflow tracking URI.',
                        type=str, default='./mlruns')
    parser.add_argument('--weight_decay', help='Weight decay.', type=float, default=0)
    args = parser.parse_args()

    # np.random.seed(args.seed)
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
        mlflow.set_tags(dict([kw.split(':') for kw in args.tags.split('+')]))
        args_dict.pop('tags')
        mlflow.set_tags({label: True for label in args.labels.split('+')})
        args_dict.pop('labels')
        mlflow.log_params(args_dict)

        # Below, we save the starting time in format 0.YYmmddHHMMSS.
        # This representation allows quick comparisons
        # in the lexicographic order.
        dt_start = datetime.datetime.now(datetime.timezone.utc)
        mlflow.log_metric(
            'timef_start',
            float(dt_start.strftime('0.%Y%m%d%H%M%S')))

        if args.exp_type == 'toy_regression1':
            toy_regression1(args, gen_data=gen_data_cubic)
        else:
            raise ValueError('Invalid exp_type: {}.'
                            ' Valid options: toy_regression1, toy_regression_quartic, toy_regression_linear.'.format(args.loss_type))

        dt_end = datetime.datetime.now(datetime.timezone.utc)
        mlflow.log_metric('timef_end', float(dt_end.strftime('0.%Y%m%d%H%M')))


def add_bool_arg(parser, name, default=False, kw_options_on={}, kw_options_off={}):
    # See https://stackoverflow.com/a/31347222
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--' + name, dest=name, action='store_true', **kw_options_on)
    group.add_argument('--no-' + name, dest=name, action='store_false', **kw_options_off)
    parser.set_defaults(**{name:default})


def toy_regression1(
            args: Any,
            gen_data: Callable[
                ...,
                Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            ]
        ) -> Dict[str, Any]:
    device: Any
    if torch.cuda.is_available():  # type:ignore
        device = torch.device('cuda:{}'.format(args.gpu_id))
    else:
        device = torch.device('cpu')

    xall: torch.Tensor
    uall: torch.Tensor
    yall: torch.Tensor

    if args.exp_type in [
        'toy_regression1',
    ]:
        xall, uall, yall = gen_data(
            n=args.n0+args.n1+args.n_te,
            xdim=args.xdim,
            noise_level_u=args.noise_level_u,
            noise_level_y=args.noise_level_y
        )
        x_val, u_val, y_val = gen_data(
            n=2*args.n_te,
            xdim=args.xdim,
            noise_level_u=args.noise_level_u,
            noise_level_y=args.noise_level_y
        )
    else:
        raise ValueError('Invalid exp_type: {}.'
                         ' Valid options: toy_regression1.'.format(args.loss_type))

    x0, x1, x_te = torch.split(xall, [args.n0, args.n1, args.n_te])  # type: ignore
    u0, u1, u_te = torch.split(uall, [args.n0, args.n1, args.n_te])  # type: ignore
    y0, y1, y_te = torch.split(yall, [args.n0, args.n1, args.n_te])  # type: ignore

    # TODO: change n_te to n_val
    x0_val, x1_val = torch.split(x_val, [args.n_te, args.n_te])  # type: ignore
    u0_val, u1_val = torch.split(u_val, [args.n_te, args.n_te])  # type: ignore
    y0_val, y1_val = torch.split(y_val, [args.n_te, args.n_te])  # type: ignore

    loader_xu_tr = DataLoader(
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

    loader_xu_val = DataLoader(
        TensorDataset(x0_val.float(), u0_val.float()),
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True
    )
    loader_uy_val = DataLoader(
        TensorDataset(u1_val.float(), y1_val.float()),
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True
    )

    res = {}

    model_JointRR = NN(
        model_f=MLP3(
            dim_in=x0.shape[1],
            dim_hid=args.d_hidden,
            dim_out=y0.shape[1]).to(device),
        model_h=MLP3(
            dim_in=u0.shape[1],
            dim_hid=args.d_hidden,
            dim_out=y0.shape[1]).to(device),
        weight_decay_f=args.weight_decay,
        weight_decay_h=args.weight_decay,
        n_epochs=args.epochs//2 if args.warm_start else args.epochs,
        w_init=.5,
        optimizer=args.optimizer,
        batch_size=args.batch_size,
        lr_f=args.learning_rate,
        lr_h=args.learning_rate,
        grad_clip=args.grad_clip,
        warm_start=args.warm_start,
        loss_type='JointRR',
        batch_norm=False,
        record_loss=True,
        two_step=False,
        log_metric_label='JointRR',
        device=device
    )
    model_JointRR.fit_indirect(loader_xu_tr, loader_uy)
    mse_JointRR = mse_x2y_y(
        predict_y_from_x=model_JointRR.predict_y_from_x,
        loader_xy=loader_xy_te,
        device=device)
    mlflow.log_metric('MSE_JointRR', mse_JointRR)
    mlflow.log_metric('w_JointRR', model_JointRR.w_.item())
    # Validation upper and lower bounds:
    losses_UB_val_JointRR = l2_SumUB_fullbatch(
                                predict_y_from_x=model_JointRR.predict_y_from_x,
                                predict_y_from_u=model_JointRR.predict_y_from_u,
                                loader_xu=loader_xu_val,
                                loader_uy=loader_uy_val,
                                device=device
                            )
    mlflow.log_metric("loss_UB_val_JointRR", losses_UB_val_JointRR.upper_loss_total)
    mlflow.log_metric("loss_LB_val_JointRR", losses_UB_val_JointRR.lower_loss_total)

    model_JointBregMU_l2 = NN(
        model_f=MLP3(
            dim_in=x0.shape[1],
            dim_hid=args.d_hidden,
            dim_out=y0.shape[1]).to(device),
        model_h=MLP3(
            dim_in=u0.shape[1],
            dim_hid=args.d_hidden,
            dim_out=y0.shape[1]).to(device),
        weight_decay_f=args.weight_decay,
        weight_decay_h=args.weight_decay,
        n_epochs=args.epochs//2 if args.warm_start else args.epochs,
        optimizer=args.optimizer,
        batch_size=args.batch_size,
        lr_f=args.learning_rate,
        lr_h=args.learning_rate,
        grad_clip=args.grad_clip,
        warm_start=args.warm_start,
        loss_type='l2_CSUB',
        batch_norm=False,
        record_loss=True,
        two_step=False,
        log_metric_label='JointBregMU_l2',
        device=device
    )
    model_JointBregMU_l2.fit_indirect(loader_xu_tr, loader_uy)
    mse_JointBregMU_l2 = mse_x2y_y(
        predict_y_from_x=model_JointBregMU_l2.predict_y_from_x,
        loader_xy=loader_xy_te,
        device=device)
    mlflow.log_metric('MSE_JointBregMU_l2', mse_JointBregMU_l2)
    mlflow.log_metric('w_JointBregMU_l2', model_JointBregMU_l2.w_.item())
    res['MSE_JointBregMU_l2'] = mse_JointBregMU_l2
    print('MSE of JointBregMU_l2: {}'.format(mse_JointBregMU_l2))
    # Validation upper and lower bounds:
    losses_UB_val_JointBregMU_l2 = l2_CSUB_fullbatch(
        predict_y_from_x=model_JointBregMU_l2.predict_y_from_x,
        predict_y_from_u=model_JointBregMU_l2.predict_y_from_u,
        loader_xu=loader_xu_val,
        loader_uy=loader_uy_val,
        device=device
    )
    mlflow.log_metric("loss_UB_val_JointBregMU_l2", losses_UB_val_JointBregMU_l2.upper_loss_total)
    mlflow.log_metric("loss_LB_val_JointBregMU_l2", losses_UB_val_JointBregMU_l2.lower_loss_total)

    # model_2StepRR = NN(
    #     model_f=MLP3(
    #         dim_in=x0.shape[1],
    #         dim_hid=args.d_hidden,
    #         dim_out=y0.shape[1]).to(device),
    #     model_h=MLP3(
    #         dim_in=u0.shape[1],
    #         dim_hid=args.d_hidden,
    #         dim_out=y0.shape[1]).to(device),
    #     weight_decay_f=args.weight_decay,
    #     weight_decay_h=args.weight_decay,
    #     n_epochs=args.epochs,
    #     optimizer=args.optimizer,
    #     batch_size=args.batch_size,
    #     lr_f=args.learning_rate,
    #     lr_h=args.learning_rate,
    #     grad_clip=args.grad_clip,
    #     two_step=True,
    #     loss_type='l2',
    #     batch_norm=False,
    #     record_loss=True,
    #     log_metric_label='2StepRR',
    #     device=device
    # )
    # model_2StepRR.fit_indirect(loader_xu_tr, loader_uy)
    # mse_2StepRR = mse_x2y_y(
    #     predict_y_from_x=model_2StepRR.predict_y_from_x,
    #     loader_xy=loader_xy_te,
    #     device=device)
    # mlflow.log_metric('MSE_2StepRR', mse_2StepRR)
    # res['MSE_2StepRR'] = mse_2StepRR
    # print('MSE of 2StepRR: {}'.format(mse_2StepRR))

    # model_Naive = Combined(
    #     model_x2u=MLP3(
    #         dim_in=x0.shape[1],
    #         dim_hid=args.d_hidden,
    #         dim_out=u0.shape[1]),
    #     model_u2y=MLP3(
    #         dim_in=u0.shape[1],
    #         dim_hid=args.d_hidden,
    #         dim_out=y0.shape[1]),
    #     batch_size=args.batch_size,
    #     n_epochs=args.epochs,
    #     optimizer=args.optimizer,
    #     weight_decay=args.weight_decay,
    #     lr_u2y=args.learning_rate,
    #     lr_x2u=args.learning_rate,
    #     record_loss=True,
    #     log_metric_label='Naive',
    #     device=device,
    # )
    # model_Naive.fit_indirect(loader_xu_tr, loader_uy)
    # mse_Naive = mse_x2y_y(
    #     predict_y_from_x=model_Naive.predict_y_from_x,
    #     loader_xy=loader_xy_te,
    #     device=device)
    # mlflow.log_metric('MSE_Naive', mse_Naive)
    # res['MSE_Naive'] = mse_Naive
    # print('MSE of Naive: {}'.format(mse_Naive))

    return res


def gen_data_cubic(
    xdim: int,
    noise_level_u: float = 0.5,
    noise_level_y: float = 0.5,
    n: int = 1000,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    def gen_x(size: Tuple[int, int]) -> torch.Tensor:
        return force2d(2 * torch.rand(size) - 1).float()

    def gen_u(x_: torch.Tensor) -> torch.Tensor:
        # noise = 2 * noise_level_u * (torch.rand_like(x_) - 1)
        noise = noise_level_u * (2 * torch.rand_like(x_) - 1)
        return force2d(x_**3 + noise).float()

    def gen_y(u_: torch.Tensor) -> torch.Tensor:
        u_norm = torch.norm(u_, dim=1)
        return force2d(u_norm**2 + noise_level_y * torch.randn_like(u_norm)).float()

    x = gen_x(size=(n, xdim))
    u = gen_u(x_=x)
    y = gen_y(u_=u)

    return x, u, y


if __name__ == '__main__':
    main()

