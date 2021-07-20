# pyright: strict

from typing import Dict, Tuple, Any, Optional, Callable, NamedTuple, List, Union
import torch
from torch.utils.data import DataLoader, random_split, TensorDataset
import torchvision.datasets as torchdata  # type: ignore
import torchvision.transforms as transforms  # type: ignore


class DataWithInfo(NamedTuple):
    loader_xy_and_size: 'Optional[Tuple[DataLoader[Tuple[torch.Tensor, ...]], int]]'
    loader_uy_and_size: 'Optional[Tuple[DataLoader[Tuple[torch.Tensor, ...]], int]]'
    loader_xu_and_size: 'Optional[Tuple[DataLoader[Tuple[torch.Tensor, ...]], int]]'
    shape_x: Tuple[int, ...]
    shape_u: Tuple[int, ...]
    shape_y: Tuple[int, ...]


class MakeFromTorchData():
    def __init__(self, args: Any, seed: int = 0):
        self.args = args
        # self.gen = torch.Generator()
        # self.gen.manual_seed(seed)
    def __call__(self, args: Any, train: bool) -> DataWithInfo:
        if self.args.transform == 'projection' and train:
            return CIFAR10_projected_train(
                n0=self.args.n0,
                udim=self.args.udim,
                batch_size=self.args.batch_size
            )
        elif self.args.transform == 'projection' and not train:  # For test data.
            return CIFAR10_projected_test()
        elif self.args.transform == 'downsampling' and train:
            # This covers CIFAR10, CIFAR100
            return benchdata_downsampled_train(
                n0=self.args.n0,
                n1=self.args.n1,
                downsampling_kernel=self.args.downsampling_kernel,
                downsampling_stride=self.args.downsampling_stride,
                batch_size=self.args.batch_size,
                dataname=self.args.dataname,
                base_path=self.args.base_path
            )
        elif self.args.transform == 'downsampling' and not train:
            # This covers CIFAR10, CIFAR100
            return benchdata_downsampled_test(
                downsampling_kernel=self.args.downsampling_kernel,
                downsampling_stride=self.args.downsampling_stride,
                batch_size=self.args.batch_size,
                dataname=self.args.dataname,
                base_path=self.args.base_path
            )
        elif self.args.transform == 'cropping' and train and args.dataname == 'CIFAR10':
            return CIFAR10_cropped_train(n0=self.args.n0, n1=self.args.n1)
        elif self.args.transform == 'cropping' and not train and args.dataname == 'CIFAR10':
            return CIFAR10_cropped_test(n0=self.args.n0, n1=self.args.n1)
        elif self.args.transform == 'cropping' and not train:  # For test data.
            assert False
        elif self.args.transform == 'pooling' and train:
            assert False
        elif self.args.transform == 'pooling' and not train:  # For test data.
            assert False
        else:
            assert False


def benchdata_downsampled_train(
        n0: int,
        n1: int,
        downsampling_kernel: int,
        downsampling_stride: int,
        batch_size: int,
        dataname: str,
        base_path: str
) -> DataWithInfo:
    data = torchdata.__dict__[dataname](
        root=base_path,
        train=True,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize_transform(dataname)
        ]),
        download=True
    )

    data0, data1, _ = random_split(data, [n0, n1, len(data) - n0 - n1])  # type: ignore
    n_classes = len(data.classes)
    batch_size = min(batch_size, n0)
    kernel_size = (downsampling_kernel, downsampling_kernel)
    stride = (downsampling_stride, downsampling_stride)
    loader_xu = DataLoader(
        dataset=data0,
        collate_fn=DownSample(
            kernel_size=kernel_size,
            stride=stride,
            n_classes=n_classes,
            batch_size=batch_size,
            include='xu'
        ),
        batch_size=batch_size,
        drop_last=False,
        shuffle=True
    )
    n_xu = len(data0)
    loader_uy = DataLoader(
        dataset=data1,
        collate_fn=DownSample(
            kernel_size=kernel_size,
            stride=stride,
            n_classes=n_classes,
            batch_size=batch_size,
            include='uy'
        ),
        batch_size=batch_size,
        drop_last=False,
        shuffle=True
    )
    n_uy = len(data1)

    xs_tmp, us_tmp, ys_tmp = DownSample(  # type: ignore
            kernel_size=kernel_size,
            stride=stride,
            n_classes=n_classes,
            batch_size=batch_size,
            include='xuy'
        )([data[0]])
    shape_x = tuple(xs_tmp.shape[1:])
    shape_u = tuple(us_tmp.shape[1:])
    shape_y = tuple(ys_tmp.shape[1:])
    del xs_tmp, us_tmp, ys_tmp

    return DataWithInfo(
        loader_xy_and_size=None,
        loader_xu_and_size=(loader_xu, n_xu),
        loader_uy_and_size=(loader_uy, n_uy),
        shape_x=shape_x,
        shape_u=shape_u,
        shape_y=shape_y,
    )


def benchdata_downsampled_test(
        batch_size: int,
        downsampling_stride: int,
        downsampling_kernel: int,
        dataname: str,
        base_path: str
):
    data = torchdata.__dict__[dataname](
        root=base_path,
        train=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize_transform(dataname)
        ]),
        download=True
    )

    n_classes = len(data.classes)
    nte = len(data)
    batch_size = min(batch_size, nte)
    kernel_size = (downsampling_kernel, downsampling_kernel)
    stride = (downsampling_stride, downsampling_stride)

    loader_xy = DataLoader(
        dataset=data,
        collate_fn=DownSample(
            kernel_size=kernel_size,
            stride=stride,
            n_classes=n_classes,
            batch_size=batch_size,
            include='xy'
        ),
        batch_size=batch_size,
        drop_last=False,
        shuffle=True
    )  # type:ignore
    n_xy = len(data)

    xs_tmp, us_tmp, ys_tmp = DownSample(  # type: ignore
            kernel_size=kernel_size,
            stride=stride,
            n_classes=n_classes,
            batch_size=batch_size,
            include='xuy'
        )([data[0]])
    shape_x = xs_tmp.shape[1:]
    shape_u = us_tmp.shape[1:]
    shape_y = ys_tmp.shape[1:]
    del xs_tmp, us_tmp, ys_tmp

    return DataWithInfo(
        loader_xy_and_size=(loader_xy, n_xy),
        loader_xu_and_size=None,
        loader_uy_and_size=None,
        shape_x=shape_x,
        shape_u=shape_u,
        shape_y=shape_y,
    )


def CIFAR10_cropped_train(n0, n1, batch_size, base_path, cropping_ratio):
    data = torchdata.__dict__['CIFAR10'](
        root=base_path,
        train=True,
        transform=transforms.Compose([
            transforms.ToTensor()
        ]),
        download=True
    )

    data0, data1, _ = random_split(data, [n0, n1, len(data) - n0 - n1])  # type: ignore
    n0, n1 = len(data0), len(data1)
    n_classes = len(data.classes)
    batch_size=min(batch_size, n0)

    loader_xu = DataLoader(
        dataset=data0,
        collate_fn=CropCIFAR10(
            cropping_ratio=cropping_ratio,
            n_classes=n_classes,
            batch_size=batch_size,
            include='xu'
        ),
        batch_size=batch_size,
        drop_last=False,
        shuffle=True
    )  # type:ignore
    n_xu = len(data0)
    loader_uy = DataLoader(
        dataset=data1,
        collate_fn=CropCIFAR10(
            cropping_ratio=self.args.cropping_ratio,
            n_classes=n_classes,
            batch_size=batch_size,
            include='uy'
        ),
        batch_size=batch_size,
        drop_last=False,
        shuffle=True
    )  # type:ignore
    n_uy = len(data1)

    xs_tmp, us_tmp, ys_tmp = CropCIFAR10(  # type: ignore
        cropping_ratio=self.args.cropping_ratio,
        n_classes=n_classes,
        batch_size=batch_size,
        include='xuy'
    )([data[0]])
    shape_x = xs_tmp.shape[1]
    shape_u = us_tmp.shape[1]
    shape_y = ys_tmp.shape[1]
    del xs_tmp, us_tmp, ys_tmp

    return DataWithInfo(
        loader_xy_and_size=None,
        loader_xu_and_size=(loader_xu, n_xu),
        loader_uy_and_size=(loader_uy, n_uy),
        shape_x=shape_x,
        shape_u=shape_u,
        shape_y=shape_y,
    )

def CIFAR10_cropped_test(
        n0: int,
        cropping_ratio: float,
        batch_size: int,
        base_path: str,
):
    data = torchdata.__dict__['CIFAR10'](
        root=base_path,
        train=False,
        transform=transforms.Compose([
            transforms.ToTensor()
        ]),
        download=True
    )

    n_classes = len(data.classes)
    nte = len(data)
    batch_size=min(batch_size, n0)

    loader_xy = DataLoader(
        dataset=data,
        collate_fn=CropCIFAR10(
            cropping_ratio=None,
            n_classes=n_classes,
            batch_size=batch_size,
            include='xy'
        ),
        batch_size=batch_size,
        drop_last=False,
        shuffle=True
    )  # type:ignore
    n_xy = len(data)

    xs_tmp, us_tmp, ys_tmp = CropCIFAR10(  # type: ignore
        cropping_ratio=cropping_ratio,
        n_classes=n_classes,
        batch_size=batch_size,
        include='xuy'
    )([data[0]])
    shape_x = xs_tmp.shape[1]
    shape_u = us_tmp.shape[1]
    shape_y = ys_tmp.shape[1]
    del xs_tmp, us_tmp, ys_tmp

    return DataWithInfo(
        loader_xy_and_size=(loader_xy, n_xy),
        loader_xu_and_size=None,
        loader_uy_and_size=None,
        shape_x=shape_x,
        shape_u=shape_u,
        shape_y=shape_y,
    )


def CIFAR10_projected_train(
        n0: int,
        udim: int,
        batch_size: int
) -> DataWithInfo:
    # TODO Finish rewriting
    raise NotImplementedError()
    data = torchdata.__dict__['CIFAR10'](
        root=self.args.base_path,
        train=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            lambda x: x.view(-1)
        ]),
        download=True
    )

    data0, data1, _ = random_split(data, [n0, n1, len(data) - n0 - n1])  # type: ignore
    n0, n1 = len(data0), len(data1)
    n_classes = len(data.classes)
    xtmp, _ = data[0]
    xdim = xtmp.shape[1]
    w = torch.randn(udim, xdim)
    batch_size=min(batch_size, n0)

    loader_xu = DataLoader(
        dataset=data0,
        collate_fn=PreprocessXUY(w=w, n_classes=n_classes, batch_size=batch_size, cover_y=True, train=True),
        batch_size=batch_size,
        drop_last=False,
        shuffle=True
    )
    loader_uy = DataLoader(
        dataset=data1,
        collate_fn=PreprocessXUY(w=w, n_classes=n_classes, batch_size=batch_size, cover_x=True, train=True),
        batch_size=batch_size,
        drop_last=False,
        shuffle=True
    )

    return DataWithInfo(
        loader_xy=None,
        loader_xu=loader_xu,
        loader_uy=loader_uy,
        x_shape=xtmp.shape,
        u_shape=utmp.shape,
        y_shape=torch.Size((n_classes,)),
        n0=n0,
        n1=n1,
        nte=None
    )


def CIFAR10_projected_test() -> DataWithInfo:
    # TODO Finish rewriting
    raise NotImplementedError()
    data = torchdata.__dict__[self.args.dataname](
        root=self.args.base_path,
        train=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            lambda x: x.view(-1)
        ]),
        download=True
    )

    n_classes = len(data.classes)
    nte = len(data)
    batch_size=min(self.args.batch_size, nte)
    loader_xy = DataLoader(
        dataset=data,
        collate_fn=PreprocessXUY(
            w=None,
            n_classes=n_classes,
            batch_size=batch_size,
            train=False
        ),
        batch_size=batch_size,
        drop_last=False,
        shuffle=True)

    return DataWithInfo(
        loader_xy=loader_xy,
        loader_xu=None,
        loader_uy=None,
        ydim=n_classes,
        n0=None,
        n1=None,
        nte=nte
    )


class DownSample():
    def __init__(
        self,
        kernel_size: Tuple[int, int],
        stride: Tuple[int, int],
        n_classes: int,
        batch_size: int,
        include: str ='xuy'
    ):
        self.kernel_size = kernel_size
        self.stride = stride
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.include = include
        self.ys_onehot = torch.FloatTensor(batch_size, self.n_classes)  # type: ignore

        self.down_sample = torch.nn.AvgPool2d(kernel_size=self.kernel_size, stride=self.stride)

    def __call__(self, batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        # `u`s are the images in the original resolution.
        us = torch.cat([u[None, :] for u, _ in batch], axis=0)  # type: ignore
        ys_raw = torch.tensor([y for _, y in batch]).long()

        # Convert y to a one-hot-vector:
        batch_size = len(batch)
        self.ys_onehot[:batch_size, :].zero_()
        self.ys_onehot[:batch_size, :].scatter_(1, ys_raw.view(-1, 1), 1)
        ys = self.ys_onehot[:batch_size, :]

        ret = []
        if 'x' in self.include:
            # Make low-resolution images xs:
            xs = self.down_sample(us)
            ret.append(xs)

        if 'u' in self.include:
            ret.append(us)

        if 'y' in self.include:
            ret.append(ys)

        return tuple(ret)


class CropCIFAR10():
    def __init__(
        self,
        cropping_ratio: Optional[float],
        #gen: Optional[torch.Tensor],
        n_classes: int,
        batch_size: int,
        include: str ='xuy'
    ) -> None:
        # self.gen = gen
        self.cropping_ratio = cropping_ratio
        self.n_classes = n_classes
        self.ys_onehot = torch.FloatTensor(batch_size, self.n_classes)  # type: ignore
        self.include = include

    def __call__(self, batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        xs = torch.cat([x[None, :] for x, _ in batch], axis=0)  # type: ignore
        ys_raw = torch.tensor([y for _, y in batch]).long()

        batch_size = len(batch)
        self.ys_onehot[:batch_size, :].zero_()
        self.ys_onehot[:batch_size, :].scatter_(1, ys_raw.view(-1, 1), 1)
        ys = self.ys_onehot[:batch_size, :]

        ret: List[torch.Tensor] = []
        if 'x' in self.include:
            ret.append(xs)

        if 'u' in self.include:
            assert self.cropping_ratio is not None
            # Make cropped images us:
            self.width = xs.shape[2]
            self.height = xs.shape[3]
            wmargin = int(self.width * (1 - self.cropping_ratio))
            hmargin = int(self.height * (1 - self.cropping_ratio))
            left = [int(a.item()) for a in torch.rand(batch_size) * (wmargin + 1)]  # type:ignore
            right = [int(self.width * self.cropping_ratio) + a for a in left]
            top = [int(a.item()) for a in torch.rand(batch_size) * (hmargin + 1)]  # type:ignore
            bottom = [int(self.height * self.cropping_ratio) + a for a in top]
            us = torch.cat([x[:, l:r, t:b][None, :] for x, l, r, t, b in zip(xs, left, right, top, bottom)], axis=0)  # type: ignore
            ret.append(us)

        if 'y' in self.include:
            ret.append(ys)

        return tuple(ret)


def normalize_transform(dataname: str):
    """ See https://gist.github.com/weiaicunzai/e623931921efefd4c331622c344d8151#gistcomment-2851662
    """
    if dataname == 'CIFAR10':
        return transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
            # mean=[0.4914, 0.4822, 0.4465],
            # std=[0.2470, 0.2435, 0.2616])
    elif dataname == 'CIFAR100':
        return transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408],
            std=[0.2675, 0.2565, 0.2761])
    elif dataname == 'MNIST':  # TODO: Use correct mean and std
        return transforms.Normalize(
            mean=[0],
            std=[1])
    elif dataname == 'FashionMNIST':  # TODO: Use correct mean and std
        return transforms.Normalize(
            mean=[0],
            std=[1])
    else:
        raise ValueError('Argument dataname must be \'CIFAR10\', \'CIFAR100\', \'MNIST\', or \'FashionMNIST\'.')


def FashionMNIST_normalize():
    return transforms.Normalize(
        mean=[0.2860],
        std=[0.3530])


class PreprocessXUY():
    def __init__(
        self,
        w: Optional[torch.Tensor],
        n_classes: int,
        train: bool,
        batch_size: int,
        cover_x: bool=False,
        cover_y: bool=False
    ) -> None:
        self.w = w
        self.n_classes = n_classes
        self.train = train
        self.cover_x = cover_x
        self.cover_y = cover_y
        self.ys_onehot = torch.FloatTensor(batch_size, self.n_classes)  # type: ignore

    def __call__(self, batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        xs = torch.cat([x.view(1, -1) for x, _ in batch], axis=0)  # type: ignore
        ys_raw = torch.tensor([y for _, y in batch]).long()

        batch_size = len(batch)
        self.ys_onehot[:batch_size, :].zero_()
        self.ys_onehot[:batch_size, :].scatter_(1, ys_raw.view(-1, 1), 1)
        ys = self.ys_onehot[:batch_size, :]

        if not self.train:
            return xs, ys
        assert isinstance(self.w, torch.Tensor)

        us = xs.mm(self.w.T)
        if self.cover_x and not self.cover_y:
            return us, ys
        elif self.cover_y and not self.cover_x:
            return xs, us
        elif not self.cover_x and not self.cover_y:
            return xs, us, ys
        elif self.cover_x and self.cover_y:
            warnings.warn('Hiding both x and y.')
            return us
        else:
            assert False
