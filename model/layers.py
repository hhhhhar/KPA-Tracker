import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
import math


## ---------- standard layers ----------


class MyBatchNorm1d(_BatchNorm):
    # r"""Applies Batch Normalization over a 2d or 3d input that is seen as a
    # mini-batch.
    # .. math::
    #     y = \frac{x - mean[x]}{ \sqrt{Var[x] + \epsilon}} * gamma + beta
    # The mean and standard-deviation are calculated per-dimension over
    # the mini-batches and gamma and beta are learnable parameter vectors
    # of size C (where C is the input size).
    # During training, this layer keeps a running estimate of its computed mean
    # and variance. The running sum is kept with a default momentum of 0.1.
    # During evaluation, this running mean/variance is used for normalization.
    # Because the BatchNorm is done over the `C` dimension, computing statistics
    # on `(N, L)` slices, it's common terminology to call this Temporal BatchNorm
    # Args:
    #     num_features: num_features from an expected input of size
    #         `batch_size x num_features [x width]`
    #     eps: a value added to the denominator for numerical stability.
    #         Default: 1e-5
    #     momentum: the value used for the running_mean and running_var
    #         computation. Default: 0.1
    #     affine: a boolean value that when set to ``True``, gives the layer learnable
    #         affine parameters. Default: ``True``
    # Shape:
    #     - Input: :math:`(N, C)` or :math:`(N, C, L)`
    #     - Output: :math:`(N, C)` or :math:`(N, C, L)` (same shape as input)
    # """

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, momentum_decay_step=None, momentum_decay=1):
        super(MyBatchNorm1d, self).__init__(num_features, eps, momentum, affine)
        self.momentum_decay_step = momentum_decay_step
        self.momentum_decay = momentum_decay
        self.momentum_original = self.momentum

    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'
                             .format(input.dim()))
        super(MyBatchNorm1d, self)._check_input_dim(input)

    def forward(self, input, epoch=None):
        if (epoch is not None) and (epoch >= 1) and (self.momentum_decay_step is not None) and (
                self.momentum_decay_step > 0):
            # perform momentum decay
            self.momentum = self.momentum_original * (self.momentum_decay ** (epoch // self.momentum_decay_step))
            if self.momentum < 0.01:
                self.momentum = 0.01

        return nn.functional.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            self.training, self.momentum, self.eps)


class MyBatchNorm2d(_BatchNorm):
    # r"""Applies Batch Normalization over a 4d input that is seen as a mini-batch
    # of 3d inputs
    # .. math::
    #     y = \frac{x - mean[x]}{ \sqrt{Var[x] + \epsilon}} * gamma + beta
    # The mean and standard-deviation are calculated per-dimension over
    # the mini-batches and gamma and beta are learnable parameter vectors
    # of size C (where C is the input size).
    # During training, this layer keeps a running estimate of its computed mean
    # and variance. The running sum is kept with a default momentum of 0.1.
    # During evaluation, this running mean/variance is used for normalization.
    # Because the BatchNorm is done over the `C` dimension, computing statistics
    # on `(N, H, W)` slices, it's common terminology to call this Spatial BatchNorm
    # Args:
    #     num_features: num_features from an expected input of
    #         size batch_size x num_features x height x width
    #     eps: a value added to the denominator for numerical stability.
    #         Default: 1e-5
    #     momentum: the value used for the running_mean and running_var
    #         computation. Default: 0.1
    #     affine: a boolean value that when set to ``True``, gives the layer learnable
    #         affine parameters. Default: ``True``
    # Shape:
    #     - Input: :math:`(N, C, H, W)`
    #     - Output: :math:`(N, C, H, W)` (same shape as input)
    # """

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, momentum_decay_step=None, momentum_decay=1):
        super(MyBatchNorm2d, self).__init__(num_features, eps, momentum, affine)
        self.momentum_decay_step = momentum_decay_step
        self.momentum_decay = momentum_decay
        self.momentum_original = self.momentum

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))
        super(MyBatchNorm2d, self)._check_input_dim(input)

    def forward(self, input, epoch=None):
        if (epoch is not None) and (epoch >= 1) and (self.momentum_decay_step is not None) and (
                self.momentum_decay_step > 0):
            # perform momentum decay
            self.momentum = self.momentum_original * (self.momentum_decay ** (epoch // self.momentum_decay_step))
            if self.momentum < 0.01:
                self.momentum = 0.01

        return nn.functional.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            self.training, self.momentum, self.eps)


class MyConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, activation=None,
                 normalization=None, momentum=0.1, bn_momentum_decay_step=None, bn_momentum_decay=1):
        super(MyConv2d, self).__init__()
        self.activation = activation
        self.normalization = normalization

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        # self.conv = XConv(in_channels, out_channels, dim=3, kernel_size=kernel_size)
        if self.normalization == 'batch':
            self.norm = MyBatchNorm2d(out_channels, momentum=momentum, affine=True,
                                      momentum_decay_step=bn_momentum_decay_step, momentum_decay=bn_momentum_decay)
        elif self.normalization == 'instance':
            self.norm = nn.InstanceNorm2d(out_channels, momentum=momentum, affine=True)
        if self.activation == 'relu':
            self.act = nn.ReLU()
        elif self.activation == 'elu':
            self.act = nn.ELU(alpha=1.0)
        elif 'leakyrelu' == self.activation:
            self.act = nn.LeakyReLU(0.01)
        elif 'selu' == self.activation:
            self.act = nn.SELU()

        self.weight_init()

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.fill_(0)
            elif isinstance(m, MyBatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, epoch=None):
        x = self.conv(x)
        if self.normalization == 'batch':
            x = self.norm(x, epoch)
        elif self.normalization is not None:
            x = self.norm(x)

        if self.activation is not None:
            x = self.act(x)
        return x


class EquivariantLayer(nn.Module):
    def __init__(self, num_in_channels, num_out_channels, activation='relu', normalization=None, momentum=0.1,
                 bn_momentum_decay_step=None, bn_momentum_decay=1, num_groups=16):
        super(EquivariantLayer, self).__init__()

        self.num_in_channels = num_in_channels
        self.num_out_channels = num_out_channels
        self.activation = activation
        self.normalization = normalization

        self.conv = nn.Conv1d(self.num_in_channels, self.num_out_channels, kernel_size=1, stride=1, padding=0)

        if 'batch' == self.normalization:
            self.norm = MyBatchNorm1d(self.num_out_channels, momentum=momentum, affine=True,
                                      momentum_decay_step=bn_momentum_decay_step, momentum_decay=bn_momentum_decay)
        elif 'instance' == self.normalization:
            self.norm = nn.InstanceNorm1d(self.num_out_channels, momentum=momentum, affine=True)
        elif 'group' == self.normalization:
            self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=self.num_out_channels)

        if 'relu' == self.activation:
            self.act = nn.ReLU()
        elif 'leakyrelu' == self.activation:
            self.act = nn.LeakyReLU(0.01)

        self.weight_init()

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                if self.activation == 'relu' or self.activation == 'leakyrelu':
                    nn.init.kaiming_normal_(m.weight, nonlinearity=self.activation)
                else:
                    m.weight.data.normal_(0, std=0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, MyBatchNorm1d) or isinstance(m, nn.InstanceNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, epoch=None):
        y = self.conv(x)

        if self.normalization == 'batch':
            y = self.norm(y, epoch)
        elif self.normalization is not None:
            y = self.norm(y)

        if self.activation is not None:
            y = self.act(y)

        return y


## ---------- nodes branch ----------
class GeneralKNNFusionModule(nn.Module):
    def __init__(self, in_channels, out_channels_list_before, out_channels_list_after,
                 activation, normalization, momentum=0.1,
                 bn_momentum_decay_step=None, bn_momentum_decay=1):
        super(GeneralKNNFusionModule, self).__init__()

        self.layers_before = nn.ModuleList()
        previous_out_channels = in_channels
        for i, c_out in enumerate(out_channels_list_before):
            self.layers_before.append(
                MyConv2d(previous_out_channels, c_out, kernel_size=1, stride=1, padding=0, bias=True,
                         activation=activation, normalization=normalization,
                         momentum=momentum, bn_momentum_decay_step=bn_momentum_decay_step,
                         bn_momentum_decay=bn_momentum_decay))
            previous_out_channels = c_out

        self.layers_after = nn.ModuleList()
        previous_out_channels = 2 * previous_out_channels
        for i, c_out in enumerate(out_channels_list_after):
            self.layers_after.append(
                MyConv2d(previous_out_channels, c_out, kernel_size=1, stride=1, padding=0, bias=True,
                         activation=activation, normalization=normalization,
                         momentum=momentum, bn_momentum_decay_step=bn_momentum_decay_step,
                         bn_momentum_decay=bn_momentum_decay))
            previous_out_channels = c_out

    def forward(self, query, database, x, K, epoch=None):
        # '''

        # :param query: Bx3xM FloatTensor
        # :param database: Bx3xN FloatTensor
        # :param x: BxCxN FloatTensor
        # :param K: K neighbors
        # :return:
        # '''
        # 1. compute knn, query -> database
        # 2. for each query, normalize neighbors with its coordinate
        # 3. FC for these normalized points
        # 4. maxpool for each query

        B, M, N, C = query.size()[0], query.size()[2], database.size()[2], x.size()[1]

        query_Mx1 = query.detach().unsqueeze(3)  # Bx3xMx1
        database_1xN = database.detach().unsqueeze(2)  # Bx3x1xN

        norm = torch.norm(query_Mx1 - database_1xN, dim=1, keepdim=False)  # Bx3xMxN -> BxMxN
        knn_D, knn_I = torch.topk(norm, k=K, dim=2, largest=False, sorted=True)  # BxMxK, BxMxK
        knn_I_3 = knn_I.unsqueeze(1).expand(B, 3, M, K).contiguous().view(B, 3, M * K)  # Bx3xMxK -> Bx3xM*K
        knn_I_C = knn_I.unsqueeze(1).expand(B, C, M, K).contiguous().view(B, C, M * K)  # BxCxMxK -> BxCxM*K

        query_neighbor_coord = torch.gather(database, dim=2, index=knn_I_3).view(B, 3, M, K)  # Bx3xMxK
        query_neighbor_feature = torch.gather(x, dim=2, index=knn_I_C).view(B, C, M, K)  # BxCxMxK

        query_neighbor_coord_decentered = (query_neighbor_coord - query_Mx1).detach()
        query_neighbor = torch.cat((query_neighbor_coord_decentered, query_neighbor_feature), dim=1)  # Bx(3+C)xMxK

        for layer in self.layers_before:
            query_neighbor = layer(query_neighbor, epoch)
        feature, _ = torch.max(query_neighbor, dim=3, keepdim=True)  # BxCxMx1

        y = torch.cat((feature.expand_as(query_neighbor), query_neighbor), dim=1)  # Bx2CxMxK
        for layer in self.layers_after:
            y = layer(y, epoch)
        feature, _ = torch.max(y, dim=3, keepdim=False)  # BxCxM

        return feature


class PointNet(nn.Module):
    def __init__(self, in_channels, out_channels_list, activation, normalization, momentum=0.1,
                 bn_momentum_decay_step=None, bn_momentum_decay=1, output_init_radius=None):
        super(PointNet, self).__init__()

        self.layers = nn.ModuleList()
        previous_out_channels = in_channels
        for i, c_out in enumerate(out_channels_list):
            if i != len(out_channels_list) - 1:
                self.layers.append(EquivariantLayer(previous_out_channels, c_out, activation, normalization,
                                                    momentum, bn_momentum_decay_step, bn_momentum_decay))
            else:
                self.layers.append(EquivariantLayer(previous_out_channels, c_out, None, None))
            previous_out_channels = c_out

        if output_init_radius is not None:
            self.layers[len(out_channels_list) - 1].conv.bias.data.uniform_(-1 * output_init_radius, output_init_radius)

    def forward(self, x, epoch=None):
        for layer in self.layers:
            x = layer(x, epoch)
        return x


## ---------- pose and coefficients branch ----------
class InstanceBranch(nn.Module):
    def __init__(self, in_channels, out_channels_list, num_basis, activation, normalization, momentum=0.1,
                 bn_momentum_decay_step=None, bn_momentum_decay=1, output_init_radius=None):
        super(InstanceBranch, self).__init__()
        # like the PointNet module but with fc
        self.num_basis = num_basis

        self.layers = nn.ModuleList()
        previous_out_channels = in_channels
        for i, c_out in enumerate(out_channels_list):
            if i != len(out_channels_list) - 1:
                self.layers.append(EquivariantLayer(previous_out_channels, c_out, activation, normalization,
                                                    momentum, bn_momentum_decay_step, bn_momentum_decay))
            else:
                self.layers.append(EquivariantLayer(previous_out_channels, c_out, None, None))
            previous_out_channels = c_out

        if output_init_radius is not None:
            self.layers[len(out_channels_list) - 1].conv.bias.data.uniform_(-1 * output_init_radius, output_init_radius)

    def forward(self, x, epoch=None):
        for layer in self.layers:
            x = layer(x, epoch)  # Bx64x300 / Bx128x300 / Bx10x300

        x = torch.max(x, 2, keepdim=True)[0]  # Bx10x1
        x = x.view(-1, x.shape[1])  # Bx10

        return x

## ---------- joint branch ----------

class JointBranch(nn.Module):
    """PointNet-like structure"""
    def __init__(self, in_channels, first_out_channels_list, second_out_channels_list,
                 activation, normalization, momentum=0.1,
                 bn_momentum_decay_step=None, bn_momentum_decay=1, output_init_radius=None):
        super(JointBranch, self).__init__()

        self.first_layers = nn.ModuleList()
        self.second_layers = nn.ModuleList()
        previous_out_channels = in_channels
        for i, c_out in enumerate(first_out_channels_list):
            if i != len(first_out_channels_list) - 1:
                self.first_layers.append(EquivariantLayer(previous_out_channels, c_out, activation, normalization,
                                                          momentum, bn_momentum_decay_step, bn_momentum_decay))
            else:
                self.first_layers.append(EquivariantLayer(previous_out_channels, c_out, None, None))
            previous_out_channels = c_out

        if output_init_radius is not None:
            self.first_layers[len(first_out_channels_list) - 1].conv.bias.data.uniform_(-1 * output_init_radius, output_init_radius)

        previous_out_channels = first_out_channels_list[-1]
        for i, c_out in enumerate(second_out_channels_list):
            if i != len(second_out_channels_list) - 1:
                self.second_layers.append(EquivariantLayer(previous_out_channels, c_out, activation, normalization,
                                                          momentum, bn_momentum_decay_step, bn_momentum_decay))
            else:
                self.second_layers.append(EquivariantLayer(previous_out_channels, c_out, None, None))
            previous_out_channels = c_out

        if output_init_radius is not None:
            self.second_layers[len(second_out_channels_list) - 1].conv.bias.data.uniform_(-1 * output_init_radius, output_init_radius)

    def forward(self, x, epoch=None):
        for layer in self.first_layers:
            x = layer(x, epoch)
        x = torch.max(x, 2, keepdim=True)[0]
        for layer in self.second_layers:
            x = layer(x, epoch)
        return x


class JointBranch2(nn.Module):
    """MLP-like structure"""
    def __init__(self, in_channels, out_channels_list, activation, normalization, momentum=0.1,
                 bn_momentum_decay_step=None, bn_momentum_decay=1, output_init_radius=None):
        super(JointBranch2, self).__init__()

        self.layers = nn.ModuleList()
        previous_out_channels = in_channels
        for i, c_out in enumerate(out_channels_list):
            if i != len(out_channels_list) - 1:
                self.layers.append(EquivariantLayer(previous_out_channels, c_out, activation, normalization,
                                                    momentum, bn_momentum_decay_step, bn_momentum_decay))
            else:
                self.layers.append(EquivariantLayer(previous_out_channels, c_out, None, None))
            previous_out_channels = c_out

        if output_init_radius is not None:
            self.layers[len(out_channels_list) - 1].conv.bias.data.uniform_(-1 * output_init_radius, output_init_radius)

    def forward(self, x, epoch=None):
        for layer in self.layers:
            x = layer(x, epoch)
        return x