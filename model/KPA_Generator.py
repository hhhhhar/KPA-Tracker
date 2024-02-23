import torch
import torch.nn as nn
import numpy as np
import sys
from torch.utils.tensorboard import SummaryWriter
sys.path.append("..")  ## 这个
import os
os.chdir(sys.path[0])
from model.layers import PointNet, GeneralKNNFusionModule, EquivariantLayer, InstanceBranch, JointBranch2
import index_max


class KPA_Generator(nn.Module):
    def __init__(self,
                 surface_normal_len=0,
                 basis_num=10,
                 node_num=16,
                 part_num=2,
                 k=1,
                 node_knn_k_1=3,
                 symtype='shape',
                 device=torch.device('cuda:0'),
                 init_n_pl=None,
                 init_basis=None
                 ):
        super(KPA_Generator, self).__init__()
        self.surface_normal_len = surface_normal_len
        self.basis_num = basis_num
        self.node_num = node_num
        self.part_num = part_num
        self.joint_num = part_num - 1
        self.k = k
        self.node_knn_k_1 = node_knn_k_1
        self.symtype = symtype  # defines the symmetric deformation space "shape" or "basis" (default: {"shape"})
        assert self.symtype in ('shape', 'basis', 'none')
        self.device = device
        assert self.node_num % self.part_num == 0, 'node number should be devided by part number'
        if self.symtype == 'shape':
            assert self.node_num % 2 == 0, 'node number must be an even number'
            assert (self.node_num // 2) % self.part_num == 0

        # ---- Nodes branch definition ----

        self.C1 = 128
        self.C2 = 512
        input_channels = self.C1 + self.C2
        output_channels = 4  # 3 coordinates + sigma
        assert self.node_knn_k_1 >= 2

        self.first_pointnet = PointNet(3 + self.surface_normal_len,
                                       [int(self.C1 / 2), int(self.C1 / 2), int(self.C1 / 2)],
                                       activation='relu',
                                       normalization='batch',
                                       momentum=0.1,
                                       bn_momentum_decay_step=None,
                                       bn_momentum_decay=1.0)

        self.second_pointnet = PointNet(self.C1, [self.C1, self.C1],
                                        activation='relu',
                                        normalization='batch',
                                        momentum=0.1,
                                        bn_momentum_decay_step=None,
                                        bn_momentum_decay=1.0)

        self.knnlayer_1 = GeneralKNNFusionModule(3 + self.C1, (int(self.C2 / 2), int(self.C2 / 2), int(self.C2 / 2)),
                                                 (self.C2, self.C2),
                                                 activation='relu',
                                                 normalization='batch',
                                                 momentum=0.1,
                                                 bn_momentum_decay_step=None,
                                                 bn_momentum_decay=1.0)

        self.node_mlp1 = EquivariantLayer(input_channels, 512,
                                          activation='relu', normalization='batch',
                                          momentum=0.1,
                                          bn_momentum_decay_step=None,
                                          bn_momentum_decay=1.0)

        self.node_mlp2 = EquivariantLayer(512, 256,
                                          activation='relu', normalization='batch',
                                          momentum=0.1,
                                          bn_momentum_decay_step=None,
                                          bn_momentum_decay=1.0)

        self.node_mlp3 = EquivariantLayer(256, output_channels, activation=None, normalization=None)

        # ---- Joint branch defination ---
        self.C0 = 64
        self.joint_net = JointBranch2(self.basis_num, [self.C0, self.C0, self.joint_num * 6],
                                      activation='relu', normalization='batch',
                                      momentum=0.1,
                                      bn_momentum_decay_step=None,
                                      bn_momentum_decay=1.0)


        # ---- Pose and coefficients branch definition ----

        self.third_pointnet2 = InstanceBranch(3 + self.surface_normal_len,
                                              [int(self.C1 / 2), self.C1, self.basis_num],
                                              self.basis_num,
                                              activation='relu',
                                              normalization='batch',
                                              momentum=0.1,
                                              bn_momentum_decay_step=None,
                                              bn_momentum_decay=1.0)

        # ---- Additional learnable parameters ----
        if self.symtype == 'shape':
            self.basis = torch.nn.Parameter(
                (torch.rand(1, 3, self.node_num // 2, self.basis_num) - 0.5).to(device), requires_grad=True)  # 1x3xM/2xK
        elif self.symtype == 'basis':
            raise NotImplementedError
        elif self.symtype == 'none':
            if init_basis is not None:
                self.basis = torch.nn.Parameter(init_basis.to(device))  # 1x3xMxK
            else:
                self.basis = torch.nn.Parameter((torch.rand(1, 3, self.node_num, self.basis_num) - 0.5).to(device))  # 1x3xMxK
        if init_n_pl is None:
            self.n_pl = torch.nn.Parameter(torch.rand(1, 2).to(device), requires_grad=True)
        else:
            self.n_pl = torch.nn.Parameter(torch.tensor(init_n_pl).to(device), requires_grad=True)

    def forward(self, x, sn, node, epoch=None):
        '''
        :param x: BxNx3 Tensor
        :param sn: BxNx3 Tensor
        :param node: BxMx3 FloatTensor
        :return:
        '''
        bs = x.size(0)
        x = x.transpose(1, 2)
        if sn is not None:
            sn = sn.transpose(1, 2)
        node = node.transpose(1, 2)
        # modify the x according to the nodes, minus the center
        mask, mask_row_max, min_idx = self.query_topk(node, x, node.size()[2],
                                                     k=self.k)  # BxkNxnode_num, Bxnode_num, BxkN
        mask_row_sum = torch.sum(mask, dim=1)  # Bxnode_num
        mask = mask.unsqueeze(1)  # Bx1xkNxnode_num

        # if necessary, stack the x
        x_stack = x.repeat(1, 1, self.k)
        if self.surface_normal_len >= 1:
            sn_stack = sn.repeat(1, 1, self.k)

        x_stack_data_unsqueeze = x_stack.unsqueeze(3)  # BxCxkNx1
        x_stack_data_masked = x_stack_data_unsqueeze * mask.float()  # BxCxkNxnode_num
        cluster_mean = torch.sum(x_stack_data_masked, dim=2) / (
                    mask_row_sum.unsqueeze(1).float() + 1e-5).detach()  # BxCxnode_num
        som_node_cluster_mean = cluster_mean

        B, N, kN, M = x.size()[0], x.size()[2], x_stack.size()[2], som_node_cluster_mean.size()[2]

        # assign each point with a center
        node_expanded = som_node_cluster_mean.unsqueeze(2)  # BxCx1xnode_num, som.node is BxCxnode_num
        centers = torch.sum(mask.float() * node_expanded, dim=3).detach()  # BxCxkN

        x_decentered = (x_stack - centers).detach()  # Bx3xkN
        if self.surface_normal_len >= 1:
            x_augmented = torch.cat((x_decentered, sn_stack), dim=1)  # Bx6xkN

        # ---- Nodes branch ----

        # First PointNet
        if self.surface_normal_len >= 1:
            first_pn_out = self.first_pointnet(x_augmented, epoch)
        else:
            first_pn_out = self.first_pointnet(x_decentered, epoch)

        with torch.cuda.device(first_pn_out.get_device()):
            first_gather_index = index_max.forward_cuda_shared_mem(first_pn_out.detach(), min_idx.int(),
                                                                   M).detach().long()
        first_pn_out_masked_max = first_pn_out.gather(dim=2, index=first_gather_index) * mask_row_max.unsqueeze(
            1).float()  # BxCxM

        # scatter the masked_max back to the kN points
        scattered_first_masked_max = torch.gather(first_pn_out_masked_max,
                                                  dim=2,
                                                  index=min_idx.unsqueeze(1).expand(B, first_pn_out.size()[1],
                                                                                    kN))  # BxCxkN
        first_pn_out_fusion = torch.cat((first_pn_out, scattered_first_masked_max), dim=1)  # Bx2CxkN

        # Second PointNet
        second_pn_out = self.second_pointnet(first_pn_out_fusion, epoch)

        with torch.cuda.device(second_pn_out.get_device()):
            second_gather_index = index_max.forward_cuda_shared_mem(second_pn_out, min_idx.int(), M).detach().long()
        second_pn_out_masked_max = second_pn_out.gather(dim=2, index=second_gather_index) * mask_row_max.unsqueeze(
            1).float()  # BxCxM

        # knn search on nodes
        knn_feature_1 = self.knnlayer_1(query=som_node_cluster_mean,
                                        database=som_node_cluster_mean,
                                        x=second_pn_out_masked_max,
                                        K=self.node_knn_k_1,
                                        epoch=epoch)
        node_feature_aggregated = torch.cat((second_pn_out_masked_max, knn_feature_1), dim=1)  # Bx(C1+C2)xM

        # mlp to calculate the per-node keypoint
        y = self.node_mlp1(node_feature_aggregated)
        point_descriptor = self.node_mlp2(y)
        keypoint_sigma = self.node_mlp3(point_descriptor)  # Bx(3+1)xkN
        nodes = keypoint_sigma[:, 0:3, :] + som_node_cluster_mean  # Bx3xM
        nodes = nodes.transpose(1, 2).contiguous()  # BxMx3
        part_nodes = nodes.reshape(bs, self.part_num, -1, 3)  # (B, K, M/K, 3)

        # -- Pose and coefficients branch --
        if self.surface_normal_len >= 1:
            x_init_augmented = torch.cat((x_stack, sn_stack), dim=1)
            coefs = self.third_pointnet2(x_init_augmented, epoch)
        else:
            coefs = self.third_pointnet2(x_stack, epoch)

        coefs_expand = coefs.clone().unsqueeze(2)
        joint_params = self.joint_net(coefs_expand).reshape(B, self.joint_num, 6)
        joint_loc = joint_params[:, :, :3]
        joint_axis = joint_params[:, :, 3:]

        return part_nodes, coefs, joint_loc, joint_axis

    def get_transformed_pred_keypoints(self, c, gt_r, gt_t):
        """The category-specific symmetric 3D keypoints are computed with the deformation function.
        (transformed based on gt_r and gt_t)

        Arguments:
            c {torch.Tensor} -- predicted def coefficients - BxK
            gt_r {torch.Tensor} -- ground truth rotation - Bx3x3
            gt_t {torch.Tensor} -- ground truth translation - Bx1x3

        Returns:
            torch.Tensor -- kpts: category-specific symmetric 3D keypoints - BXpart_numx(M/part_num)X3
        """
        refl_mat = self.get_reflection_operator(self.n_pl)
        if self.symtype != "none":
            basis_half = self.basis
        else:
            basis = self.basis
        c = c.unsqueeze_(1).unsqueeze_(1)  # (B, 1, 1, K)
        bs = c.size(0)

        if self.symtype == "shape":
            refl_batch = refl_mat.repeat(c.shape[0], 1, 1)  # (B, 3, 3)
            kpts_half = torch.sum(c * basis_half, 3)  # (B, 3, M/2)
            kpts_half_reflected = torch.matmul(refl_batch, kpts_half)  # (B, 3, M/2)

            part_kpts_half = kpts_half.reshape(bs, 3, self.part_num, -1)  # (B, 3, part_num, M/2/part_num)
            part_kpts_half_reflected = kpts_half_reflected.reshape(bs, 3, self.part_num, -1)  # (B, 3, part_num, M/2/part_num)
            part_kpts = torch.cat((part_kpts_half, part_kpts_half_reflected), dim=-1)  # (B, 3, part_num, M/part_num)
            kpts = part_kpts.reshape(bs, 3, -1)  # (B, 3, M)
        elif self.symtype == "basis":
            raise NotImplementedError
        elif self.symtype == "none":
            kpts = torch.sum(c * basis, 3)  # (B, 3, M)
        else:
            raise NotImplementedError

        kpts = torch.bmm(gt_r, kpts).transpose(1, 2) + gt_t  # (BxMx3)
        part_kpts = kpts.reshape(bs, self.part_num, -1, 3)  # (B, part_num, M/part_num, 3)
        return part_kpts

    @staticmethod
    def get_transformed_joint_params(pred_joint_loc, pred_joint_axis, gt_r, gt_t):
        """
        transform predicted joint params based on gt_r and gt_t
        :param pred_joint_loc: joint location, BxJx3 Tensor
        :param pred_joint_axis: joint axis, BxJx3 Tensor
        :param gt_r: ground truth rotation matrix, Bx3x3 Tensor
        :param gt_t: ground truth translation, Bx1x3 Tensor
        :return:
            trans_joint_loc: transformed joint location
            trans_joint_axis: transformed joint axis
        """
        trans_joint_loc = torch.bmm(gt_r, pred_joint_loc.transpose(1, 2)).transpose(1, 2) + gt_t
        trans_joint_axis = torch.bmm(gt_r, pred_joint_axis.transpose(1, 2)).transpose(1, 2) + gt_t
        return trans_joint_loc, trans_joint_axis

    @staticmethod
    def get_reflection_operator(n_pl):
        """ The reflection operator is parametrized by the normal vector
        of the plane of symmetry passing through the origin. """
        norm_npl = torch.norm(n_pl, 2)
        n_x = n_pl[0, 0] / norm_npl  # torch.tensor(1.0).cuda()
        n_y = torch.tensor(0.0, device=n_pl.device)
        n_z = n_pl[0, 1] / norm_npl
        refl_mat = torch.stack(
            [
                1 - 2 * n_x * n_x,
                -2 * n_x * n_y,
                -2 * n_x * n_z,
                -2 * n_x * n_y,
                1 - 2 * n_y * n_y,
                -2 * n_y * n_z,
                -2 * n_x * n_z,
                -2 * n_y * n_z,
                1 - 2 * n_z * n_z,
            ],
            dim=0,
        ).reshape(1, 3, 3)

        return refl_mat

    @staticmethod
    def query_topk(node, x, M, k):
        '''
        :param node: SOM node of BxCxM tensor
        :param x: input data BxCxN tensor
        :param M: number of SOM nodes
        :param k: topk
        :return: mask: Nxnode_num
        '''
        # ensure x, and other stored tensors are in the same device
        device = x.device
        node = node.to(x.device)
        node_idx_list = torch.from_numpy(np.arange(M).astype(np.int64)).to(device)  # node_num LongTensor

        # expand as BxCxNxnode_num
        node = node.unsqueeze(2).expand(x.size(0), x.size(1), x.size(2), M)
        x_expanded = x.unsqueeze(3).expand_as(node)

        # calcuate difference between x and each node
        diff = x_expanded - node  # BxCxNxnode_num
        diff_norm = (diff ** 2).sum(dim=1)  # BxNxnode_num

        # find the nearest neighbor
        _, min_idx = torch.topk(diff_norm, k=k, dim=2, largest=False, sorted=False)  # BxNxk
        min_idx_expanded = min_idx.unsqueeze(2).expand(min_idx.size()[0], min_idx.size()[1], M, k)  # BxNxnode_numxk

        node_idx_list = node_idx_list.unsqueeze(0).unsqueeze(0).unsqueeze(3).expand_as(
            min_idx_expanded).long()  # BxNxnode_numxk
        mask = torch.eq(min_idx_expanded, node_idx_list).int()  # BxNxnode_numxk
        # mask = torch.sum(mask, dim=3)  # BxNxnode_num

        # debug
        B, N, M = mask.size()[0], mask.size()[1], mask.size()[2]
        mask = mask.permute(0, 2, 3, 1).contiguous().view(B, M, k * N).permute(0, 2,
                                                                               1).contiguous()  # BxMxkxN -> BxMxkN -> BxkNxM
        min_idx = min_idx.permute(0, 2, 1).contiguous().view(B, k * N)

        mask_row_max, _ = torch.max(mask, dim=1)  # Bxnode_num, this indicates whether the node has nearby x

        return mask, mask_row_max, min_idx
