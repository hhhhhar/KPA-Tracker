import argparse
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
import sys
sys.path.append("..")  ## 这个
import os
import os.path as osp
os.chdir(sys.path[0])
from model.pointnet_plus_cuda import PointNetPlus
from model.layers import EquivariantLayer, JointBranch2

from IPython import embed




class KPA_Tracker(nn.Module):
    def __init__(self, device,
                 params_dict,
                 num_points,
                 num_kp,
                 num_parts=2,
                 init_dense_soft_factor=1.0,
                 num_basis=10,
                 symtype='shape',
                 use_attention=True,
                 use_normal=False,
                 use_background=False):
        super(KPA_Tracker, self).__init__()
        self.device = device
        assert 'basis' in params_dict and 'n_pl' in params_dict and 'joint_net' in params_dict
        self.num_points = num_points
        self.num_parts = num_parts
        self.num_joints = num_parts - 1
        self.num_basis = num_basis
        self.symtype = symtype
        assert symtype in ('shape', 'none')
        self.num_kp = num_kp
        self.num_parts = num_parts
        self.num_classes = self.num_parts + 1 if use_background else self.num_parts
        self.use_background = use_background
        self.num_kp_per_part = self.num_kp // self.num_parts
        assert self.num_kp % self.num_parts == 0
        self.dense_soft_factor = init_dense_soft_factor
        self.use_attention = use_attention
        self.use_normal = use_normal

        if self.use_normal:
            self.pointnet = PointNetPlus(in_channel=6)
        else:
            self.pointnet = PointNetPlus()
        self.pointnet = self.pointnet.to(device)

        self.cls_1 = EquivariantLayer(128, 64, activation='relu', normalization='batch')
        self.cls_2 = EquivariantLayer(64, self.num_classes, activation=None, normalization=None)

        self.kp_1 = EquivariantLayer(128, 64, activation='relu', normalization='batch')
        self.kp_2 = EquivariantLayer(64, 3 * num_kp, activation=None, normalization=None)

        self.kp_att_1 = EquivariantLayer(128, 64, activation='relu', normalization='batch')
        self.kp_att_2 = EquivariantLayer(64, num_kp, activation=None, normalization=None)

        self.conv1_r = EquivariantLayer(128, 128, activation='relu', normalization='batch')
        self.conv2_r = EquivariantLayer(128, 64, activation='relu', normalization='batch')
        self.conv3_r = EquivariantLayer(64, 4, activation=None, normalization=None)  # quaternion (w, x, y, z)

        self.conv1_t = EquivariantLayer(128, 64, activation='relu', normalization='batch')
        self.conv2_t = EquivariantLayer(64, 3, activation=None, normalization=None)  # translation (x, y, z)

        self.joint_state_1 = EquivariantLayer(128, 64, activation='relu', normalization='batch')
        # self.joint_state_2 = EquivariantLayer(64, 8, activation=None, normalization=None)
        self.joint_state_2 = EquivariantLayer(64, self.num_joints, activation=None, normalization=None)

        self.joint_att_1 = EquivariantLayer(128, 64, activation='relu', normalization='batch')
        self.joint_att_2 = EquivariantLayer(64, self.num_joints, activation=None, normalization=None)

        self.beta_1 = EquivariantLayer(128, 128, activation='relu', normalization='batch')
        self.beta_2 = EquivariantLayer(128, 64, activation='relu', normalization='batch')
        self.beta_3 = EquivariantLayer(64, self.num_basis, activation=None, normalization=None)

        # Additional parameters(loaded)
        self.basis = params_dict['basis'].clone().detach().to(self.device)
        self.n_pl = params_dict['n_pl'].clone().detach().to(self.device)
        self.joint_net = JointBranch2(self.num_basis, [64, 64, self.num_joints * 6],
                                      activation='relu', normalization='batch')
        for param in self.joint_net.parameters():
            param.requires_grad = False
        self.joint_net.load_state_dict(params_dict['joint_net'])

    def set_device(self, device):
        self.device = device
        self.basis = self.basis.to(device)
        self.n_pl = self.n_pl.to(device)

    def forward(self, pts_all, cls_label=None):
        self.joint_net.eval()  # freeze BN
        bs, n, dim = pts_all.shape
        if dim > 3 and self.use_normal:
            pts = pts_all[:, :, :3]
            feat = pts_all[:, :, 3:]
        else:
            pts = pts_all[:, :, :3]
            feat = None
        # TODO pointnet
        dense_pts_feat = self.pointnet(pts, feat).contiguous()  # (bs, 128, n)
        global_pts_feat = torch.max(dense_pts_feat, dim=-1, keepdim=True)[0]  # (bs, 128, 1)

        dense_cls_score = self.cls_2(self.cls_1(dense_pts_feat)).transpose(1, 2).contiguous()  # (bs, n, K)

        dense_kp_offset = self.kp_2(self.kp_1(dense_pts_feat)).transpose(1, 2).contiguous()  # (bs, n, M*3)
        dense_kp_offset = dense_kp_offset.reshape(bs, n, -1, 3)  # (bs, n, M, 3)

        dense_kp_coords = pts.reshape(bs, n, 1, 3).expand(bs, n, self.num_kp,
                                                          3) + dense_kp_offset  # (bs, n, M, 3)
        if self.use_attention:
            dense_kp_score = self.kp_att_2(self.kp_att_1(dense_pts_feat))\
                .transpose(1, 2).contiguous().reshape(bs, n, -1, 1)  # (bs, n, M, 1)
            dense_kp_prob = F.softmax(dense_kp_score * self.dense_soft_factor, dim=1)  # (bs, n, M, 1)
        else:
            dense_kp_prob = torch.ones_like(dense_kp_coords) / n
        pred_kp = torch.sum(dense_kp_prob * dense_kp_coords, dim=1, keepdim=True)  # (bs, 1, M, 3)
        pred_part_kp = pred_kp.reshape(bs, self.num_parts, -1, 3)  # (bs, K, M/k, 3)

        base_idx = 1 if self.use_background else 0
        if cls_label is None:
            base_global_feat_list = []
            for idx in range(bs):
                base_cls_idxs = torch.argmax(dense_cls_score[idx].detach(), dim=1) == base_idx  # (n, )
                if torch.nonzero(base_cls_idxs).shape[0] == 0:
                    base_global_feat_list.append(torch.zeros(dense_pts_feat.shape[1], 1, device=dense_pts_feat.device))  # (128, 1)
                else:
                    base_global_feat_list.append(torch.mean(dense_pts_feat[idx, :, base_cls_idxs], dim=-1, keepdim=True))  # (128, 1)
            base_global_feat = torch.stack(base_global_feat_list, dim=0)  # (bs, 128, 1)
        else:
            base_global_feat_list = []
            for idx in range(bs):
                base_cls_idxs = cls_label[idx] == base_idx  # (n, )
                if torch.nonzero(base_cls_idxs).shape[0] == 0:
                    base_global_feat_list.append(torch.zeros(dense_pts_feat.shape[1], 1, device=dense_pts_feat.device))  # (128, 1)
                else:
                    base_global_feat_list.append(
                        torch.mean(dense_pts_feat[idx, :, base_cls_idxs], dim=-1, keepdim=True))  # (128, 1)
            # TODO base_global_feat = zeros
            base_global_feat = torch.stack(base_global_feat_list, dim=0)  # (bs, 128, 1)
        rx = self.conv3_r(self.conv2_r(self.conv1_r(base_global_feat)))  # (bs, 4, 1)

        quat = (rx / torch.norm(rx, dim=1, keepdim=True)).squeeze(2)  # (bs, 4) - (w, x, y, z)
        base_r = torch.cat(((1.0 - 2.0 * (quat[:, 2] ** 2 + quat[:, 3] ** 2)).view(-1, 1),
                          (2.0 * quat[:, 1] * quat[:, 2] - 2.0 * quat[:, 0] * quat[:, 3]).view(-1, 1),
                          (2.0 * quat[:, 0] * quat[:, 2] + 2.0 * quat[:, 1] * quat[:, 3]).view(-1, 1),
                          (2.0 * quat[:, 1] * quat[:, 2] + 2.0 * quat[:, 3] * quat[:, 0]).view(-1, 1),
                          (1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 3] ** 2)).view(-1, 1),
                          (-2.0 * quat[:, 0] * quat[:, 1] + 2.0 * quat[:, 2] * quat[:, 3]).view(-1, 1),
                          (-2.0 * quat[:, 0] * quat[:, 2] + 2.0 * quat[:, 1] * quat[:, 3]).view(-1, 1),
                          (2.0 * quat[:, 0] * quat[:, 1] + 2.0 * quat[:, 2] * quat[:, 3]).view(-1, 1),
                          (1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)).view(-1, 1)),
                         dim=-1).contiguous().reshape(bs, 3, 3)  # (bs, 3, 3)

        tx = self.conv2_t(self.conv1_t(base_global_feat))  # (bs, 3, 1)
        base_t = tx.squeeze(-1)  # (bs, 3)

        # joint_state_all = self.joint_state_3(self.joint_state_2(self.joint_state_1(dense_pts_feat))).transpose(1, 2)  # (bs, n, K-1)
        joint_state_all = self.joint_state_2(self.joint_state_1(dense_pts_feat)).transpose(1, 2)

        joint_conf = F.softmax(self.joint_att_2(self.joint_att_1(dense_pts_feat)), dim=2).transpose(1, 2)  # (bs, n, K-1)
        joint_state = torch.sum(joint_state_all * joint_conf, dim=1)  # (bs, K-1)

        beta = self.beta_3(self.beta_2(self.beta_1(global_pts_feat))).squeeze(-1)  # (bs, basis_num)
        norm_part_kp = self.get_norm_keypoints(beta)  # (bs, K, M/k, 3)
        norm_joint_loc, norm_joint_axis = self.get_norm_joint_params(beta)  # (bs, K-1, 3), (bs, K-1, 3)

        return dense_cls_score, pred_part_kp, quat, base_r, base_t, joint_state, \
               beta, norm_part_kp, norm_joint_loc, norm_joint_axis

    def get_norm_keypoints(self, beta):
        """The category-specific symmetric 3D keypoints in normalized(zero-centered) space
        are computed with the deformation function.

        Arguments:
            beta {torch.Tensor} -- predicted def coefficients - Bxbasis_num

        Returns:
            torch.Tensor -- kpts: category-specific symmetric 3D keypoints - BXpart_numx(M/part_num)X3
        """
        refl_mat = self.get_reflection_operator(self.n_pl)
        if self.symtype != "none":
            basis_half = self.basis
        else:
            basis = self.basis
        c = beta.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, basis_num)
        bs = c.size(0)

        if self.symtype == "shape":
            refl_batch = refl_mat.repeat(c.shape[0], 1, 1)  # (B, 3, 3)
            kpts_half = torch.sum(c * basis_half, 3)  # (B, 3, M/2)
            kpts_half_reflected = torch.matmul(refl_batch, kpts_half)  # (B, 3, M/2)

            part_kpts_half = kpts_half.reshape(bs, 3, self.num_parts, -1)  # (B, 3, part_num, M/2/part_num)
            part_kpts_half_reflected = kpts_half_reflected.reshape(bs, 3, self.num_parts, -1)  # (B, 3, part_num, M/2/part_num)
            part_kpts = torch.cat((part_kpts_half, part_kpts_half_reflected), dim=-1)  # (B, 3, part_num, M/part_num)
            kpts = part_kpts.reshape(bs, 3, -1)  # (B, 3, M)
        elif self.symtype == "basis":
            raise NotImplementedError
        elif self.symtype == "none":
            kpts = torch.sum(c * basis, 3)  # (B, 3, M)
        else:
            raise NotImplementedError

        part_kpts = kpts.transpose(1, 2).contiguous().reshape(bs, self.num_parts, -1, 3)  # (B, part_num, M/part_num, 3)
        return part_kpts

    def get_norm_joint_params(self, beta):
        """
        get joint params(joint location, joint axis) in normalized(zero-centerd) space in rest state
        :param beta: {torch.Tensor} -- predicted def coefficients - Bxbais_num
        :return:
            norm_joint_loc: {torch.Tensor} -- predicted joint location - Bxjoint_numx3
            norm_joint_axis: {torch.Tensor} -- predicted joint axis(direction) - Bxjoint_numx3
        """
        norm_joint_params = self.joint_net(beta.unsqueeze(-1)).reshape(-1, self.num_joints, 6)
        norm_joint_loc = norm_joint_params[:, :, :3]  # (bs, K-1, 3)
        norm_joint_axis = norm_joint_params[:, :, 3:]  # (bs, K-1, 3)
        return norm_joint_loc, norm_joint_axis

    @staticmethod
    def get_reflection_operator(n_pl):
        """ The reflection operator is parametrized by the normal vector
        of the plane of symmetry passing through the origin. """
        norm_npl = torch.norm(n_pl, 2)
        n_x = n_pl[0, 0] / norm_npl
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=str, default='URDF', help='dataset root dir')
    parser.add_argument('--params_dir', type=str, help='the dir for params and kp annotations')
    parser.add_argument('--resume', type=str, default=None, help='resume model')
    parser.add_argument('--category', type=int, default=1, help='category to train')
    parser.add_argument('--num_points', type=int, default=1024, help='points')
    parser.add_argument('--num_cates', type=int, default=5, help='number of categories')
    parser.add_argument('--num_parts', type=int, default=2, help='number of parts')
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
    parser.add_argument('--num_kp', type=int, default=12, help='number of all keypoints')
    parser.add_argument('--num_basis', type=int, default=10, help='number of shape basis')
    parser.add_argument('--bs', type=int, default=16, help='batch size')
    parser.add_argument('--dense_soft_factor', type=float, default=1.0, help='the factor of dense softmax')
    parser.add_argument('--loc_weight', type=float, default=5.0, help='the weight of pts loc weight')
    parser.add_argument('--base_weight', type=float, default=0.2, help='the weight of base rotation loss')
    parser.add_argument('--cls_weight', type=float, default=1.0, help='the weight of segmentation loss')
    parser.add_argument('--joint_state_weight', type=float, default=5.0, help='the weight of joint state loss')
    parser.add_argument('--shape_weight', type=float, default=3.0, help='the weight of shape loss')
    parser.add_argument('--joint_param_weight', type=float, default=3.0, help='the weight of joint param loss')
    parser.add_argument('--reg_weight', type=float, default=0.01, help='the weight of regularization loss')
    parser.add_argument('--no_att', action='store_true', help='whether to not use attention map')
    parser.add_argument('--symtype', type=str, default='shape', choices=['shape', 'none'], help='the symmetry type')
    parser.add_argument('--work_dir', type=str, default='work_dir/base', help='save dir')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    opt = parser.parse_args()

    device = torch.device("cuda:0")
    # print(osp.join(opt.params_dir, 'params.pth'))
    params_dict = torch.load(osp.join('..', opt.params_dir, 'params.pth'))
    model = KPA_Tracker(device=device, params_dict=params_dict,
                       num_points=opt.num_points, num_kp=opt.num_kp, num_parts=opt.num_parts,
                       init_dense_soft_factor=opt.dense_soft_factor, num_basis=opt.num_basis, symtype=opt.symtype,
                       use_attention=not opt.no_att)
    model = model.to(device)
