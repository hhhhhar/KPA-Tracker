import torch
import torch.nn as nn
import torch.nn.functional as F


class PartChamferLoss_Brute(nn.Module):
    def __init__(self, num_parts):
        super(PartChamferLoss_Brute, self).__init__()
        self.dimension = 3
        self.num_parts = num_parts

    def forward(self, part_pc_src_input, part_pc_dst_input):
        '''
        :param part_pc_src_input: BxKX3x(M/K) Tensor in GPU
        :param part_pc_dst_input: BxKX3x(N/k) Tensor in GPU
        :return:
        '''
        chamfer_pure_list = []
        for part_idx in range(self.num_parts):
            pc_src_input = part_pc_src_input[:, part_idx, :, :]
            pc_dst_input = part_pc_dst_input[:, part_idx, :, :]
            B, M = pc_src_input.size()[0], pc_src_input.size()[2]
            N = pc_dst_input.size()[2]

            pc_src_input_expanded = pc_src_input.unsqueeze(3).expand(B, 3, M, N)
            pc_dst_input_expanded = pc_dst_input.unsqueeze(2).expand(B, 3, M, N)

            # the gradient of norm is set to 0 at zero-input. There is no need to use custom norm anymore.
            diff = torch.norm(pc_src_input_expanded - pc_dst_input_expanded, dim=1, keepdim=False)  # BxMxN

            # pc_src vs selected pc_dst, M
            src_dst_min_dist, _ = torch.min(diff, dim=2, keepdim=False)  # BxM
            forward_loss = src_dst_min_dist.mean()

            # pc_dst vs selected pc_src, N
            dst_src_min_dist, _ = torch.min(diff, dim=1, keepdim=False)  # BxN
            backward_loss = dst_src_min_dist.mean()

            chamfer_pure = forward_loss + backward_loss
            chamfer_pure_list.append(chamfer_pure)

        chamfer_all = torch.mean(torch.stack(chamfer_pure_list))
        return chamfer_all


def smooth_l1_loss(pred, target, beta=1.0, reduction='mean'):
    assert reduction in ('mean', 'none')
    assert beta > 0
    assert pred.size() == target.size()
    if target.numel() == 0:
        return pred * 0.
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
                       diff - 0.5 * beta)
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'none':
        return loss


class CoverageLoss(nn.Module):
    """Directly calculate loss based on max and min, not scale"""
    def __init__(self, beta=0.1, use_relative_coverage=False):
        super(CoverageLoss, self).__init__()
        self.beta = beta
        self.use_relative_coverage = use_relative_coverage

    def forward(self, kp, pc):
        if self.use_relative_coverage:
            # volume
            val_max_pc, _ = torch.max(pc, 2)
            val_min_pc, _ = torch.min(pc, 2)

            val_max_kp, _ = torch.max(kp, 2)
            val_min_kp, _ = torch.min(kp, 2)

            scale_pc = val_max_pc - val_min_pc
            scale_kp = val_max_kp - val_min_kp

            cov_loss = (smooth_l1_loss(val_max_kp / scale_pc, val_max_pc / scale_pc, beta=self.beta) +
                        smooth_l1_loss(val_min_kp / scale_pc, val_min_pc / scale_pc, beta=self.beta) +
                        smooth_l1_loss(torch.log(scale_kp), torch.log(scale_pc), beta=self.beta)
                        ) / 3
        else:
            # volume
            val_max_pc, _ = torch.max(pc, 2)
            val_min_pc, _ = torch.min(pc, 2)

            val_max_kp, _ = torch.max(kp, 2)
            val_min_kp, _ = torch.min(kp, 2)

            cov_loss = (smooth_l1_loss(val_max_kp, val_max_pc, beta=self.beta) +
                        smooth_l1_loss(val_min_kp, val_min_pc, beta=self.beta))/2
        return cov_loss


class SurfaceLoss(nn.Module):
    def __init__(self):
        super(SurfaceLoss, self).__init__()
        self.single_side_chamfer = SingleSideChamferLoss_Brute()

    def forward(self, keypoint, pc):
        loss = self.single_side_chamfer(keypoint, pc)

        return torch.mean(loss)


class SingleSideChamferLoss_Brute(nn.Module):
    def __init__(self):
        super(SingleSideChamferLoss_Brute, self).__init__()

    def forward(self, pc_src_input, pc_dst_input):
        '''
        :param pc_src_input: Bx3xM Variable in GPU
        :param pc_dst_input: Bx3xN Variable in GPU
        :return:
        '''

        B, M = pc_src_input.size()[0], pc_src_input.size()[2]
        N = pc_dst_input.size()[2]

        pc_src_input_expanded = pc_src_input.unsqueeze(3).expand(B, 3, M, N)
        pc_dst_input_expanded = pc_dst_input.unsqueeze(2).expand(B, 3, M, N)

        diff = torch.norm(pc_src_input_expanded - pc_dst_input_expanded, dim=1, keepdim=False)  # BxMxN

        # pc_src vs selected pc_dst, M
        src_dst_min_dist, _ = torch.min(diff, dim=2, keepdim=False)  # BxM

        return src_dst_min_dist


class JointLoss(nn.Module):
    def __init__(self, loc_weight=1.0, axis_weight=0.5):
        super(JointLoss, self).__init__()
        self.loc_weight = loc_weight
        self.axis_weight = axis_weight

    def forward(self, pred_joint_loc, pred_joint_axis, gt_joint_loc, gt_joint_axis):
        loss_axis = 1 - F.cosine_similarity(pred_joint_axis, gt_joint_axis, dim=-1).mean(dim=-1).mean(dim=-1)

        norm_gt_joint_axis = gt_joint_axis / torch.norm(gt_joint_axis, dim=-1, keepdim=True)
        p = gt_joint_loc
        q = gt_joint_loc + norm_gt_joint_axis
        r = pred_joint_loc
        x = p - q
        loss_loc = torch.norm(((r-q)*x).sum(-1, keepdim=True)/(x*x).sum(-1, keepdim=True)*(p-q)+(q-r), dim=-1).mean(-1).mean(-1)

        loss_joint = loss_loc * self.loc_weight + loss_axis * self.axis_weight
        return loss_joint


class Loss_OMAD_PriorNet(torch.nn.Module):
    def __init__(self,
                 device,
                 num_kp, num_parts, num_cate=5,
                 loss_chamfer_weight=1.0,
                 loss_coverage_weight=1.0,
                 loss_surface_weight=5.0,
                 loss_joint_weight=1.0,
                 loss_reg_weight=0.01,
                 loss_sep_weight=2.0,
                 sep_factor=8,
                 beta=0.1,
                 joint_type='revolute',
                 use_relative_coverage=False,
                 ):
        super(Loss_OMAD_PriorNet, self).__init__()
        self.num_kp = num_kp
        self.num_cate = num_cate
        self.num_parts = num_parts
        self.num_joints = num_parts - 1
        assert self.num_kp % self.num_parts == 0
        self.num_kp_per_part = self.num_kp // self.num_parts
        self.joint_type = joint_type
        self.device = device
        assert joint_type in ('revolute', 'prismatic')

        self.loss_chamfer_weight = loss_chamfer_weight
        self.loss_coverage_weight = loss_coverage_weight
        self.loss_surface_weight = loss_surface_weight
        self.loss_joint_weight = loss_joint_weight
        self.loss_reg_weight = loss_reg_weight
        self.loss_sep_weight = loss_sep_weight
        self.sep_factor = sep_factor
        self.use_relative_coverage = use_relative_coverage

        self.part_chamfer_criteria = PartChamferLoss_Brute(num_parts=num_parts)
        self.surface_criteria = SurfaceLoss()
        self.coverage_criteria = CoverageLoss(beta=beta, use_relative_coverage=use_relative_coverage)
        self.joint_criteria = JointLoss(loc_weight=0. if self.joint_type == 'prismatic' else 1.0)

        self.zeros = torch.tensor(
            [0.0 for _ in range(self.num_kp_per_part - 1) for _ in range(self.num_kp_per_part)]).to(
            device)
        self.select1 = torch.tensor(
            [i for _ in range(self.num_kp_per_part - 1) for i in range(self.num_kp_per_part)]).to(
            device)
        self.select2 = torch.tensor([(i % self.num_kp_per_part) for j in range(1, self.num_kp_per_part)
                                     for i in range(j, j + self.num_kp_per_part)]).to(device)

    def forward(self, coefs, part_pred_kps, part_nodes, cloud, cloud_cls,
                pred_joint_loc, pred_joint_axis, gt_joint_loc, gt_joint_axis, gt_part_scale):
        loss_dict = dict()
        bs = part_pred_kps.size(0)

        part_pred_kps_trans = part_pred_kps.transpose(2, 3)  # (B, K, 3, M/K)
        part_nodes_trans = part_nodes.transpose(2, 3)  # (B, K, 3, M/K)
        nodes_trans = part_nodes.reshape(bs, -1, 3).transpose(1, 2)  # (B, 3, M）
        cloud_trans = cloud.transpose(1, 2)  # (B, 3, N)
        chf_loss = self.part_chamfer_criteria(part_pred_kps_trans, part_nodes_trans)

        cov_loss_list = []
        sep_loss_list = []
        for part_idx in range(self.num_parts):
            for bs_idx in range(bs):
                singe_part_nodes_trans = part_nodes_trans[bs_idx, part_idx, :, :].unsqueeze(0)  # (1, 3, m')
                cloud_inds = (cloud_cls[bs_idx] == part_idx)  # (N)
                single_part_cloud_trans = cloud_trans[bs_idx, :, cloud_inds].unsqueeze(0)  # (1, 3, n')
                part_cov_loss = self.coverage_criteria(singe_part_nodes_trans, single_part_cloud_trans)
                cov_loss_list.append(part_cov_loss)

            """Separation Loss"""
            max_rad = torch.norm(gt_part_scale[:, part_idx, :], dim=-1)
            max_dist = max_rad / self.sep_factor
            max_thr = max_dist.unsqueeze(1)  # (bs, 1)

            pred_kp_select1 = torch.index_select(part_nodes[:, part_idx, :, :], 1, self.select1).contiguous()
            pred_kp_select2 = torch.index_select(part_nodes[:, part_idx, :, :], 1, self.select2).contiguous()
            dist_sep = torch.norm((pred_kp_select1 - pred_kp_select2), dim=2)
            loss_sep = torch.max(self.zeros.reshape(1, -1).expand(dist_sep.shape), max_thr - dist_sep)
            loss_sep = torch.mean(loss_sep)

            sep_loss_list.append(loss_sep)

        cov_loss = torch.mean(torch.stack(cov_loss_list))
        sep_loss = torch.mean(torch.stack(sep_loss_list))
        surf_loss = self.surface_criteria(nodes_trans, cloud_trans)
        joint_loss = self.joint_criteria(pred_joint_loc, pred_joint_axis, gt_joint_loc, gt_joint_axis)
        reg_loss = (coefs * coefs).mean()

        """SUM UP"""
        loss = chf_loss * self.loss_chamfer_weight + cov_loss * self.loss_coverage_weight + \
               surf_loss * self.loss_surface_weight + joint_loss * self.loss_joint_weight + \
               reg_loss * self.loss_reg_weight + sep_loss * self.loss_sep_weight
        score = loss.item()
        loss_dict['loss_chamfer'] = chf_loss.item()
        loss_dict['loss_coverage'] = cov_loss.item()
        loss_dict['loss_surface'] = surf_loss.item()
        loss_dict['loss_joint'] = joint_loss.item()
        loss_dict['loss_reg'] = reg_loss.item()
        loss_dict['loss_sep'] = sep_loss.item()

        return loss, score, loss_dict
