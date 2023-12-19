import torch
import torch.nn.functional as F
import numpy as np


class Loss_OMAD_Net(torch.nn.Module):
    def __init__(self, num_key_per_part, num_parts, device, num_cate=5,
                 loss_loc_weight=5.0,
                 loss_cls_weight=1.0,
                 loss_base_weight=0.2,
                 loss_joint_state_weight=5.0,
                 loss_shape_weight=3.0,
                 loss_joint_param_weight=3.0,
                 loss_reg_weight=0.01,
                 joint_type='revolute',
                 use_background=False,
                 use_state_penalty = True
                 ):
        super(Loss_OMAD_Net, self).__init__()
        self.num_key_per_part = num_key_per_part
        self.num_cate = num_cate
        self.num_parts = num_parts
        self.num_classes = self.num_parts + 1 if use_background else self.num_parts
        self.num_joints = num_parts - 1
        self.device = device
        self.joint_type = joint_type
        assert joint_type in ('revolute', 'prismatic')

        self.loss_loc_weight = loss_loc_weight
        self.loss_cls_weight = loss_cls_weight
        self.loss_base_weight = loss_base_weight
        self.loss_joint_state_weight = loss_joint_state_weight
        self.loss_shape_weight = loss_shape_weight
        self.loss_joint_param_weight = loss_joint_param_weight
        self.loss_reg_weight = loss_reg_weight
        self.use_state_penalty = use_state_penalty

    def forward(self,
                pred_trans_part_kp, dense_part_cls_score, pred_base_quat, pred_base_t, 
                pred_norm_part_kp, pred_joint_loc, pred_joint_axis, pred_joint_state, 
                pred_beta, gt_part_cls, gt_part_quat, gt_part_r, gt_part_t, gt_norm_part_kp, 
                gt_joint_loc, gt_joint_axis, gt_joint_state, gt_tran_part_kp):
        losses = []
        loss_dict = dict(loss_loc=[], loss_shape=[])
        loss_base_list = []
        loss_joint_state_list = []
        loss_joint_param_list = []
        for part_idx in range(self.num_parts):
            pred_trans_kp = pred_trans_part_kp[:, part_idx, :, :]
            pred_norm_kp = pred_norm_part_kp[:, part_idx, :, :]
            gt_norm_kp = gt_norm_part_kp[:, part_idx, :, :]
            gt_tran_kp = gt_tran_part_kp[:, part_idx, :, :]
            gt_r = gt_part_r[:, part_idx, :, :]
            gt_inv_r = gt_r.transpose(1, 2)
            gt_t = gt_part_t[:, part_idx, :, :]

            """Location Loss"""
            # inv_pred_kp = torch.bmm(gt_inv_r, (pred_trans_kp - gt_t).transpose(1, 2)).transpose(1, 2).contiguous()
            # loss_loc = torch.mean(torch.mean(torch.norm((inv_pred_kp - gt_norm_kp), dim=2), dim=1))
            loss_loc = torch.mean(torch.mean(torch.norm((pred_trans_kp - gt_tran_kp.contiguous()), dim=2), dim=1))

            """Base Transformation Loss"""
            if part_idx == 0:  # Only count for base part
                loss_base_r = torch.mean(1 - torch.sum(pred_base_quat * gt_part_quat[:, 0, :], dim=-1))
                loss_base_t = torch.mean(torch.norm(pred_base_t - gt_t[:, 0, :], dim=-1))
                loss_base = loss_base_r + 0.5*loss_base_t
                loss_base_list.append(loss_base)

            """Joint State Loss"""
            if part_idx != 0:  # Only count for child part\
                pn_map = pred_joint_state[:, part_idx - 1] * gt_joint_state[:, part_idx] # (bs, )
                diff_joint_state = (pred_joint_state[:, part_idx - 1] - gt_joint_state[:, part_idx]) # / np.pi
                penalty_map = torch.ones_like(diff_joint_state, dtype=torch.float32)
                if self.use_state_penalty:
                    ng_index = torch.nonzero(pn_map <= 0)
                    penalty_map[ng_index] = 20.
                loss_joint = torch.mean(torch.abs(diff_joint_state * penalty_map))
                # print(torch.abs(diff_joint_state * penalty_map))
                # print(torch.mean(torch.abs(diff_joint_state * penalty_map)))
                # input()
                loss_joint_state_list.append(loss_joint)
            # print(f'pred:{pred_joint_state[0, 0]}, gt:{gt_joint_state[0, 1]}\n')

            """Shape Loss"""
            loss_shape = torch.mean(torch.mean(torch.norm(pred_norm_kp - gt_norm_kp, dim=2), dim=-1))

            """Joint Params Loss"""
            if part_idx != 0:  # Only count for child part
                loss_joint_axis = 1 - F.cosine_similarity(pred_joint_axis[:, part_idx - 1, :],
                                                    gt_joint_axis[:, part_idx - 1, :], dim=-1).mean(dim=-1).mean(dim=-1)
                norm_gt_joint_axis = gt_joint_axis[:, part_idx - 1, :] / torch.norm(gt_joint_axis[:, part_idx - 1, :],
                                                                                    dim=-1, keepdim=True)
                if self.joint_type == 'revolute':
                    p = gt_joint_loc[:, part_idx - 1, :]
                    q = gt_joint_loc[:, part_idx - 1, :] + norm_gt_joint_axis
                    r = pred_joint_loc[:, part_idx - 1, :]
                    x = p - q
                    loss_joint_loc = torch.norm(
                        ((r - q) * x).sum(-1, keepdim=True) / (x * x).sum(-1, keepdim=True) * (p - q) + (q - r),
                        dim=-1).mean(-1).mean(-1)
                    loss_joint_param = loss_joint_loc + loss_joint_axis
                elif self.joint_type == 'prismatic':
                    loss_joint_param = loss_joint_axis
                loss_joint_param_list.append(loss_joint_param)

            """SUM UP"""
            loss = loss_loc * self.loss_loc_weight + loss_shape * self.loss_shape_weight
            loss_dict['loss_loc'].append(loss_loc.item())
            loss_dict['loss_shape'].append(loss_shape.item())
            losses.append(loss)

        loss_all = torch.mean(torch.stack(losses, dim=0), dim=0)

        loss_base_all = torch.mean(torch.stack(loss_base_list))
        loss_all += self.loss_base_weight * loss_base_all
        loss_dict['loss_base'] = [loss_base_all.item()]

        loss_joint_state_all = torch.mean(torch.stack(loss_joint_state_list))
        loss_all += self.loss_joint_state_weight * loss_joint_state_all
        loss_dict['loss_joint_state'] = [loss_joint_state_all.item()]

        loss_joint_param_all = torch.mean(torch.stack(loss_joint_param_list))
        loss_all += self.loss_joint_param_weight * loss_joint_param_all
        loss_dict['loss_joint_param'] = [loss_joint_param_all.item()]

        for key in loss_dict.keys():
            loss_dict[key] = np.mean(loss_dict[key])

        """Classification(segmentation) Loss"""
        epsilon = 1e-5
        torch.clamp(dense_part_cls_score, epsilon)
        torch.clamp(gt_part_cls, epsilon)
        loss_cls = torch.mean(F.cross_entropy(dense_part_cls_score.view(-1, self.num_classes), gt_part_cls.view(-1)))
        loss_dict['loss_cls'] = loss_cls.item()
        loss_all += self.loss_cls_weight * loss_cls

        """Regularization Loss"""
        loss_reg = (pred_beta * pred_beta).mean()
        loss_dict['loss_reg'] = loss_reg.item()
        loss_all += self.loss_reg_weight * loss_reg

        scores_all = loss_all.item()
        return loss_all, scores_all, loss_dict
