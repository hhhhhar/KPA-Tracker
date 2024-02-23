import math
import torch
from scipy.spatial.transform import Rotation as R
import numpy as np
import open3d as o3d

def visual(kp):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(kp.reshape(-1, 3))
    return pcd

def cross(R,V):
    h = torch.tensor([R[1] * V[2] - R[2] * V[1],
         R[2] * V[0] - R[0] * V[2],
         R[0] * V[1] - R[1] * V[0]])
    return h

def cal_n_vec(kps):
    a = kps[0]
    b = kps[1]
    c = kps[2]
    v1 = (a - c).reshape(-1)
    v1 = v1 / torch.norm(v1)
    v2 = (b - c).reshape(-1)
    v2 = v2 / torch.norm(v2)
    norm_vec = torch.cross(v1, v2)
    norm_vec = norm_vec  # / torch.norm(norm_vec)
    return norm_vec

def RotateAnyAxis(v1, v2, step):
    axis = v2 - v1
    axis = axis / torch.norm(axis)

    a, b, c = v1[0], v1[1], v1[2]
    u, v, w = axis[0], axis[1], axis[2] 

    cos = torch.cos(-step)
    sin = torch.sin(-step)

    rot = torch.cat([torch.stack([u*u+(v*v+w*w)*cos, u*v*(1-cos)-w*sin, u*w*(1-cos)+v*sin,
                                                   (a*(v*v+w*w)-u*(b*v+c*w))*(1-cos)+(b*w-c*v)*sin,
                                                   u*v*(1-cos)+w*sin, v*v+(u*u+w*w)*cos, v*w*(1-cos)-u*sin,
                                                   (b*(u*u+w*w)-v*(a*u+c*w))*(1-cos)+(c*u-a*w)*sin,
                                                   u*w*(1-cos)-v*sin, v*w*(1-cos)+u*sin, w*w+(u*u+v*v)*cos,
                                                   (c*(u*u+v*v)-w*(a*u+b*v))*(1-cos)+(a*v-b*u)*sin]).reshape(3, 4),
                                                   torch.tensor([[0., 0., 0., 1.]], device='cpu')], dim=0)

    return rot

class PoseEstimator(torch.nn.Module):
    def __init__(self, model, num_parts, num_kp, init_base_r, init_base_t, init_joint_state,
                 norm_kp, norm_jt_loc, norm_jt_axis, device, cate_id=1, debug=False, sort_part=1):
        super(PoseEstimator, self).__init__()
        self.model = model.to(device)
        self.model.eval()
        self.num_parts = num_parts
        self.num_kp = num_kp
        self.num_joints = num_parts - 1
        self.norm_kp = norm_kp
        self.norm_jt_loc = norm_jt_loc
        self.norm_jt_axis = norm_jt_axis
        self.device = device
        self.cate_id = cate_id
        assert cate_id in range(1,6)

        x, y, z, w = R.from_matrix(init_base_r.cpu().numpy()).as_quat()
        self.base_r_quat = torch.nn.Parameter(torch.tensor(
            [w, x, y, z], device=device, dtype=torch.float))  # q=a+bi+ci+di
        self.base_t = torch.nn.Parameter(init_base_t.clone().detach().to(device))
        self.joint_state = torch.nn.Parameter(init_joint_state.clone().detach().to(device))
        self.debug = debug
        self.turns = 0
        self.sort_part = sort_part

    def forward(self, pred_kp, mode='base'):
        isDebug = self.debug and (self.turns % 20 == 0)
        self.turns += 1
        assert mode in ('base', 'joint_single', 'all')
        norm_kp = self.norm_kp  # bs=1
        
        # homo_norm_kp = torch.cat([norm_kp, torch.ones(norm_kp.shape[0], norm_kp.shape[1], 1, device=norm_kp.device)], dim=-1)
        # homo_pred_kp = torch.cat([pred_kp, torch.ones(pred_kp.shape[0], pred_kp.shape[1], 1, device=pred_kp.device)], dim=-1)
        homo_norm_kp = norm_kp
        homo_pred_kp = pred_kp

        base_r_quat = self.base_r_quat / torch.norm(self.base_r_quat)
        a, b, c, d = base_r_quat[0], base_r_quat[1], base_r_quat[2], base_r_quat[3]  # q=a+bi+ci+di
        base_rot_matrix = torch.stack([1 - 2 * c * c - 2 * d * d, 2 * b * c - 2 * a * d, 2 * a * c + 2 * b * d,
                                       2 * b * c + 2 * a * d, 1 - 2 * b * b - 2 * d * d, 2 * c * d - 2 * a * b,
                                       2 * b * d - 2 * a * c, 2 * a * b + 2 * c * d,
                                       1 - 2 * b * b - 2 * c * c]).reshape(3, 3)
        base_transform = torch.cat([torch.cat([base_rot_matrix, self.base_t.transpose(0, 1)], dim=1),
                                    torch.tensor([[0., 0., 0., 1.]], device=self.device)], dim=0)
        base_objective_dis = torch.mean(
            torch.norm((base_rot_matrix @ (homo_norm_kp[0] + self.base_t).T).T - homo_pred_kp[0], dim=-1))
        base_objective_rot = torch.var(
            torch.norm((base_rot_matrix @ (homo_norm_kp[0] + self.base_t).T).T - homo_pred_kp[0], dim=-1))
        base_objective_dis = base_objective_dis * 1.5 # + base_objective_rot * 5
        
        trans_norm_base_kp = (base_rot_matrix @ (homo_norm_kp[0] + self.base_t).T).T

        if isDebug:
            trans_norm_kp = torch.unsqueeze(trans_norm_base_kp, dim=0).detach().cpu().numpy()
            _pcd1 = visual(trans_norm_kp)
            _pcd1.paint_uniform_color([0., 0., 1.])
            _pcd2 = visual(torch.unsqueeze(homo_pred_kp[0], dim=0).detach().cpu().numpy())
            _pcd2.paint_uniform_color([1., 0., 0.])
            coord_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

        norm_joint_loc_all, norm_joint_axis_all = self.norm_jt_loc, self.norm_jt_axis  
        norm_joint_axis_all = norm_joint_axis_all / torch.norm(norm_joint_axis_all, dim=-1, keepdim=True)
        new_joint_anchor_list = []
        new_joint_axis_list = []
        kp_child_rt_list = []
        pcds1 = []
        pcds2 = []
        child_loss = 0.

        # optim joint state
        for joint_idx in range(self.num_joints):
            part_idx = joint_idx + 1

            # support kinematic tree depth > 2
            norm_joint_loc, norm_joint_axis = norm_joint_loc_all[joint_idx], norm_joint_axis_all[joint_idx]  # bs=1
            # homo_joint_anchor = torch.cat([norm_joint_loc, torch.ones(1, device=self.device)]).unsqueeze(1)
            homo_joint_anchor = norm_joint_loc
            new_joint_anchor = (base_rot_matrix @ (self.base_t + homo_joint_anchor).T).T # + self.jloc_t
            new_joint_axis = base_rot_matrix.matmul(norm_joint_axis)
            new_joint_anchor = torch.reshape(new_joint_anchor, new_joint_axis.size())
            start_point = new_joint_anchor.detach().cpu().numpy()
            end_point = start_point + new_joint_axis.detach().cpu().numpy()
            new_joint_anchor_list.append(new_joint_anchor.detach())
            new_joint_axis_list.append(new_joint_axis.detach())

            # if self.cate_id == 1:
            #     start_point = trans_norm_base_kp[4].detach().cpu().numpy()
            #     end_point = trans_norm_base_kp[10].detach().cpu().numpy()
            #     axis = end_point - start_point
            #     start_point += (axis / 2)
            #     axis = axis / np.linalg.norm(axis, axis=-1, keepdims=True)
            # else:
            #     start_point = new_joint_anchor.detach().cpu().numpy()
            #     end_point = start_point + new_joint_axis.detach().cpu().numpy()
            start_point = new_joint_anchor.detach().cpu().numpy()
            end_point = start_point + new_joint_axis.detach().cpu().numpy()

            if isDebug:
                line_points = np.stack([start_point.reshape(-1), end_point.reshape(-1)])
                lines = [[0, 1]]  # Right leg
                colors = [[0, 0, 1] for i in range(len(lines))]
                line_pcd = o3d.geometry.LineSet()
                line_pcd.lines = o3d.utility.Vector2iVector(lines)
                line_pcd.colors = o3d.utility.Vector3dVector(colors)
                line_pcd.points = o3d.utility.Vector3dVector(line_points)

            if self.cate_id not in [3, 4]:
                relative_transform = RotateAnyAxis(torch.from_numpy(start_point), torch.from_numpy(end_point), self.joint_state[joint_idx])
            else:
                relative_transform = torch.cat([torch.cat([torch.eye(3, device=self.device),
                                                             (new_joint_axis*self.joint_state[joint_idx]).unsqueeze(1)], dim=1),
                                                torch.tensor([[0., 0., 0., 1.]], device=self.device)], dim=0)
            kp_child_rt_list.append(relative_transform.detach())
            temp_rt = relative_transform.matmul(base_transform)
            if self.cate_id == 1:
                child_objective = torch.mean(torch.norm((temp_rt[:3, :3] @ (temp_rt[:3, 3] + homo_norm_kp[part_idx]).T).T - homo_pred_kp[part_idx],
                            dim=-1))
                child_loss += child_objective
            else:
                child_objective = torch.mean(torch.norm((((temp_rt[:3, :3] @ homo_norm_kp[part_idx].T).T) + temp_rt[:3, 3]) - homo_pred_kp[part_idx],
                            dim=-1))
                c_weight = 1.2 if part_idx == self.sort_part else 0.5
                child_loss += child_objective * c_weight

            if isDebug:
                trans_norm_kp = temp_rt[:3, 3] + (temp_rt[:3, :3] @ homo_norm_kp[part_idx].T).T
                trans_norm_kp = torch.unsqueeze(trans_norm_kp, dim=0).detach().cpu().numpy()
                pcd1 = visual(trans_norm_kp)
                pcd1.paint_uniform_color([1., 0., 0.])
                pcd2 = visual(torch.unsqueeze(homo_pred_kp[part_idx], dim=0).detach().cpu().numpy())
                pcd2.paint_uniform_color([0., 0., 1.])
                pcds1.append(pcd1)
                pcds2.append(pcd2)

        if isDebug:
            # o3d.visualization.draw_geometries([vec_pcd1, need_pcd])s
            o3d.visualization.draw_geometries([coord_pcd, line_pcd, _pcd1, _pcd2] + pcds1 + pcds2)
        new_joint_params_all = (torch.stack(new_joint_anchor_list, dim=0), torch.stack(new_joint_axis_list, dim=0))
        return base_objective_dis, child_loss, base_transform.detach(), new_joint_params_all


def optimize_pose(estimator, pred_kp, use_initial=False):
    if use_initial:
        pass
    else:
        estimator.base_r_quat.requires_grad_(True)
        estimator.base_t.requires_grad_(True)
        optimizer = torch.optim.Adam(estimator.parameters(), lr=1e-2)
        last_loss = 0.
        for iter in range(80):  # base transformation(r+t) + joint state
            base_loss, _, _, _ = estimator(pred_kp.detach(), mode='all')
            if iter % 20 == 0:
                # print('base_r + base_t + joint state + beta: iter {}, loss={:05f}'.format(iter, loss.item()))
                if abs(last_loss - base_loss.item()) < 1e-4:
                    break
                last_loss = base_loss.item()
            optimizer.zero_grad()
            base_loss.backward()
            optimizer.step()
        estimator.base_r_quat.requires_grad_(False)
        estimator.base_t.requires_grad_(False)
        estimator.joint_state.requires_grad_(True)
        for iter in range(200):  # base transformation(r+t) + joint state
            _, child_loss, _, _ = estimator(pred_kp.detach(), mode='all')
            if iter % 20 == 0:
                # print(f'child loss:{child_loss}')
                # print('base_r + base_t + joint state + beta: iter {}, loss={:05f}'.format(iter, loss.item()))
                # if child_loss < 1e-3:
                if abs(last_loss - child_loss.item()) < 1e-4:
                    break
                last_loss = child_loss.item()
            optimizer.zero_grad()
            child_loss.backward()
            optimizer.step()
    _, _, base_transform, new_joint_params_all = estimator(pred_kp.detach())
    joint_state = estimator.joint_state.detach()
    
    return base_transform, new_joint_params_all, joint_state
