from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
import numpy as np
import open3d as o3d
import copy

def visual(kp):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(kp.reshape(-1, 3))
    return pcd

class Optim():
    # use scipy.leats_squares
    def __init__(self, num_parts, norm_jts_loc, norm_jts_axis, cate_id=2, isFranka=False, isDebug=False, isArtImage=True):
        self.num_parts = num_parts
        self.num_joints = num_parts - 1
        self.norm_jts_loc = norm_jts_loc
        self.norm_jts_axis = norm_jts_axis / np.linalg.norm(norm_jts_axis, axis=-1, keepdims=True)
        self.dynamic_parent_rt = np.ones([4, 4])
        self.cate_id = cate_id
        self.new_joint_anchor_list = []
        self.new_joint_axis_list = []
        self.line_pcds = []
        self.isFranka = isFranka
        self.isDebug = isDebug
        self.pri_idx = [4] if isArtImage else [3, 4]

    def RotateAnyAxis_np(self, v1, v2, step):
        axis = v2 - v1
        axis = axis / np.linalg.norm(axis)

        a, b, c = v1[0], v1[1], v1[2]
        u, v, w = axis[0], axis[1], axis[2] 

        cos = np.cos(-step)
        sin = np.sin(-step)

        rot = np.concatenate([np.stack([u*u+(v*v+w*w)*cos, u*v*(1-cos)-w*sin, u*w*(1-cos)+v*sin,
                                                    (a*(v*v+w*w)-u*(b*v+c*w))*(1-cos)+(b*w-c*v)*sin,
                                                    u*v*(1-cos)+w*sin, v*v+(u*u+w*w)*cos, v*w*(1-cos)-u*sin,
                                                    (b*(u*u+w*w)-v*(a*u+c*w))*(1-cos)+(c*u-a*w)*sin,
                                                    u*w*(1-cos)-v*sin, v*w*(1-cos)+u*sin, w*w+(u*u+v*v)*cos,
                                                    (c*(u*u+v*v)-w*(a*u+b*v))*(1-cos)+(a*v-b*u)*sin]).reshape(3, 4),
                                                    np.array([[0., 0., 0., 1.]])], axis=0)

        return rot


    def former_rt_child_state(self, state, joint_idx):
        if joint_idx == 0: # remove not last scipy iter turns' joint param
            self.new_joint_axis_list = []
            self.new_joint_anchor_list = []
        norm_jt_loc, norm_jt_axis = self.norm_jts_loc[joint_idx], self.norm_jts_axis[joint_idx]
        new_joint_anchor = (self.dynamic_parent_rt[:3, :3] @ (self.dynamic_parent_rt[:3, 3] + norm_jt_loc).T).T
        new_joint_axis = self.dynamic_parent_rt[:3, :3] @ norm_jt_axis.reshape(3, 1).reshape(3)
        new_joint_anchor = np.reshape(new_joint_anchor, new_joint_axis.shape)
        start_point = new_joint_anchor
        end_point = start_point + new_joint_axis
        if self.isDebug:
            line_points = np.stack([start_point.reshape(-1), end_point.reshape(-1)])
            lines = [[0, 1]]  # Right leg
            colors = [[0, 0, 1] for i in range(len(lines))]
            line_pcd = o3d.geometry.LineSet()
            line_pcd.lines = o3d.utility.Vector2iVector(lines)
            line_pcd.colors = o3d.utility.Vector3dVector(colors)
            line_pcd.points = o3d.utility.Vector3dVector(line_points)
            self.line_pcds.append(line_pcd)
        self.new_joint_anchor_list.append(new_joint_anchor)
        self.new_joint_axis_list.append(new_joint_axis)
        if self.isFranka or self.cate_id not in self.pri_idx:
            relative_transform = self.RotateAnyAxis_np(start_point, end_point, state)
        else:
            relative_transform = np.concatenate([np.concatenate([np.identity(3), np.expand_dims((new_joint_axis*state), 1)], axis=1),
                                                np.array([[0., 0., 0., 1.]])], axis=0)
        child_rt = relative_transform @ self.dynamic_parent_rt
        if self.isFranka:
            self.dynamic_parent_rt = child_rt

        return child_rt


    def base_dif_func(self, base_param, base_norm_kp, base_pred_kp):
        # param is np.array([quat, trans])
        # dif is l2 between pred_kp and norm_kp with transform
        base_r_quat, base_t = base_param[:4], base_param[4:]
        base_r_quat /= np.linalg.norm(base_r_quat)
        a, b, c, d = base_r_quat[0], base_r_quat[1], base_r_quat[2], base_r_quat[3]  # q=a+bi+ci+di
        base_rot_matrix = np.stack([1 - 2 * c * c - 2 * d * d, 2 * b * c - 2 * a * d, 2 * a * c + 2 * b * d,
                                       2 * b * c + 2 * a * d, 1 - 2 * b * b - 2 * d * d, 2 * c * d - 2 * a * b,
                                       2 * b * d - 2 * a * c, 2 * a * b + 2 * c * d,
                                       1 - 2 * b * b - 2 * c * c]).reshape(3, 3)
        base_dis = np.linalg.norm((base_rot_matrix @ (base_norm_kp + base_t).T).T - base_pred_kp, axis=-1)
        return base_dis
    

    def child_dif_func(self, state, norm_kp, pred_kp):
        child_dis_list = []
        for part_idx in range(1, self.num_parts):
            joint_idx = part_idx - 1
            child_rt = self.former_rt_child_state(state[joint_idx], joint_idx)
            child_dis = np.linalg.norm((child_rt[:3, :3] @ (norm_kp[part_idx] + child_rt[:3, 3]).T).T - pred_kp[part_idx], axis=-1)
            if joint_idx == 5:
                child_dis *= 0.2
            child_dis_list.append(child_dis)
        all_child_dis = np.concatenate(child_dis_list)

        return all_child_dis
    
    def optim_func(self, init_base_r, init_base_t, state, norm_kp, pred_kp):
        x, y, z, w = R.from_matrix(init_base_r).as_quat()
        base_r_quat = np.array([w, x, y, z])
        init_base_param = np.concatenate([base_r_quat, init_base_t])

        res_base = least_squares(self.base_dif_func, init_base_param, loss='soft_l1', args=(norm_kp[0], pred_kp[0]))
        new_base_param = res_base.x
        new_base_quat, new_base_t = new_base_param[:4], new_base_param[4:]
        a, b, c, d = new_base_quat[0], new_base_quat[1], new_base_quat[2], new_base_quat[3]  # q=a+bi+ci+di
        base_rot_matrix = np.stack([1 - 2 * c * c - 2 * d * d, 2 * b * c - 2 * a * d, 2 * a * c + 2 * b * d,
                                       2 * b * c + 2 * a * d, 1 - 2 * b * b - 2 * d * d, 2 * c * d - 2 * a * b,
                                       2 * b * d - 2 * a * c, 2 * a * b + 2 * c * d,
                                       1 - 2 * b * b - 2 * c * c]).reshape(3, 3)
        new_base_transform = np.concatenate([np.concatenate([base_rot_matrix, new_base_t.reshape(3, 1)], axis=1),
                                    np.array([[0., 0., 0., 1.]])], axis=0)
        self.dynamic_parent_rt = copy.deepcopy(new_base_transform)
        if self.isDebug:
            trans_norm_kp = (base_rot_matrix @ (norm_kp[0] +new_base_t).T).T
            _pcd1 = visual(trans_norm_kp)
            _pcd1.paint_uniform_color([0., 0., 1.])
            _pcd2 = visual(pred_kp[0])
            _pcd2.paint_uniform_color([1., 0., 0.])
            coord_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        
        res_child = least_squares(self.child_dif_func, state, loss='soft_l1', f_scale=0.1, args=(norm_kp, pred_kp))
        new_joint_state = res_child.x
        
        if self.isDebug:
            pts_child_rts = []
            for joint_idx in range(self.num_joints):
                start_point = self.new_joint_anchor_list[joint_idx]
                new_joint_axis = self.new_joint_axis_list[joint_idx]
                end_point = start_point + new_joint_axis
                if self.cate_id not in self.pri_idx:
                    relative_transform = self.RotateAnyAxis_np(start_point, end_point, new_joint_state[joint_idx])
                elif self.cate_id not in self.pri_idx:
                    relative_transform = np.concatenate([np.concatenate([np.identity(3),
                                                        np.expand_dims((new_joint_axis*new_joint_state[joint_idx]), 1)], axis=1),
                                                    np.array([[0., 0., 0., 1.]])], axis=0)
                pts_child_rts.append(relative_transform)
            pred_r_list = [new_base_transform[:3, :3]] + [
                            (pts_child_rts[joint_idx] @ new_base_transform)[:3, :3]
                            for joint_idx in range(self.num_joints)]
            pred_t_list = [new_base_transform[:3, -1]] + [
                            (pts_child_rts[joint_idx] @ new_base_transform)[:3, -1]
                            for joint_idx in range(self.num_joints)]
            
            pcds1 = []
            pcds2 = []
            for part_idx in range(self.num_parts):
                trans_norm_kp = (pred_r_list[part_idx] @ (pred_t_list[part_idx].reshape(3, 1) + norm_kp[part_idx].T)).T
                pcd1 = visual(trans_norm_kp)
                pcd1.paint_uniform_color([0., 0., 1.])
                pcd2 = visual(pred_kp[part_idx])
                pcd2.paint_uniform_color([1., 0., 0.])
                pcds1.append(pcd1)
                pcds2.append(pcd2)
            o3d.visualization.draw_geometries([coord_pcd, _pcd1, _pcd2] + self.line_pcds + pcds1 + pcds2)

        all_joint_param = [self.new_joint_anchor_list, self.new_joint_axis_list]
        return new_base_transform, all_joint_param, new_joint_state


