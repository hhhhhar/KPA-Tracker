import math
import time
import open3d as o3d
import torch
import numpy as np
import os
import os.path as osp
import argparse
import mmcv
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation
import tqdm
import sys
sys.path.append('../')
from dataset.dataset1_Tracker_camera import SapienDataset_OMADNet
from model.KPA_Tracker import OMAD_Net
import pycocotools.mask as maskUtils
from libs.loss_omadnet import Loss_OMAD_Net
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
import copy
import cv2
from libs.utils import iou_3d, get_part_bbox_from_kp, get_part_bbox_from_corners
from optimization.scipy_optim import Optim
from libs.utils import calc_joint_errors

#Consolas, 'Courier New', monospace

'''
have: T0
'''

CLASSES = ['laptop', 'eyeglasses', 'dishwasher', 'drawer', 'scissors']

def rot_diff_rad(rot1, rot2):
    if np.abs((np.trace(np.matmul(rot1, rot2.T)) - 1) / 2) > 1.:
        print('Something wrong in rotation error!')
    return np.arccos((np.trace(np.matmul(rot1, rot2.T)) - 1) / 2) % (2*np.pi)

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

def RotateAnyAxis_np(v1, v2, step):
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

def rot_diff_degree(rot1, rot2):
    return rot_diff_rad(rot1, rot2) / np.pi * 180

def visual(kp):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(kp.reshape(-1, 3))
    return pcd

def get_vis_resource(index, result, norm_kp, model, opt, device, complete_all_mesh_pts, fathest_sampler): # no_concat

    pred_cls = torch.from_numpy(result['pred_cls']).to(device)
    pred_base_r = torch.from_numpy(result['pred_base_r']).to(device)
    pred_base_t = torch.from_numpy(result['pred_base_t']).to(device)
    pred_joint_state = torch.from_numpy(result['pred_joint_state']).to(device)
    pred_trans_part_kp = torch.from_numpy(result['pred_trans_part_kp']).to(device)
    pred_norm_part_kp = torch.from_numpy(result['pred_norm_part_kp']).to(device)
    gt_norm_joint_loc = torch.from_numpy(result['gt_norm_joint_loc']).to(device)
    gt_norm_joint_axis = torch.from_numpy(result['gt_norm_joint_axis']).to(device)
    gt_trans_part_kp = torch.from_numpy(result['gt_trans_part_kp']).to(device)
    sort_part = result['sort_part']
    cate = torch.from_numpy(result['cate']).to(device)
    urdf_id = torch.from_numpy(result['urdf_id']).to(device)

    norm_kp = torch.from_numpy(norm_kp).to(device)


    if index == 0:
        print()
        print('URDF id={}'.format(urdf_id))
        # print('{} valid keypoints!'.format(torch.sum(init_part_kp_weight).to(torch.int).item()))

    init_base_t = pred_base_t.unsqueeze(0)
    init_base_r = pred_base_r

    # support prismatic joints and kinematic tree depth > 2
    num_joints = opt.num_parts - 1
    if opt.cate_id != 4:
        init_joint_state = pred_joint_state * torch.pi / 180.
    else:
        init_joint_state = pred_joint_state
    joint_type = 'revolute' if opt.cate_id != 4 else 'prismatic'
    if index == 0:
        if joint_type == 'revolute':
            print('init_joint_state={} rad'.format((init_joint_state.cpu().numpy()).tolist()))
        else:
            print('init_joint_state={}'.format((init_joint_state.cpu().numpy()).tolist()))


    scipy_optim = Optim(opt.num_parts, gt_norm_joint_loc, gt_norm_joint_axis, opt.cate_id, isDebug=opt.show_omad)
    base_transform, pred_trans_joint_params_all, new_pred_joint_state = \
        scipy_optim.optim_func(init_base_r, init_base_t, init_joint_state, norm_kp, pred_trans_part_kp)
    
    pts_child_rts = []
    for joint_idx in range(num_joints):
        start_point = pred_trans_joint_params_all[0][joint_idx]
        new_joint_axis = pred_trans_joint_params_all[1][joint_idx]
        end_point = start_point + new_joint_axis
        if opt.cate_id != 4:
            relative_transform = RotateAnyAxis_np(start_point, end_point, new_pred_joint_state[joint_idx])
        elif opt.cate_id == 4:
            relative_transform = np.concatenate([np.concatenate([np.identity(3),
                                                np.expand_dims((new_joint_axis*new_pred_joint_state[joint_idx]), 1)], axis=1),
                                            np.array([[0., 0., 0., 1.]])], axis=0)
        pts_child_rts.append(relative_transform)

    pred_r_list = [base_transform[:3, :3]] + [
                    (pts_child_rts[joint_idx] @ base_transform)[:3, :3]
                    for joint_idx in range(num_joints)]
    pred_t_list = [base_transform[:3, -1]] + [
                    (pts_child_rts[joint_idx] @ base_transform)[:3, -1]
                    for joint_idx in range(num_joints)]
    pred_rt_list = [base_transform] + [
                    (pts_child_rts[joint_idx] @ base_transform)
                    for joint_idx in range(num_joints)]

    # num_all_pts = complete_all_mesh_pts.shape[0]
    # if num_all_pts >= opt.num_points:
    #     choose = np.random.choice(np.arange(num_all_pts), opt.num_points, replace=False)
    # else:
    #     choose = np.pad(np.arange(num_all_pts), (0, opt.num_points - num_all_pts), 'wrap')
    # print(complete_all_mesh_pts.shape)
    all_mesh_pts = copy.deepcopy(complete_all_mesh_pts)

    nodes_list = []
    for part_idx in range(opt.num_parts):
        part_cloud = all_mesh_pts[part_idx]
        pts_num = part_cloud.shape[0]

        part_nodes = fathest_sampler.sample(
            part_cloud[np.random.choice(part_cloud.shape[0], pts_num, replace=False)],
            opt.node_num // opt.num_parts,
        )
        nodes_list.append(part_nodes)

    result_dict = dict()
    result_dict['cloud'] = result['cloud']
    result_dict['pred_cls'] = np.stack(pred_cls.cpu(), axis=0)
    result_dict['pred_r_list'] = np.stack(pred_r_list, axis=0)
    result_dict['pred_t_list'] = np.stack(pred_t_list, axis=0)
    result_dict['pred_rt_list'] = np.stack(pred_rt_list, axis=0)
    result_dict['pred_trans_part_kp'] = pred_trans_part_kp.cpu().numpy()
    result_dict['nodes_list'] = np.stack(nodes_list, axis=0)
    result_dict['pred_trans_joint_params_all'] = [pred_trans_joint_params_all[0].cpu().numpy(), pred_trans_joint_params_all[1].cpu().numpy()]
    result_dict['new_pred_joint_state'] = new_pred_joint_state.cpu().numpy()

    if index == 0:
        print()
    
    return result_dict

def get_child_trans():
    pass

def vis_func(trans_info, cam_fix_rt, pred_cam_rt):
    pred_cls = trans_info['pred_cls']
    pred_r_list = trans_info['pred_r_list']
    pred_t_list = trans_info['pred_t_list']
    pred_trans_part_kp = trans_info['pred_trans_part_kp']
    nodes_list = trans_info['nodes_list']
    pred_trans_joint_params_all = trans_info['pred_trans_joint_params_all']
    num_joints = pred_trans_joint_params_all[0].shape[0]
    cloud = trans_info['cloud']

    base_colors = np.array([(207/255, 37/255, 38/255), (28/255, 108/255, 171/255),
                                        (38/255, 148/255, 36/255), (254/255, 114/255, 16/255)] * 2)
    cmap = cm.get_cmap("jet", opt.num_kp)
    kp_colors = cmap(np.linspace(0, 1, opt.num_kp, endpoint=True))[:, :3]

    pred_trans_norm_kp_mesh_list = []
    pred_urdf_pts_list = []

    for part_idx in range(opt.num_parts):
        # optimized beta + pred r,t
        # temp_kp = ((f_fix @ r_former[part_idx]) @ new_pred_norm_kp[part_idx, :, :].T).T
        # pred_trans_norm_kp = (pred_r_list[part_idx] @ temp_kp.T +
        #                         (pred_t_list[part_idx][np.newaxis, :] + t_former[part_idx][np.newaxis, :]).T).T
        pred_trans_norm_kp_mesh_list += \
            [o3d.geometry.TriangleMesh.create_sphere(radius=0.02, resolution=5).translate((x, y, z)) for
            x, y, z
            in pred_trans_part_kp[part_idx]]

        # urdf_pts + pred r,t 
        nodes_pcd = o3d.geometry.PointCloud()
        # temp_nodes = ((f_fix @ r_former[part_idx]) @ nodes_list[part_idx, :, :].T).T
        # pred_gt_pts = (pred_r_list[part_idx] @ temp_nodes.T +
        #                         (pred_t_list[part_idx][np.newaxis, :] + t_former[part_idx][np.newaxis, :]).T).T
        pred_gt_pts = (pred_r_list[part_idx] @ (nodes_list[part_idx, :, :]).T).T + pred_t_list[part_idx]
        nodes_pcd.points = o3d.utility.Vector3dVector(pred_gt_pts)
        # nodes_pcd.colors = o3d.utility.Vector3dVector(base_colors[pred_cls, :])
        nodes_pcd.paint_uniform_color([0., 0., 1.])
        pred_urdf_pts_list.append(nodes_pcd)

    sphere_pts_num = np.asarray(pred_trans_norm_kp_mesh_list[0].vertices).shape[0]
    for idx, mesh in enumerate(pred_trans_norm_kp_mesh_list):
        mesh.vertex_colors = o3d.utility.Vector3dVector(
            kp_colors[np.newaxis, idx, :].repeat(sphere_pts_num, axis=0))

    line_pcd_list = []
    for joint_idx in range(num_joints):
        start_point = pred_trans_joint_params_all[0][joint_idx]
        end_point = start_point + pred_trans_joint_params_all[1][joint_idx]
        # start_point = ((f_fix @ r_former[0]) @ start_point.reshape(-1,3).T + t_former[0][np.newaxis, :].T).T
        # end_point = ((f_fix @ r_former[0]) @ end_point.reshape(-1,3).T + t_former[0][np.newaxis, :].T).T
        line_points = np.stack([start_point.reshape(-1), end_point.reshape(-1)])
        lines = [[0, 1]]  # Right leg
        colors = [[0, 0, 1] for i in range(len(lines))]
        line_pcd = o3d.geometry.LineSet()
        line_pcd.lines = o3d.utility.Vector2iVector(lines)
        line_pcd.colors = o3d.utility.Vector3dVector(colors)
        line_pcd.points = o3d.utility.Vector3dVector(line_points)
        line_pcd_list.append(line_pcd)
        coord_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    
    cloud_pcd = o3d.geometry.PointCloud()
    cloud_pcd.points = o3d.utility.Vector3dVector(cloud)
    cloud_pcd.paint_uniform_color([1., 0., 0.])


    # o3d.visualization.draw_geometries(pred_kp_pcd_list + [pcd] + line_pcd_list
    #                                   + pred_trans_norm_kp_pcd_list)
    # for l_pcd in line_pcd_list:
    #     if index % 29 == 0:
    #         vis.clear_geometries()
    #         vis.add_geometry(l_pcd)
    #     else:
    #         vis.update_geometry(l_pcd)
    # for kp_pcd in pred_trans_norm_kp_mesh_list:
    #     if index % 29 == 0:
    #         vis.add_geometry(kp_pcd)
    #     else:
    #         vis.update_geometry(kp_pcd)
    # for pts_pcd in pred_urdf_pts_list:
    #     if index % 29 == 0:
    #         vis.add_geometry(pts_pcd)
    #     else:
    #         vis.update_geometry(pts_pcd)
    #         vis.poll_events()
    #         vis.update_renderer()

    o3d.visualization.draw_geometries([coord_pcd, cloud_pcd] + line_pcd_list + pred_trans_norm_kp_mesh_list + pred_urdf_pts_list)

    cam_pcds = []
    for id in range(num_parts):
        norm_pcd = o3d.geometry.PointCloud()
        norm_pcd.points = o3d.utility.Vector3dVector(nodes_list[id])
        norm_pcd.transform(cam_fix_rt[id])
        cam_pcds.append(norm_pcd)

    pred_cam_pcds = []
    for id in range(num_parts):
        norm_pcd = o3d.geometry.PointCloud()
        norm_pcd.points = o3d.utility.Vector3dVector(nodes_list[id])
        norm_pcd.transform(pred_cam_rt[id])
        norm_pcd.paint_uniform_color([0., 0., 1.])
        pred_cam_pcds.append(norm_pcd)
    o3d.visualization.draw_geometries([coord_pcd] + cam_pcds + pred_cam_pcds)


def compose_rt(rotation, translation):
    aligned_RT = np.zeros((4, 4), dtype=np.float32)
    aligned_RT[:3, :3] = rotation[:3, :3]
    aligned_RT[:3, 3] = translation
    aligned_RT[3, 3] = 1
    return aligned_RT

class FarthestSampler:
    def __init__(self):
        pass

    def calc_distances(self, p0, points):
        return ((p0 - points) ** 2).sum(axis=1)

    def sample(self, pts, k):
        farthest_pts = np.zeros((k, 3))
        # use center as initial point
        init_point = (np.max(pts, axis=0, keepdims=True) + np.min(pts, axis=0, keepdims=True)) / 2
        distances = self.calc_distances(init_point, pts)
        for i in range(0, k):
            farthest_pts[i] = pts[np.argmax(distances)]
            distances = np.minimum(distances, self.calc_distances(farthest_pts[i], pts))
        return farthest_pts

def calErr(pred_r, pred_t, gt_r, gt_t, sort_part):
    num_parts = gt_r.shape[0]
    r_errs = []
    t_errs = []
    for i in range(num_parts):
        if i in [0, sort_part]:
            r_err = rot_diff_degree(gt_r[i], pred_r[i])
            t_err = np.linalg.norm(gt_t[i] - pred_t[i])
            r_errs.append(r_err)
            t_errs.append(t_err)
    return r_errs, t_errs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()  
    parser.add_argument('--cate_id', type=int, default=1, help='category to show')
    parser.add_argument('--params_dir', type=str, help='the dir for params and kp annotations')
    parser.add_argument('--checkpoint', type=str, default=None, help='test checkpoint')
    parser.add_argument('--num_parts', type=int, default=2, help='number of parts')
    parser.add_argument('--num_cates', type=int, default=5, help='number of cates')
    parser.add_argument('--num_kp', type=int, default=12, help='number of all keypoints')
    parser.add_argument('--num_basis', type=int, default=10, help='number of shape basis')
    parser.add_argument('--symtype', type=str, default='shape', choices=['shape', 'none'], help='the symmetry type')
    parser.add_argument('--dense_soft_factor', type=float, default=1.0, help='the factor of dense softmax')
    parser.add_argument('--data_dir', type=str, default='../ArtImage', help='prior and inputs path')
    parser.add_argument('--use_p1_aug', type=bool, default=True)
    parser.add_argument('--node_num', type=int, default=512, help='urdf nodes')
    parser.add_argument('--work_dir', type=str, default='work_dir/base', help='working dir')
    parser.add_argument('--no_att', action='store_true', help='whether to not use attention map')
    parser.add_argument('--num_points', type=int, default=1024, help='number of points')
    parser.add_argument('--debug', action='store_true', help='wheter to debug')
    parser.add_argument('--use_raw_kp', action='store_true', help='wheter to use raw keypoints')
    parser.add_argument('--kp_thr', type=float, default=0.1, help='the threshold for kp weight')
    parser.add_argument('--loc_weight', type=float, default=5.0, help='the weight of pts loc weight')
    parser.add_argument('--base_weight', type=float, default=0.2, help='the weight of base rotation loss')
    parser.add_argument('--cls_weight', type=float, default=1.0, help='the weight of segmentation loss')
    parser.add_argument('--joint_state_weight', type=float, default=2.0, help='the weight of joint state loss')
    parser.add_argument('--shape_weight', type=float, default=3.0, help='the weight of shape loss')
    parser.add_argument('--joint_param_weight', type=float, default=3.0, help='the weight of joint param loss')
    parser.add_argument('--reg_weight', type=float, default=0.01, help='the weight of regularization loss')
    parser.add_argument('--data_tag', type=str, default='val')
    parser.add_argument('--show', action='store_true', help='whether to visualize')
    parser.add_argument('--show_omad', action='store_true', help='whether to visualize')
    parser.add_argument('--use_gt_kp', action='store_true', help='wheter to use gt kp')
    parser.add_argument('--use_pn', action='store_true')
    parser.add_argument('--use_initial', action='store_true', help='wheter to use initial prediction without refinement')
    opt = parser.parse_args()

    num_parts = opt.num_parts
    device = torch.device("cuda:0")
    urdf_dir = f'../ArtImage/{CLASSES[opt.cate_id-1]}/urdf'
    intrinsics_path = osp.join(f'../ArtImage/camera_intrinsic.json')
    camera_intrinsic = o3d.io.read_pinhole_camera_intrinsic(intrinsics_path)
    cam_cx, cam_cy = camera_intrinsic.get_principal_point()
    cam_fx, cam_fy = camera_intrinsic.get_focal_length()
    width = camera_intrinsic.width
    height = camera_intrinsic.height

    xmap = np.array([[j for _ in range(width)] for j in range(height)])
    ymap = np.array([[i for i in range(width)] for _ in range(height)])

    valid_flag_path = osp.join(opt.data_dir, CLASSES[opt.cate_id-1], 'train.txt'
            if opt.data_tag == 'train' else 'test.txt')
    annotation_valid_flags = dict()
    with open(valid_flag_path, 'r') as f:
        annotation_valid_flags[opt.cate_id-1] = f.readlines()
    for idx in range(len(annotation_valid_flags[opt.cate_id-1])):
        annotation_valid_flags[opt.cate_id-1][idx] = annotation_valid_flags[opt.cate_id-1][idx].split('\n')[0]


    annotation_list = []
    annotation_list_path = osp.join(opt.data_dir, f'{CLASSES[opt.cate_id-1]}','train', 'annotations')
    for file in sorted(os.listdir(annotation_list_path)):
        if '.json' in file and file in annotation_valid_flags[opt.cate_id-1]:
            annotation = mmcv.load(osp.join(annotation_list_path, file))
            annotation_list.append(annotation)
    urdf_id = annotation_list[0]['instances'][0]['urdf_id']
    dir = str(urdf_id)


    params_dict = torch.load(osp.join(opt.params_dir, 'params.pth'))
    model = OMAD_Net(device=device, params_dict=params_dict,
                        num_points=opt.num_points, num_kp=opt.num_kp, num_parts=opt.num_parts,
                        init_dense_soft_factor=opt.dense_soft_factor, num_basis=opt.num_basis, symtype=opt.symtype)
    model = model.to(device)

    assert opt.checkpoint is not None
    model.load_state_dict(torch.load(osp.join(opt.work_dir, opt.checkpoint), map_location=device))

    test_kp_anno_path = osp.join(opt.params_dir, 'unsup_test_keypoints.pkl')
    test_dataset = SapienDataset_OMADNet(opt.data_tag, data_root=opt.data_dir, add_noise=False, num_pts=opt.num_points,
                                         num_parts=opt.num_parts, num_cates=opt.num_cates, cate_id=opt.cate_id,
                                         device=torch.device("cpu"), data_tag='train', kp_anno_path=test_kp_anno_path)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    k_fix = []
    test_count = 0
    pred_s = []
    gt_s = []
    sample_id = 0
    farthest_sampler = FarthestSampler()

    new_model = copy.deepcopy(model)
    ini_base_r_error_all = 0.
    ini_sort_child_r_error_all = 0.
    ini_base_t_error_all = 0.
    ini_sort_child_t_error_all = 0.

    new_base_r_error_all = 0.
    new_sort_child_r_error_all = 0.
    new_base_t_error_all = 0.
    new_sort_child_t_error_all = 0.

    cam_base_r_error_all = 0.
    cam_sort_child_r_error_all = 0.
    cam_base_t_error_all = 0.
    cam_sort_child_t_error_all = 0.

    last_cam_base_r_error_all = 0.
    last_cam_sort_child_r_error_all = 0.
    last_cam_base_t_error_all = 0.
    last_cam_sort_child_t_error_all = 0.

    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    # coord_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    # vis.add_geometry(coord_pcd)
    rt_key = [_ for _ in range(num_parts)]
    r_key = [_ for _ in range(num_parts)]
    t_key = [_ for _ in range(num_parts)]
    key_j_state = torch.tensor(0.)
    turns = 0
    video_num = 0
    for i, data in enumerate(test_dataloader):
        turns += 1
        cloud = []
        clouds, norm_part_pts, gt_part_cls, gt_part_r, gt_part_quat, gt_part_t, gt_joint_state, gt_norm_joint_loc, gt_norm_joint_axis, \
        gt_norm_part_kp, gt_scale, gt_center, gt_norm_part_corners, cate, urdf_id, sort_part, link_category_to_idx_map = data
        link_category_to_idx_map = link_category_to_idx_map.cpu().numpy()[0]
        sort_part = sort_part.cpu().numpy()[0]
        sort_part = link_category_to_idx_map[sort_part]

        angles = np.array([np.random.uniform(-5., 5.),
                        np.random.uniform(-5., 5.),
                        np.random.uniform(-5., 5.)])
        base_fix_t = np.array([np.random.uniform(-0.1, 0.1),
                                    np.random.uniform(-0.1, 0.1),
                                    np.random.uniform(-0.1, 0.1)])
        base_fix_r = Rotation.from_euler('yxz', angles, degrees=True).as_matrix()
        base_fix_rt = compose_rt(base_fix_r, base_fix_t)
        flag = 0
        inner_index = i % 29
        print(f'inner_index: {inner_index}')
        if inner_index == 0:
            k_fix = base_fix_rt
            flag = 1
            r_key = copy.deepcopy(gt_part_r[0].numpy())
            t_key = copy.deepcopy(gt_part_t[0].numpy())

            key_j_state = gt_joint_state[0][sort_part]
            turn_error_all = 0.
            key_error_all = 0.
            new_key_error_all = 0.
            turn_num = 0
            real_rt_key = [_ for _ in range(num_parts)]

            for part_idx in range(num_parts):
                rt_key[part_idx] = compose_rt(r_key[part_idx], t_key[part_idx])
                real_rt_key[part_idx] = copy.deepcopy(rt_key[part_idx])
                rt_key[part_idx] = rt_key[part_idx] @ k_fix
        
        key_inner_id = ((inner_index - 1) // 5) * 5 if flag == 0 else 0
        key_dis = inner_index - key_inner_id
        
        clouds = clouds[0].numpy()
        cloud_pcds = []
        gt_part_r = gt_part_r[0].numpy()
        gt_part_t = gt_part_t[0].numpy()
        gt_part_rt = [_ for _ in range(num_parts)]
        gt_ori_jstate = copy.deepcopy(gt_joint_state[0])

        cam_fix_rt = [_ for _ in range(num_parts)]
        cam_fix_r = [_ for _ in range(num_parts)]
        cam_fix_t = [_ for _ in range(num_parts)]

        real_rt = [_ for _ in range(num_parts)]
        real_r = [_ for _ in range(num_parts)]
        real_t = [_ for _ in range(num_parts)]
        noise_rt = np.linalg.inv(k_fix) @ base_fix_rt
        for part_idx in range(num_parts):
            gt_part_rt[part_idx] = compose_rt(gt_part_r[part_idx], gt_part_t[part_idx])
            real_rt[part_idx] = copy.deepcopy(gt_part_rt[part_idx])

            cam_fix_rt[part_idx] = real_rt[part_idx] @ base_fix_rt
            cam_fix_r[part_idx] = cam_fix_rt[part_idx][:3, :3]
            cam_fix_t[part_idx] = cam_fix_rt[part_idx][:3, 3]

            if part_idx not in [0, sort_part]:
                rt_key[part_idx] = gt_part_rt[part_idx] @ k_fix
                cam_fix_rt[part_idx] = cam_fix_rt[0]
                cam_fix_r[part_idx] = cam_fix_rt[part_idx][:3, :3]
                cam_fix_t[part_idx] = cam_fix_rt[part_idx][:3, 3]
            
            gt_part_rt[part_idx] = np.linalg.inv(rt_key[part_idx]) @ gt_part_rt[part_idx]
            gt_part_rt[part_idx] = base_fix_rt @ gt_part_rt[part_idx]
            gt_part_r[part_idx] = gt_part_rt[part_idx][:3, :3]
            gt_part_t[part_idx] = gt_part_rt[part_idx][:3, 3]

            if part_idx == sort_part:
                gt_joint_state[0][part_idx] = gt_joint_state[0][part_idx] - key_j_state
            else:
                gt_joint_state[0][part_idx] = torch.tensor(0.)
            
            cloud_pcd = o3d.geometry.PointCloud()
            cloud_pcd.points = o3d.utility.Vector3dVector(clouds[part_idx])
            cloud_pcd.transform(np.linalg.inv(rt_key[part_idx]))
            cloud_pcd.transform(base_fix_rt)
            cloud_pcd.paint_uniform_color([1., 0., 0.])
            cloud_pcds.append(cloud_pcd)
            clouds[part_idx] = np.asarray(cloud_pcd.points)
            if part_idx not in [0, sort_part]:
                rt_key[part_idx] = rt_key[0]


        temp_cloud = np.concatenate(clouds, axis=0)
        if key_dis == 5 or inner_index == 0:
            real_rt_key = copy.deepcopy(real_rt)    

        c_len = temp_cloud.shape[0]
        if c_len > opt.num_points:
            c_mask = np.random.choice(np.arange(c_len), opt.num_points, replace=False)
            temp_cloud = temp_cloud[c_mask]
        cloud.append(temp_cloud)

        if flag == 1:
            key_cloud = cloud[0] # (part_num, -1, 3)

        model.eval()

        coord_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)

        gt_norm_kp = gt_norm_part_kp[0].cpu().numpy()
        gt_trans_part_kp = [_ for _ in range(num_parts)]
        for part_idx in range(num_parts):
            gt_trans_part_kp[part_idx] = (gt_part_r[part_idx] @ gt_norm_kp[part_idx].T).T + gt_part_t[part_idx]
        gt_trans_part_kp = np.stack(gt_trans_part_kp)

        if opt.show:
            gt_trans_kp_mesh_list = []
            for j in range(num_parts):
                gt_trans_kp_mesh_list += \
                    [o3d.geometry.TriangleMesh.create_sphere(radius=0.02, resolution=5).translate((x, y, z)) for
                    x, y, z
                    in gt_trans_part_kp[j]]
            o3d.visualization.draw_geometries([coord_pcd] + cloud_pcds + gt_trans_kp_mesh_list)

        
        gt_part_r = np.expand_dims(gt_part_r, axis=0)
        gt_part_t = np.expand_dims(gt_part_t, axis=0)
        raw_part_target_quat = np.stack([Rotation.from_matrix(gt_part_r[0][i]).as_quat()
                            for i in range(opt.num_parts)], axis=0)  # (x, y, z, w)
        gt_part_quat = np.concatenate([
            raw_part_target_quat[:, -1][:, np.newaxis], raw_part_target_quat[:, :3]], axis=-1)  # (w, x, y, z)

        cloud = torch.from_numpy(np.stack(cloud).astype(np.float32).reshape(1, -1, 3)).contiguous()
        gt_part_r = torch.from_numpy(gt_part_r).contiguous()
        gt_part_t = torch.from_numpy(gt_part_t).contiguous()
        gt_part_quat = torch.from_numpy(np.expand_dims(gt_part_quat, 0)).contiguous()
        gt_s.append(gt_joint_state.cpu()[0][1])

        cloud, norm_part_pts, gt_part_cls, gt_part_r, gt_part_quat, gt_part_t, gt_joint_state, \
        gt_norm_joint_loc, gt_norm_joint_axis, gt_norm_part_kp = \
            cloud.to(device), norm_part_pts.to(device), gt_part_cls.to(device), gt_part_r.to(device), gt_part_quat.to(device), \
            gt_part_t.to(device), gt_joint_state.to(device), gt_norm_joint_loc.to(device), \
            gt_norm_joint_axis.to(device), gt_norm_part_kp.to(device)

        with torch.no_grad():
            dense_part_cls_score, pred_trans_part_kp, pred_base_quat, pred_base_r, pred_base_t, pred_joint_state, \
                pred_beta, pred_norm_part_kp, pred_joint_loc, pred_joint_axis = model(cloud, None)
            pred_cls = torch.argmax(dense_part_cls_score, dim=-1)

        pred_jstate = 10 * pred_joint_state.cpu().numpy()[0]
        gt_norm_part_kp = gt_norm_part_kp[0].cpu().numpy()
        gt_part_r = gt_part_r[0].cpu().numpy()
        gt_part_t = gt_part_t[0].cpu().numpy()
        gt_norm_joint_loc = gt_norm_joint_loc[0].cpu().numpy()
        gt_norm_joint_axis = gt_norm_joint_axis[0].cpu().numpy()
        pred_trans_part_kp = pred_trans_part_kp[0].cpu().numpy()
        pred_base_r = pred_base_r[0].cpu().numpy()
        pred_base_t = pred_base_t[0].cpu().numpy()
        pred_joint_axis = pred_joint_axis[0].cpu().numpy()
        pred_joint_axis = pred_joint_axis / np.linalg.norm(pred_joint_axis)
        base_transform = compose_rt(pred_base_r, pred_base_t)
        j_state_d = pred_joint_state.cpu().numpy()[0]


        line_pcd_list = []
        pred_trans_kp_mesh_list = []      

        pred_chlid_rt = [_ for _ in range(num_parts-1)]
        for joint_idx in range(num_parts-1):
            start = (pred_base_r @ (gt_norm_joint_loc[joint_idx] + pred_base_t).T).T
            axis_len = (pred_base_r @ (gt_norm_joint_axis[joint_idx] + pred_base_t).T).T
            if opt.cate_id != 4:
                pred_chlid_rt[joint_idx] = RotateAnyAxis_np(start, start + axis_len, -1. * j_state_d[sort_part-1] * 10.)
            else:
                pred_chlid_rt[joint_idx] = np.concatenate([np.concatenate([np.eye(3),
                                            np.expand_dims(pred_joint_axis[joint_idx] * j_state_d[joint_idx], 1)], axis=1),
                                            np.array([[0., 0., 0., 1.]])], axis=0)
        
        pred_r_list = [base_transform[:3, :3]] + [
                    np.matmul(pred_chlid_rt[joint_idx], base_transform)[:3, :3]
                    for joint_idx in range(num_parts-1)]
        pred_t_list = [base_transform[:3, -1]] + [
                        np.matmul(pred_chlid_rt[joint_idx], base_transform)[:3, -1]
                        for joint_idx in range(num_parts-1)]
        
        errs = calErr(pred_r_list, pred_t_list, gt_part_r, gt_part_t, sort_part)
        base_r_err = errs[0][0]
        ini_base_r_error_all += base_r_err
        child_r_err = errs[0][1]
        ini_sort_child_r_error_all += child_r_err
        if base_r_err > 30:
            turns -= 1
            continue
        print(f'sort part: {sort_part}')
        print(f'ini base r_err: {base_r_err}')
        print(f'ini child r_err: {child_r_err}')

        base_t_err = errs[1][0]
        ini_base_t_error_all += base_t_err
        child_t_err = errs[1][1]
        ini_sort_child_t_error_all += child_t_err
        print(f'ini base t_err: {base_t_err}')
        print(f'ini child t_err: {child_t_err}')

        norm_part_pts = norm_part_pts[0].cpu().numpy()

        # next key flame

        if key_dis == 5:
            turn_num += 1
            k_fix = base_fix_rt

        gt_trans_part_kp = [_ for _ in range(num_parts)]
        for part_idx in range(num_parts):
            gt_trans_part_kp[part_idx] = (gt_part_r[part_idx] @ gt_norm_part_kp[part_idx].T).T + gt_part_t[part_idx]
        gt_trans_part_kp = np.stack(gt_trans_part_kp)
            
        result = dict(
                    sample_id=i,
                    pred_cls=pred_cls[0].cpu().numpy(),
                    pred_trans_part_kp=pred_trans_part_kp,
                    pred_base_r=pred_base_r,
                    pred_base_t=pred_base_t,
                    pred_joint_state=pred_joint_state[0].cpu().numpy(),
                    pred_norm_part_kp=pred_norm_part_kp[0].cpu().numpy(),

                    cloud=cloud[0].cpu().numpy(),
                    gt_norm_joint_loc=gt_norm_joint_loc,
                    gt_norm_joint_axis=gt_norm_joint_axis,
                    gt_norm_part_kp=gt_norm_part_kp,
                    gt_trans_part_kp=gt_trans_part_kp,
                    cate=cate[0].cpu().numpy(),
                    urdf_id=urdf_id[0].cpu().numpy(),
                    sort_part=sort_part)
        sample_id += 1
        # if opt.debug:
        #     if not osp.exists(f'./temp/{CLASSES[opt.cate_id-1]}/test_{i}'):
        #         os.makedirs(f'./temp/{CLASSES[opt.cate_id-1]}/test_{i}')
        if flag == 1:
            norm_kp = pred_trans_part_kp
            print()
        optimize_result = get_vis_resource(i, result, norm_kp, new_model, opt, 'cpu', norm_part_pts, farthest_sampler)

        new_pred_r = optimize_result['pred_r_list']
        new_pred_t = optimize_result['pred_t_list']
        new_pred_rt = optimize_result['pred_rt_list']
        new_errs = calErr(new_pred_r, new_pred_t, gt_part_r, gt_part_t, sort_part)
        base_r_err = new_errs[0][0]
        child_r_err = new_errs[0][1]
        if not (np.isnan(base_r_err) or np.isnan(child_r_err)):
            new_sort_child_r_error_all += child_r_err
            new_base_r_error_all += base_r_err
            print(f'new base r_err: {base_r_err}')
            print(f'new child r_err: {child_r_err}')

        base_t_err = new_errs[1][0]
        new_base_t_error_all += base_t_err
        child_t_err = new_errs[1][1]
        new_sort_child_t_error_all += child_t_err
        print(f'new base t_err: {base_t_err}')
        print(f'new child t_err: {child_t_err}')
        
        pred_cam_rt = [np.matmul(rt_key[part_idx], new_pred_rt[part_idx])
                        for part_idx in range(num_parts)]
        pred_cam_r = [pred_cam_rt[part_idx][:3, :3] for part_idx in range(num_parts)]
        pred_cam_t = [pred_cam_rt[part_idx][:3, 3] for part_idx in range(num_parts)]
        
        cam_errs = calErr(np.stack(pred_cam_r), np.stack(pred_cam_t), np.stack(cam_fix_r), np.stack(cam_fix_t), sort_part)
        cam_base_r_err = cam_errs[0][0]
        cam_child_r_err = cam_errs[0][1]
        if not (np.isnan(cam_base_r_err) or np.isnan(cam_child_r_err)):
            cam_base_r_error_all += cam_base_r_err
            cam_sort_child_r_error_all += cam_child_r_err
            print(f'cam base r_err: {cam_base_r_err}')
            print(f'cam child r_err: {cam_child_r_err}')

        cam_base_t_err = cam_errs[1][0]
        cam_base_t_error_all += cam_base_t_err
        cam_child_t_err = cam_errs[1][1]
        cam_sort_child_t_error_all += cam_child_t_err
        print(f'cam base t_err: {cam_base_t_err}')
        print(f'cam child t_err: {cam_child_t_err}')
        print()

        if inner_index == 28:
            if not (np.isnan(cam_base_r_err) or np.isnan(cam_child_r_err)):
                video_num += 1
                last_cam_base_r_error_all += cam_base_r_err
                last_cam_sort_child_r_error_all += cam_child_r_err
                last_cam_base_t_error_all += cam_base_t_err
                last_cam_sort_child_t_error_all += cam_child_t_err

        if opt.show:
            vis_func(optimize_result, cam_fix_rt, pred_cam_rt)

        if opt.cate_id != 4:
            new_joint_state = 180 * optimize_result['new_pred_joint_state'] / torch.pi
        else:
            new_joint_state = optimize_result['new_pred_joint_state']
        print()
        if key_dis == 5:
            for part_idx in  range(num_parts):
                pred_rt = compose_rt(new_pred_r[part_idx], new_pred_t[part_idx]) # bs = 1
                rt_key[part_idx] = rt_key[part_idx] @ pred_rt
                r_key[part_idx] = rt_key[part_idx][:3, :3]
                t_key[part_idx] = rt_key[part_idx][:3, 3]

    # vis.destory_window()
    print(f'turn:{turns}')
    print(f"initial base r error mean:{ini_base_r_error_all/turns}")
    print(f"initial child r error mean:{ini_sort_child_r_error_all/turns}")
    print(f"initial base t error mean:{ini_base_t_error_all/turns}")
    print(f"initial child t error mean:{ini_sort_child_t_error_all/turns}")
    print()
    print(f"new base r error mean:{new_base_r_error_all/turns}")
    print(f"new child r error mean:{new_sort_child_r_error_all/turns}")
    print(f"new base t error mean:{new_base_t_error_all/turns}")
    print(f"new child t error mean:{new_sort_child_t_error_all/turns}")
    print()
    print(f"cam base r error mean:{cam_base_r_error_all/turns}")
    print(f"cam child r error mean:{cam_sort_child_r_error_all/turns}")
    print(f"cam base t error mean:{cam_base_t_error_all/turns}")
    print(f"cam child t error mean:{cam_sort_child_t_error_all/turns}")
    print()
    print(f"last cam base r error mean:{last_cam_base_r_error_all/video_num}")
    print(f"last cam child r error mean:{last_cam_sort_child_r_error_all/video_num}")
    print(f"last cam base t error mean:{last_cam_base_t_error_all/video_num}")
    print(f"last cam child t error mean:{last_cam_sort_child_t_error_all/video_num}")
    print()




        
                
            


