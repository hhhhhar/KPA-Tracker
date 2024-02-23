import pickle
import re
import torch.utils.data as data
import os
import os.path as osp
import torch
import numpy as np
import open3d as o3d
import copy
import xml.etree.ElementTree as ET
import mmcv
import pycocotools.mask as maskUtils
import cv2
from scipy.spatial.transform import Rotation
from IPython import embed


class SapienDataset_OMADNet(data.Dataset):
    CLASSES = ('background', 'box', 'stapler', 'cutter', 'drawer', 'scissor')
    TEST_URDF_IDs = (0, 1) + (10, 11) + (20, 21) + (30, 31) + (40, 41)

    def __init__(self, mode, data_root,
                 num_pts, num_cates, cate_id, num_parts, add_noise=False, use_p1_aug=False, 
                 use_pn=False, debug=False, device='cuda:0', data_tag='train',
                 kp_anno_path='unsup_train_keypoints.pkl', use_background=False):
        assert mode in ('train', 'val')
        self.data_root = data_root
        self.mode = mode
        self.num_pts = num_pts
        self.num_cates = num_cates
        self.num_parts = num_parts
        self.debug = debug
        self.device = device
        self.data_tag = data_tag
        self.cate_id = cate_id
        self.add_noise = add_noise
        self.use_p1_aug = use_p1_aug
        self.norm_part_kp_annotions = dict()
        self.norm_part_kp_annotions[self.cate_id] = mmcv.load(kp_anno_path)
        self.use_pn = use_pn
        self.use_background=use_background

        valid_flag_path = osp.join(self.data_root, 'Data', 'new_anno_test', 'flag', self.CLASSES[cate_id], 'train.txt'
            if mode == 'train' else 'test.txt')
        self.annotation_valid_flags = dict()
        with open(valid_flag_path, 'r') as f:
            self.annotation_valid_flags[self.cate_id] = f.readlines()
        for idx in range(len(self.annotation_valid_flags[self.cate_id])):
            self.annotation_valid_flags[self.cate_id][idx] = self.annotation_valid_flags[self.cate_id][idx].split('\n')[0]

        self.obj_list = {}
        self.obj_name_list = {}

        intrinsics_path = osp.join(self.data_root, 'Data','camera_intrinsic.json')
        self.camera_intrinsic = o3d.io.read_pinhole_camera_intrinsic(intrinsics_path)
        self.annotation_dir = osp.join(self.data_root, 'Data','new_anno_test', self.CLASSES[cate_id])
        self.obj_annotation_list = []
        self.obj_urdf_id_list = []
        self.num_samples = 0
        # print(sorted(os.listdir(self.annotation_dir)))
        # pause = input()
        for video in sorted(os.listdir(self.annotation_dir)):
            video_path = osp.join(self.annotation_dir, video)
            for file in sorted(os.listdir(video_path)):
                if '.json' in file and file in self.annotation_valid_flags[self.cate_id]:
                # if '.json' in file:
                    # print(file)
                    # pause = input()
                    annotation = mmcv.load(osp.join(video_path, file))
                    annotation['mask_path'] = osp.join(self.data_root, 'Data', 'category_mask', annotation['depth_path'].replace('depth', 'category_mask'))
                    annotation['depth_path'] = osp.join(self.data_root, 'Data', 'depth', annotation['depth_path'])
                    annotation['color_path'] = osp.join(self.data_root, 'Data', 'color', annotation['color_path'])
                    instances = annotation['instances']
                    assert len(instances) == 1, 'Only support one instance per image'
                    instance = instances[0]
                    urdf_id = instance['urdf_id']
                    if (self.mode == 'train' and urdf_id not in self.TEST_URDF_IDs) \
                            or (self.mode == 'val' and urdf_id in self.TEST_URDF_IDs):
                        self.obj_annotation_list.append(annotation)
                        self.obj_urdf_id_list.append(urdf_id)
                        self.num_samples += 1
        print('Finish loading {} annotations!'.format(self.num_samples))

        self.urdf_ids_dict = dict()
        self.urdf_ids_dict[self.cate_id] = []
        self.urdf_dir = osp.join(self.data_root, 'URDF', self.CLASSES[cate_id])
        self.urdf_rest_transformation_dict = dict()
        self.urdf_rest_transformation_dict[self.cate_id] = dict()
        self.raw_urdf_joint_loc_dict = dict()
        self.raw_urdf_joint_loc_dict[self.cate_id] = dict()
        self.raw_urdf_joint_axis_dict = dict()
        self.raw_urdf_joint_axis_dict[self.cate_id] = dict()
        self.all_norm_obj_joint_loc_dict = dict()
        self.all_norm_obj_joint_loc_dict[self.cate_id] = dict()  # (joint anchor - center) -> normalized joint anchor
        self.all_norm_obj_joint_axis_dict = dict()
        self.all_norm_obj_joint_axis_dict[self.cate_id] = dict()  # numpy array, the same as raw joint axis
        self.all_obj_raw_scale = dict()
        self.all_obj_raw_scale[self.cate_id] = dict()  # raw mesh scale(rest state)
        self.all_obj_raw_center = dict()  # raw mesh center(rest state)
        self.all_obj_raw_center[self.cate_id] = dict()  # raw mesh center(rest state)
        self.norm_part_obj_corners = dict()
        self.norm_part_obj_corners[self.cate_id] = dict()  # raw mesh corners(rest state), part-level
        self.all_raw_obj_pts_dict = dict()  # raw complete obj pts(rest state)
        self.all_raw_obj_pts_dict[self.cate_id] = dict()
        self.norm_part_obj_pts_dict = dict()  # zero centered complete obj pts(rest state)
        self.norm_part_obj_pts_dict[self.cate_id] = dict()
        self.part_obj_raw_scale = dict()
        self.part_obj_raw_scale[self.cate_id] = dict()  # raw mesh scale(rest state), part-level
        self.part_obj_pts_dict = dict()
        self.part_obj_pts_dict[self.cate_id] = dict()
        for dir in sorted(os.listdir(self.urdf_dir)):
            if osp.isdir(osp.join(self.urdf_dir, dir)):
                urdf_id = int(dir)
                if (self.mode == 'train' and urdf_id not in self.TEST_URDF_IDs) \
                        or (self.mode == 'val' and urdf_id in self.TEST_URDF_IDs):
                    self.part_obj_pts_dict[self.cate_id][urdf_id] = [None for _ in range(self.num_parts)]
                    self.part_obj_raw_scale[self.cate_id][urdf_id] = [None for _ in range(self.num_parts)]
                    if urdf_id not in self.urdf_ids_dict[self.cate_id]:
                        self.urdf_ids_dict[self.cate_id].append(urdf_id)
                    new_urdf_file = osp.join(self.urdf_dir, dir, 'mobility_for_unity_align.urdf')
                    # more flexible
                    compute_relative = True if cate_id == 5 else False  # only applied for scissors
                    self.raw_urdf_joint_loc_dict[self.cate_id][urdf_id], \
                        self.raw_urdf_joint_axis_dict[self.cate_id][urdf_id] = \
                        self.parse_joint_info(new_urdf_file, self.cate_id)
                    with open(osp.join(self.urdf_dir, dir, 'model_pts.pkl'), 'rb') as f:
                        pts = pickle.load(f)
                    assert len(pts) - 1 == self.num_parts
                    for part_idx in range(len(pts) - 1):
                        self.part_obj_pts_dict[self.cate_id][urdf_id][part_idx] = pts[part_idx + 1]

                    for part_idx in range(self.num_parts):
                        _, part_scale, _ = self.get_norm_factor(self.part_obj_pts_dict[self.cate_id][urdf_id][part_idx])
                        self.part_obj_raw_scale[self.cate_id][urdf_id][part_idx] = part_scale

                    self.all_raw_obj_pts_dict[self.cate_id][urdf_id] = pts[0]

                    center, scale, _ = self.get_norm_factor(self.all_raw_obj_pts_dict[self.cate_id][urdf_id])
                    self.norm_part_obj_pts_dict[self.cate_id][urdf_id] = [
                        (self.part_obj_pts_dict[self.cate_id][urdf_id][part_idx] - center[np.newaxis, :])
                        for part_idx in range(self.num_parts)]
                    self.norm_part_obj_corners[self.cate_id][urdf_id] = [None for _ in range(self.num_parts)]
                    for part_idx in range(self.num_parts):
                        _, _, self.norm_part_obj_corners[self.cate_id][urdf_id][part_idx] = \
                            self.get_norm_factor(self.norm_part_obj_pts_dict[self.cate_id][urdf_id][part_idx])
                    self.norm_part_obj_corners[self.cate_id][urdf_id] = np.stack(
                        self.norm_part_obj_corners[self.cate_id][urdf_id], axis=0)

                    self.all_obj_raw_center[self.cate_id][urdf_id] = center
                    self.all_obj_raw_scale[self.cate_id][urdf_id] = scale

                    self.all_norm_obj_joint_loc_dict[self.cate_id][urdf_id] = []
                    self.all_norm_obj_joint_axis_dict[self.cate_id][urdf_id] = []
                    for part_idx in range(self.num_parts):
                        if part_idx in self.raw_urdf_joint_loc_dict[self.cate_id][urdf_id]:
                            self.all_norm_obj_joint_loc_dict[self.cate_id][urdf_id].append(
                                (self.raw_urdf_joint_loc_dict[self.cate_id][urdf_id][part_idx] - center))
                            self.all_norm_obj_joint_axis_dict[self.cate_id][urdf_id].append(
                                self.raw_urdf_joint_axis_dict[self.cate_id][urdf_id][part_idx]
                            )
                    self.all_norm_obj_joint_loc_dict[self.cate_id][urdf_id] = np.stack(
                        self.all_norm_obj_joint_loc_dict[self.cate_id][urdf_id], axis=0)
                    self.all_norm_obj_joint_axis_dict[self.cate_id][urdf_id] = np.stack(
                        self.all_norm_obj_joint_axis_dict[self.cate_id][urdf_id], axis=0)

        self.num_objs = len(self.part_obj_pts_dict[self.cate_id])
        self.samples_per_obj = self.num_samples // self.num_objs
        print('Finish loading {} objects!'.format(self.num_objs))

        self.cam_cx, self.cam_cy = self.camera_intrinsic.get_principal_point()
        self.cam_fx, self.cam_fy = self.camera_intrinsic.get_focal_length()
        self.width = self.camera_intrinsic.width
        self.height = self.camera_intrinsic.height

        self.xmap = np.array([[j for _ in range(self.width)] for j in range(self.height)])
        self.ymap = np.array([[i for i in range(self.width)] for _ in range(self.height)])


    @staticmethod
    def load_depth(depth_path):
        depth = cv2.imread(depth_path, -1)
        # print(depth)
        # pause = input()

        if len(depth.shape) == 3:
            depth16 = np.uint16(depth[:, :, 1]*256) + np.uint16(depth[:, :, 2])
            depth16 = depth16.astype(np.uint16)
        elif len(depth.shape) == 2 and depth.dtype == 'uint16':
            depth16 = depth
        else:
            assert False, '[ Error ]: Unsupported depth type.'

        return depth16

    def parse_joint_info(self, urdf_file, cate_id):
        # support base joint
        tree = ET.parse(urdf_file)
        root_urdf = tree.getroot()
        joint_loc_dict = dict()
        joint_axis_dict = dict()
        for i, joint in enumerate(root_urdf.iter('joint')):
            if joint.attrib['type'] == 'fixed' or joint.attrib['type'] == '0':
                continue
            if cate_id != 5:
                child_name = joint.attrib['name'].split('_')[-2]
            else:
                child_name = joint.attrib['name'].split('_')[0]
            child_idx = re.findall('\d+', child_name)[0]
            for axis in joint.iter('axis'):
                r, p, y = [float(x) for x in axis.attrib['xyz'].split()][::-1]
                axis = np.array([p, r, y])
                axis /= np.linalg.norm(axis)
                u, v, w = axis
                joint_axis_dict[int(child_idx)] = np.array([u, v, w])
                a, b, c = 0., 0., 0.
                joint_loc_dict[int(child_idx)] = np.array([a, b, c])
            for origin in joint.iter('origin'):
                x, y, z = [float(x) for x in origin.attrib['xyz'].split()][::-1]
                a, b, c = y, x, z
                joint_loc_dict[int(child_idx)] = np.array([a, b, c])
        return joint_loc_dict, joint_axis_dict

    @staticmethod
    def get_urdf_mobility(dir, filename='mobility_for_unity_align.urdf'):
        urdf_ins = {}
        tree_urdf = ET.parse(os.path.join(dir, filename))
        num_real_links = len(tree_urdf.findall('link'))
        root_urdf = tree_urdf.getroot()

        rpy_xyz = {}
        list_type = [None] * (num_real_links - 1)
        list_parent = [None] * (num_real_links - 1)
        list_child = [None] * (num_real_links - 1)
        list_xyz = [None] * (num_real_links - 1)
        list_rpy = [None] * (num_real_links - 1)
        list_axis = [None] * (num_real_links - 1)
        list_limit = [[0, 0]] * (num_real_links - 1)
        # here we still have to read the URDF file
        for joint in root_urdf.iter('joint'):
            joint_index = int(joint.attrib['name'].split('_')[1])
            list_type[joint_index] = joint.attrib['type']

            for parent in joint.iter('parent'):
                link_name = parent.attrib['link']
                if link_name == 'base':
                    link_index = 0
                else:
                    # link_index = int(link_name.split('_')[1]) + 1
                    link_index = int(link_name) + 1
                list_parent[joint_index] = link_index
            for child in joint.iter('child'):
                link_name = child.attrib['link']
                if link_name == 'base':
                    link_index = 0
                else:
                    # link_index = int(link_name.split('_')[1]) + 1
                    link_index = int(link_name) + 1
                list_child[joint_index] = link_index
            for origin in joint.iter('origin'):
                if 'xyz' in origin.attrib:
                    list_xyz[joint_index] = [float(x) for x in origin.attrib['xyz'].split()]
                else:
                    list_xyz[joint_index] = [0, 0, 0]
                if 'rpy' in origin.attrib:
                    list_rpy[joint_index] = [float(x) for x in origin.attrib['rpy'].split()]
                else:
                    list_rpy[joint_index] = [0, 0, 0]
            for axis in joint.iter('axis'):  # we must have
                list_axis[joint_index] = [float(x) for x in axis.attrib['xyz'].split()]
            for limit in joint.iter('limit'):
                list_limit[joint_index] = [float(limit.attrib['lower']), float(limit.attrib['upper'])]

        rpy_xyz['type'] = list_type
        rpy_xyz['parent'] = list_parent
        rpy_xyz['child'] = list_child
        rpy_xyz['xyz'] = list_xyz
        rpy_xyz['rpy'] = list_rpy
        rpy_xyz['axis'] = list_axis
        rpy_xyz['limit'] = list_limit

        urdf_ins['joint'] = rpy_xyz
        urdf_ins['num_links'] = num_real_links

        return urdf_ins

    @staticmethod
    def compose_rt(rotation, translation):
        aligned_RT = np.zeros((4, 4), dtype=np.float32)
        aligned_RT[:3, :3] = rotation[:3, :3]
        aligned_RT[:3, 3] = translation
        aligned_RT[3, 3] = 1
        return aligned_RT

    @staticmethod
    def get_norm_factor(obj_pts):
        xmin, xmax = np.min(obj_pts[:, 0]), np.max(obj_pts[:, 0])
        ymin, ymax = np.min(obj_pts[:, 1]), np.max(obj_pts[:, 1])
        zmin, zmax = np.min(obj_pts[:, 2]), np.max(obj_pts[:, 2])

        x_scale = xmax - xmin
        y_scale = ymax - ymin
        z_scale = zmax - zmin

        center = np.array([(xmin + xmax)/2., (ymin + ymax)/2., (zmin + zmax)/2.])
        scale = np.array([x_scale, y_scale, z_scale])
        corners = np.array([xmin, xmax, ymin, ymax, zmin, zmax])
        return center, scale, corners

    def get_frame(self, choose_frame_annotation):
        assert choose_frame_annotation['width'] == self.width
        assert choose_frame_annotation['height'] == self.height
        # print(choose_frame_annotation['color_path'])

    
        raw_transform_matrix = [np.array(choose_frame_annotation['instances'][0]['links'][i]['transformation'])
                                 for i in range(self.num_parts)]
        rest_transform_matrix = [np.diag([1., 1., 1., 1.]) for _ in range(self.num_parts)]
        joint_state = [0. for _ in range(self.num_parts)]
        urdf_id = choose_frame_annotation['instances'][0]['urdf_id']
        # TODO: more flexible
        joint_type = 'prismatic' if self.cate_id in [3, 4] else 'revolute'
        for link_idx in range(self.num_parts):
            if 'state' in choose_frame_annotation['instances'][0]['links'][link_idx]:
                if joint_type == 'revolute':
                    joint_state[link_idx] = choose_frame_annotation['instances'][0]['links'][link_idx]['state'] * 180 / np.pi
                else:
                    joint_state[link_idx] = \
                    choose_frame_annotation['instances'][0]['links'][link_idx]['state']

        rest_transform_matrix = np.array(rest_transform_matrix)
        joint_state = np.array(joint_state)
        all_center = copy.deepcopy(self.all_obj_raw_center[self.cate_id][urdf_id][np.newaxis, :])
        # new transform matrix for zero-centerd pts in rest state
        part_transform_matrix = [_ for _ in range(self.num_parts)]
        # X' = R_rest @ X_raw - Center(C)
        for part_idx in range(self.num_parts):
            # R' = R @ (R_rest)^-1
            transform_matrix = raw_transform_matrix[part_idx] @ np.linalg.inv(rest_transform_matrix[part_idx])
            # T' = T + R' @ C
            transform_matrix[:3, -1] = transform_matrix[:3, -1] + (transform_matrix[:3, :3] @ all_center.T)[:, 0]
            part_transform_matrix[part_idx] = transform_matrix

        part_target_r = np.stack([part_transform_matrix[i][:3, :3] for i in range(self.num_parts)], axis=0)
        raw_part_target_quat = np.stack([Rotation.from_matrix(part_target_r[i]).as_quat()
                                     for i in range(self.num_parts)], axis=0)  # (x, y, z, w)
        part_target_quat = np.concatenate([
            raw_part_target_quat[:, -1][:, np.newaxis], raw_part_target_quat[:, :3]], axis=-1)  # (w, x, y, z)
        part_target_t = np.stack([part_transform_matrix[i][:3, 3]
                                  for i in range(self.num_parts)], axis=0)[:, np.newaxis, :]

        depth = np.array(self.load_depth(choose_frame_annotation['depth_path'])) / 1000.0
        img_height, img_width = choose_frame_annotation['height'], choose_frame_annotation['width'] 
        # print((x1, y1, x2, y2))
        # pause = input()

        part_mask = mmcv.imread(choose_frame_annotation['mask_path'])[:, :, 0]
        clouds = [_ for _ in range(self.num_parts)]

        cam_scale = 1.0
        for i in range(self.num_parts):
            part_seg = choose_frame_annotation['instances'][0]['links'][i]['segmentation']

            clouds[i] = np.zeros([self.num_pts, 3])

            try:
                rle = maskUtils.frPyObjects(part_seg, img_height, img_width)
                part_label = np.sum(maskUtils.decode(rle), axis=2).clip(max=1).astype(np.uint8)


                part_depth = depth * part_label
                # part_mask = part_mask[y1:y2, x1:x2]

                choose = (part_depth.flatten() != 0.0).nonzero()[0]
                if len(choose) >= self.num_pts:
                    c_mask = np.random.choice(np.arange(len(choose)), self.num_pts, replace=False)
                    choose = choose[c_mask]
                else:
                    choose = np.pad(choose, (0, self.num_pts - len(choose)), 'wrap')

                xmap_masked = self.xmap.flatten()[choose][:, np.newaxis].astype(np.float32)
                ymap_masked = self.ymap.flatten()[choose][:, np.newaxis].astype(np.float32)

                depth_masked = part_depth.flatten()[choose][:, np.newaxis].astype(np.float32)
                pt2 = depth_masked / cam_scale
                pt0 = (ymap_masked - self.cam_cx) * pt2 / self.cam_fx
                pt1 = (xmap_masked - self.cam_cy) * pt2 / self.cam_fy

                clouds[i] = np.concatenate((pt0, pt1, pt2), axis=1)
            except:
                pass



        choose = (depth.flatten() != 0.0).nonzero()[0]
        assert len(choose) > 0
        if len(choose) >= self.num_pts:
            c_mask = np.random.choice(np.arange(len(choose)), self.num_pts, replace=False)
            choose = choose[c_mask]
        else:
            choose = np.pad(choose, (0, self.num_pts - len(choose)), 'wrap')

        part_cls = part_mask.flatten()[choose]

        
        return part_cls, clouds, part_target_r, part_target_quat, part_target_t, joint_state

    def __getitem__(self, index):
        choose_urdf_id = self.obj_urdf_id_list[index]
        # annotations of zero-centerd rest-state  keypoints in aligned model space
        norm_part_kp = copy.deepcopy(self.norm_part_kp_annotions[self.cate_id][choose_urdf_id])
        # zero-centered joint location and axis in rest state
        norm_joint_loc = copy.deepcopy(self.all_norm_obj_joint_loc_dict[self.cate_id][choose_urdf_id])
        norm_joint_axis = copy.deepcopy(self.all_norm_obj_joint_axis_dict[self.cate_id][choose_urdf_id])

        choose_frame_annotation = self.obj_annotation_list[index]
        try:  
            sort_part = choose_frame_annotation['sort_part']
        except:
            sort_part = 1

        part_cls, clouds, part_r, part_quat, part_t, joint_state = self.get_frame(choose_frame_annotation)

        cloud = np.concatenate([clouds[i] for i in range(len(clouds))], axis=0)  # (num_pts, 3)
        c_len = cloud.shape[0]
        if c_len > self.num_pts:
            c_mask = np.random.choice(np.arange(c_len), self.num_pts, replace=False)
            cloud = cloud[c_mask]
        norm_part_obj_pts = self.norm_part_obj_pts_dict[self.cate_id][choose_urdf_id]
        norm_pts_len = 200
        for i in range(self.num_parts):
            if norm_part_obj_pts[i].shape[0] >= norm_pts_len:
                c_mask = np.random.choice(np.arange(norm_part_obj_pts[i].shape[0]), norm_pts_len, replace=False)
                norm_part_obj_pts[i] = norm_part_obj_pts[i][c_mask]
            else:
                norm_part_obj_pts[i] = np.pad(norm_part_obj_pts[i], (0, norm_pts_len - norm_part_obj_pts[i].shape[0]), 'wrap')

        self.it = 0
        if self.debug:
            print(index % 29)
            colors = [(1, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 1)]
            gt_part_obj_pts = [norm_part_obj_pts[i] for i in range(self.num_parts)]
            # gt_part_obj_pts = [(part_r[i] @ norm_part_obj_pts[i].T).T + part_t[i] for i in range(self.num_parts)]
            gt_part_pts_pcd_list = [o3d.geometry.PointCloud() for _ in range(self.num_parts)]
            for part_idx in range(self.num_parts):
                gt_part_pts_pcd_list[part_idx].points = o3d.utility.Vector3dVector(gt_part_obj_pts[part_idx])
                gt_part_pts_pcd_list[part_idx].paint_uniform_color(colors[part_idx])
                # o3d.io.write_point_cloud(f'{temp_root}/gt_part_pcd_list/{part_idx}.ply', gt_part_pts_pcd_list[part_idx])

            cloud_pcd = o3d.geometry.PointCloud()
            cloud_pcd.points = o3d.utility.Vector3dVector(cloud)
            coord_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size = 0.5)

            o3d.visualization.draw_geometries([cloud_pcd, coord_pcd])

        class_gt = np.array([self.cate_id-1])
        raw_scale = self.all_obj_raw_scale[self.cate_id][choose_urdf_id]
        raw_center = self.all_obj_raw_center[self.cate_id][choose_urdf_id]
        norm_part_corners = self.norm_part_obj_corners[self.cate_id][choose_urdf_id]

        return torch.from_numpy(np.stack(clouds).astype(np.float32)).to(self.device), \
               torch.from_numpy(np.stack(norm_part_obj_pts).astype(np.float32)).to(self.device), \
               torch.from_numpy(part_cls).to(device=self.device, dtype=torch.long), \
               torch.from_numpy(part_r.astype(np.float32)).to(self.device), \
               torch.from_numpy(part_quat.astype(np.float32)).to(self.device), \
               torch.from_numpy(part_t.astype(np.float32)).to(self.device), \
               torch.from_numpy(joint_state.astype(np.float32)).to(self.device), \
               torch.from_numpy(norm_joint_loc.astype(np.float32)).to(self.device), \
               torch.from_numpy(norm_joint_axis.astype(np.float32)).to(self.device), \
               torch.from_numpy(norm_part_kp.astype(np.float32)).to(self.device), \
               torch.from_numpy(raw_scale.astype(np.float32)).to(self.device), \
               torch.from_numpy(raw_center.astype(np.float32)).to(self.device), \
               torch.from_numpy(norm_part_corners.astype(np.float32)).to(self.device), \
               torch.from_numpy(class_gt.astype(np.int32)).to(torch.long).to(self.device), \
               torch.tensor(choose_urdf_id, dtype=torch.long, device=self.device), \
               torch.tensor(sort_part, dtype=torch.long, device=self.device)

    def __len__(self):
        return self.num_samples


if __name__ == '__main__':
    dataset = SapienDataset_OMADNet(mode='train',
                                           data_root='/mnt/7797b2ec-a944-4795-abb2-f472a7fc833e/har/dataset_2',
                                           num_pts=1024,
                                           num_cates=5,
                                           cate_id=2,
                                           num_parts=2,
                                           kp_anno_path='../work_dir/ReArt_priornet_stapler_kp12/unsup_train_keypoints.pkl',
                                           device='cpu',
                                           debug=True)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    for i, data in enumerate(data_loader):
       cloud, norm_part_obj_pts, part_cls, part_r, part_quat, part_t, joint_state, norm_joint_loc, norm_joint_axis, \
        norm_part_kp, scale, center, norm_part_corners, cate, urdf_id, sort_part = data
       pass