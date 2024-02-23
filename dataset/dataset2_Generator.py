import pickle
import torch.utils.data as data
import os
import os.path as osp
import torch
import numpy as np
import open3d as o3d
import copy
import re
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation
import mmcv


class SapienDataset_OMADPriorNet(data.Dataset):
    CLASSES = ('background', 'box', 'stapler', 'cutter', 'drawer', 'scissor')
    TEST_URDF_IDs = (0, 1) + (10, 11) + (20, 21) + (30, 31) + (40, 41)

    def __init__(self, mode, data_root, num_pts, num_cates, cate_id, num_parts, add_noise=True,
                 debug=False, device='cuda:0', data_tag='train', num_samples=10000,
                 node_num=16, use_scale_aug=True, use_rot_aug=False, scale_aug_max_range=0.2):
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
        self.node_num = node_num
        assert self.node_num % self.num_parts == 0
        self.use_scale_aug = use_scale_aug
        self.use_rot_aug = use_rot_aug
        self.scale_aug_max_range = scale_aug_max_range

        self.obj_list = {}
        self.obj_name_list = {}
        self.num_samples = num_samples
        self.fathest_sampler = FarthestSampler()

        # point_cloud origin
        self.part_obj_pts_dict = dict()
        self.part_obj_pts_dict[self.cate_id] = dict()
        self.urdf_ids_dict = dict()
        self.urdf_ids_dict[self.cate_id] = []
        self.urdf_dir = osp.join(self.data_root, 'URDF', self.CLASSES[cate_id])
        self.urdf_joint_loc_dict = dict()
        self.urdf_joint_loc_dict[self.cate_id] = dict()
        self.urdf_joint_axis_dict = dict()
        self.urdf_joint_axis_dict[self.cate_id] = dict()
        self.all_norm_obj_joint_loc_dict = dict()
        self.all_norm_obj_joint_loc_dict[self.cate_id] = dict()  # (joint anchor - center) -> normalized joint anchor
        self.all_norm_obj_joint_axis_dict = dict()
        self.all_norm_obj_joint_axis_dict[self.cate_id] = dict()  # numpy array, the same as raw joint axis
        self.all_obj_raw_scale = dict()
        self.all_obj_raw_scale[self.cate_id] = dict()  # raw mesh scale(rest state)
        self.all_obj_raw_center = dict()  # raw mesh center(rest state)
        self.all_obj_raw_center[self.cate_id] = dict()  # raw mesh center(rest state)
        self.part_obj_raw_scale = dict()
        self.part_obj_raw_scale[self.cate_id] = dict()  # raw mesh scale(rest state), part-level
        self.all_raw_obj_pts_dict = dict()  # raw complete obj pts(rest state)
        self.all_raw_obj_pts_dict[self.cate_id] = dict()
        self.all_norm_obj_pts_dict = dict()  # zero centered complete obj pts(rest state)
        self.all_norm_obj_pts_dict[self.cate_id] = dict()
        self.all_part_cls_dict = dict()  # part classification label for complete pts(rest state)
        self.all_part_cls_dict[self.cate_id] = dict()
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
                    self.urdf_joint_loc_dict[self.cate_id][urdf_id],\
                        self.urdf_joint_axis_dict[self.cate_id][urdf_id] = \
                        self.parse_joint_info(new_urdf_file, self.cate_id)
                    with open(osp.join(self.urdf_dir, dir, 'model_pts.pkl'), 'rb') as f:
                        pts = pickle.load(f)
                    assert len(pts) - 1 == self.num_parts
                    for part_idx in range(len(pts) - 1):
                        self.part_obj_pts_dict[self.cate_id][urdf_id][part_idx] = pts[part_idx + 1]

                    for part_idx in range(self.num_parts):
                        _, part_scale = self.get_norm_factor(self.part_obj_pts_dict[self.cate_id][urdf_id][part_idx])
                        self.part_obj_raw_scale[self.cate_id][urdf_id][part_idx] = part_scale

                    self.all_raw_obj_pts_dict[self.cate_id][urdf_id] = pts[0]

                    center, scale = self.get_norm_factor(self.all_raw_obj_pts_dict[self.cate_id][urdf_id])
                    self.all_norm_obj_pts_dict[self.cate_id][urdf_id] = \
                        (self.all_raw_obj_pts_dict[self.cate_id][urdf_id] - center[np.newaxis, :])
                    # self.all_norm_obj_pts_dict[self.cate_id][urdf_id] = \
                    #     (self.all_raw_obj_pts_dict[self.cate_id][urdf_id])
                    self.all_obj_raw_center[self.cate_id][urdf_id] = center
                    self.all_obj_raw_scale[self.cate_id][urdf_id] = scale

                    self.all_part_cls_dict[self.cate_id][urdf_id] = []
                    self.all_norm_obj_joint_loc_dict[self.cate_id][urdf_id] = []
                    self.all_norm_obj_joint_axis_dict[self.cate_id][urdf_id] = []
                    for part_idx in range(self.num_parts):
                        part_pts_num = self.part_obj_pts_dict[self.cate_id][urdf_id][part_idx].shape[0]
                        self.all_part_cls_dict[self.cate_id][urdf_id].append(part_idx * np.ones((part_pts_num,)))
                        if part_idx in self.urdf_joint_loc_dict[self.cate_id][urdf_id]:
                            self.all_norm_obj_joint_loc_dict[self.cate_id][urdf_id].append(
                                (self.urdf_joint_loc_dict[self.cate_id][urdf_id][part_idx] - center))
                            self.all_norm_obj_joint_axis_dict[self.cate_id][urdf_id].append(
                                self.urdf_joint_axis_dict[self.cate_id][urdf_id][part_idx]
                            )
                    self.all_part_cls_dict[self.cate_id][urdf_id] = \
                        np.concatenate(self.all_part_cls_dict[self.cate_id][urdf_id]).astype(np.int32)
                    self.all_norm_obj_joint_loc_dict[self.cate_id][urdf_id] = np.stack(
                        self.all_norm_obj_joint_loc_dict[self.cate_id][urdf_id], axis=0)
                    self.all_norm_obj_joint_axis_dict[self.cate_id][urdf_id] = np.stack(
                        self.all_norm_obj_joint_axis_dict[self.cate_id][urdf_id], axis=0)

        self.num_objs = len(self.part_obj_pts_dict[self.cate_id])
        print('Finish loading {} objects!'.format(self.num_objs))


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
        return center, scale

    def get_frame(self, complete_all_mesh_pts, complete_all_part_cls, raw_joint_loc, raw_joint_axis):
        if self.use_rot_aug:
            angles = np.array([np.random.uniform(-7., 7.),
                               np.random.uniform(-7., 7.),
                               np.random.uniform(-7., 7.)])
        else:
            angles = np.array([0., 0., 0.])
        part_target_r = Rotation.from_euler('yxz', angles, degrees=True).as_matrix()
        part_target_inv_r = part_target_r.T

        num_all_pts = complete_all_mesh_pts.shape[0]
        if num_all_pts >= self.num_pts:
            choose = np.random.choice(np.arange(num_all_pts), self.num_pts, replace=False)
        else:
            choose = np.pad(np.arange(num_all_pts), (0, self.num_pts - num_all_pts), 'wrap')
        all_mesh_pts = copy.deepcopy(complete_all_mesh_pts[choose, :])
        all_part_cls = copy.deepcopy(complete_all_part_cls[choose])

        if self.use_scale_aug:
            scale_factor = 2 * self.scale_aug_max_range * np.random.rand(1, 3) + (1-self.scale_aug_max_range)  # [1-x, 1+x)
        else:
            scale_factor = np.ones((1, 3))
        cloud = (part_target_r @ (all_mesh_pts * scale_factor).T).T
        joint_loc = (part_target_r @ (copy.deepcopy(raw_joint_loc) * scale_factor).T).T
        joint_axis = (part_target_r @ (copy.deepcopy(raw_joint_axis) * scale_factor).T).T

        if self.mode == 'train' and self.add_noise:
            # add random noise to point cloud
            cloud = cloud + np.random.normal(loc=0.0, scale=0.003, size=cloud.shape)

        nodes_list = []
        for part_idx in range(self.num_parts):
            part_cloud = cloud[all_part_cls == part_idx]
            pts_num = part_cloud.shape[0]
            part_nodes = self.fathest_sampler.sample(
                part_cloud[np.random.choice(part_cloud.shape[0], pts_num, replace=False)],
                self.node_num // self.num_parts,
            )
            nodes_list.append(part_nodes)
        nodes = np.concatenate(nodes_list, axis=0)

        return cloud, nodes, scale_factor, all_part_cls, part_target_r, part_target_inv_r, joint_loc, joint_axis

    def __getitem__(self, index):
        # print(index)
        choose_urdf_id = np.random.choice(self.urdf_ids_dict[self.cate_id], 1)[0]
        # use normalized(zero-centered) all obj points
        choose_obj_pts = self.all_norm_obj_pts_dict[self.cate_id][choose_urdf_id]
        # classification(part-label)
        choose_obj_cls = self.all_part_cls_dict[self.cate_id][choose_urdf_id]
        # normazlied joint location and axis
        choose_joint_loc = self.all_norm_obj_joint_loc_dict[self.cate_id][choose_urdf_id]
        choose_joint_axis = self.all_norm_obj_joint_axis_dict[self.cate_id][choose_urdf_id]

        cloud, nodes, scale_factor, part_cls, part_r, part_inv_r, joint_loc, joint_axis = \
            self.get_frame(choose_obj_pts, choose_obj_cls, choose_joint_loc, choose_joint_axis)
        self.it = 1
        if self.debug and choose_urdf_id:
            # print(self.num_parts - 1)
            print(choose_urdf_id)
            colors = [(1., 0., 0.), (0., 1., 0.), (0., 0., 1.), (1., 1., 0.), (1., 0., 1.)]
            cloud_pcd = o3d.geometry.PointCloud()
            cloud_pcd.points = o3d.utility.Vector3dVector(cloud)
            cloud_pcd.paint_uniform_color([0., 0., 0.])
            pcd_color = np.zeros((self.num_pts, 3))
            for part_idx in range(self.num_parts):
                part_num = np.sum(part_cls == part_idx)
                pcd_color[part_cls == part_idx, :] = np.array(colors[part_idx])[np.newaxis, :].repeat(part_num, axis=0)
            cloud_pcd.colors = o3d.utility.Vector3dVector(pcd_color)

            nodes_list = [
                o3d.geometry.TriangleMesh.create_sphere(radius=0.002, resolution=10).translate((x, y, z)) for x, y, z
                in nodes]
            sphere_pts_num = np.asarray(nodes_list[0].vertices).shape[0]
            for idx, mesh in enumerate(nodes_list):
                part_idx = idx // (self.node_num // self.num_parts)
                mesh.vertex_colors = o3d.utility.Vector3dVector(
                    np.array(colors[part_idx])[np.newaxis, :].repeat(sphere_pts_num, axis=0))

            line_pcd_list = []
            for joint_idx in range(self.num_parts - 1):
                start_point = joint_loc[joint_idx]
                end_point = start_point + joint_axis[joint_idx]
                line_points = np.stack([start_point, end_point])
                lines = [[0, 1]]  # Right leg
                colors = [[0, 0, 1] for _ in range(len(lines))]
                line_pcd = o3d.geometry.LineSet()
                line_pcd.lines = o3d.utility.Vector2iVector(lines)
                line_pcd.colors = o3d.utility.Vector3dVector(colors)
                line_pcd.points = o3d.utility.Vector3dVector(line_points)
                line_pcd_list.append(line_pcd)

            self.it += 1
            coord_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
            
            o3d.visualization.draw_geometries([cloud_pcd, coord_mesh] + nodes_list + line_pcd_list)
            


        class_gt = np.array([self.cate_id-1])
        raw_part_scale = np.stack(self.part_obj_raw_scale[self.cate_id][choose_urdf_id], axis=0)
        raw_center = self.all_obj_raw_center[self.cate_id][choose_urdf_id]

        return torch.from_numpy(cloud.astype(np.float32)).to(self.device), \
               torch.from_numpy(nodes.astype(np.float32)).to(self.device), \
               torch.from_numpy(part_cls.astype(np.int32)).to(torch.long).to(self.device), \
               torch.from_numpy(part_inv_r.astype(np.float32)).to(self.device), \
               torch.from_numpy(joint_loc.astype(np.float32)).to(self.device), \
               torch.from_numpy(joint_axis.astype(np.float32)).to(self.device), \
               torch.from_numpy(scale_factor.astype(np.float32)).to(self.device), \
               torch.from_numpy(raw_part_scale.astype(np.float32)).to(self.device), \
               torch.from_numpy(raw_center.astype(np.float32)).to(self.device), \
               torch.from_numpy(class_gt.astype(np.int32)).to(torch.long).to(self.device), \
               torch.tensor(choose_urdf_id, dtype=torch.long, device=self.device)

    def __len__(self):
        return self.num_samples


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


if __name__ == '__main__':
    dataset = SapienDataset_OMADPriorNet(mode='train',
                                         data_root='/mnt/7797b2ec-a944-4795-abb2-f472a7fc833e/har/dataset_2',
                                         num_pts=1024,
                                         num_cates=5,
                                         cate_id=2,
                                         num_parts=2,
                                         device='cpu',
                                         debug=False,
                                         node_num=8,
                                         use_scale_aug=False,
                                         use_rot_aug=False)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    try:
        for i, data in enumerate(data_loader):
            cloud, nodes, part_cls, part_inv_r, joint_loc, joint_axis, \
                scale_factor, raw_part_scale, raw_center, cate, urdf_id = data
            pass
    except Exception as e:
        print(e)
