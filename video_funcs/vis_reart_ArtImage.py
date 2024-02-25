import os
import os.path as osp
import copy
import open3d as o3d
import cv2
import numpy as np
import json
import random
from pycocotools.coco import COCO
from tqdm import tqdm
import mmcv
import matplotlib.pyplot as plt
import seaborn as sns
# TODO:

# def fetch_factors_nocs(root_dset):
#     urdf_metas = json.load(open(root_dset + '/urdf_metas_franka.json', 'r'))['urdf_metas']
#     norm_factors = np.array(urdf_metas[0]['norm_factors'])
#     corner_pts = np.array(urdf_metas[0]['corner_pts'])

#     return norm_factors, corner_pts

def fetch_factors_nocs(urdf_metas_path, urdf_id, link_id):
    urdf_metas = json.load(open(urdf_metas_path, 'r'))['urdf_metas']
    for urdf_meta in urdf_metas:
        if urdf_meta['id'] == urdf_id:
            u = urdf_meta
    # print(u['object_name'])
    norm_factors = np.array(u['norm_factors'][link_id+1])
    corner_pts = np.array(u['corner_pts'][link_id+1])

    return norm_factors, corner_pts

def fetch_joints_params(root_dset):
    joint_ins = {'xyz': [], 'axis': []}
    joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
    joint_info = json.load(open(root_dset + '/joint_infos.json', 'r'))
    for joint_name in joint_names:
        joint_ins['xyz'].append(joint_info[joint_name]['xyz'])
        joint_ins['axis'].append(joint_info[joint_name]['axis'])

    return joint_ins


def compose_rt(rotation, translation):
    aligned_RT = np.zeros((4, 4), dtype=np.float32)
    aligned_RT[:3, :3] = rotation[:3, :3]
    aligned_RT[:3, 3]  = translation
    aligned_RT[3, 3]   = 1
    return aligned_RT


def transform_coordinates_3d(coordinates, RT):
    assert coordinates.shape[0] == 3
    coordinates = np.vstack([coordinates, np.ones((1, coordinates.shape[1]), dtype=np.float32)])
    new_coordinates = RT @ coordinates
    new_coordinates = new_coordinates[:3, :] / new_coordinates[3, :]
    return new_coordinates


def calculate_2d_projections(coordinates_3d, intrinsics):
    projected_coordinates = intrinsics @ coordinates_3d
    projected_coordinates = projected_coordinates[:2, :] / projected_coordinates[2, :]
    projected_coordinates = projected_coordinates.transpose()
    projected_coordinates = np.array(projected_coordinates, dtype=np.int32)

    return projected_coordinates

if __name__ == '__main__':
#     norm_factors, corner_pts = fetch_factors_nocs('.')
#     joint_ins = fetch_joints_params('.')
    # kp_colors = [[255, 97, 0], [61,145,64], [138,43,226]]
    kp_colors = [(0,0,255), (255,0,0)]
    
    path = 'ReArt/Demo'
    category_name = 'drawer'
    
    camera_intrinsic = o3d.io.read_pinhole_camera_intrinsic('ReArt/Demo/camera_intrinsic.json')
    image_list = os.listdir('{}/color'.format(path))
    image_list.sort()
    # random.shuffle(image_list)
    kp_path = './work_dir/ReArt_priornet_drawer_kp16'
    link_kps = mmcv.load(osp.join(kp_path, 'unsup_test_keypoints.pkl'))
    
    for j, image_name in enumerate(image_list):
        if j == 29:
            break
        image = cv2.imread('{}/color/{}'.format(path, image_name))
        annotation = json.load(open('{}/annotations_compose/{}'.format(path, image_name.split('.')[0]+'.json'), 'r'))
        instance_info = annotation['instances'][1]
        urdf_id = instance_info['urdf_id']
        link_kp = link_kps[urdf_id]
        link_infos = instance_info['links']
        transformations = np.load(f'./temp/Demo/{category_name}/{j}/rts.npy')

        for i, link_info in enumerate(link_infos):
            link_id = link_info['id']
            norm_factors, corner_pts = fetch_factors_nocs('ReArt/URDF/urdf_metas.json', urdf_id, link_id)
            transformation = transformations[i]
            
            xyz_axis = 0.03 * np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]]).transpose()
            xyz_offset = corner_pts[:,0,:].mean(axis=0)
            xyz_axis += xyz_offset.reshape(3,1)
            
            transformed_axes = transform_coordinates_3d(xyz_axis, transformation)
            projected_axes = calculate_2d_projections(transformed_axes, camera_intrinsic.intrinsic_matrix)

            cv2.line(image, tuple(projected_axes[0]), tuple(projected_axes[1]), (0, 0, 255), 3)
            cv2.line(image, tuple(projected_axes[0]), tuple(projected_axes[3]), (255, 0, 0), 3)
            cv2.line(image, tuple(projected_axes[0]), tuple(projected_axes[2]), (0, 255, 0), 3)  ## y last
            
            # 3D box
            # corner_pt = corner_pts[i + 1]
            x1, y1, z1 = corner_pts[0][0]
            x2, y2, z2 = corner_pts[1][0]
            amodal_box3d = np.array([
                    [x1, y1, z1],  # 0
                    [x1, y2, z1],  # 1
                    [x1, y1, z2],  # 2
                    [x1, y2, z2],  # 3
                    [x2, y1, z1],  # 4
                    [x2, y1, z2],  # 5
                    [x2, y2, z1],  # 6
                    [x2, y2, z2]  # 7
                ]).transpose()
            
            transformed_box = transform_coordinates_3d(amodal_box3d, transformation)
            projected_box = calculate_2d_projections(transformed_box, camera_intrinsic.intrinsic_matrix)
            color = (0, 255, 255)

            cv2.line(image, tuple(projected_box[0]), tuple(projected_box[1]), color, 1)
            cv2.line(image, tuple(projected_box[0]), tuple(projected_box[2]), color, 1)
            cv2.line(image, tuple(projected_box[0]), tuple(projected_box[4]), color, 1)
            cv2.line(image, tuple(projected_box[1]), tuple(projected_box[3]), color, 1)
            cv2.line(image, tuple(projected_box[1]), tuple(projected_box[6]), color, 1)
            cv2.line(image, tuple(projected_box[2]), tuple(projected_box[5]), color, 1)
            cv2.line(image, tuple(projected_box[2]), tuple(projected_box[3]), color, 1)
            cv2.line(image, tuple(projected_box[3]), tuple(projected_box[7]), color, 1)
            cv2.line(image, tuple(projected_box[4]), tuple(projected_box[6]), color, 1)
            cv2.line(image, tuple(projected_box[4]), tuple(projected_box[5]), color, 1)
            cv2.line(image, tuple(projected_box[5]), tuple(projected_box[7]), color, 1)
            cv2.line(image, tuple(projected_box[6]), tuple(projected_box[7]), color, 1)

                    # joint
            #             if i != 0:
            #                 joint_axis = np.array(joint_ins['axis'][i-1]) * 0.2
            #                 joint_xyz = np.array(joint_ins['xyz'][i-1])
            #                 line_start = [(joint_xyz[j] + joint_axis[j]) for j in range(len(joint_xyz))]
            #                 line_end = [(joint_xyz[j] - joint_axis[j]) for j in range(len(joint_xyz))]
            #                 joint = np.array([line_start, line_end]).transpose()

            #                 transformed_joint = transform_coordinates_3d(joint, transformation)
            #                 projected_joint = calculate_2d_projections(transformed_joint, camera_intrinsic.intrinsic_matrix)

            #                 cv2.arrowedLine(image, tuple(projected_joint[1]), tuple(projected_joint[0]), (255, 0, 255), 2,
            #                                 cv2.LINE_AA, 0, tipLength=0.1)
        
            transformed_kps = transform_coordinates_3d(link_kp[i].T, transformation)
            projected_kps = calculate_2d_projections(transformed_kps, camera_intrinsic.intrinsic_matrix)
            for projected_kp in projected_kps:
                cv2.circle(image, projected_kp, 8, kp_colors[i], -1)
            
            
        # cv2.imshow('vis', image)
        # cv2.waitKey(0)
        if not os.path.lexists('{}/demo_video5'.format(path)):
            os.makedirs('{}/demo_video5'.format(path))
        output_image_name = '{}/demo_video5/{}'.format(path, image_name)
        mmcv.imwrite(image, output_image_name)
