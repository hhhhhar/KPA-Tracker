import numpy as np
import itertools
import torch
import torch.nn.functional as F

def iou_3d(bbox1, bbox2, nres=50):
    bmin = np.min(np.concatenate((bbox1, bbox2), 0), 0)
    bmax = np.max(np.concatenate((bbox1, bbox2), 0), 0)
    xs = np.linspace(bmin[0], bmax[0], nres)
    ys = np.linspace(bmin[1], bmax[1], nres)
    zs = np.linspace(bmin[2], bmax[2], nres)
    pts = np.array([x for x in itertools.product(xs, ys, zs)])
    flag1 = pts_inside_box(pts, bbox1)
    flag2 = pts_inside_box(pts, bbox2)
    intersect = np.sum(np.logical_and(flag1, flag2))
    union = np.sum(np.logical_or(flag1, flag2))
    if union == 0:
        return 1
    else:
        return intersect/float(union)


def pts_inside_box(pts, bbox):
    # pts: N x 3
    # bbox: 8 x 3 (-1, 1, 1), (1, 1, 1), (1, -1, 1), (-1, -1, 1), (-1, 1, -1), (1, 1, -1), (1, -1, -1), (-1, -1, -1)
    u1 = bbox[5, :] - bbox[4, :]
    u2 = bbox[7, :] - bbox[4, :]
    u3 = bbox[0, :] - bbox[4, :]

    up = pts - np.reshape(bbox[4, :], (1, 3))
    p1 = np.matmul(up, u1.reshape((3, 1)))
    p2 = np.matmul(up, u2.reshape((3, 1)))
    p3 = np.matmul(up, u3.reshape((3, 1)))
    p1 = np.logical_and(p1>0, p1<np.dot(u1, u1))
    p2 = np.logical_and(p2>0, p2<np.dot(u2, u2))
    p3 = np.logical_and(p3>0, p3<np.dot(u3, u3))
    return np.logical_and(np.logical_and(p1, p2), p3)


def get_part_bbox_from_kp(part_kp):
    if torch.is_tensor(part_kp):
        part_kp = part_kp.detach().cpu().numpy()
    num_parts = part_kp.shape[0]
    part_bbox_list = []
    for part_idx in range(num_parts):
        kp = part_kp[part_idx]
        xmin, xmax = np.min(kp[:, 0]), np.max(kp[:, 0])
        ymin, ymax = np.min(kp[:, 1]), np.max(kp[:, 1])
        zmin, zmax = np.min(kp[:, 2]), np.max(kp[:, 2])
        bbox = np.array([[xmin, ymax, zmax],
                         [xmax, ymax, zmax],
                         [xmax, ymin, zmax],
                         [xmin, ymin, zmax],
                         [xmin, ymax, zmin],
                         [xmax, ymax, zmin],
                         [xmax, ymin, zmin],
                         [zmin, ymin, zmin]])
        part_bbox_list.append(bbox)
    part_bbox = np.stack(part_bbox_list, axis=0)
    return part_bbox


def get_part_bbox_from_corners(corners):
    if torch.is_tensor(corners):
        corners = corners.detach().cpu().numpy()
    num_parts = corners.shape[0]
    part_bbox_list = []
    for part_idx in range(num_parts):
        xmin, xmax, ymin, ymax, zmin, zmax = corners[part_idx]
        bbox = np.array([[xmin, ymax, zmax],
                         [xmax, ymax, zmax],
                         [xmax, ymin, zmax],
                         [xmin, ymin, zmax],
                         [xmin, ymax, zmin],
                         [xmax, ymax, zmin],
                         [xmax, ymin, zmin],
                         [zmin, ymin, zmin]])
        part_bbox_list.append(bbox)
    part_bbox = np.stack(part_bbox_list, axis=0)
    return part_bbox


def calc_joint_errors(pred_joint_loc, pred_joint_axis, gt_joint_loc, gt_joint_axis, pred_joint_state, gt_joint_state,
                      joint_type='revolute'):
    axis_err = torch.acos(F.cosine_similarity(pred_joint_axis, gt_joint_axis, dim=-1).mean(dim=-1).mean(dim=-1)).item() \
               / np.pi * 180

    orth_vect = torch.cross(pred_joint_axis, gt_joint_axis)
    product = torch.sum(orth_vect * (pred_joint_loc - gt_joint_loc))
    loc_err = torch.abs(product / torch.norm(orth_vect, dim=-1)).item()

    state_err = torch.abs(pred_joint_state - gt_joint_state).item()
    # if joint_type == 'revolute':
    #     state_err = torch.abs(pred_joint_state - gt_joint_state).item() / np.pi * 180
    # else:
    #     state_err = torch.abs(pred_joint_state - gt_joint_state).item()
    return loc_err, axis_err, state_err