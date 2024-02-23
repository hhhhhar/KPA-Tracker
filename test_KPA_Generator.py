import argparse
import numpy as np
import os
import os.path as osp
import open3d as o3d
import tqdm
# from tensorboardX import SummaryWriter
import mmcv
import torch.utils.data
from matplotlib import cm

from model.KPA_Generator import KPA_Generator
from dataset.dataset1_Generator import SapienDataset_OMADPriorNet

cate_list = ['laptop', 'eyeglasses', 'dishwasher', 'drawer', 'scissors']  

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=str, default='ArtImage', help='dataset root dir')
    parser.add_argument('--checkpoint', type=str, default=None, help='test checkpoint')
    parser.add_argument('--category', type=int, default=1, help='category to train')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'val'],
                        help='the mode of training or validation')
    parser.add_argument('--num_points', type=int, default=2048, help='points')
    parser.add_argument('--num_cates', type=int, default=1, help='number of categories')
    parser.add_argument('--num_parts', type=int, default=2, help='number of parts')
    parser.add_argument('--num_basis', type=int, default=10, help='number of shape basis')
    parser.add_argument('--workers', type=int, default=0, help='number of data loading workers')
    parser.add_argument('--bs', type=int, default=1, help='batch size')
    parser.add_argument('--num_kp', type=int, default=8, help='number of all keypoints')
    parser.add_argument('--symtype', type=str, default='shape', choices=['shape', 'none'], help='the symmetry type')
    parser.add_argument('--work_dir', type=str, default='work_dir/base', help='working dir')
    parser.add_argument('--out', action='store_true', help='whether to save outputs')
    parser.add_argument('--show', action='store_true', help='wheter to visualize keypoints')
    parser.add_argument('--use_gpu', action='store_true', help='wheter to use GPU')
    opt = parser.parse_args()

    device = torch.device("cuda") if opt.use_gpu else torch.device("cpu")
    model = KPA_Generator(node_num=opt.num_kp, basis_num=opt.num_basis, part_num=opt.num_parts, device=device,
                        symtype=opt.symtype)
    model = model.to(device)

    assert opt.checkpoint is not None
    model.load_state_dict(torch.load(osp.join(opt.work_dir, opt.checkpoint), map_location=device))

    test_dataset = SapienDataset_OMADPriorNet(opt.mode,
                                        data_root=opt.dataset_root,
                                        add_noise=False,
                                        num_pts=opt.num_points,
                                        num_parts=opt.num_parts,
                                        num_cates=opt.num_cates, cate_id=opt.category,
                                        num_samples=1000 if opt.mode == 'train' else 1000,
                                        node_num=opt.num_kp,
                                        use_scale_aug=False,
                                        use_rot_aug=False,
                                        device=torch.device("cpu"))
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.bs, shuffle=False,
                                                  num_workers=opt.workers)
    test_dataloader = tqdm.tqdm(test_dataloader)

    assert opt.num_kp % opt.num_parts == 0, 'number of keypoints must be divided by number of parts'

    model.eval()
    results = dict()
    coefs_list = []
    for j, data in enumerate(test_dataloader):
        cloud, init_nodes, cloud_cls, gt_inv_r, gt_joint_loc, gt_joint_axis, scale_factor, \
            raw_scale, raw_center, cate, urdf_id = data
        cloud, init_nodes, gt_inv_r, cloud_cls, gt_joint_loc, gt_joint_axis = \
            cloud.to(device), init_nodes.to(device), gt_inv_r.to(device), cloud_cls.to(device), \
            gt_joint_loc.to(device), gt_joint_axis.to(device)
        scale_factor, raw_scale, raw_center = scale_factor.to(device), raw_scale.to(device), raw_center.to(device)

        bs_now = cloud.shape[0]
        gt_r = gt_inv_r.transpose(1, 2)
        gt_t = torch.zeros(bs_now, 1, 3, device=device)

        with torch.no_grad():
            final_nodes, coefs, pred_joint_loc, pred_joint_axis = model(cloud, None, init_nodes)
            # with SummaryWriter(log_dir="network_visualization") as w:
            #     w.add_graph(model, (cloud, torch.zeros_like(cloud), init_nodes, ))
            # w.close()
            part_pred_kps = model.get_transformed_pred_keypoints(coefs, gt_r, gt_t)
            trans_joint_loc, trans_joint_axis = model.get_transformed_joint_params(pred_joint_loc, pred_joint_axis,
                                                                                   gt_r, gt_t)

        coefs_list.append(coefs.squeeze().cpu().numpy())
        pred_kps = part_pred_kps.reshape(bs_now, -1, 3)
        trans_Kp = torch.bmm(gt_inv_r, (pred_kps - gt_t).transpose(1, 2)).transpose(1, 2).contiguous()
        if opt.category == 2 and opt.num_kp == 12:
            trans_Kp = trans_Kp.reshape(bs_now, opt.num_parts, -1, 3)
            child_kps = trans_Kp[:, 1:, [0, 3, 2, 1], :]
            new_second = torch.cat((child_kps[:, :, 0, :], child_kps[:, :, 1, :]), dim=1)
            new_frist = torch.cat((child_kps[:, :, 2, :], child_kps[:, :, 3, :]), dim=1)
            new_child_kps = torch.stack((new_frist, new_second), dim=1)
            trans_Kp[:, 1:, :] = new_child_kps
            trans_Kp = trans_Kp.reshape(bs_now, -1, 3)
        align_kp = trans_Kp / scale_factor  # zero-centered keypoints
        part_align_kp = align_kp.reshape(bs_now, opt.num_parts, -1, 3)
        trans_cloud = torch.bmm(gt_inv_r, (cloud - gt_t).transpose(1, 2)).transpose(1, 2).contiguous()

        for idx, id in enumerate(urdf_id.cpu().numpy()):
            if id not in results:
                results[id] = []
            results[id].append(part_align_kp[idx].unsqueeze(0).cpu().numpy())

        if opt.show and opt.bs == 1:
            print('urdf id = {}'.format(urdf_id.cpu().numpy()))
            base_colors = np.array([(1., 0., 0.), (0., 1., 0.), (0., 0., 1.), (0., 1., 1.)] * 2)
            cls_colors = np.array([(1., 0., 0.), (0., 1., 0.), (0., 0., 1.)])
            cmap = cm.get_cmap("jet", opt.num_kp)
            kp_colors = cmap(np.linspace(0, 1, opt.num_kp, endpoint=True))[:, :3]

            trans_pcd = o3d.geometry.PointCloud()
            trans_pcd.points = o3d.utility.Vector3dVector(trans_cloud[0].cpu().numpy())
            trans_pcd.paint_uniform_color([0, 0, 0])

            trans_pcd_kp_list = [
                o3d.geometry.TriangleMesh.create_sphere(radius=0.008, resolution=10).translate((x, y, z)) for
                x, y, z
                in trans_Kp[0].detach().cpu().numpy()]

            sphere_pts_num = np.asarray(trans_pcd_kp_list[0].vertices).shape[0]
            for idx, mesh in enumerate(trans_pcd_kp_list):
                mesh.vertex_colors = o3d.utility.Vector3dVector(
                    kp_colors[np.newaxis, idx, :].repeat(sphere_pts_num, axis=0))

            # TODO: support prismatic joints
            line_pcd_list = []
            for joint_idx in range(opt.num_parts - 1):
                start_point = trans_joint_loc[0, joint_idx, :].cpu().numpy()
                end_point = start_point + trans_joint_axis[0, joint_idx, :].cpu().numpy()
                line_points = np.stack([start_point, end_point], axis=0)
                lines = [[0, 1]]  # Right leg
                colors = [[1, 0, 0] for _ in range(len(lines))]
                line_pcd = o3d.geometry.LineSet()
                line_pcd.lines = o3d.utility.Vector2iVector(lines)
                line_pcd.colors = o3d.utility.Vector3dVector(colors)
                line_pcd.points = o3d.utility.Vector3dVector(line_points)
                line_pcd_list.append(line_pcd)

            part_kp_num = int(opt.num_kp / opt.num_parts)
            for i in range(len(trans_pcd_kp_list[:(opt.num_parts-1) * part_kp_num])):
                print(i)
                o3d.visualization.draw_geometries([trans_pcd, trans_pcd_kp_list[i+part_kp_num]] + line_pcd_list )
            o3d.visualization.draw_geometries([trans_pcd] + line_pcd_list + trans_pcd_kp_list)

    for id in results.keys():
        print('urdf {} has {} valid results!'.format(id, len(results[id])))
        results[id] = np.mean(np.concatenate(results[id], axis=0), axis=0)

    coefs_all = np.concatenate(coefs_list, axis=0)

    if opt.out:
        tag = 'train' if opt.mode == 'train' else 'test'
        kp_path = osp.join(opt.work_dir, 'unsup_{}_keypoints.pkl'.format(tag))
        mmcv.dump(results, kp_path)
        print('Saving keypoint results to {}!'.format(kp_path))
        coefs_path = osp.join(opt.work_dir, 'unsup_{}_coefs.pkl'.format(tag))
        mmcv.dump(coefs_all, coefs_path)
        print('Saving coefs results to {}!'.format(coefs_path))

        save_params_dict = dict()
        save_params_dict['joint_net'] = model.joint_net.state_dict()
        save_params_dict['basis'] = model.state_dict()['basis']
        save_params_dict['n_pl'] = model.state_dict()['n_pl']
        params_path = osp.join(opt.work_dir, 'params.pth')
        torch.save(save_params_dict, params_path)
        print('Saving params to {}!'.format(params_path))

        if opt.show:
            base_colors = np.array([(1., 0., 0.), (0., 1., 0.), (0., 0., 1.), (0., 1., 1.)] * 2)
            points_dir = osp.join(opt.work_dir, f'urdf_normal_points_{opt.mode}')
            os.mkdir(points_dir)
            for id in results.keys():
                print('urdf id={}'.format(id))
                all_align_pts = test_dataset.all_norm_obj_pts_dict[opt.category][id]
                trans_pcd = o3d.geometry.PointCloud()
                trans_pcd.points = o3d.utility.Vector3dVector(all_align_pts)
                trans_pcd.paint_uniform_color([0, 0, 0])
                o3d.io.write_point_cloud(osp.join(points_dir, f'{id}.ply'), trans_pcd)

                # trans_pcd_kp_list = []
                # for part_idx in range(opt.num_parts):
                #     trans_pcd_kp_list += [
                #         o3d.geometry.TriangleMesh.create_sphere(radius=0.01, resolution=10).translate((x, y, z)) for
                #         x, y, z
                #         in results[id][part_idx]]

                #     sphere_pts_num = np.asarray(trans_pcd_kp_list[0].vertices).shape[0]
                #     part_kp_num = opt.num_kp // opt.num_parts
                #     for idx, mesh in enumerate(trans_pcd_kp_list[-part_kp_num:]):
                #         mesh.vertex_colors = o3d.utility.Vector3dVector(
                #             base_colors[np.newaxis, part_idx].repeat(sphere_pts_num, axis=0))

                # o3d.visualization.draw_geometries(trans_pcd_kp_list + [trans_pcd])

