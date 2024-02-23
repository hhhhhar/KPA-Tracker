import argparse
import numpy as np
import os.path as osp
from torch.utils.tensorboard import SummaryWriter

import torch.optim as optim
import torch.utils.data
import tqdm
import time

from model.KPA_Generator import KPA_Generator
from libs.loss_omad_priornet import Loss_OMAD_PriorNet
from dataset.dataset1_Generator import SapienDataset_OMADPriorNet

cate_list = ['laptop', 'eyeglasses', 'dishwasher', 'drawer', 'scissors']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=str, default='./dataset1', help='dataset root dir')
    parser.add_argument('--resume', type=str, default=None, help='resume model')
    parser.add_argument('--category', type=int, default=1, help='category to train')
    parser.add_argument('--num_samples', type=int, default=50000, help='number of samples of training dataset')
    parser.add_argument('--num_points', type=int, default=2048, help='points')
    parser.add_argument('--num_cates', type=int, default=5, help='number of categories')
    parser.add_argument('--num_parts', type=int, default=2, help='number of parts')
    parser.add_argument('--workers', type=int, default=0, help='number of data loading workers')
    parser.add_argument('--num_kp', type=int, default=8, help='number of all keypoints')
    parser.add_argument('--num_basis', type=int, default=10, help='number of shape basis')
    parser.add_argument('--chamfer_weight', type=float, default=1.0, help='weight of chamfer loss')
    parser.add_argument('--coverage_weight', type=float, default=1.0, help='weight of coverage loss')
    parser.add_argument('--surface_weight', type=float, default=5.0, help='weight of surface loss')
    parser.add_argument('--joint_weight', type=float, default=1.0, help='weight of joint loss')
    parser.add_argument('--reg_weight', type=float, default=0.01, help='weight of regularization loss')
    parser.add_argument('--sep_weight', type=float, default=2.0, help='weight of seperation loss')
    parser.add_argument('--sep_factor', type=float, default=8.0, help='the sep factor')
    parser.add_argument('--scale_aug_range', type=float, default=0.5, help='the range of scale augmentation')
    parser.add_argument('--use_relative_coverage', action='store_true', default=False,
                        help='whether to use relative coverage loss')
    parser.add_argument('--beta', type=float, default=0.1, help='the beta of smooth l1 loss for coverege loss')
    parser.add_argument('--symtype', type=str, default='shape', choices=['shape', 'none'], help='the symmetry type')
    parser.add_argument('--bs', type=int, default=32, help='batch size')
    parser.add_argument('--work_dir', type=str, default='work_dir/base', help='save dir')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    opt = parser.parse_args()

    device = torch.device("cuda")
    model = KPA_Generator(node_num=opt.num_kp, basis_num=opt.num_basis, part_num=opt.num_parts, device=device,
                         symtype=opt.symtype, init_n_pl=((1., 0.), ))
    model = model.to(device)

    if opt.resume is not None:
        model.load_state_dict(torch.load(osp.join(opt.work_dir, opt.resume)))

    train_dataset = SapienDataset_OMADPriorNet('train',
                                            data_root=opt.dataset_root, add_noise=False, num_pts=opt.num_points,
                                            num_parts=opt.num_parts,
                                            num_cates=opt.num_cates, cate_id=opt.category,
                                            num_samples=opt.num_samples,
                                            node_num=opt.num_kp,
                                            use_scale_aug=False,
                                            use_rot_aug=False,
                                            device=torch.device("cpu"))
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.bs, shuffle=True, num_workers=opt.workers)
    test_dataset = SapienDataset_OMADPriorNet('val', data_root=opt.dataset_root, add_noise=False, num_pts=opt.num_points,
                                           num_parts=opt.num_parts,
                                           num_cates=opt.num_cates, cate_id=opt.category,
                                           num_samples=5000,
                                           node_num=opt.num_kp,
                                           use_scale_aug=False,
                                           use_rot_aug=False,
                                           scale_aug_max_range=opt.scale_aug_range,
                                           device=torch.device("cpu"))
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.bs, shuffle=False, num_workers=opt.workers)

    assert opt.num_kp % opt.num_parts == 0, 'number of keypoints must be divided by number of parts'
    joint_type = 'revolute' if opt.category != 4 else 'prismatic'
    criterion = Loss_OMAD_PriorNet(device=device,
                       num_kp=opt.num_kp, num_cate=opt.num_cates, num_parts=opt.num_parts,
                       loss_chamfer_weight=opt.chamfer_weight,
                       loss_coverage_weight=opt.coverage_weight,
                       loss_surface_weight=opt.surface_weight,
                       loss_joint_weight=opt.joint_weight,
                       loss_reg_weight=opt.reg_weight,
                       loss_sep_weight=opt.sep_weight,
                       sep_factor=opt.sep_factor,
                       beta=opt.beta,
                       joint_type=joint_type,
                       use_relative_coverage=opt.use_relative_coverage
                       )

    best_test = np.Inf
    optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0)

    writer = SummaryWriter(log_dir=opt.work_dir)
    total_iter = 0
    for epoch in range(0, 100):
        model.train()
        train_count = 0

        optimizer.zero_grad()

        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        for i, data in enumerate(train_dataloader):
            cloud, init_nodes, cloud_cls, gt_inv_r, gt_joint_loc, gt_joint_axis, scale_factor, \
            raw_part_scale, raw_center, cate, urdf_id = data
            cloud, init_nodes, gt_inv_r, cloud_cls, gt_joint_loc, gt_joint_axis, raw_part_scale = \
                cloud.to(device), init_nodes.to(device), gt_inv_r.to(device), cloud_cls.to(device), \
                gt_joint_loc.to(device), gt_joint_axis.to(device), raw_part_scale.to(device)

            bs_now = cloud.shape[0]
            gt_r = gt_inv_r.transpose(1, 2)
            gt_t = torch.zeros(bs_now, 1, 3, device=device)

            cloud.requires_grad_()
            init_nodes.requires_grad_()
            # input = []
            # input.append(torch.as_tensor(cloud))
            # input.append(None)
            # input.append(torch.as_tensor(init_nodes))
            # with SummaryWriter(log_dir="network_visualization") as w:
            #     w.add_graph(model, list(input))
            # w.close()
            final_nodes, coefs, pred_joint_loc, pred_joint_axis = model(cloud, None, init_nodes)
            
            part_pred_kps = model.get_transformed_pred_keypoints(coefs, gt_r, gt_t)
            trans_joint_loc, trans_joint_axis = model.get_transformed_joint_params(pred_joint_loc, pred_joint_axis, gt_r, gt_t)

            loss, _, loss_dict = criterion(coefs, part_pred_kps, final_nodes, cloud, cloud_cls,
                                           trans_joint_loc, trans_joint_axis, gt_joint_loc, gt_joint_axis, raw_part_scale)
            loss.backward()

            train_count += 1
            total_iter += 1
            optimizer.step()
            optimizer.zero_grad()
            if train_count % 100 == 0:
                print('{}, Epoch: {}, iter: {}/{}, loss_all:{:05f}, loss_chamfer:{:05f}, '
                      'loss_coverage:{:05f}, loss_surface:{:05f}, loss_joint:{:05f}, loss_reg:{:05f}, loss_sep:{:05f}'.format(
                    time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())),
                    epoch, train_count, len(train_dataset) // opt.bs, float(loss.item()),
                    loss_dict['loss_chamfer'],
                    loss_dict['loss_coverage'],
                    loss_dict['loss_surface'],
                    loss_dict['loss_joint'],
                    loss_dict['loss_reg'],
                    loss_dict['loss_sep']
                ))
                for key, value in loss_dict.items():
                    writer.add_scalar('{}/train'.format(key), value, total_iter)
                writer.add_scalar('loss_all/train', loss.item(), total_iter)

        torch.save(model.state_dict(), osp.join(opt.work_dir, 'model_current_{}.pth'.format(cate_list[opt.category - 1])))

        # change lr
        scheduler.step()
        optimizer.zero_grad()

        if (epoch == 0) or ((epoch + 1) % 5 == 0):
            model.eval()
            score = []
            print('>>>>>>>>>>>>Testing epoch {}>>>>>>>>>>>>>'.format(epoch))
            sum_test_loss_dict = dict(loss_chamfer=0., loss_coverage=0., loss_surface=0., loss_joint=0., loss_reg=0.,
                                      loss_sep=0.)
            for j, data in enumerate(tqdm.tqdm(test_dataloader)):
                cloud, init_nodes, cloud_cls, gt_inv_r, gt_joint_loc, gt_joint_axis, scale_factor, \
                raw_part_scale, raw_center, cate, urdf_id = data
                cloud, init_nodes, gt_inv_r, cloud_cls, gt_joint_loc, gt_joint_axis, raw_part_scale = \
                    cloud.to(device), init_nodes.to(device), gt_inv_r.to(device), cloud_cls.to(device), \
                    gt_joint_loc.to(device), gt_joint_axis.to(device), raw_part_scale.to(device)

                bs_now = cloud.shape[0]
                gt_r = gt_inv_r.transpose(1, 2)
                gt_t = torch.zeros(bs_now, 1, 3, device=device)

                with torch.no_grad():
                    final_nodes, coefs, pred_joint_loc, pred_joint_axis = model(cloud, None, init_nodes)
                    part_pred_kps = model.get_transformed_pred_keypoints(coefs, gt_r, gt_t)
                    trans_joint_loc, trans_joint_axis = model.get_transformed_joint_params(pred_joint_loc, pred_joint_axis, gt_r, gt_t)

                _, item_score, loss_dict = criterion(coefs, part_pred_kps, final_nodes, cloud, cloud_cls,
                                                     trans_joint_loc, trans_joint_axis, gt_joint_loc, gt_joint_axis, raw_part_scale)
                for key, value in loss_dict.items():
                    sum_test_loss_dict[key] += value
                score.append(item_score)

            for key, value in sum_test_loss_dict.items():
                writer.add_scalar('{}/test'.format(key), value / len(score), total_iter)

            test_score = np.mean(np.array(score))
            writer.add_scalar('score/test', test_score, total_iter)

            if test_score < best_test:
                best_test = test_score
                torch.save(model.state_dict(),
                           '{0}/model_{1}_{2}_{3}.pth'.format(opt.work_dir, epoch, test_score, cate_list[opt.category - 1]))
                print('epoch:', epoch, ' >>>>>>>>----------BEST TEST MODEL SAVED---------<<<<<<<<')
    writer.close()
