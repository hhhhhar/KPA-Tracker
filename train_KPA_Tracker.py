import argparse
import numpy as np
import os
import os.path as osp
import sys
sys.path.append("..")
os.chdir(sys.path[0])

import torch.optim as optim
import torch.utils.data
import tqdm
import time

from torch.utils.tensorboard import SummaryWriter
from model.OMAD_KPA_Tracker import OMAD_KPA_Tracker
from libs.loss_omadnet import Loss_OMAD_Net
from dataset.dataset_Tracker_conrdinate import SapienDataset_OMADNet

cate_list = ['laptop', 'eyeglasses', 'dishwasher', 'drawer', 'scissors']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=str, default='ArtImage', help='dataset root dir')
    parser.add_argument('--params_dir', type=str, help='the dir for params and kp annotations')
    parser.add_argument('--resume', type=str, default=None, help='resume model')
    parser.add_argument('--category', type=int, default=1, help='category to train')
    parser.add_argument('--num_points', type=int, default=1024, help='points')
    parser.add_argument('--num_cates', type=int, default=5, help='number of categories')
    parser.add_argument('--num_parts', type=int, default=2, help='number of parts')
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
    parser.add_argument('--num_kp', type=int, default=12, help='number of all keypoints')
    parser.add_argument('--num_basis', type=int, default=10, help='number of shape basis')
    parser.add_argument('--bs', type=int, default=16, help='batch size')
    parser.add_argument('--dense_soft_factor', type=float, default=1.0, help='the factor of dense softmax')
    parser.add_argument('--loc_weight', type=float, default=5.0, help='the weight of pts loc weight')
    parser.add_argument('--base_weight', type=float, default=0.2, help='the weight of base rotation loss')
    parser.add_argument('--cls_weight', type=float, default=1.0, help='the weight of segmentation loss')
    parser.add_argument('--joint_state_weight', type=float, default=5.0, help='the weight of joint state loss')
    parser.add_argument('--shape_weight', type=float, default=3.0, help='the weight of shape loss')
    parser.add_argument('--joint_param_weight', type=float, default=3.0, help='the weight of joint param loss')
    parser.add_argument('--reg_weight', type=float, default=0.01, help='the weight of regularization loss')
    parser.add_argument('--no_att', action='store_true', help='whether to not use attention map')
    parser.add_argument('--symtype', type=str, default='shape', choices=['shape', 'none'], help='the symmetry type')
    parser.add_argument('--work_dir', type=str, default='work_dir/base', help='save dir')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--use_background', action='store_true', help='whether to use background')
    opt = parser.parse_args()

    device = torch.device("cuda")

    # print(osp.join(opt.params_dir, 'params.pth'))
    params_dict = torch.load(osp.join(opt.params_dir, 'params.pth'))
    model = OMAD_KPA_Tracker(device=device, params_dict=params_dict,
                       num_points=opt.num_points, num_kp=opt.num_kp, num_parts=opt.num_parts,
                       init_dense_soft_factor=opt.dense_soft_factor, num_basis=opt.num_basis, symtype=opt.symtype,
                       use_attention=not opt.no_att)
    model = model.to(device)

    if opt.resume is not None:
        model.load_state_dict(torch.load(osp.join(opt.work_dir, opt.resume)))

    train_kp_anno_path = osp.join(opt.params_dir, 'unsup_train_keypoints.pkl')
    train_dataset = SapienDataset_OMADNet('train', data_root=opt.dataset_root, use_p1_aug=True, num_pts=opt.num_points,
                                          num_parts=opt.num_parts, num_cates=opt.num_cates, cate_id=opt.category,
                                          device=torch.device("cpu"), kp_anno_path=train_kp_anno_path, use_background=opt.use_background)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.bs, shuffle=True, num_workers=opt.workers, drop_last=True)

    test_kp_anno_path = osp.join(opt.params_dir, 'unsup_test_keypoints.pkl')
    test_dataset = SapienDataset_OMADNet('val', data_root=opt.dataset_root, use_p1_aug=True, num_pts=opt.num_points,
                                         num_parts=opt.num_parts, num_cates=opt.num_cates, cate_id=opt.category,
                                         device=torch.device("cpu"), kp_anno_path=test_kp_anno_path, use_background=opt.use_background)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.bs, shuffle=False, num_workers=opt.workers, drop_last=True)

    assert opt.num_kp % opt.num_parts == 0, 'number of keypoints must be divided by number of parts'
    # TODO check gt
    criterion = Loss_OMAD_Net(num_key_per_part=opt.num_kp // opt.num_parts, num_cate=opt.num_cates, num_parts=opt.num_parts,
                                     loss_loc_weight=opt.loc_weight,
                                     loss_cls_weight=opt.cls_weight,
                                     loss_base_weight=opt.base_weight,
                                     loss_joint_state_weight=opt.joint_state_weight,
                                     loss_shape_weight=opt.shape_weight,
                                     loss_joint_param_weight=opt.joint_param_weight,
                                     loss_reg_weight=opt.reg_weight,
                                     device=device)

    best_test = np.Inf
    optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150, eta_min=0)
    writer = SummaryWriter(log_dir=opt.work_dir)
    total_iter = 0
    for epoch in range(0, 170):
        model.train()
        train_count = 0

        optimizer.zero_grad()

        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        for i, data in enumerate(train_dataloader):
            with torch.autograd.set_detect_anomaly(True):
                cloud, gt_part_cls, gt_part_r, gt_part_quat, gt_part_t, gt_joint_state, gt_norm_joint_loc, gt_norm_joint_axis, \
                gt_norm_part_kp, gt_scale, gt_center, gt_norm_part_corners, cate, urdf_id, gt_tran_part_kp, rt_key = data

                cloud.requires_grad_()

                cloud, gt_part_cls, gt_part_r, gt_part_quat, gt_part_t, gt_joint_state, \
                gt_norm_joint_loc, gt_norm_joint_axis, gt_norm_part_kp, gt_tran_part_kp= \
                    cloud.to(device), gt_part_cls.to(device), gt_part_r.to(device), gt_part_quat.to(device), \
                    gt_part_t.to(device), gt_joint_state.to(device), gt_norm_joint_loc.to(device), \
                    gt_norm_joint_axis.to(device), gt_norm_part_kp.to(device), gt_tran_part_kp.to(device)

                dense_part_cls_score, pred_trans_part_kp, pred_base_quat, pred_base_r, pred_base_t, pred_joint_state,\
                    pred_beta, pred_norm_part_kp, pred_joint_loc, pred_joint_axis = model(cloud, gt_part_cls)

                epsilon = 1e-5
                torch.clamp(pred_beta, epsilon)

                loss, _, loss_dict = criterion(pred_trans_part_kp, dense_part_cls_score, pred_base_quat, pred_base_t, pred_norm_part_kp,
                                            pred_joint_loc, pred_joint_axis, pred_joint_state, pred_beta,
                                            gt_part_cls, gt_part_quat,
                                            gt_part_r, gt_part_t, gt_norm_part_kp,
                                            gt_norm_joint_loc, gt_norm_joint_axis, gt_joint_state, gt_tran_part_kp)
                if not torch.isnan(loss):
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=35)
                    optimizer.step()

            train_count += 1
            total_iter += 1
            optimizer.zero_grad()
            if train_count % 5 == 0:
                print('{}, Epoch: {}, iter: {}/{}, loss_all:{:05f}, loss_loc:{:05f}, loss_cls:{:05f}, '
                      'loss_base:{:05f}, loss_joint_state:{:05f}, loss_shape:{:05f}, loss_joint_param:{:05f}, '.format(
                    time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())),
                    epoch, train_count, len(train_dataset) // opt.bs, float(loss.item()),
                    loss_dict['loss_loc'],
                    loss_dict['loss_cls'],
                    loss_dict['loss_base'],
                    loss_dict['loss_joint_state'],
                    loss_dict['loss_shape'],
                    loss_dict['loss_joint_param']
                    # loss_dict['loss_reg']
                ))
                for key, value in loss_dict.items():
                    writer.add_scalar('{}/train'.format(key), value, total_iter)
                writer.add_scalar('loss_all/train', loss.item(), total_iter)

        torch.save(model.state_dict(), osp.join(opt.work_dir, 'model_current_{}.pth'.format(cate_list[opt.category - 1])))

        # change lr
        scheduler.step(epoch)
        optimizer.zero_grad()

        if (epoch + 1) % 5 == 0:
            model.eval()
            score = []
            print('>>>>>>>>>>>>Testing epoch {}>>>>>>>>>>>>>'.format(epoch))
            sum_test_loss_dict = dict(loss_loc=0., loss_cls=0., loss_base=0., loss_joint_state=0.,
                                      loss_shape=0., loss_joint_param=0., loss_reg=0.)
            for j, data in enumerate(tqdm.tqdm(test_dataloader)):
                cloud, gt_part_cls, gt_part_r, gt_part_quat, gt_part_t, gt_joint_state, gt_norm_joint_loc, gt_norm_joint_axis, \
                gt_norm_part_kp, gt_scale, gt_center, gt_norm_part_corners, cate, urdf_id, gt_tran_part_kp, rt_key = data

                cloud, gt_part_cls, gt_part_r, gt_part_quat, gt_part_t, gt_joint_state, \
                gt_norm_joint_loc, gt_norm_joint_axis, gt_norm_part_kp, gt_tran_part_kp= \
                    cloud.to(device), gt_part_cls.to(device), gt_part_r.to(device), gt_part_quat.to(device), \
                    gt_part_t.to(device), gt_joint_state.to(device), gt_norm_joint_loc.to(device), \
                    gt_norm_joint_axis.to(device), gt_norm_part_kp.to(device), gt_tran_part_kp.to(device)

                with torch.no_grad():
                    dense_part_cls_score, pred_trans_part_kp, pred_base_quat, pred_base_r, pred_base_t, pred_joint_state, \
                        pred_beta, pred_norm_part_kp, pred_joint_loc, pred_joint_axis = model(cloud, None)

                _, item_score, loss_dict = criterion(pred_trans_part_kp, dense_part_cls_score, pred_base_quat, pred_base_t,
                                               pred_norm_part_kp,
                                               pred_joint_loc, pred_joint_axis, pred_joint_state, pred_beta,
                                               gt_part_cls, gt_part_quat,
                                               gt_part_r, gt_part_t, gt_norm_part_kp,
                                               gt_norm_joint_loc, gt_norm_joint_axis, gt_joint_state, gt_tran_part_kp)
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
