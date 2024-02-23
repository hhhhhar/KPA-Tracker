import os
import os.path as osp
from shutil import move, rmtree
from tqdm import tqdm

classes = ('background', 'box', 'stapler', 'cutter', 'drawer', 'scissor')
cate_id = 1
cate = classes[cate_id]

work_path = osp.join('annotations_test', cate)
anno_list = os.listdir(work_path)
pbar_anno = tqdm(sorted(anno_list))
for idx, one_anno in enumerate(pbar_anno):
    if idx % 29 == 0:
        video_idx = str(idx // 29).zfill(5)
        video_path = osp.join('new_anno_test', cate, video_idx)
        os.makedirs(video_path)
    ori_anno_path = osp.join(work_path, one_anno)
    move(ori_anno_path, video_path)
# rmtree(osp.join(work_path, 'annotations'))

# work_path = osp.join('category_mask', cate)
# mask_list = os.listdir(work_path)
# pbar_mask = tqdm(sorted(mask_list))
# for idx, one_anno in enumerate(pbar_mask):
#     if idx == 0:
#         idx = idx - 1
#     if (idx + 1) % 30 == 0:
#         video_idx = str((idx + 1) // 30).zfill(5)
#         video_path = osp.join(work_path, video_idx, 'category_mask')
#         os.makedirs(video_path)
#     ori_mask_path = osp.join(work_path, 'category_mask', one_anno)
#     move(ori_mask_path, video_path)
# # rmtree(osp.join(work_path, 'category_mask'))
    
# color_list = os.listdir(osp.join(work_path, 'color'))
# pbar_color = tqdm(sorted(color_list))
# for idx, one_anno in enumerate(pbar_color):
#     if idx == 0:
#         idx = idx - 1
#     if (idx + 1) % 30 == 0:
#         video_idx = str((idx + 1) // 30).zfill(5)
#         video_path = osp.join(work_path, video_idx, 'color')
#         os.makedirs(video_path)
#     ori_color_path = osp.join(work_path, 'color', one_anno)
#     move(ori_color_path, video_path)
# # rmtree(osp.join(work_path, 'color'))

# depth_list = os.listdir(osp.join(work_path, 'depth'))
# pbar_depth = tqdm(sorted(depth_list))
# for idx, one_anno in enumerate(pbar_depth):
#     if idx == 0:
#         idx = idx - 1
#     if (idx + 1) % 30 == 0:
#         video_idx = str((idx + 1) // 30).zfill(5)
#         video_path = osp.join(work_path, video_idx, 'depth')
#         os.makedirs(video_path)
#     ori_depth_path = osp.join(work_path, 'depth', one_anno)
#     move(ori_depth_path, video_path)
# # rmtree(osp.join(work_path, 'depth'))

# depth_list = os.listdir(osp.join(work_path, 'depth_foreground'))
# pbar_depth = tqdm(sorted(depth_list))
# for idx, one_anno in enumerate(pbar_depth):
#     if idx % 30 == 0:
#         video_idx = str(idx // 30).zfill(5)
#         video_path = osp.join(work_path, video_idx, 'depth_foreground')
#         os.makedirs(video_path)
#     ori_depth_path = osp.join(work_path, 'depth_foreground', one_anno)
#     move(ori_depth_path, video_path)
# # rmtree(osp.join(work_path, 'depth_foreground'))

# rawan_list = os.listdir(osp.join(work_path, 'annotations_raw'))
# pbar_depth = tqdm(sorted(rawan_list))
# for idx, one_anno in enumerate(pbar_depth):
#     if idx % 30 == 0:
#         video_idx = str(idx // 30).zfill(5)
#         video_path = osp.join(work_path, video_idx, 'annotations_raw')
#         os.makedirs(video_path)
#     ori_depth_path = osp.join(work_path, 'annotations_raw', one_anno)
#     move(ori_depth_path, video_path)
# rmtree(osp.join(work_path, 'annotations_raw'))

# os.makedirs(osp.join(work_path, 'demo', cate))
# os.makedirs(osp.join(work_path, 'demo_train', cate))
# move(osp.join(work_path, 'demo', cate), 'demo')
# move(osp.join(work_path, 'demo_train', cate), 'demo_train')
