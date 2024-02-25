# KPA-Tracker: Towards Robust and Real-Time Category-Level Articulated Object 6D Pose Tracking

This repository is the official implementation of the paper KPA-Tracker: Towards Robust and Real-Time Category-Level Articulated Object 6D Pose Tracking. This paper has been accepted to AAAI 2024.

![Overview](assets/OMAD.png)
![Visulization](assets/qualitative%20results.png)

## Datasets

[the dataset](https://1drv.ms/u/s!As7BgFXGjZFgmRnf80as8a35G2v6?e=SZfD5e) dataset contains the synthetic images generated from Unity along with the following annotations:

- RGB image
- depth map
- part mask
- part pose

This dataset also contains **URDF** articulated object models of five categories from PartNet-Mobility,
which is re-annotated by us to align the rest state in the same category.

## Usage

### Installation

Environments:

- Python >= 3.7
- Pytorch > 1.10.0
- CUDA >= 10.0
- open3d >= 0.13.0

```bash
git clone https://github.com/hhhhhar/KPA-Tracker.git
cd KPA-Tracker
```

Install the dependencies listed in ``requirements.txt``

```
pip install -r requirements.txt
```

Then, compile CUDA module - index_max:

```bash
cd model/index_max_ext
python setup.py install
```

And, building only the CUDA kernels:

```bash
cd model/Pointnet2_PyTorch_master
pip install pointnet2_ops_lib/.
```

Finally, download [the dataset1](https://1drv.ms/u/s!As7BgFXGjZFgmRnf80as8a35G2v6?e=SZfD5e)/[the dataset2](https://1drv.ms/u/s!As7BgFXGjZFgmRojXaRSwtosu81I?e=Dja8Ld) Dataset and put it in `KPA-Tracker/dataset1(or2)` folder.

Now you are ready to go!

### Training of KPA_Generator

```bash
python train_KPA_Generator.py --num_kp  8  --work_dir  work_dir/KPA_generator_laptop_kp8  --category 1 --num_parts 2  --use_relative_coverage  --symtype shape
```

### Testing of KPA_Generator

```bash
python test_KPA_Generator.py --num_kp  8 --checkpoint  model_current_laptop.pth  --work_dir  work_dir/KPA_generator_laptop_kp8  --bs  16  --workers  0  --use_gpu  --symtype shape --out  --mode train

python test_KPA_Generator.py --num_kp  8 --checkpoint  model_current_laptop.pth  --work_dir  work_dir/KPA_generator_laptop_kp8  --bs  16  --workers  0  --use_gpu  --symtype shape --out  --mode val
```

### Training of KPA_Tracker

```bash
python  train_KPA_Tracker.py --num_kp 8  --work_dir  work_dir/KPA_tracker_laptop_kp8  --params_dir  work_dir/KPA_generator_laptop_kp8  --num_basis  10  --symtype shape
```

### Testing of KPA_Tracker and Visualization

```bash
python video_funcs/video_func_dataset1.py  --num_kp 8 --checkpoint model_current_laptop.pth --work_dir work_dir/KPA_tracker_laptop_kp8   --params_dir work_dir/KPA_generator_laptop_kp8  --cate_id 1 --num_basis 10 --num_parts 2 --reg_weight 0 --kp_thr 0.1 --show
```
