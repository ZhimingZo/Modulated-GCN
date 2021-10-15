# Modulated Graph Convolutional Network for 3D Human Pose Estimation (ICCV 2021)

This repository holds the Pytorch implementation of [Modulated Graph Convolutional Network for 3D Human Pose Estimation](https://openaccess.thecvf.com/content/ICCV2021/papers/Zou_Modulated_Graph_Convolutional_Network_for_3D_Human_Pose_Estimation_ICCV_2021_paper.pdf) by Zhiming Zou and Wei Tang. If you find our code useful in your research, please consider citing:

```
@InProceedings{Zou_2021_ICCV,
    author    = {Zou, Zhiming and Tang, Wei},
    title     = {Modulated Graph Convolutional Network for 3D Human Pose Estimation},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {11477-11487}
}
```

## Introduction

The proposed Modulated Graph Convolutional Network (Modulated GCN) is a graph convolutional network architecture that operates on regression tasks with graph-structured data. We evaluate our model for 3D human pose estimation on the [Human3.6M Dataset](http://vision.imar.ro/human3.6m/).

In this repository, only 2D joints of the human pose are exploited as inputs. We utilize the method described in Pavllo et al. [2] to normalize 2D and 3D poses in the dataset. To be specific, 2D poses are scaled according to the image resolution and normalized to [-1, 1]; 3D poses are aligned with respect to the root joint. Please refer to the corresponding part in Pavllo et al. [2] for more details. For the 2D ground truth, we predict 16 joints (as the skeleton in Martinez et al. [1] and Zhao et al. [3] without the 'Neck/Nose' joint). For the 2D pose detections, the 'Neck/Nose' joint is reserved. 

### Results on Human3.6M

Under Protocol 1 (mean per-joint position error) and Protocol 2 (mean per-joint position error after rigid alignment).

| Method | 2D Detections | # of Epochs | # of Parameters | MPJPE (P1) | P-MPJPE (P2) |
|:-------|:-------:|:-------:|:-------:|:-------:|:-------:|
| SemGCN | Ground truth | 50 | 0.27M | 42.14 mm | 33.53 mm |
| SemGCN (w/ Non-local) | Ground truth | 30 | 0.43M | 40.78 mm | 31.46 mm |
| Modulated GCN   | Ground truth | 50 |  0.29M  | **38.25 mm** | **30.06 mm** |

Results using Ground truth are reported. 
The results are borrowed from [SemGCN](https://github.com/garyzhao/SemGCN).

## Quickstart

This repository is build upon Python v3.7 and Pytorch v1.7.0 on Ubuntu 18.04. All experiments are conducted on a single NVIDIA RTX 2080 Ti GPU. See [`requirements.txt`](requirements.txt) for other dependencies. We recommend installing Python v3.7 from [Anaconda](https://www.anaconda.com/) and installing Pytorch (>= 1.7.0) following guide on the [official instructions](https://pytorch.org/) according to your specific CUDA version. Then you can install dependencies with the following commands.

```
git clone https://github.com/ZhimingZo/Modulated-GCN.git
cd Modulated-GCN
pip install -r requirements.txt
```

### Benchmark setup
CPN 2D detections for Human3.6M datasets are provided by [VideoPose3D](https://github.com/facebookresearch/VideoPose3D) Pavllo et al. [2], which can be downloaded by the following steps:

```
cd dataset
wget https://dl.fbaipublicfiles.com/video-pose-3d/data_2d_h36m_cpn_ft_h36m_dbb.npz
wget https://dl.fbaipublicfiles.com/video-pose-3d/data_2d_h36m_detectron_ft_h36m.npz
cd ..
```
3D labels and ground truth can be downloaded
```
cd dataset
gdown --id 1P7W3ldx2lxaYJJYcf3RG4Y9PsD4EJ6b0
cd ..
```
### Evaluating our pre-trained models
The pre-trained models can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1HoNd2YPc8BdGvrN46GR_N2OchahzLx4I?usp=sharing). Put `ckpt` in the project root directory.

To evaluate Modulated GCN with CPN detectors as input, run:
```
python main_graph.py  --post_refine --module_gcn_reload 1  --previous_dir './ckpt/best_model' --save_dir './ckpt_best_model' --post_refine_reload 1 --nepoch 2 --show_protocol2
```

Note that the results will be reported in an **action-wise** manner.

### Training from scratch
If you want to reproduce the results of our pretrained models, run the following commands.
For Modulated GCN:
```
python main_graph.py  --pro_train 1 --save_model 1  --save_dir './ckpt'
```
By default the application runs in testing mode.
If you want to try different network settings, please refer to [`opt1.py`](opt1.py) for more details. Note that the 
default setting of hyper-parameters is used for training model with CPN detectors as input, please refer to the paper for implementation details.


### GT setup 

GT 2D keypoints for Human3.6M datasets are obtained from [SemGCN](https://github.com/garyzhao/SemGCN) Zhao et al. [3], which can be downloaded by the following steps:
```
cd data
pip install gdown
gdown https://drive.google.com/uc?id=1Ac-gUXAg-6UiwThJVaw6yw2151Bot3L1
python prepare_data_h36m.py --from-archive h36m.zip
cd ..
```
After this step, you should end up with two files in the dataset directory: data_3d_h36m.npz for the 3D poses, and data_2d_h36m_gt.npz for the ground-truth 2D poses.

### GT Evaluation 
```
python main_gcn.py  --eva checkpoint/ckpt_Modulated_GCN_128_38.25_30.06.pth.tar
```
### GT Training 
```
python main_gcn.py
```


### References

[1] Martinez et al. [A simple yet effective baseline for 3d human pose estimation](https://arxiv.org/pdf/1705.03098.pdf). ICCV 2017.

[2] Pavllo et al. [3D human pose estimation in video with temporal convolutions and semi-supervised training](https://arxiv.org/pdf/1811.11742.pdf). CVPR 2019.

[3] Zhao et al. [Semantic Graph Convolutional Networks for 3D Human Pose Regression](https://arxiv.org/pdf/1904.03345.pdf). CVPR 2019.

[3] Cai et al. [Exploiting Spatial-temporal Relationships for 3D Pose Estimation via Graph Convolutional Networks](https://openaccess.thecvf.com/content_ICCV_2019/papers/Cai_Exploiting_Spatial-Temporal_Relationships_for_3D_Pose_Estimation_via_Graph_Convolutional_ICCV_2019_paper.pdf). ICCV 2019.

### Acknowledgement
This code is extended from the following repositories.
- [3d-pose-baseline](https://github.com/una-dinosauria/3d-pose-baseline)
- [3d_pose_baseline_pytorch](https://github.com/weigq/3d_pose_baseline_pytorch)
- [VideoPose3D](https://github.com/facebookresearch/VideoPose3D)
- [Semantic GCN](https://github.com/garyzhao/SemGCN)
- [Local-to-Global GCN](https://github.com/vanoracai/Exploiting-Spatial-temporal-Relationships-for-3D-Pose-Estimation-via-Graph-Convolutional-Networks)

We thank the authors for releasing their code. Please also consider citing their work.
