﻿# FastPoseFPN - Multi Pose Estimation with support for multi gpu 

This repo is a pytorch reproduce-version for paper "MultiPoseNet: Fast Multi-Person Pose
Estimation using Pose Residual Network", ECCV 2018. https://arxiv.org/pdf/1807.04067.pdf


The network include four parts: backbone, keypoint regression subnet, person detection part and prn part for multi-person pose assignment.

![avatar](img/img.png)


## Improvements
- (Speed) Caching on hdd 
- (Speed) Multi-gpu
- (Programming) Training easy to customize
- (Programming) Code formatting following the IYO patterns

## Datasets

MSCOCO2017:

The keypoint subnet and retinanet need MSCOCO2017.You should download the data from official MSCOCO's website. 

## Requirements

- python3
- pytorch == 1.0.1
- pycocotools
- numpy
- tqdm
- progress
- scikit-image


## Train
The backbone and two fpn-based model has been realized in this repo. The keypoint subnet is defined in posenet.py , the person detection network is defined in retinanet.py. You can train them to get keypoint heatmap and person detection bbox.  

(MULTI-GPU recommended )If you want to train the posenet (keypoint subnet) for keypoint estimation with batch size 100 on 4 gpus:
```
python fpn_pose/__init__.py --batch_size 100 --devices 0,1,2,3
```

(SINGLE GPU )If you want to train the posenet (keypoint subnet) for keypoint estimation with batch size 8 on 1 gpu:
```
python fpn_pose/__init__.py --batch_size 8 --devices 0
```

This version support multi-gpu 
## Acknowledgement

Thanks for the author of "MultiPoseNet: Fast Multi-Person Pose Estimation using Pose Residual Network".
Thanks for the author of the following repository https://github.com/IcewineChen/pytorch-MultiPoseNet that I reused as a starting point.


For any question please contact me at j.cadic@protonmail.ch



