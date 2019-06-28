# MultiPoseNet

This repo is a pytorch reproduce-version for paper "MultiPoseNet: Fast Multi-Person Pose
Estimation using Pose Residual Network", ECCV 2018.

The paper link is here multiposenet-paper: (https://arxiv.org/pdf/1807.04067.pdf). Now multiposenet has archived the state-of-the-art of multi-person pose estimation in bottom-up methods.

Now network has been defined and test with CUDA 8 and ubuntu 16.04(include keypoint subnet,retinanet and pose residual network).


## Network Architecture

The network include four parts: backbone, keypoint regression subnet, person detection part and prn part for multi-person pose assignment.

The architecture of the network:

![avatar](http://wx1.sinaimg.cn/mw690/005uXRWzly1fua75w1y62j30ul08vwk2.jpg)

The architecture of the keypoint subnet:

![avatar](http://wx4.sinaimg.cn/mw690/005uXRWzly1fua75sh9xaj30ub072755.jpg)

## Datasets

MSCOCO2017:

The keypoint subnet and retinanet need MSCOCO2017.You should download the data from official MSCOCO's website. 

## Requirements

- python3
- pytorch == 0.4.0
- pycocotools
- numpy
- tqdm
- progress
- scikit-image

## Train
The backbone and two fpn-based model has been realized in this repo. The keypoint subnet is defined in posenet.py , the person detection network is defined in retinanet.py. You can train them to get keypoint heatmap and person detection bbox.  

If you want to train the posenet (keypoint subnet) for keypoint estimation, just(and if you want to train retinanet with coco, just modify the dataloader.py to get bounding box annotations):
```
python train_posenet.py
```

This PRN model has been set in prn_train direcory. MSCOCO dataset's annotations contain person bbox and coordinate of keypoints. You can just train the PRN network by 
```
python prn_train/train.py
```

The options of training settings in prn_train/opt.py.

Thanks for the [repo](https://github.com/salihkaragoz/pose-residual-network-pytorch). We have realized and contribute few code of the independent PRN module based to this repo. Most code of PRN module are from this repo. I fixed some bugs when run the code from the authors' code.
 
## Acknowledgement

Thanks for the author of "MultiPoseNet: Fast Multi-Person Pose
Estimation using Pose Residual Network".





