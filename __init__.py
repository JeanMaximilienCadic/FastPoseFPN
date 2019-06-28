from torch.nn.parallel import DataParallel as DDP
import argparse
from pycocotools.coco import COCO
import numpy as np

from pose_fpn.nn.trainers import NetTrainerDefault as NetTrainer
from pose_fpn.nn.modules import FPNet
from pose_fpn.data.dataset import KeypointDataset
from pose_fpn.nn.optim.lr import adjust_lr
from pose_fpn.nn.modules import FPN101, FPN50, Concat
from pose_fpn.data.dataset import KeypointDataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Relative to data
    parser.add_argument('--num_of_keypoints', type=int, default=3,
                        help='Minimum number of keypoints for each bbox in training')
    parser.add_argument('--cocopath', type=str, default='__data__/train2017/')
    parser.add_argument("--keypoints_json", default='__data__/json/person_keypoints_train2017.json')

    # Relative training
    parser.add_argument('--number_of_epoch', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=12)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument("--devices", default="0,1,2,3", type=str)
    parser.add_argument('--lr', type=float, default=1.0e-3, help='Learning Rate')
    parser.add_argument('--lr_gamma', type=float, default=0.9, help='Gamma Rate')

    # Parse
    args = parser.parse_args()
    args.devices = np.array(args.devices.split(","), dtype=int)

    # Create the model
    trainer = NetTrainer(model=FPNet(101) if len(args.devices)==1 else DDP(FPNet(101)),
                         dataset=KeypointDataset(coco_train=COCO(args.keypoints_json),
                                                 num_of_keypoints=args.num_of_keypoints,
                                                 cocopath=args.cocopath,
                                                 cache_dir="/tmp"),
                         batch_size=args.batch_size,
                         num_workers=args.num_workers,
                         shuffle=True)
    # Train
    trainer.run(epochs=args.number_of_epoch,
                lr=args.lr,
                lr_gamma=args.lr_gamma,
                checkpoint="checkpoint")

    # Save
    trainer.save()

