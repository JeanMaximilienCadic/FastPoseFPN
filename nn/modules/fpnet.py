# Libaries
import os
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DataParallel as DDP
import argparse
from tqdm import tqdm
from pycocotools.coco import COCO
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
# Project
from pose_fpn.nn.optim.lr import adjust_lr
from pose_fpn.nn.modules import FPN101, FPN50, Concat
from pose_fpn.data.dataset import KeypointDataset

class FPNet(nn.Module):
    def __init__(self, layers):
        super(FPNet, self).__init__()
        self.device="cuda:0"

        self.fpn = FPN50() if layers == 50 else FPN101()

        self.latlayer4 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)

        self.convt1 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.convt2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.convt3 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.convt4 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.convs1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.convs2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.convs3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.convs4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.upsample1 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.concat = Concat()
        self.conv2 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.convfin = nn.Conv2d(256, 17, kernel_size=1, stride=1, padding=0)
        print(self)

    def forward(self, x):
        p2, p3, p4, p5 = self.fpn(x)[0]
        dt5 = self.convt1(p5)
        d5 = self.convs1(dt5)
        dt4 = self.convt2(p4)
        d4 = self.convs2(dt4)
        dt3 = self.convt3(p3)
        d3 = self.convs3(dt3)
        dt2 = self.convt4(p2)
        d2 = self.convs4(dt2)

        up5 = self.upsample1(d5)
        up4 = self.upsample2(d4)
        up3 = self.upsample3(d3)

        concat = self.concat(up5, up4, up3, d2)
        smooth = F.relu(self.conv2(concat))
        predict = self.convfin(smooth)
        predict = F.upsample(predict, scale_factor=4, mode='bilinear', align_corners=True)

        return predict

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear') + y
