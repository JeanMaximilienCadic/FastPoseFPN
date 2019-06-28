import torch
import torch.nn as nn

class Concat(nn.Module):
    def __init__(self):
        super(Concat,self).__init__()

    def forward(self,up1,up2,up3,up4):
        return torch.cat((up1,up2,up3,up4),1)