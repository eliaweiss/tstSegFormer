# DiceLoss.py
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-8):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, prediction, target):

        
        intersection = (prediction * target).sum()
        dice = (2. * intersection + self.smooth) / (
            (prediction.sum() + target.sum()) + self.smooth
        )
        return 1 - dice
