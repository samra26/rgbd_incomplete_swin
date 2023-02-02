import torch
import torch.nn as nn
from cswin import CSWinTransformer
import torch.nn.functional as F
from functools import partial
from torchsummary import summary
from timm.models.layers import DropPath, trunc_normal_
import os
import cv2
import numpy
import numpy as np
import time
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
writer = SummaryWriter('log/run' + time.strftime("%d-%m"))
im_size=(320,320)


class RGBDInModule(nn.Module):
    def __init__(self, backbone):
        super(RGBDInModule, self).__init__()
        self.backbone = backbone
        

    def load_pretrained_model(self, model_path):
        pretrained_dict = torch.load(model_path)
        model_dict = self.backbone.state_dict()
        pretrained_dict = {ka: va for ka, va in pretrained_dict.items() if ka in model_dict}
        model_dict.update(pretrained_dict)
        self.backbone.load_state_dict(model_dict, strict=True)
        

    def forward(self, x):
        feature_stage=[]
        x,x1= self.backbone(x)
        a=[1,5,37,40]
        for i in a:
            print(i,'The backbone features are',x1[i].shape)
            B, new_HW, C = x1[i].shape
            print(B,new_HW,C)
            H = W = int(np.sqrt(new_HW))
            print(H,W)
            temp=x.transpose(-2, -1).contiguous()
            print(temp.shape)
            temp2=temp.view(B, C, H, W)
            print(temp2.shape)
            feature_stage.append(temp.view(B, C, H, W))
            
        print(feature_stage)
        return feature_stage


class RGBD_incomplete(nn.Module):
    def __init__(self,RGBDInModule):
        super(RGBD_incomplete, self).__init__()
        
        self.RGBDInModule = RGBDInModule

        
    def forward(self, f_all):
        feat_rgb = self.RGBDInModule(f_all)
        return feat_rgb[0]


def build_model(network='cswin', base_model_cfg='cswin'):
    backbone = CSWinTransformer(patch_size=4, embed_dim=144, depth=[2,4,32,2],
        split_size=[1,2,12,12], num_heads=[6,12,24,24], mlp_ratio=4.0)
      
   

    return RGBD_incomplete(RGBDInModule(backbone))
