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


class JLModule(nn.Module):
    def __init__(self, backbone):
        super(JLModule, self).__init__()
        self.backbone = backbone
        

    def load_pretrained_model(self, model_path):
        pretrained_dict = torch.load(model_path)
        model_dict = self.backbone.state_dict()
        pretrained_dict = {ka: va for ka, va in pretrained_dict.items() if ka in model_dict}
        model_dict.update(pretrained_dict)
        self.backbone.load_state_dict(model_dict, strict=True)
        

    def forward(self, x):

        x,x1= self.backbone(x)
        for i in range(len(x1)):
            print('The backbone features are',x1[i].shape)
        return x


class JL_DCF(nn.Module):
    def __init__(self,JLModule):
        super(JL_DCF, self).__init__()
        
        self.JLModule = JLModule

        
    def forward(self, f_all):
        x = self.JLModule(f_all)
        return x


def build_model(network='cswin', base_model_cfg='cswin'):
    backbone = CSWinTransformer(patch_size=4, embed_dim=144, depth=[2,4,32,2],
        split_size=[1,2,12,12], num_heads=[6,12,24,24], mlp_ratio=4.0)
      
   

    return JL_DCF(JLModule(backbone))
