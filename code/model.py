import torch
import torch.nn as nn
import torch.nn.utils as utils

from torchvision.models import resnet50, resnet152


class AttrEncoder(nn.Module):
    '''
    Attribute predictor class (encoder)
    
    '''
    def __init__(self, outdims=40):
        super(AttrEncoder, self).__init__()
        self.resnet = resnet152(pretrained=True)
        self.reslayers = list(self.resnet.children())[:-1]
        self.model = nn.Sequential(*self.reslayers)
        self.affine = nn.Linear(2048, outdims)
        self.act = nn.Sigmoid()

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        with torch.no_grad():
            resout = self.model(x)
        resout = resout.view(resout.size(0), -1)
        out = self.act(self.affine(resout))
        return out
