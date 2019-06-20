import torch.nn as nn
import torch
from TinyMind.TMDetecter.TMBackbone import TMBackbone
class TMRPN(nn.Module):
    def __init__(self,num_classes=1,criterion=None,backbone=TMBackbone,fpn=True):
        super(TMRPN,self).__init__()
        self.backbone=backbone(fpn=fpn)
        self.num_classes=num_classes+1
        if fpn==True:
            in_channels=256
        else:
            in_channels=self.backbone.inplanes
        self.cls_head=self._make_head(in_channels,self.num_classes)#self.backbone.inplanes
        self.loc_head=self._make_head(in_channels,4)
        self.criterion=criterion
    def forward(self, input,loc_targets=None,cls_targets=None):
        output=self.backbone(input)
        cls_preds=[]
        loc_preds=[]
        for out in output:
            cls_pred=self.cls_head(out).permute(0,2,3,1).reshape(input.size(0),-1,self.num_classes)
            loc_pred=self.loc_head(out).permute(0,2,3,1).reshape(input.size(0),-1,4)
            cls_preds.append(cls_pred)
            loc_preds.append(loc_pred)
        loc_preds=torch.cat(loc_preds,dim=1)
        cls_preds=torch.cat(cls_preds,dim=1)
        """
        cls_preds=self.cls_head(output).permute(0,2,3,1).reshape(input.size(0),-1,self.num_classes)
        loc_preds=self.loc_head(output).permute(0,2,3,1).reshape(input.size(0),-1,4)
        """
        if self.criterion is None:
            return loc_preds,cls_preds
        else:
            loss=self.criterion(loc_preds,loc_targets,cls_preds,cls_targets)
            return loss,cls_preds
    def _make_head(self,in_channels,out_channels):
        layers=[]
        for i in range(1):
            layers.append(nn.Sequential(nn.Conv2d(in_channels,256,3,1,1),
                                        nn.ReLU(inplace=True)))
        layers.append(nn.Conv2d(256,out_channels,3,1,1))
        return nn.Sequential(*layers)