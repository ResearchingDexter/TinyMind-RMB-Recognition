import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from TinyMind.Logging import *
__all__=['FocalLoss']
def sigmoid_focal_loss(preds:torch.Tensor,targets:torch.Tensor,gamma:float=2,alpha:float=0.25):
    num_classes=preds.size(1)
    print('num_class:{}'.format(num_classes))
    dtype=targets.dtype
    device=targets.device
    class_range=torch.arange(0,num_classes,dtype=dtype,device=device).unsqueeze(0)
    print('class_range:{}'.format(class_range))
    t=targets.unsqueeze(1)
    print(t==class_range)
    p=torch.sigmoid(preds).clamp(min=1e-10,max=0.999999999)
    term1=(1-p)**gamma*torch.log(p)
    term2=p**gamma*torch.log(1-p)
    loss=-(t==class_range).float()*term1*alpha-((t!=class_range)*(t>=0)).float()*term2*(1-alpha)
    return loss
class _FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25,eps=1e-7):
        super(_FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha=alpha
        self.eps = eps
    def forward(self, input, target):
        background_num=(target==0).sum()
        print('target:{}|background_num:{}'.format(len(target),background_num))
        loss=sigmoid_focal_loss(input,target,gamma=self.gamma,alpha=self.alpha)
        return loss.sum()
class FocalLoss(nn.Module):
    def __init__(self,num_classes=20):
        super(FocalLoss,self).__init__()
        self.num_classes=num_classes+1
    def forward(self, loc_preds,loc_targets,cls_preds,cls_targets):
        num_classes=cls_preds.size(-1)
        N=cls_targets.size(1)
        batch_size,num_boxes=cls_targets.size()
        logging.debug("loc_pred'size:{}|loc_target'size:{}|cls_pred's size:{}|cls_target's size:{}".format(loc_preds.size(),loc_targets.size(),cls_preds.size(),cls_targets.size()))
        pos=cls_targets>0#[N,anchors]
        num_pos=pos.data.long().sum()
        #loc_loss
        mask=pos.unsqueeze(2).expand_as(loc_preds)
        #logging.debug('mask:{}'.format(mask.cpu().data))
        masked_loc_preds=loc_preds[mask].contiguous().view(-1,4)
        masked_loc_target=loc_targets[mask].contiguous().view(-1,4)
        print('masked_loc_preds:{}|mask_loc_target:{}'.format(masked_loc_preds[:4,:],masked_loc_target[:4,:]))
        logging.debug("the positive example's number:{}".format(masked_loc_target.size(0)))
        loc_loss=F.smooth_l1_loss(masked_loc_preds,masked_loc_target,reduction='sum')
        #focal_loss
        pos_neg=cls_targets>-1
        logging.debug("cls_pred's size:{}|cls_target's size:{}".format(cls_preds.size(),cls_targets.size()))
        mask=pos_neg.unsqueeze(2).expand_as(cls_preds)#[N,anchors,classes]
        masked_cls_preds=cls_preds[mask].view(-1,num_classes)
        logging.debug("the example's number:{}".format(masked_cls_preds.size(0)))
        #cls_loss=self.focal_loss(masked_cls_preds,cls_targets[pos_neg])
        cls_loss=_FocalLoss(2)(masked_cls_preds,cls_targets[pos_neg])
        logging.debug('loc_loss:{:.6f}|cls_loss:{:.3f}'.format(loc_loss.data.item(),cls_loss.data.item()))
        loss=loc_loss/num_pos+cls_loss/(num_pos+N)
        #reference https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/modeling/rpn/retinanet/loss.py
        #loss=cls_loss/num_pos
        return loss
