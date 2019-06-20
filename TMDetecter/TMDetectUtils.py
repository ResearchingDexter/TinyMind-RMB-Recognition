import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from typing import Tuple,List
import math
import copy
from FoveaBox.FBDataSet import FBDataSet
import numpy as np
import pdb
from FoveaBox.FBConfigure import TOTAL_NUM,POSITIVE_NUM
#from CrowdHuman.CHUtils import resize_img
from PIL import Image,ImageDraw,ImageFont
POSITIVE_NUM=400
TOTAL_NUM=800
def change_box_order(boxes_:torch.Tensor,order:str)->torch.Tensor:
    assert order in ['ltwh2ltrb','ltrb2ltwh','ltrb2xywh']
    a=boxes_[:,:2]
    b=boxes_[:,2:]
    if order=='ltrb2ltwh':
        return torch.cat([a,b-a],1)
    elif order=='ltrb2xywh':
        boxes=torch.cat([a,b-a],1)
        a=boxes[:,:2].float()
        b=boxes[:,2:].float()
        return torch.cat([a+b*0.5,b],1)
    return torch.cat([a,a+b],1)
def meshgrid(x:int,y:int,row_major=True):
    a=torch.arange(0,x)
    b=torch.arange(0,y)
    xx=a.repeat(y).view(-1,1)
    yy=b.view(-1,1).repeat(1,x).view(-1,1)
    return torch.cat([xx,yy],1) if row_major else torch.cat([yy,xx],1)
def cal_S(tuple_l:Tuple)->List:
    S=[]
    S0=4#16
    for l in range(tuple_l[0],tuple_l[1]+1):
        S.append(S0*4**l)
    return S
def cal_boxes_area(coordinates:torch.Tensor)->torch.Tensor:
    whs=coordinates[:,2:]-coordinates[:,:2]
    areas=whs[:,0]*whs[:,1]
    return areas
def scale_assignment(areas:torch.Tensor,Sl:float,eta:float=2.5)->torch.Tensor:
    if Sl==4**5*4:
        keep=areas>Sl/(eta**2)
        return keep
    keep=(areas>Sl/(eta**2)) & (areas<Sl*(eta**2))
    return keep
def location_transform(location:torch.Tensor,keep_coordinates:torch.Tensor,Sl:float,l:int)->torch.Tensor:
    #print('before transform location:{}|keep coordinates:{}Sl:{}|l:{}'.format(location,keep_coordinates,Sl,l))
    """
    location[:,:,:2]=((2**l)*(location[:,:,0:2]+0.5)-keep_coordinates[[1,0]])/math.sqrt(Sl)
    location[:,:,2:]=(keep_coordinates[[3,2]]-(2**l)*(location[:,:,2:]+0.5))/math.sqrt(Sl)
    """
    location[:,:,:2]=((2**l)*(location[:,:,:2]+0.5)-keep_coordinates[[1,0]])/math.sqrt(Sl)
    location[:,:,2:]=torch.log(keep_coordinates[[3,2]]/math.sqrt(Sl))
    #print('after transform location:{}'.format(location.max()))
    return location
def encode_cls_loc(l:int,sigma1:float,sigma2:float,Sl:float,keep_coordinates:torch.Tensor,keep_labels:torch.Tensor,fm_size:torch.IntTensor)->Tuple[torch.Tensor,torch.Tensor]:
    a_spur = keep_coordinates[:, :2] / (2 ** l)
    b_spur = keep_coordinates[:, 2:] / (2 ** l)
    center = a_spur + 0.5 * (b_spur - a_spur)
    a_spur2 = (center - 0.5 * (b_spur - a_spur) * sigma1).floor().int()#x1,y1 of positive
    b_spur2 =( center + 0.5 * (b_spur - a_spur) * sigma1).ceil().int()#x2,y2 of positive
    neg_a_spur2=(center - 0.5 * (b_spur - a_spur) * sigma2).floor().int()#x1,y1 of negative
    neg_b_spur2=(center + 0.5 * (b_spur - a_spur) * sigma2).ceil().int()#x2,y2 of negative
    #print(a_spur2,b_spur2,a_spur2.floor())
    matrix = torch.zeros((fm_size[1].item(),fm_size[0].item()))#*(-1)#h,w
    matrix_4=meshgrid(fm_size[0].item(),fm_size[1].item(),False).reshape(fm_size[1].item(),fm_size[0].item(),2)
    matrix_4=torch.cat([matrix_4,matrix_4],dim=-1).float()
    matrix_4_copy=copy.deepcopy(matrix_4)
    """"""
    keep_coordinates_xywh=change_box_order(keep_coordinates,'ltrb2xywh')
    for i,label in enumerate(keep_labels):
        matrix[neg_a_spur2[i][1]:neg_b_spur2[i][1],neg_a_spur2[i][0]:neg_b_spur2[i][0]]=0
        matrix[a_spur2[i][1]:b_spur2[i][1],a_spur2[i][0]:b_spur2[i][0]]=label
        matrix_4[a_spur2[i][1]:b_spur2[i][1],a_spur2[i][0]:b_spur2[i][0], :]=matrix_4_copy[a_spur2[i][1]:b_spur2[i][1],a_spur2[i][0]:b_spur2[i][0],:]
        location_transform(matrix_4[a_spur2[i][1]:b_spur2[i][1],a_spur2[i][0]:b_spur2[i][0],:],keep_coordinates_xywh[i],Sl,l)##
    return matrix.reshape(1,-1).squeeze(0),matrix_4.reshape(-1,4)

def boxes2fovea(coordinates:torch.Tensor,labels:torch.Tensor,fms_size:torch.Tensor,sigma1:float=0.3,sigma2=0.4,tuple_l:Tuple=(3,7))->Tuple[torch.Tensor,torch.Tensor]:
    S=cal_S(tuple_l)#(3,7)
    areas=cal_boxes_area(coordinates)
    #print('areas:{}'.format(areas))
    # sort the areas in descending order for the large areas' boxes first
    areas,descending_index=areas.sort(descending=True)
    coordinates=coordinates[descending_index]
    labels=labels[descending_index]
    #
    cls_target=[]
    loc_target=[]
    for l in range(tuple_l[0],tuple_l[1]+1):
        fm_size = (fms_size / (2 ** l)).ceil().int()
        keep=scale_assignment(areas,S[l-tuple_l[0]])
        keep_coordinates=coordinates[keep]
        if len(keep_coordinates)==0:
            cls_target.append(torch.Tensor([0]*fm_size[0].item()*fm_size[1].item()))
            loc_target.append(torch.zeros((fm_size[0].item()*fm_size[1].item(),4)))
            print('keep coordinates is none:',l)
            continue
        keep_labels=labels[keep]
        cls,loc=encode_cls_loc(l,sigma1,sigma2,S[l-tuple_l[0]],keep_coordinates,keep_labels,fm_size)
        cls_target.append(cls)
        loc_target.append(loc)
    return torch.cat(cls_target),torch.cat(loc_target,0)
#for inference
def get_densebox(fms_size:torch.Tensor,tuple_l:Tuple=(3,7))->torch.Tensor:
    densebox=[]
    S = cal_S(tuple_l)
    for l in range(tuple_l[0],tuple_l[1]+1):
        fm_size = (fms_size / (2 ** l)).ceil().int()##
        matrix_2 = (meshgrid(fm_size[0].item(), fm_size[1].item(), False).reshape(fm_size[1].item(), fm_size[0].item(), 2).float()+0.5)*(2**l)
        matrix_1=torch.ones((fm_size[1].int().item(),fm_size[0].int().item(),2))* math.sqrt(S[l-tuple_l[0]])#->l-tuple_l[0]
        #matrix_1[:,:,-1]*=2.1
        #matrix_1[:,:,-2]*=1.2
        #print(matrix_2.size(),matrix_1.size())
        densebox.append(torch.cat([matrix_2,matrix_1],dim=-1).reshape(-1,4))
    return torch.cat(densebox,0)#.reshape(-1,3)
def fovea2boxes(loc_preds:torch.Tensor,cls_preds:torch.Tensor,fms_size:torch.Tensor,tuple_l:Tuple=(3,7))->Tuple[List,List,List]:
    batch_coordinate=[]
    batch_cls=[]
    batch_score=[]
    for i,cls_pred in enumerate(cls_preds):
        cls_pred=cls_pred.softmax(dim=-1)
        scores,index_pred=cls_pred.max(-1)
        #print("scores:{}".format(scores))
        index_pos=index_pred>0
        scores_pos=scores[index_pos]
        cls_pos=index_pred[index_pos]
        loc_pos=loc_preds[i][index_pos]
        #loc_pos=loc_preds[i][index_pos]
        densebox=get_densebox(fms_size,tuple_l)
        densebox_pos=densebox[index_pos]
        #"""
        scores_pos_index=scores_pos>0.2
        scores_pos=scores_pos[scores_pos_index]
        cls_pos=cls_pos[scores_pos_index]
        loc_pos=loc_pos[scores_pos_index]
        densebox_pos=densebox_pos[scores_pos_index]
        #"""
        #print(loc_pos.size(),densebox_pos.size())

        #"""
        #location[:,:,:2]=((2**l)*(location[:,:,:2]+0.5)-keep_coordinates[[1,0]])/math.sqrt(Sl)
        #location[:,:,2:]=torch.log(keep_coordinates[[3,2]]/math.sqrt(Sl))
        """
        loc_coordinate_lt=densebox_pos[:,:2]-loc_pos[:,:2]*densebox_pos[:,2:]#y1,x1
        loc_coordinate_rb=densebox_pos[:,:2]+loc_pos[:,2:]*densebox_pos[:,2:]#y2,x2
        """
        loc_coordinate_xy=densebox_pos[:,:2]-loc_pos[:,:2]*densebox_pos[:,2:]#y1,x1
        loc_coordinate_wh=loc_pos[:,2:].exp()*densebox_pos[:,2:]#y2,x2
        loc_coordinate_lt=((loc_coordinate_xy-loc_coordinate_wh/2.)*1).floor()
        loc_coordinate_rb=((loc_coordinate_xy+loc_coordinate_wh/2.)*1.0).ceil()
        """"""
        #print('xy:{}|wh:{}|lt:{}|rb:{}'.format(loc_coordinate_xy,loc_coordinate_wh,loc_coordinate_lt,loc_coordinate_rb))
        loc_coordinate=torch.cat([loc_coordinate_lt,loc_coordinate_rb],dim=-1)
        batch_coordinate.append(loc_coordinate[:,[1,0,3,2]])
        batch_cls.append(cls_pos)
        batch_score.append(scores_pos)
    return batch_cls,batch_score,batch_coordinate
#end inference
#nms
def box_nms(bboxes:torch.Tensor,scores:torch.Tensor,threshold=0.5,mode='union'):
    x1=bboxes[:,0]
    y1=bboxes[:,1]
    x2=bboxes[:,2]
    y2=bboxes[:,3]
    areas=(x2-x1+1)*(y2-y1+1)
    _,order=scores.sort(0,descending=True)
    #print(order)
    keep=[]
    while order.numel()>0:
        if order.numel()!=1:
            i=order[0].item()
        else:
            i=order
        keep.append(i)
        if order.numel()==1:
            break
        xx1=x1[order[1:]].clamp(min=x1[i].item())
        yy1=y1[order[1:]].clamp(min=y1[i].item())
        xx2=x2[order[1:]].clamp(max=x2[i].item())
        yy2=y2[order[1:]].clamp(max=y2[i].item())
        w=(xx2-xx1+1).clamp(min=0)
        h=(yy2-yy1+1).clamp(min=0)
        intersection=w*h

        if mode=='union':
            overlap=intersection/(areas[i]+areas[order[1:]]-intersection)
        elif mode=='min':
            overlap=intersection/areas[order[1:]].clamp(max=areas[i])
        else:
            raise TypeError('Unkown nms mode :{}'.format(mode))
        ids=(overlap<=threshold).nonzero().squeeze()
        if ids.numel()==0:
            break
        order=order[ids+1]
        #print('keep:{}'.format(order))
    return torch.LongTensor(keep)
#for loading labels
def default_load_label(label_path:str,img_name:str)->Tuple[torch.Tensor,torch.Tensor]:
    with open(label_path+img_name.split('.')[0]+'.txt','r') as f:
        label=f.readlines()
    coordinates=[]
    cls=[]
    for l in label:
        l=l.strip().split(',')[:8]
        l=list(map(int,l))
        if l[4]!=0:
            coordinate=l[:4]
            #coordinate=change_box_order(torch.Tensor(coordinate),'xywh2xyxy')
            coordinates.append(torch.Tensor(coordinate))
            cls.append(l[5])
    coordinates=torch.stack(coordinates,dim=0)
    coordinates=change_box_order(coordinates,'ltwh2ltrb')
    return coordinates,torch.Tensor(cls)
def default_target_transform(coordinates:torch.Tensor,classes:torch.Tensor,img_size:List,tuple_l:Tuple=(3,7),cfg=None)->Tuple[torch.Tensor,torch.Tensor]:
    cls_target,loc_target=boxes2fovea(coordinates,classes,torch.Tensor(img_size),tuple_l=tuple_l)
    pos_index=(cls_target>0).nonzero().squeeze(-1)
    neg_index=(cls_target==0).nonzero().squeeze(-1)
    if cfg is not None:
        pos_num = cfg.POSITIVE_NUM  # 400
        total_num =cfg.TOTAL_NUM  # 800
    else:
        pos_num=POSITIVE_NUM#400
        total_num=TOTAL_NUM#800
    if len(pos_index)>pos_num:
        pos_index_index=torch.randperm(pos_index.size(0))[:(len(pos_index)-pos_num)]
        cls_target[pos_index[pos_index_index]]=-1
    neg_num=total_num-min(len(pos_index),pos_num)
    if len(neg_index)>neg_num:
        neg_index_index=torch.randperm(neg_index.size(0))[:len(neg_index)-neg_num]
        cls_target[neg_index[neg_index_index]]=-1
    return loc_target,cls_target

def resize_img(img:Image.Image,boxes:torch.Tensor,expected_size:Tuple)->Tuple[Image.Image,torch.Tensor]:#expected_size=(w,h)
    w,h=img.size
    w_ratio,h_ratio=expected_size[0]/w,expected_size[1]/h
    img_=img.resize(expected_size,resample=Image.BICUBIC)
    #print('ratio:{}|{}'.format(w_ratio,h_ratio))
    boxes=boxes.float()
    if w_ratio==h_ratio:
        boxes*=w_ratio
        boxes[:, [0, 1]] = boxes[:, [0, 1]].floor()
        boxes[:, [2, 3]] = boxes[:, [2, 3]].ceil()
        return img_,boxes
    else:
        #print('before change:{}'.format(boxes))
        boxes[:,[0,2]]*=w_ratio
        boxes[:,[1,3]]*=h_ratio
        boxes[:, [0, 1]]=boxes[:,[0,1]].floor()
        boxes[:, [2, 3]]=boxes[:,[2,3]].ceil()
        #print('after change:{}'.format(boxes))
        return img_,boxes

if __name__=='__main__':


    temp=meshgrid(4,5,False).view(5,4,2)
    #print(torch.cat([temp,temp],-1))





