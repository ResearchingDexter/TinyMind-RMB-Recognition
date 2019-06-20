from torch.utils.data import Dataset
from PIL import Image
from typing import Any,Optional,Tuple,Callable,List,TypeVar,Iterable
import torch
function=TypeVar('function')
import os
import numpy as np
from random import randint
tuple_l=(3,5)#(4,4)
def load_img(path:str)->Image.Image:
    return Image.open(path).convert('RGB')
def resize_img(img:Image.Image,boxes:torch.Tensor,expected_size:Tuple)->Tuple[Image.Image,torch.Tensor]:#expected_size=(w,h)
    w,h=img.size
    w_ratio,h_ratio=expected_size[0]/w,expected_size[1]/h
    img=img.resize(expected_size,resample=Image.BICUBIC)
    #print('ratio:{}|{}'.format(w_ratio,h_ratio))
    boxes=boxes.float()
    #print('before',boxes[:2])
    if w_ratio==h_ratio:
        boxes*=w_ratio
        boxes[:, [0, 1]] = boxes[:, [0, 1]].floor()
        boxes[:, [2, 3]] = boxes[:, [2, 3]].ceil()
        #print('after',boxes[:2])
        return img,boxes
    else:
        #print('before change:{}'.format(boxes))
        boxes[:,[0,2]]*=w_ratio
        boxes[:,[1,3]]*=h_ratio
        boxes[:, [0, 1]]=boxes[:,[0,1]].floor()
        boxes[:, [2, 3]]=boxes[:,[2,3]].ceil()
        #print('after change:{}'.format(boxes))
        return img,boxes
def img_transform(img:Image.Image,expected_size:Tuple)->Tuple[Image.Image,float]:
    img_size = img.size
    ratio = [e / i for e, i in zip(expected_size, img_size)]
    min_ratio = min(ratio)
    img_size = list(map(lambda x: round(x * min_ratio), img_size))
    # img_size=expected_size
    img_ = img.resize(img_size, resample=Image.BICUBIC)
    return img_,min_ratio
class FBDataSet(Dataset):
    def __init__(self,img_path:str,label_path:Optional[str]=None,load_label:Optional[Callable[[str,str] ,Tuple]]=None,target_transform:Optional[Callable[...,Tuple]]=None,
                 expected_img_size:Optional[Tuple]=None,img_transform:Callable[[Image.Image],torch.Tensor]=None,load_img:Callable[[str],Image.Image]=load_img,
                 cfg:Optional[classmethod]=None,labels_index=False,train:bool=True):
        super(FBDataSet,self).__init__()
        self.img_path=img_path
        self.img_transform=img_transform
        self.load_img=load_img
        self.labels_index=labels_index
        self.train=train
        self.cfg=cfg
        if self.train==False:
            self.labels_index=False
        self.expected_img_size=expected_img_size#(w,h)
        if self.train==True:
            self.label_path=label_path
            self.load_label=load_label
            self.target_transfrom=target_transform
            if self.labels_index==True:
                self.labels_name=os.listdir(label_path)#
            else:
                self.imgs_name=os.listdir(img_path)
            #self.labels_name=self.load_label(self.label_path)
        else:
            self.imgs_name = os.listdir(img_path)
    def __getitem__(self, item):
        if self.labels_index==False:
            img_name=self.imgs_name[item]
        else:
            img_name=self.labels_name[item].split('.')[0]+'.jpg'
        img=self.load_img(self.img_path+img_name)
        image=img
        print('img_name:{}'.format(img_name))
        img_size=img.size#(w,h)
        temp=randint(0,10)
        print('rand:{}'.format(temp))
        if self.expected_img_size is not None:
            expected_size=self.expected_img_size
        else:
            expected_size=(800,600)#
        ratio=[e/i for e,i in zip(expected_size,img_size)]
        min_ratio=min(ratio)
        img_size=list(map(lambda x:round(x*min_ratio),img_size))
        if self.train==True:
            coordinates,classes=self.load_label(self.label_path,img_name)
            img,coordinates=resize_img(img,coordinates,img_size)#
            img=self._pad_img(img,expected_size)
            img_size=img.size
            #print('img_size:{}'.format(img_size))
            if self.img_transform is not None:
                img = self.img_transform(img)
            if self.cfg is not None:
                tuple_l=self.cfg.TUPLE_L
            else:
                tuple_l=(3,5)
            loc_target, cls_target=self.target_transfrom(coordinates,classes,img_size,tuple_l=tuple_l,cfg=self.cfg)#
            print('max loc_target:{}|min loc_target:{}|tuple_l:{}'.format(loc_target.max(),loc_target.min(),tuple_l))
            return img,loc_target,cls_target,img_name
        if self.expected_img_size is not None:
            img=img.resize(img_size,resample=Image.BICUBIC)
            img=self._pad_img(img,expected_size)
        if self.img_transform is not None:
            img=self.img_transform(img)
        return img,img_name,min_ratio,image
    def __len__(self):
        if self.labels_index==False:
            return len(self.imgs_name)
        else:
            return len(self.labels_name)
    def collate(self,batch_data:List):
        elem_type=batch_data[0]
        if isinstance(batch_data[0],str):
            return batch_data
        elif isinstance(batch_data[0],Image.Image):
            return batch_data
        elif isinstance(batch_data[0],int):
            return torch.LongTensor(batch_data)
        elif isinstance(batch_data[0],float):
            return batch_data
        elif isinstance(batch_data[0],torch.Tensor):
            return torch.stack(batch_data,dim=0)
        elif isinstance(batch_data[0],Iterable):
            transposed=zip(*batch_data)
            return [self.collate(samples) for samples in transposed]
        else:
            raise TypeError('Expected get Image or string but get the type:{}'.format(elem_type))
    def _pad_img(self,img:Image.Image,img_size_padded:tuple):#img_size_padded=(w,h)
        w,h=img.size
        img_array=np.array(img)
        img_array=img_array.transpose((2,0,1))
        img_array=np.pad(img_array,((0,0),(0,img_size_padded[1]-h),(0,img_size_padded[0]-w)),mode='constant',constant_values=0)
        img_array=img_array.transpose((1,2,0))
        img=Image.fromarray(img_array)
        #img1.show()
        return img
if __name__=='__main___':
    a=FBDataSet('SS','SS')

    #def a(s=1):
        #print(s)
    print(a)