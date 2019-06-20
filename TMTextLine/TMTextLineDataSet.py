from torch.utils.data import Dataset
import torch
from PIL import Image
from typing import Any,Optional,List,Iterable,Tuple
import numpy as np
import pandas as pd
import os
import math
def load_img(path:str)->Image.Image:
    return Image.open(path).convert('RGB')
class TMTextLineDataSet(Dataset):
    def __init__(self,img_path:str,dictionary:Optional[dict]=None,expected_img_size:Optional[Tuple]=None,img_transform:Any=None,train=True,load_img=load_img):
        super(TMTextLineDataSet,self).__init__()
        self.img_path=img_path
        self.dictionary=dictionary
        self.expected_img_size=expected_img_size
        self.img_transform=img_transform
        self.train=train
        self.load_img=load_img
        if train==True:
            self.labels=pd.read_csv('train_id_crop_label.csv',sep=',')
        else:
            self.img_names=os.listdir(img_path)
    def __getitem__(self, item):
        if self.expected_img_size is not None:
            e_w, e_h = self.expected_img_size
        else:
            e_w, e_h = 128, 16
        if self.train==True:
            name_label=self.labels.iloc[item,:].tolist()
            name,label=name_label[0],name_label[1]
            """
            figure_label=[]
            for s in label:
                s_=self.dictionary.get(s,default=-1)
                if s==-1:
                    print('name:{}|label:{}'.format(name,label))
                figure_label.append(int(s_)+1)#0 for blank"""
            img=self.load_img(self.img_path+name)
            #theata=random.random()#
            img=self._resize_img(img=img,w=e_w,h=e_h)#376=47*8
            w,h=img.size
            if w<e_w:
                img=self._pad_img(img,(e_w,e_h))
            if self.img_transform is not None:
                img=self.img_transform(img)
            return img,label,len(label),name    #image,the label ,the length of the label
        else:
            img_name=self.img_names[item]
            img=self.load_img(self.img_path+img_name)
            print(img.size)
            img=self._resize_img(img=img,w=e_w,h=e_h)
            w,h=img.size
            if w<e_w:
                img=self._pad_img(img,(e_w,e_h))
            if self.img_transform is not None:
                img=self.img_transform(img)
            return img,img_name
    def __len__(self):
        if self.train==True:
            return len(self.labels)
        else:
            return len(self.img_names)
    def _resize_img(self,img:Image.Image,w=376,h=32):
        img_w,img_h=img.size
        if img_h<h:
            theata=math.ceil(h/img_h)
            img_w*=theata
        elif img_h>h:
            theata=h/img_h
            img_w=math.ceil(img_w*theata)
        #print('img_w:{}'.format(img_w))
        img=img.resize((img_w,h))
        if img_w>w:
            return img.resize((w,h))
        return img
    def _pad_img(self,img,img_size_padded:tuple):#img_size_padded=(w,h)
        w,h=img.size
        #print(img_size_padded)
        img_array=np.array(img)
        img_array=img_array.transpose((2,0,1))
        img_w_left=img_size_padded[0]-w
        img_w_right=img_w_left//2
        img_w_left-=img_w_right
        img_array=np.pad(img_array,((0,0),(0,img_size_padded[1]-h),(img_w_left,img_w_right)),mode='constant',constant_values=0)
        #img_array=np.pad(img_array,((0,0),(0,img_size_padded[1]-h),(0,img_size_padded[0]-w)),mode='constant',constant_values=0)
        img_array=img_array.transpose((1,2,0))
        img=Image.fromarray(img_array)
        #img1.show()
        return img
    def transform_label(self,batch_name):
        batch_label=[]
        one_dim_label_figure=[]
        for name in batch_name:
            label=[]
            for s in name:
                s_=self.dictionary.get(s,-1)
                if s_==-1 or s_==' ':
                    print('name:{}|'.format(s))
                label.append(int(s_))#0 for blank
                one_dim_label_figure.append(int(s_))
            batch_label.append(torch.LongTensor(label))
        """
        one_dim_label=''.join(batch_name)
        one_dim_label_figure=[]
        for s in one_dim_label:
            s_ = self.dictionary.get(s, default=-1)
            if s_ == -1:
                print('name:{}|'.format(s))
            one_dim_label_figure.append(int(s_) + 1)  # 0 for blank"""
        return torch.LongTensor(one_dim_label_figure),batch_label
if __name__=='__main__':
    a=ICDARRecTs_2DataSet('SS')
    print(a)
    """
    print(type(load_img))
    with open('label.txt','r',encoding='UTF-8') as f:
        temp=f.readlines()
        print(temp[0].split(',')[-1][:-1])"""
