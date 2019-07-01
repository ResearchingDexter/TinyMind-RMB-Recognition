import os
import pandas as pd
import json
import torch
import pdb
import numpy as np
from PIL import Image,ImageDraw
from typing import Tuple,List,Optional,Union
from tqdm import tqdm
def get_label():
    path=r'E:\Files\TinyMind\\'
    file_name='train_face_value_label.csv'
    dictionary={'0.1':0,'0.2':1,'0.5':2,'1':3,'2':4,'5':5,'10':6,'20':7,'50':8,'100':9}
def alleviate_json():
    input_path=r'E:\Files\TinyMind\train_data\label\\'
    output_path=r'E:\Files\TinyMind\train_data\train_label_1\\'
    if os.path.exists(output_path)==False:
        os.makedirs(output_path)
    label_names=os.listdir(input_path)
    for label_name in label_names:
        with open(input_path+label_name,'r') as f:
            json_file:dict=json.load(f)
        shape=json_file.get('shapes')[0]
        dict_file={}
        list_file=shape['points'][0]
        list_file.extend(shape['points'][1])
        #print(list_file)
        dict_file.setdefault('points',list_file)
        with open(output_path+label_name,'w') as f:
            json.dump(dict_file,f)
        #break
    print('finished')
def TMcal_labels_length(path:str,label_name:str,crop_path:Optional[str]=None)->None:
    labels:pd.DataFrame=pd.read_csv(path+label_name,sep=',')
    if crop_path is not None:
        crop_imgs_name=set(os.listdir(crop_path))
    dictonary={' ':0}
    img_labels,img_names=[],[]
    for i in range(len(labels)):
        img_name=labels.iloc[i,0]
        img_label=labels.iloc[i,1]
        for s in img_label:
            if dictonary.__contains__(s)==False:
                dictonary[s]=len(dictonary)
        if crop_path is not None and img_name in crop_imgs_name:
            img_names.append(img_name)
            img_labels.append(img_label)
    dictonary_inv={v:k for k,v in dictonary.items()}
    print('length:{}'.format(len(dictonary)))
    with open('dictionary.json','w') as f:
        json.dump(dictonary,f)
    with open('dictionary_inv.json','w') as f:
        json.dump(dictonary_inv,f)
    if crop_path is not None:
        pd.DataFrame({'name':img_names,'label':img_labels},index=0).to_csv(path+'train_id_crop_label.csv',index=0)


def TMcrop_img(img:Image.Image,coordinate:List,save:bool=True,img_name:Optional[str]=None,path:Optional[str]=None)->Optional[Image.Image]:
    if isinstance(coordinate,(torch.Tensor,list)):
        coordinate=np.array(coordinate)
    img_size=img.size
    """
    coordinate[:2]=coordinate[:2].clip(min=0)
    coordinate[2]=min(img_size[0],coordinate[2])
    coordinate[3]=min(img_size[1],coordinate[3])
    """
    if save==True:
        if path is None or img_name is None:
            raise FileNotFoundError('must assign a path and img_name when save is {}'.format(save))
        if os.path.exists(path)==False:
            os.makedirs(path)
        """
        img_draw=ImageDraw.Draw(img)
        img_draw.rectangle(coordinate,outline=(255,0,0))
        print(img_size,coordinate)
        img.show()
        pdb.set_trace()
        """
        img.crop(coordinate).save(path+img_name)
    else:
        return img.crop(coordinate)
def TMwrite_submissions(cls_classes:torch.Tensor,score_classes:torch.Tensor,coordinates:torch.Tensor,img_name:str,cfg:classmethod)->None:
    _,index=score_classes.max(dim=0)
    cls_class=cls_classes[index].tolist()
    coordinate=coordinates[index].tolist()
    with open(cfg.SUBMISSIONS_PATH+img_name,'a+') as f:
        json.dump(coordinate,f)
def TMload_label(label_path:str,img_name:str)->Tuple[torch.Tensor,torch.Tensor]:
    #print(os.path.exists(label_path+img_name.split('.')[0]+'.json'))
    with open(label_path+img_name.split('.')[0]+'.json') as f:
        labels:dict=json.load(f)
    #print(labels)
    coordinate:list=labels["points"]#labels.get('points')
    print('points:{}'.format(coordinate))
    coordinate=torch.Tensor(coordinate)
    coordinate[:2].floor_()
    coordinate[2:].ceil_()
    return coordinate.unsqueeze(0),torch.Tensor([1])
if __name__=='__main__':
    #alleviate_json()
    #with open('./TMTextLine/task2.txt','r') as f:
    with open('./submission/submission.txt', 'r') as f:

        recogniton=f.read().split('\n')
    total=os.listdir(r'E:\Files\TinyMind\private_test_data\private_test_data\\')
    name=[]
    label=[]
    for r in recogniton:
        if r==recogniton[-1]:
            break
        print('r:{}'.format(r))
        r=r.split(',')
        name.append(r[0])
        label.append(r[1])
    name_set=set(name)
    for t in total:
        if t not in name_set:
            name.append(t)
            label.append('LK39482832')
    pd.DataFrame({'name':name,'label':label}).to_csv('./submission/TMIDSubmission_private6_0.99x1.01_VGGN13BNLSTM_200_32.csv',sep=',',index=0)
    print('finished')
    #TMcal_labels_length(r'E:\Files\TinyMind\\','train_id_label.csv')
    """
    path=r'E:\Files\Mybook\kaggle\MicrosoftMalwarePrediction\train\\'
    file_names=os.listdir(path)
    length=len(file_names)
    range_=tqdm(range(0,length))
    length=len(file_names)
    for i in range_:
        range_.set_description('length{}|i:{}'.format(length,i))
        os.remove(path+file_names[i])
    """
