import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import os
import pdb
from datetime import datetime
from torchvision import transforms
import json
from TMTextLine.TMTextLineDataSet import TMTextLineDataSet
from TMTextLine.TMTextLineNN import ResNetLSTM
import sys
from IPython.display import clear_output
sys.path.append('../')
from Logging import *
torch.backends.cudnn.benchmark = True
os.environ['CUDA_VISIBLE_DEVICES']='0'
class cfg:
    DEVICE='cuda'
    BATCH_SIZE=1
    PATH=r'E:\Files\TinyMind\train_data\\'
    DICTIONARY_NAME='dictionary_inv.json'
    COORDINATES_PATH=r'E:\Files\ICDAR2019RecTs\ReCTS_test_part1\ReCTS_test_part1\Task2\coordinates\\'
    IMAGE_PATH=r'E:\Files\TinyMind\train_data\multi_test_crop\\'
    MODEL_PATH=r'E:\Files\TinyMind\train_data\\'
    MODEL_NAME='TMResNet101LSTM_2_512.pkl'
    NUM_CLASS=36
def test(model=ResNetLSTM,DataSet=TMTextLineDataSet,cfg=cfg):
    if cfg.DEVICE=='cuda':
        if torch.cuda.is_available()==False:
            logging.error("can't find a GPU device")
            pdb.set_trace()
    if os.path.exists(cfg.MODEL_PATH+cfg.MODEL_NAME)==False:
        logging.error("can't find a pretrained model")
        pdb.set_trace()
    if os.path.exists(cfg.PATH+cfg.DICTIONARY_NAME)==False:
        logging.error("can't find the dictionary")
        pdb.set_trace()
    with open(cfg.PATH+cfg.DICTIONARY_NAME,'r') as f:
        dictionary_inv=json.load(f)
    model=model(cfg.NUM_CLASS)
    model.load_state_dict(torch.load(cfg.MODEL_PATH+cfg.MODEL_NAME,map_location=cfg.DEVICE))#transforms.ColorJitter(brightness=0.5,contrast=0.5,saturation=0.5,hue=0.3),
    model.to(cfg.DEVICE).eval()
    dataset=DataSet(cfg.IMAGE_PATH,img_transform=transforms.Compose([
                                                                                        transforms.ToTensor(),
                                                                                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]),train=False)
    dataloader=DataLoader(dataset,batch_size=cfg.BATCH_SIZE,num_workers=0)
    length=len(dataloader)
    file=open('task2.txt','w',encoding='utf-8')
    for step,data in enumerate(dataloader):
        step_time=datetime.now()
        imgs,names=data
        imgs=Variable(imgs).to(cfg.DEVICE)
        preds=model(imgs)
        preds=preds.permute(1,0,2)
        batch_size=preds.size(0)
        preds=preds.cpu()
        _, preds = preds.max(2)
        for i in range(batch_size):
            pred,_=condense(preds[i])
            pred_str=[]
            for p in pred:
                s=dictionary_inv.get(str(p))
                pred_str.append(s)
            pred_str=''.join(pred_str)
            if len(pred_str)==0:
                pred_str='1'
            logging.info('length:{}|step:{}|i:{}|predicting time:{}'.format(length,step,i,datetime.now()-step_time))
            logging.info("image's name:{}|predicting character:{}".format(names[i],pred_str))
            name=names[i]
            file.write(name+','+pred_str+'\n')
        if step%200==0:
            clear_output(wait=True)
    logging.info('ended')
    f.close()
def condense(pred):
    result=[]
    original_pred=[]
    for i,p in enumerate(pred):
        original_pred.append(p.item())
        if p!=0 and (not(i>0 and pred[i-1]==pred[i])):
            result.append(p.item())
    return result,original_pred
if __name__=='__main__':
    test()

