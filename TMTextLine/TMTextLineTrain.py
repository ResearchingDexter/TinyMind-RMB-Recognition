import torch
from torch.autograd import Variable
from datetime import datetime
from torch.utils.data import DataLoader
from torch.nn import CTCLoss,CrossEntropyLoss
from torch.optim import Adam,Adadelta
import json
from torchvision import transforms
from IPython.display import clear_output
from TMTextLine.TMTextLineDataSet import TMTextLineDataSet
from TMTextLine.TMTextLineNN import ResNetLSTM
import pdb
import os
import sys
sys.path.append('../')
from Logging import *
from TMTextLine.TMTextLineConfigure import *
cfg=cfg()
torch.backends.cudnn.benchmark = True
os.environ['CUDA_VISIBLE_DEVICES']='0'
def train(pretrain=cfg.PRETRAIN,model=ResNetLSTM,DataSet=TMTextLineDataSet,cfg=cfg):
    logging.info('pretrain:{}'.format(pretrain))
    if cfg.DEVICE=='cuda':
        if torch.cuda.is_available()==False:
            logging.error("can't find a GPU device")
            pdb.set_trace()
    model=model(cfg.NUM_CLASS)
    if os.path.exists(cfg.MODEL_PATH)==False:
        os.makedirs(cfg.MODEL_PATH)
    if os.path.exists(cfg.PATH+cfg.DICTIONARY_NAME)==False:
        logging.error("can't find the dictionary")
        pdb.set_trace()
    with open(cfg.PATH+cfg.DICTIONARY_NAME,'r') as f:
        dictionary=json.load(f)
    if pretrain==True:
        model.load_state_dict(torch.load(cfg.MODEL_PATH+cfg.MODEL_NAME,map_location=cfg.DEVICE))
    model.to(cfg.DEVICE).train()
    model.register_backward_hook(backward_hook)#transforms.Resize((32,400))
    dataset=DataSet(cfg.IMAGE_PATH,dictionary,cfg.EXPECTED_IMG_SIZE,img_transform=transforms.Compose([transforms.ColorJitter(brightness=0.5,contrast=0.5,saturation=0.5,hue=0.3),
                                                                                        transforms.ToTensor(),
                                                                                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]),
                    cfg=cfg)
    dataloader=DataLoader(dataset,batch_size=cfg.BATCH_SIZE,shuffle=True,num_workers=4,drop_last=False)#collate_fn=dataset.collate
    #optimizer=Adam(model.parameters(),lr=LR,betas=(0.9,0.999),weight_decay=0)
    optimizer=Adadelta(model.parameters(),lr=0.01,rho=0.9,weight_decay=0)
    if cfg.LOSS=='CTC':
        criterion=CTCLoss(blank=0)
    else:
        criterion=CrossEntropyLoss()
    length=len(dataloader)
    max_accuracy=0
    if os.path.exists('max_accuracy.txt')==True:
        with open('max_accuracy.txt','r') as f:
            max_accuracy=float(f.read())
    for epoch in range(cfg.EPOCH):
        epoch_time=datetime.now()
        epoch_correct=0
        epoch_loss=0
        min_loss=100
        for step,data in enumerate(dataloader):
            step_time=datetime.now()
            imgs,names,label_size,img_name=data
            #print(names,label_size)
            logging.debug("imgs' size:{}".format(imgs.size()))
            imgs=Variable(imgs,requires_grad=True).to(cfg.DEVICE)
            label,batch_label=dataset.transform_label(batch_name=names)
            label=Variable(label).to(cfg.DEVICE)
            label_size=Variable(label_size).to(cfg.DEVICE)
            preds=model(imgs)
            logging.debug("preds size:{}".format(preds.size()))
            if cfg.LOSS=='CTC':
                preds_size=Variable(torch.LongTensor([preds.size(0)]*cfg.BATCH_SIZE)).to(cfg.DEVICE)
                loss=criterion(preds,label,preds_size,label_size)
                num_same = if_same(preds.cpu().data, batch_label)

            else:
                loss=criterion(preds,label)
                preds_index=preds.max(1)[1]
                num_same = (preds_index==label).sum().item()
            epoch_loss+=loss.item()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()
            if min_loss>loss.item():
                min_loss=loss.item()
                torch.save(model.state_dict(),cfg.MODEL_PATH+cfg.MODEL_NAME)
            #num_same=if_same(preds.cpu().data,batch_label)
            epoch_correct+=num_same
            logging.debug("Epoch:{}|length:{}|step:{}|num_same:{}|loss:{:.4f}|min loss:{:.4f}".format(epoch,length,step,num_same,loss.item(),min_loss))
            logging.debug("the time of one step:{}".format(datetime.now()-step_time))
            if step%100==0:
                clear_output(wait=True)
        accuracy=epoch_correct/(length)*cfg.BATCH_SIZE
        if accuracy>max_accuracy:
            max_accuracy=accuracy
            with open('max_accuracy.txt','w') as f:
                f.write(str(max_accuracy))
            torch.save(model.state_dict(),cfg.MODEL_PATH+cfg.MODEL_NAME)
            torch.save(model.state_dict(),cfg.MODEL_PATH+'optimal'+str(max_accuracy)+cfg.MODEL_NAME)
        mean_loss=epoch_loss/length
        logging.info('Epoch:{}|accuracy:{}|mean loss:{}|the time of one epoch:{}|max accuracy:{}'.format(epoch,accuracy,mean_loss,datetime.now()-epoch_time,max_accuracy))
        with open('accuracy.txt','a+') as f:
            f.write('Epoch:{}|accuracy:{}|mean loss:{}|the time of one epoch:{}|max accuracy:{}\n'.format(epoch,accuracy,mean_loss,datetime.now()-epoch_time,max_accuracy))
def backward_hook(module,grad_input,grad_output):
    for g in grad_input:
        #print('g:{}'.format(g))
        g[g!=g]=0#replace all nan or inf in gradients to zero
def if_same(preds,batch_label):
    #print(batch_label)
    t,b,n_class=preds.size()
    preds=preds.permute(1,0,2)
    _,preds=preds.max(2)
    count=0
    def condense(pred):
        result=[]
        original_pred=[]
        for i,p in enumerate(pred):
            original_pred.append(p.item())
            if p!=0 and (not(i>0 and pred[i-1]==pred[i])):
                result.append(p.item())
        return result,original_pred
    for pred,label in zip(preds,batch_label):
        flag=0
        pred,original_pred=condense(pred)
        label,_=condense(label)
        if(len(pred)==len(label)):
            for i,p in enumerate(pred):
                if(p!=label[i]):
                    flag=1
                    break
        if(flag==0 and len(pred)==len(label)):
            count+=1
        """if(count==1):
            print('label:{}'.format(label))
            print('pred:{}'.format(pred))
            print('original pred:{}'.format(original_pred))"""
        print('label:{}'.format(label))
        print('pred:{}'.format(pred))
        if(len(pred)==0):
            pass
            #return (0,1)
    return count
if __name__=='__main__':
    train(cfg.PRETRAIN)

