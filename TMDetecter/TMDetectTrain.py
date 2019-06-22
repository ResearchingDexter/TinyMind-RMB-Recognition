import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from datetime import datetime
import torch.optim as optim
from torchvision import transforms
import os
import pdb
from IPython.display import clear_output
from TMDetecter.TMDetectDataSet import FBDataSet
from TMDetecter.TMDetectLoss import FocalLoss
from TMDetecter.TMDetectUtils import default_target_transform#default_load_label
from TMUtils import TMload_label
from TMDetecter.TMRPN import TMRPN
from Logging import *
from TMDetecter.TMDetectConfigure import *
torch.backends.cudnn.benchmark = True
#os.environ['CUDA_VISIBLE_DEVICES']='0'
def train(RetinaNet=TMRPN,default_load_label=TMload_label,default_target_transform=default_target_transform,cfg=TMcfg):
    pretrain=cfg.PRETRAIN
    logging.info('pretrain:{}'.format(pretrain))
    if cfg.DEVICE[:4]=='cuda':
        if torch.cuda.is_available()==False:
            logging.error("can't find a GPU device")
            pdb.set_trace()
    if os.path.exists(cfg.MODEL_PATH)==False:
        os.makedirs(cfg.MODEL_PATH)
    model=RetinaNet(cfg.NUM_CLASS,criterion=FocalLoss())
    if pretrain==True:
        model.load_state_dict(torch.load(cfg.MODEL_PATH + cfg.MODEL_NAME, map_location=cfg.DEVICE))
    if cfg.DISTRIBUTED==True:
        model=nn.DataParallel(model,device_ids=cfg.DEVICE_ID,output_device=cfg.DEVICE_ID[0]).cuda().train()
        #DEVICE=DEVICE_ID[-1]
    else:
        model.to(cfg.DEVICE).train()
    img_transform=transforms.Compose([transforms.ColorJitter(brightness=0.5,contrast=0.5,saturation=0.5,hue=0.3),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    dataset=FBDataSet(cfg.IMAGE_PATH,cfg.LABEL_PATH,default_load_label,default_target_transform,expected_img_size=cfg.EXPECTED_IMG_SIZE,img_transform=img_transform,cfg=cfg,labels_index=cfg.LABEL_INDEX)
    dataloader=DataLoader(dataset,cfg.BATCH_SIZE,shuffle=False,num_workers=0)
    #criterion=FocalLoss()
    #optimizer=optim.Adam([{'params':model.fpn.parameters(),'lr':LR/10},{'params':model.loc_head.parameters()},{'params':model.cls_head.parameters()}],lr=LR,betas=(0.9,0.999),weight_decay=1e-6)
    optimizer=optim.SGD(model.parameters(),lr=cfg.LR,momentum=0.9,weight_decay=1e-5)
    print(optimizer.state_dict())
    scheduler=optim.lr_scheduler.CosineAnnealingLR(optimizer,cfg.EPOCH,1e-7)
    #scheduler=WarmupMultiStepLR(optimizer,[50,100],warmup_iters=10,last_epoch=10)
    #optimizer=optim.SGD(model.parameters(),lr=LR,momentum=0.9)
    length=len(dataloader)
    accumulate_step=8
    max_accuracy = cfg.MAX_ACCURACY
    min_loss = 100
    for epoch in range(cfg.EPOCH):
        epoch_time=datetime.now()
        epoch_loss=0
        epoch_correct=0
        scheduler.step()
        for step,data in enumerate(dataloader):
            #scheduler.step()
            step_time=datetime.now()
            imgs,loc_targets,cls_targets,img_names=data
            imgs=Variable(imgs,requires_grad=True).to(cfg.DEVICE)
            loc_targets=Variable(loc_targets).to(cfg.DEVICE)
            cls_targets=Variable(cls_targets).to(cfg.DEVICE)
            """
            loc_preds,cls_preds=model(imgs)
            batch_size=loc_preds.size(0)
            loss=criterion(loc_preds,loc_targets,cls_preds,cls_targets)
            """
            loss,cls_preds=model(imgs,loc_targets,cls_targets)
            batch_size = cls_preds.size(0)
            loss=loss.sum()
            nan=torch.isnan(loss)
            inf=torch.isinf(loss)
            if nan.item()==1 or inf.item()==1:
                loss.cpu()
                cls_preds.cpu()
                model.zero_grad()
                optimizer.zero_grad()
                torch.cuda.empty_cache()
                continue
            epoch_loss+=loss.item()
            #accumulate gradient
            """
            loss.backward()
            if step%accumulate_step==0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                optimizer.step()
                optimizer.zero_grad()
            #end accumulation
            """
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()#"""
            _,cls_preds_index=cls_preds.max(-1)
            batch_correct=(cls_targets==cls_preds_index.float()).sum().item()
            epoch_correct+=batch_correct
            logging.debug('epoch:{}|length:{}|step:{}|batch_correct:{}|loss:{}|step_time:{}|memory:{:.3f}MB'.format(epoch,length,step,batch_correct,loss.item(),datetime.now()-step_time,torch.cuda.max_memory_allocated()/1024/1024))
            if loss.item()<min_loss:
                min_loss=loss.item()
                if cfg.DISTRIBUTED==True:
                    torch.save(model.module.state_dict(),cfg.MODEL_PATH+cfg.MODEL_NAME)
                else:
                    torch.save(model.state_dict(),cfg.MODEL_PATH+cfg.MODEL_NAME)
            if step%200==0:
                clear_output(wait=True)
        accuracy=epoch_correct/(cfg.BATCH_SIZE*length*cfg.TOTAL_NUM)
        mean_loss=epoch_loss/length
        logging.info('epoch:{}|mean loss:{:.4f}|accuracy:{:.6f}|epoch_time:{}|memory:{}'.format(epoch,mean_loss,accuracy,datetime.now()-epoch_time,torch.cuda.max_memory_allocated()/1024/1024))
        with open(cfg.PATH+'accuracy.txt','a+') as f:
            f.write('epoch:{}|mean loss:{:.4f}|accuracy:{:.6f}|epoch_time:{}'.format(epoch,mean_loss,accuracy,datetime.now()-epoch_time))
        if max_accuracy<accuracy:
            max_accuracy=accuracy
            with open(cfg.PATH+'max_accuracy.txt','w') as f:
                f.write(max_accuracy)
            if cfg.DISTRIBUTED==True:
                torch.save(model.module.state_dict(), cfg.MODEL_PATH +'_'+str(max_accuracy)[:5]+'_'+ cfg.MODEL_NAME)
            else:
                torch.save(model.state_dict(), cfg.MODEL_PATH +'_'+str(max_accuracy)[:5]+'_'+ cfg.MODEL_NAME)
if __name__=='__main__':
    train(RetinaNet=TMRPN,default_load_label=TMload_label)

