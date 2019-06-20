import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import pdb
from typing import List
from torch.autograd import Variable
from IPython.display import clear_output
from datetime import datetime
from torchvision.transforms import transforms
from TMDetecter.TMDetectDataSet import FBDataSet,img_transform,load_img
from Logging import *
from TMDetecter.TMDetectUtils import fovea2boxes,box_nms
from TMUtils import TMcrop_img
from TMDetecter.TMRPN import TMRPN
from TMDetecter.TMDetectConfigure import *
@torch.no_grad()
def test(RetinaNet=TMRPN,cfg=TMcfg):
    if cfg.DEVICE[:4]=='cuda':
        if torch.cuda.is_available()==False:
            logging.info("can't find a GPU device")
            pdb.set_trace()
    model=RetinaNet(cfg.NUM_CLASS)#transforms.ColorJitter(brightness=0.5,contrast=0.5,saturation=0.5,hue=0.3),
    model.load_state_dict(torch.load(cfg.MODEL_PATH+cfg.MODEL_NAME,map_location=cfg.DEVICE))
    dataset=FBDataSet(cfg.IMAGE_PATH,expected_img_size=cfg.EXPECTED_IMG_SIZE,
                      img_transform=transforms.Compose([
                                                        transforms.ToTensor(),
                                                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]),
                      cfg=cfg,train=False)
    dataloader=DataLoader(dataset,batch_size=cfg.BATCH_SIZE,num_workers=4,collate_fn=dataset.collate)
    if cfg.DISTRIBUTED==True:
        model=nn.DataParallel(module=model,device_ids=cfg.DEVICE_ID).cuda().eval()
    else:
        model.to(cfg.DEVICE).eval()
    length=len(dataloader)
    start_time=datetime.now()
    empty_num=0
    f=open('cannot_detect.txt','w')
    multiscale_imgs=[]
    for step,data in enumerate(dataloader):
        step_time=datetime.now()
        imgs,imgs_name,min_ratioes,images=data
        keep_images=list(range(imgs.size(0)))
        imgs = Variable(imgs, requires_grad=False).to(cfg.DEVICE)
        batch_size=imgs.size(0)
        h,w=imgs.size(2),imgs.size(3)
        loc_preds,cls_preds=model(imgs)
        batch_cls,batch_score,batch_coordinate=fovea2boxes(loc_preds.cpu(),cls_preds.cpu(),torch.Tensor([w,h]),tuple_l=cfg.TUPLE_L)
        logging.debug("length:{}|step:{}|imgs_name:{}".format(length,step,imgs_name))
        for b in range(batch_size):
            cls,score,coordinate=batch_cls[b],batch_score[b],batch_coordinate[b]
            cls_list,score_list,coordinate_list=[],[],[]
            for num_class in range(1,cfg.NUM_CLASS+1):
                num_class_index=(cls==num_class)
                cls_class,score_class,coordinate_class=cls[num_class_index],score[num_class_index],coordinate[num_class_index]
                keep=box_nms(coordinate_class,score_class,threshold=0.3)
                cls_class,score_class,coordinate_class=cls_class[keep],score_class[keep],coordinate_class[keep]
                coordinate_class/=min_ratioes[b]
                coordinate_class[:,[0,1]].floor_()
                coordinate_class[:,[2,3]].ceil_()
                if len(score_class)==0:
                    multiscale_imgs.append(imgs_name[b])
                    empty_num+=1
                    f.write(imgs_name[b]+'\n')
                    continue
                else:
                    keep_images.pop(b)
                _,index=score_class.max(0)
                TMcrop_img(images[b],coordinate_class[index].tolist(),img_name=imgs_name[b],path=cfg.CROP_PATH)
        if step%200==0:
            clear_output(wait=True)
        logging.debug("step_time cost :{}".format(datetime.now()-step_time))
    multiscale_test(multiscale_imgs,model,cfg=cfg)
    f.close()
    logging.info("finshed and total cost of time is :{}|number of empty:{}".format(datetime.now()-start_time,empty_num))
@torch.no_grad()
def multiscale_test(imgs_name:List,model:nn.Module,cfg=TMcfg):#148KLEW0：fliped
    img2tensor = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    f=open('multi_cannot_detect.txt','w')
    logging.info("starting multiscale test")
    for i,img_name in enumerate(imgs_name):
        try:
            image = load_img(cfg.IMAGE_PATH + img_name)
        except FileNotFoundError:
            break
        flag = False
        break_time=0
        for j,scale in enumerate(cfg.MULTISCALE_SIZE):
            if flag==True:
                break
            #image = load_img(cfg.IMAGE_PATH + img_name)
            logging.debug("length:{}|step:{}|img_name:{}|scale:{}".format(len(imgs_name),i,img_name,scale))
            img,min_ratio=img_transform(image,scale)
            #w,h=img.size
            #pad=transforms.Pad(padding=(0,0,scale[0]-w,scale[1]-h),fill=0)
            #img=pad(img_)
            img_tensor=img2tensor(img)
            h,w=img_tensor.size(1),img_tensor.size(2)
            loc_preds,cls_preds=model(img_tensor.unsqueeze(0).to(cfg.DEVICE))
            batch_cls,batch_score,batch_coordinate=fovea2boxes(loc_preds.cpu(),cls_preds.cpu(),torch.Tensor([w,h]),tuple_l=cfg.TUPLE_L)
            cls,score,coordinate=batch_cls[0],batch_score[0],batch_coordinate[0]
            for num_class in range(1,cfg.NUM_CLASS+1):
                num_class_index=(cls==num_class)
                cls_class,score_class,coordinate_class=cls[num_class_index],score[num_class_index],coordinate[num_class_index]
                if len(cls_class)==0:
                    break_time+=1
                    continue
                print('cls_class:{}'.format(cls_class))
                keep=box_nms(coordinate_class,score_class,threshold=0.3)
                cls_class,score_class,coordinate_class=cls_class[keep],score_class[keep],coordinate_class[keep]
                print('coordinate:{}'.format(coordinate_class))
                coordinate_class/=min_ratio
                coordinate_class[:,[0,1]]=coordinate_class[:,[0,1]].floor()
                coordinate_class[:,[2,3]]=coordinate_class[:,[2,3]].ceil()
                print('coordinate:{}'.format(coordinate_class))
                if len(score_class) == 0:
                    break_time+=1
                    break
                _, index = score_class.max(0)
                print('index:{}|coordinate_classs:{}'.format(index,coordinate_class[index]))
                #TMcrop_img(image,temp,img_name=img_name,path=cfg.CROP_PATH)
                TMcrop_img(image,coordinate_class[index],img_name=img_name,path=cfg.CROP_PATH_COMPLEMENT)
                flag=True
            if break_time==len(cfg.MULTISCALE_SIZE):
                f.write(img_name + '\n')
        #image.show(title='test')
        #break
        logging.info("flag:{}".format(break_time))
    f.close()
def write_submission(cls_class:List,score_class:List,coordinate:List,img_name:str)->None:
    f=open(img_name+'.txt','a+')
    for i in range(len(cls_class)):
        coordinate_i=coordinate[i]
        coordinate_i=list(map(str,coordinate_i))
        f.write(','.join(coordinate_i)+','+str(cls_class[i])+','+str(score_class[i]))
    f.close()
if __name__=='__main__':
    #test(TMRPN,TMcfg)

    with open('cannot_detect.txt','r') as f:
        imgs_name=f.read()
    #print(imgs_name.split('\n'))

    model = TMRPN(TMcfg.NUM_CLASS)  # transforms.ColorJitter(brightness=0.5,contrast=0.5,saturation=0.5,hue=0.3),
    model.load_state_dict(torch.load(TMcfg.MODEL_PATH + TMcfg.MODEL_NAME, map_location=TMcfg.DEVICE))
    model.eval()
    multiscale_test(imgs_name.split('\n'),model.to(TMcfg.DEVICE))
