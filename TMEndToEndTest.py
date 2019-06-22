import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import pdb
import os
import json
from typing import List
from torch.autograd import Variable
from IPython.display import clear_output
from datetime import datetime
from torchvision.transforms import transforms
from TMDetecter.TMDetectDataSet import FBDataSet,img_transform,load_img
from TMTextLine.TMTextLineDataSet import TMTextLineDataSet
from Logging import *
from TMDetecter.TMDetectUtils import fovea2boxes,box_nms
from TMUtils import TMcrop_img
from TMDetecter.TMRPN import TMRPN
from TMTextLine.TMTextLineNN import ResNetLSTM,VGGLSTM
from TMTextLine.TMTextLineTest import condense,distill_condense
from TMEndToEndConfigure import *
@torch.no_grad()
def end_to_end_test(detect_model:nn.Module=TMRPN(TMETEcfg.DETECT_NUM_CLASS),
                    detect_dataset=FBDataSet,
                    recognize_model:nn.Module=VGGLSTM(TMETEcfg.RECOGNIZE_NUM_CLASS),
                    cfg=TMETEcfg()):
    if cfg.DEVICE=='cuda':
        if torch.cuda.is_available()==False:
            logging.error("can't find a GPU device")
            pdb.set_trace()
    if os.path.exists(cfg.PATH+cfg.DETECT_MODEL_NAME)==False:
        logging.error("can't find a pretrained model:{}".format(cfg.PATH+cfg.DETECT_MODEL_NAME))
        pdb.set_trace()
    if os.path.exists(cfg.PATH+cfg.RECOGNIZE_MODEL_NAME)==False:
        logging.error("can't find a pretrained model:{}".format(cfg.PATH+cfg.RECOGNIZE_MODEL_NAME))
        pdb.set_trace()
    if os.path.exists(cfg.PATH+cfg.DICTIONARY_NAME)==False:
        logging.error("can't find the dictionary{}".format(cfg.PATH+cfg.DICTIONARY_NAME))
        pdb.set_trace()
    with open(cfg.PATH+cfg.DICTIONARY_NAME,'r') as f:
        dictionary_inv=json.load(f)
    cfg.__setattr__('dictionary_inv',dictionary_inv)
    detect_model.load_state_dict(torch.load(cfg.PATH+cfg.DETECT_MODEL_NAME,map_location=cfg.DEVICE))
    recognize_model.load_state_dict(torch.load(cfg.PATH+cfg.RECOGNIZE_MODEL_NAME,map_location=cfg.DEVICE))
    detect_model.to(cfg.DEVICE).eval()
    recognize_model.to(cfg.DEVICE).eval()
    dataset = detect_dataset(cfg.TEST_IMAGE_PATH, expected_img_size=cfg.DETECT_EXPECTED_IMG_SIZE,
                        img_transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]),
                        cfg=cfg, train=False)
    dataloader=DataLoader(dataset,batch_size=cfg.BATCH_SIZE,num_workers=4,collate_fn=dataset.collate)
    length = len(dataloader)
    start_time = datetime.now()
    f = open(cfg.OUTPUT_PATH+'cannot_detect.txt', 'w')
    r_e_w,r_e_h=cfg.RECOGNIZE_EXPECTED_IMG_SIZE
    multiscale_imgs_name=[]
    img2tensor = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    for step, data in enumerate(dataloader):
        step_time = datetime.now()
        imgs, imgs_name, min_ratioes, images = data
        imgs = Variable(imgs, requires_grad=False).to(cfg.DEVICE)
        batch_size = imgs.size(0)
        h, w = imgs.size(2), imgs.size(3)
        loc_preds, cls_preds = detect_model(imgs)
        batch_cls, batch_score, batch_coordinate = fovea2boxes(loc_preds.cpu(), cls_preds.cpu(), torch.Tensor([w, h]),
                                                               tuple_l=cfg.TUPLE_L)
        logging.debug("length:{}|step:{}|imgs_name:{}|memory:{:.3f}MB".format(length, step, imgs_name,torch.cuda.max_memory_allocated()/1024/1024))
        crop_imgs,crop_imgs_name=[],[]
        for b in range(batch_size):
            cls, score, coordinate = batch_cls[b], batch_score[b], batch_coordinate[b]
            cls_list, score_list, coordinate_list = [], [], []
            for num_class in range(1, cfg.DETECT_NUM_CLASS + 1):
                num_class_index = (cls == num_class)
                cls_class, score_class, coordinate_class = cls[num_class_index], score[num_class_index], coordinate[
                    num_class_index]
                keep = box_nms(coordinate_class, score_class, threshold=0.3)
                cls_class, score_class, coordinate_class = cls_class[keep], score_class[keep], coordinate_class[keep]
                coordinate_class /= min_ratioes[b]
                coordinate_class[:, [0, 1]].floor_()
                coordinate_class[:, [2, 3]].ceil_()
                if len(score_class) == 0:
                    multiscale_imgs_name.append(imgs_name[b])
                    f.write(imgs_name[b] + '\n')
                    continue
                _, index = score_class.max(0)
                """start cropping"""
                crop_img=TMcrop_img(images[b], coordinate_class[index].tolist(), img_name=imgs_name[b],save=False, path=cfg.CROP_PATH)
                crop_img=TMTextLineDataSet.resize_img(crop_img,r_e_w,r_e_h)
                crop_img=TMTextLineDataSet.pad_img(crop_img,(r_e_w,r_e_h))
                crop_img=img2tensor(crop_img)
                crop_imgs.append(crop_img)
                crop_imgs_name.append(imgs_name[b])
        """start recognizing"""
        recognizer(recognize_model,crop_imgs,crop_imgs_name,dictionary_inv,cfg)
        if step % 200 == 0:
            clear_output(wait=True)
        logging.debug("step_time cost :{}".format(datetime.now() - step_time))
    """multiscale test to detecte the image that cannot be detected by single scale"""
    logging.info("finshed and single detect cost of time is :{}|".format(datetime.now() - start_time))
    if len(multiscale_imgs_name)>1:
        multiscale_test(multiscale_imgs_name, detect_model,recognize_model=recognize_model, cfg=cfg)
    f.close()
    logging.info("finshed and total cost of time is :{}|".format(datetime.now() - start_time))
def recognizer(model:nn.Module,crop_imgs:List,names:List,dictionary_inv:dict,cfg):
    f=open(cfg.OUTPUT_PATH+'submission.txt','a+')
    imgs=torch.stack(crop_imgs,dim=0)
    imgs = Variable(imgs).to(cfg.DEVICE)
    preds = model(imgs)
    preds = preds.permute(1, 0, 2)
    batch_size = preds.size(0)
    preds = preds.cpu()
    _, preds = preds.max(2)
    for i in range(batch_size):
        pred, _ = condense(preds[i])
        if len(pred) > 10:
            distill_condense(pred)
        pred_str = []
        for p in pred:
            s = dictionary_inv.get(str(p))
            pred_str.append(s)
        pred_str = ''.join(pred_str)
        if len(pred_str) == 0:
            pred_str = '1'
        logging.info("image's name:{}|predicting character:{}".format(names[i], pred_str))
        name = names[i]
        f.write(name + ',' + pred_str + '\n')
    logging.info('recognizer ended')
    f.close()

@torch.no_grad()
def multiscale_test(imgs_name: List, model: nn.Module,recognize_model, cfg=TMETEcfg()):  # 148KLEW0：fliped
    img2tensor = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    f = open(cfg.OUTPUT_PATH+'multi_cannot_detect.txt', 'w')
    logging.info("starting multiscale test")
    r_e_w,r_e_h=cfg.RECOGNIZE_EXPECTED_IMG_SIZE
    for i, img_name in enumerate(imgs_name):
        try:
            image = load_img(cfg.TEST_IMAGE_PATH + img_name)
        except FileNotFoundError:
            break
        flag = False
        break_time = 0
        for j, scale in enumerate(cfg.MULTISCALE_SIZE):
            if flag == True:
                break
            # image = load_img(cfg.IMAGE_PATH + img_name)
            logging.debug("length:{}|step:{}|img_name:{}|scale:{}".format(len(imgs_name), i, img_name, scale))
            img, min_ratio = img_transform(image, scale)
            # w,h=img.size
            # pad=transforms.Pad(padding=(0,0,scale[0]-w,scale[1]-h),fill=0)
            # img=pad(img_)
            img_tensor = img2tensor(img)
            h, w = img_tensor.size(1), img_tensor.size(2)
            loc_preds, cls_preds = model(img_tensor.unsqueeze(0).to(cfg.DEVICE))
            batch_cls, batch_score, batch_coordinate = fovea2boxes(loc_preds.cpu(), cls_preds.cpu(),
                                                                   torch.Tensor([w, h]), tuple_l=cfg.TUPLE_L)
            cls, score, coordinate = batch_cls[0], batch_score[0], batch_coordinate[0]
            for num_class in range(1, cfg.DETECT_NUM_CLASS + 1):
                num_class_index = (cls == num_class)
                cls_class, score_class, coordinate_class = cls[num_class_index], score[num_class_index], coordinate[
                    num_class_index]
                if len(cls_class) == 0:
                    break_time += 1
                    continue
                print('cls_class:{}'.format(cls_class))
                keep = box_nms(coordinate_class, score_class, threshold=0.3)
                cls_class, score_class, coordinate_class = cls_class[keep], score_class[keep], coordinate_class[keep]
                print('coordinate:{}'.format(coordinate_class))
                coordinate_class /= min_ratio
                coordinate_class[:, [0, 1]] = coordinate_class[:, [0, 1]].floor()
                coordinate_class[:, [2, 3]] = coordinate_class[:, [2, 3]].ceil()
                print('coordinate:{}'.format(coordinate_class))
                if len(score_class) == 0:
                    break_time += 1
                    break
                _, index = score_class.max(0)
                print('index:{}|coordinate_classs:{}'.format(index, coordinate_class[index]))
                # TMcrop_img(image,temp,img_name=img_name,path=cfg.CROP_PATH)
                crop_img=TMcrop_img(image, coordinate_class[index], img_name=img_name, save=False,path=cfg.CROP_PATH_COMPLEMENT)
                crop_img = TMTextLineDataSet.resize_img(crop_img, r_e_w, r_e_h)
                crop_img = TMTextLineDataSet.pad_img(crop_img, (r_e_w, r_e_h))
                crop_img=img2tensor(crop_img)
                recognizer(recognize_model, [crop_img], [img_name], cfg.dictionary_inv, cfg)
                flag = True
            if break_time == len(cfg.MULTISCALE_SIZE):
                f.write(img_name + '\n')
        # image.show(title='test')
        # break
        logging.info("flag:{}".format(break_time))
    f.close()
if __name__=='__main__':
    end_to_end_test()

