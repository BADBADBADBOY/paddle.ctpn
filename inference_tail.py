"""
#!-*- coding=utf-8 -*-
@author: BADBADBADBADBOY
@contact: 2441124901@qq.com
@software: PyCharm Community Edition
@file: inference.py
@time: 2020/4/5 9:25

"""
import os
import sys 
sys.path.append('/home/aistudio/external-libraries')
import cv2
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import paddle
import paddle.nn.functional as F
from utils.rpn_msr.proposal_layer_tail import proposal_layer
from utils.text_connector.detectors_tail import TextDetector
from models.ctpn_tail import *
from tqdm import tqdm
from PIL import Image
from cal_recall.script import cal_recall_precison_f1


def Add_Padding(image,top, bottom, left, right, color):
    padded_image = cv2.copyMakeBorder(image, top, bottom,
                                      left, right, cv2.BORDER_CONSTANT, value=color)
    return padded_image

def rotate(img, angle):
    w, h = img.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((h / 2, w / 2), angle, 1)
    img_rotation = cv2.warpAffine(img, rotation_matrix, (h, w))
    return img_rotation

def resize_image(img,min_size=1200,color=(0,0,0)):
    
    img_size = img.shape
    im_size_min = np.min(img_size[0:2])
    im_scale = float(min_size) / float(im_size_min)
    new_h = int(img_size[0] * im_scale)
    new_w = int(img_size[1] * im_scale)

    new_h_h = new_h if new_h // 16 == 0 else (new_h // 16 + 1) * 16
    new_w_w = new_w if new_w // 16 == 0 else (new_w // 16 + 1) * 16

    re_im = cv2.resize(img, (new_w_w, new_h_h), interpolation=cv2.INTER_LINEAR)
           
    return re_im, (im_scale*(new_h_h/new_h),im_scale*(new_w_w/new_w))

def toTensorImage(image):
    image =  paddle.to_tensor(image)
    image = image.astype(paddle.float32).transpose((2,0,1)).unsqueeze(0)
    return image


class DetectImg():
    def load_model(self, model_file,detect_type):
        model_dict = paddle.load(model_file)
        model = CTPN_Model()
        model.set_state_dict(model_dict)
        self.model = model
        self.detect_type = detect_type
        self.model.eval()

    def detect(self, img_file,model=None):
        img = Image.open(img_file).convert('RGB')
        img = np.array(img)
        img_ori, (rh, rw) = resize_image(img)
        h, w, c = img_ori.shape
        im_info = np.array([h, w, c]).reshape([1, 3])
        img = toTensorImage(img_ori)
        # img = (img/255-0.5)/0.5
        with paddle.no_grad():
            if model is None:
                pre_score,tail_score, pre_reg = self.model(img)
            else:
                pre_score,tail_score, pre_reg,= model(img)

        pre_score = pre_score.reshape((pre_score.shape[0], 10, 2, pre_score.shape[2], pre_score.shape[3])).squeeze(0).transpose((0, 2, 3, 1)).reshape((-1, 2))
        pre_score = F.softmax(pre_score, 1)
        pre_score = pre_score.reshape((10, pre_reg.shape[2], -1, 2))
        pre_score = pre_score.transpose((1, 2, 0, 3)).reshape([pre_reg.shape[2], pre_reg.shape[3], -1]).unsqueeze(0).cpu().numpy()

        tail_score = tail_score.reshape((tail_score.shape[0], 10, 2, tail_score.shape[2], tail_score.shape[3])).squeeze(0).transpose((0, 2, 3, 1)).reshape((-1, 2))
        tail_score = F.softmax(tail_score, 1)
        tail_score = tail_score.reshape((10, pre_reg.shape[2], -1, 2))
        tail_score = tail_score.transpose((1, 2, 0, 3)).reshape([pre_reg.shape[2], pre_reg.shape[3], -1]).unsqueeze(0).cpu().numpy()

        pre_reg = pre_reg.transpose((0, 2, 3, 1)).cpu().numpy()

        textsegs, _  = proposal_layer(pre_score,tail_score, pre_reg, im_info)
        scores = textsegs[:, 1]
        label_anchors = textsegs[:,0]
        textsegs = textsegs[:, 2:]

        textdetector = TextDetector(DETECT_MODE = self.detect_type)
        boxes, text_proposals,label_anchors = textdetector.detect(textsegs, scores[:, np.newaxis],label_anchors[:, np.newaxis],img_ori.shape[:2])
        boxes = np.array(boxes, dtype=np.int32)
        text_proposals = text_proposals.astype(np.int32)
        return boxes, text_proposals,label_anchors, rh, rw
    
    
def valdetect(img_file,model,detect_type='O'):
    img = Image.open(img_file).convert('RGB')
    img = np.array(img)
    img_ori, (rh, rw) = resize_image(img)
    h, w, c = img_ori.shape
    im_info = np.array([h, w, c]).reshape([1, 3])
    img = toTensorImage(img_ori)
    # img = (img/255-0.5)/0.5
    with paddle.no_grad():
        pre_score,tail_score, pre_reg,= model(img)

    pre_score = pre_score.reshape((pre_score.shape[0], 10, 2, pre_score.shape[2], pre_score.shape[3])).squeeze(0).transpose((0, 2, 3, 1)).reshape((-1, 2))
    pre_score = F.softmax(pre_score, 1)
    pre_score = pre_score.reshape((10, pre_reg.shape[2], -1, 2))
    pre_score = pre_score.transpose((1, 2, 0, 3)).reshape([pre_reg.shape[2], pre_reg.shape[3], -1]).unsqueeze(0).cpu().numpy()

    tail_score = tail_score.reshape((tail_score.shape[0], 10, 2, tail_score.shape[2], tail_score.shape[3])).squeeze(0).transpose((0, 2, 3, 1)).reshape((-1, 2))
    tail_score = F.softmax(tail_score, 1)
    tail_score = tail_score.reshape((10, pre_reg.shape[2], -1, 2))
    tail_score = tail_score.transpose((1, 2, 0, 3)).reshape([pre_reg.shape[2], pre_reg.shape[3], -1]).unsqueeze(0).cpu().numpy()

    pre_reg = pre_reg.transpose((0, 2, 3, 1)).cpu().numpy()

    textsegs, _  = proposal_layer(pre_score,tail_score, pre_reg, im_info)
    scores = textsegs[:, 1]
    label_anchors = textsegs[:,0]
    textsegs = textsegs[:, 2:]

    textdetector = TextDetector(DETECT_MODE = detect_type)
    boxes, text_proposals,label_anchors = textdetector.detect(textsegs, scores[:, np.newaxis],label_anchors[:, np.newaxis],img_ori.shape[:2])
    boxes = np.array(boxes, dtype=np.int32)
    text_proposals = text_proposals.astype(np.int32)
    return boxes, text_proposals,label_anchors, rh, rw


def val(model,dir_path='/home/aistudio/work/icdar/aistudio/work/data/icdar/test_img'
,gt_path='/home/aistudio/work/icdar/aistudio/work/data/icdar/test_gt'):
    img_save_path = './result'
    txt_save_path = './pre_gt'
    files = os.listdir(dir_path)
    bar = tqdm(total=len(files))
    for file in files:
        bar.update(1)
        fid = open(os.path.join(txt_save_path, 'res_' + file.split('.')[0] + '.txt'), 'w+', encoding='utf-8')
        im_file = os.path.join(dir_path,file)
        boxes, text_proposals,label_anchors, rh, rw = valdetect(im_file,model)
        for i, box in enumerate(boxes):
            box = box[:8].reshape(4, 2)
            box[:, 0] = box[:, 0] / rw
            box[:, 1] = box[:, 1] / rh
            box = box.reshape(1, 8).astype(np.int32)
            box = [str(x) for x in box.reshape(-1).tolist()]
            fid.write(','.join(box) + '\n')
        fid.close()
        show_img(img_save_path, im_file, boxes, text_proposals,label_anchors)
    bar.close()
    val_result = cal_recall_precison_f1(gt_path,'./pre_gt/', show_result=False)
    return val_result


def show_img(save_path, im_file, boxes, text_proposals,label_anchors):
    img_ori = cv2.imread(im_file)
    img_ori, (rh, rw) = resize_image(img_ori)
    im_name = im_file.split('/')[-1].split('.')[0]
    tag = 0
    for item in text_proposals:
        color =  (0,255,255)
        if(label_anchors[tag]==0):
            color =  (255,255,0)
        img_ori = cv2.rectangle(img_ori, (item[0], item[1]), (item[2], item[3]), color)
        tag+=1

    img_ori = cv2.resize(img_ori, None, None, fx=1.0 / rw, fy=1.0 / rh, interpolation=cv2.INTER_LINEAR)
    for i, box in enumerate(boxes):
        color = (np.random.randint(0, 255),np.random.randint(0, 255),np.random.randint(0, 255))
        cv2.polylines(img_ori, [box[:8].astype(np.int32).reshape((-1, 1, 2))], True, color=color,
                      thickness =2)
    img_ori = cv2.resize(img_ori, None, None, fx=0.9, fy=0.9, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(os.path.join(save_path, im_name + '.jpg'), img_ori)


if __name__ == "__main__":


    dir_path = '/home/aistudio/work/icdar/aistudio/work/data/icdar/test_img'
    val_gt_path = '/home/aistudio/work/icdar/aistudio/work/data/icdar/test_gt'
    fidlog = open('testlog.txt','w+',encoding='utf-8')

    for j in range(1,50):
        model_file = './model_save/ctpn_'+str(j)+'.pdparams'
        img_save_path = './result'
        txt_save_path = './pre_gt'
        detect_type = 'O' # 'O' or 'H'
        
        detect_obj = DetectImg()
        detect_obj.load_model(model_file,detect_type)
        
        files = os.listdir(dir_path)
        bar = tqdm(total=len(files))
        for file in files:
            bar.update(1)
            fid = open(os.path.join(txt_save_path, 'res_' + file.split('.')[0] + '.txt'), 'w+', encoding='utf-8')
            im_file = os.path.join(dir_path,file)
            boxes, text_proposals, rh, rw = detect_obj.detect(im_file)
            for i, box in enumerate(boxes):
                box = box[:8].reshape(4, 2)
                box[:, 0] = box[:, 0] / rw
                box[:, 1] = box[:, 1] / rh
                box = box.reshape(1, 8).astype(np.int32)
                box = [str(x) for x in box.reshape(-1).tolist()]
                fid.write(','.join(box) + '\n')
            fid.close()
            show_img(img_save_path, im_file, boxes, text_proposals)
        bar.close()
        val_result = cal_recall_precison_f1(val_gt_path,'./pre_gt/', show_result=False)
        save_str = 'recall:'+str(val_result['recall'])+'\t'+'precision:'+str(val_result['precision'])+'\t'+'hmean:'+str(val_result['hmean'])
        fidlog.write('epoch'+str(j)+':'+'\t'+save_str+'\n')
        print(val_result)
    fidlog.close()   



