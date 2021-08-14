"""
#!-*- coding=utf-8 -*-
@author: BADBADBADBADBOY
@contact: 2441124901@qq.com
@software: PyCharm Community Edition
@file: loss.py
@time: 2020/4/5 9:28

"""


import paddle
import paddle.nn.functional as F
import paddle.nn as nn
import numpy as np
import math
import itertools

def get_select(ori_data,index,_shape=1):
    if _shape==1:
        return paddle.masked_select(ori_data,index)
    else:
        new_data = paddle.masked_select(ori_data, paddle.tile(paddle.reshape(index, [-1, 1]), [1, _shape]))
        if new_data.shape[0] == 0:
            return new_data
        else:
            return paddle.reshape(new_data,[-1,_shape])


def smooth_l1_loss(inputs,targets,beta=1. / 9,size_average=True):
    """
    https://github.com/facebookresearch/maskrcnn-benchmark
    """
    diff = paddle.abs(inputs - targets)
    loss = paddle.where(
        diff < beta,
        0.5 * diff ** 2 / beta,
        diff - 0.5 * beta
    )
    if size_average:
        return loss.mean()
    return loss.sum()

def hard_negative_mining(loss, labels, neg_pos_ratio):
    pos_mask = labels > 0
    num_pos = pos_mask.astype(paddle.int32).sum()
    num_neg = num_pos * neg_pos_ratio
    loss[pos_mask] = -math.inf
    loss_sort = loss.sort(axis=0, descending=True)
    neg_mask = loss > loss_sort[num_neg]
    return paddle.logical_or(pos_mask ,neg_mask)

def cal_class_loss_ohem(confidence,labels):
    loss_cls = F.cross_entropy(confidence, labels.astype(paddle.int64),reduction='none')
    pos_mask = labels > 0
    num_pos = pos_mask.astype(paddle.int32).sum()
    if num_pos < 1:
        return paddle.to_tensor(0.0)
    loss_pos = get_select(loss_cls, pos_mask)
    neg_mask = labels == 0
    num_neg = neg_mask.astype(paddle.int32).sum()
    if num_neg < 1:
        return paddle.to_tensor(0.0)
    loss_neg, _ = get_select(loss_cls, neg_mask).topk(min(3 * num_pos,num_neg))
    loss_cls = (loss_pos.sum() + loss_neg.sum()) / (num_pos+min(3 * num_pos,num_neg)) 
    return loss_cls

def cal_class_loss_hard(confidence,labels,predicted_locations,gt_locations,neg_pos_ratio=3):
    loss = -F.log_softmax(confidence, axis=1)[:, 0]
    loss.stop_gradient=True
    mask = hard_negative_mining(loss, labels, neg_pos_ratio)
    if mask.astype(paddle.int32).sum()<1:
        mask = labels>=0
    labels = get_select(labels, mask)
    confidence = get_select(confidence, mask,2)
    loss_cls = F.cross_entropy(confidence,labels.astype(paddle.int64),reduction='mean') if labels.numel() > 0 else paddle.to_tensor(0.0)

    predicted_locations = get_select(predicted_locations, mask, 4)
    gt_locations = get_select(gt_locations, mask, 4)
    return loss_cls,labels,predicted_locations,gt_locations


class CTPNLoss(nn.Layer):
    def __init__(self, neg_pos_ratio=3):
        """Implement SSD Multibox Loss.
        Basically, Multibox loss combines classification loss
         and Smooth L1 regression loss.
        """
        super(CTPNLoss, self).__init__()
        self.neg_pos_ratio = neg_pos_ratio

    def forward(self, confidence,confidence_tail, predicted_locations, labels, gt_locations):
        # cls_out, reg_out, anchor_gt_labels, anchor_gt_locations

        """
        Compute classification loss and smooth l1 loss.
        Args:
            confidence (batch_size, num_anchors, num_classes): class predictions.
            locations (batch_size, num_anchors, 4*seq_len): predicted locations.
            labels (batch_size, num_anchors): real labels of all the anchors.
            boxes (batch_size, num_anchors, 4*seq_len): real boxes corresponding all the anchors.


        """
        # import pdb
        # pdb.set_trace()

        batch_size = confidence.shape[0]
        confidence = confidence.reshape((batch_size,-1,2))
        confidence_tail = confidence_tail.reshape((batch_size,-1,2))
        predicted_locations = predicted_locations.reshape((batch_size,-1,4))
        labels = labels.reshape((batch_size,-1))
        gt_locations = gt_locations.reshape((batch_size,-1,4))

        #################################################

        #################################################
        labels = labels.reshape([-1])
        confidence = confidence.reshape([labels.shape[0],-1])
        confidence_tail = confidence_tail.reshape([labels.shape[0],-1])
        predicted_locations = predicted_locations.reshape([labels.shape[0], -1])
        gt_locations = gt_locations.reshape([labels.shape[0], -1])

        label_tail = labels.clone()
        labels[labels>0]=1

        mask = labels>=0
        labels = get_select(labels,mask)
        confidence = get_select(confidence,mask,2)
        predicted_locations = get_select(predicted_locations,mask,4)
        gt_locations = get_select(gt_locations,mask,4)

        if labels.shape[0]==0:
            return paddle.to_tensor(0.),paddle.to_tensor(0.),paddle.to_tensor(0.),paddle.to_tensor(0.)

        ### cal cl loss
        loss_cls = cal_class_loss_ohem(confidence,labels)
        # loss_cls,labels,predicted_locations,gt_locations = cal_class_loss_hard(confidence,labels,predicted_locations,gt_locations)


        head_tail_mask = label_tail > 0
        label_tail = get_select(label_tail, head_tail_mask) - 1
        confidence_tail = get_select(confidence_tail, head_tail_mask,2)
        loss_cls_tail = F.cross_entropy(confidence_tail,label_tail.astype(paddle.int64),reduction='mean') if label_tail.numel() > 0 else paddle.to_tensor(0.0)


        ### cal location loss
        if labels.max()<1:
             return loss_cls,paddle.to_tensor(0.),paddle.to_tensor(0.),paddle.to_tensor(0.)

        mask = labels > 0
        predicted_locations = get_select(predicted_locations, mask, 4)
        gt_locations = get_select(gt_locations, mask, 4)
        
        gt_locations_ver = paddle.concat((gt_locations[:, 1].unsqueeze(1), gt_locations[:, 3].unsqueeze(1)), 1)
        predicted_locations_ver = paddle.concat((predicted_locations[:, 1].unsqueeze(1), predicted_locations[:, 3].unsqueeze(1)), 1)

        # gt_locations_ref = paddle.concat((gt_locations[:, 0].unsqueeze(1), gt_locations[:, 2].unsqueeze(1)), 1)
        # predicted_locations_ref = paddle.concat((predicted_locations[:, 0].unsqueeze(1), predicted_locations[:, 2].unsqueeze(1)), 1)
        gt_locations_ref = gt_locations[:, 0]
        predicted_locations_ref = predicted_locations[:, 0]
        
        loss_ver = smooth_l1_loss(predicted_locations_ver, gt_locations_ver) if gt_locations_ver.numel() > 0 else paddle.to_tensor(0.0)
        loss_refine = smooth_l1_loss(predicted_locations_ref, gt_locations_ref) if gt_locations_ref.numel() > 0 else paddle.to_tensor(0.0)

        return loss_cls,loss_cls_tail, loss_ver, loss_refine
