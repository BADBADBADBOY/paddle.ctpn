"""
#!-*- coding=utf-8 -*-
@author: BADBADBADBADBOY
@contact: 2441124901@qq.com
@software: PyCharm Community Edition
@file: train.py
@time: 2020/4/5 9:24

"""
import os
import argparse
import sys 
sys.path.append('/home/aistudio/external-libraries')
import warnings
warnings.filterwarnings('ignore')
import paddle.optimizer as optim
import paddle
from paddle.io import DataLoader
import numpy as np
from dataLoader.dataLoad_tail import IC15Loader,get_bboxes
from models.loss_tail import CTPNLoss
from models.ctpn_tail import CTPN_Model
from utils.rpn_msr.anchor_target_layer_tail import anchor_target_layer
from tools.Log import Logger
from inference_tail import val

random_seed = 2020
np.random.seed(random_seed) 


def DrawLoss(loss_data,epoch_list):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _key = []
    _key_bin = []
    plt.xlabel('epoch')
    plt.ylabel('loss')
    for key in loss_data.keys():
       n, = plt.plot(epoch_list,loss_data[key])
       _key.append(key)
       _key_bin.append(n)
    plt.legend(_key_bin,_key)
    plt.savefig('loss.png')

def toTensor(item):
    item = paddle.to_tensor(item)
    return item

def main(args):
    log_write = Logger('./log.txt', 'LogFile')
    log_write.set_names(['Total loss', 'Cls loss','cls_tail loss','Y_loc_loss','X_Ref_loss','Lr'])
    data_loader = IC15Loader(args.size_list)
    gt_files = data_loader.gt_paths
    train_loader = DataLoader(
                data_loader,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_worker,
                drop_last=True)

    model = CTPN_Model()
    critetion = CTPNLoss()
    
    scheduler = optim.lr.StepDecay(args.lr, step_size=args.step_size, gamma=args.gamma)
    if(args.optimizer=='SGD'):
        optimizer = optim.SGD(parameters=model.parameters(), learning_rate=scheduler, weight_decay=5e-4)
    elif(args.optimizer=='Momentum'):
        optimizer = optim.Momentum(learning_rate=scheduler, momentum=0.99, parameters=model.parameters(), use_nesterov=False, weight_decay=5e-4)
    elif(args.optimizer=='RMSProp'):
        optimizer = optim.RMSProp(learning_rate = scheduler,parameters = model.parameters())
    else:
        optimizer = optim.Adam(parameters=model.parameters(), learning_rate=scheduler, weight_decay=5e-4)

    start_epoch = 0
    if args.restore is True:
        model.set_state_dict(paddle.load(os.path.join(args.checkpoint, 'ctpn_' + str(args.restore_epoch) + '.pdparams')))
        optimizer.set_state_dict(paddle.load(os.path.join(args.checkpoint,'optimizer.pdparams')))
        start_epoch = args.restore_epoch
        print('restore to train model !!!!!')

    log_epoch = []
    log_loss = {}
    log_loss['cls'] = []
    log_loss['cls_tail'] = []
    log_loss['loc_y'] = []
    log_loss['loc_x'] = []
    log_loss['total'] = []
    best_hmean = 0
    for epoch in range(start_epoch,args.train_epochs):
        model.train()
        loss_total_list = []
        loss_cls_list = []
        loss_cls_tail_list = []
        loss_ver_list = []
        loss_refine_list = []

        for batch_idx, (imgs, img_scales,im_shapes, gt_path_indexs,im_infos) in enumerate(train_loader):
            data_loader.get_random_train_size()
            image = toTensor(imgs)
            score_pre, score_tail_pre,vertical_pred = model(image)

            score_pre = score_pre.transpose((0, 2, 3, 1))
            score_tail_pre = score_tail_pre.transpose((0, 2, 3, 1))
            vertical_pred = vertical_pred.transpose((0, 2, 3, 1))

            batch_res_polys = get_bboxes(imgs,gt_files, gt_path_indexs, img_scales,im_shapes)

            batch_loss_cls = []
            batch_loss_cls_tail = []
            batch_loss_ver = []
            batch_loss_refine = []

            for i in range(image.shape[0]):
                image_ori =  (imgs[i].numpy()).transpose((1,2,0)).copy()
                gt_boxes = np.array(batch_res_polys[i])
                if gt_boxes.shape[0]==0:
                    continue
                rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = anchor_target_layer(
                    image_ori, score_pre[i].cpu().unsqueeze(0), gt_boxes, im_infos[i].numpy())

                rpn_labels = toTensor(rpn_labels)
                rpn_bbox_targets = toTensor(rpn_bbox_targets)

                loss_cls, loss_cls_tail,loss_ver, loss_refine = critetion(score_pre[i].unsqueeze(0), score_tail_pre[i].unsqueeze(0), vertical_pred[i].unsqueeze(0),rpn_labels, rpn_bbox_targets)

                batch_loss_cls.append(loss_cls)
                batch_loss_cls_tail.append(loss_cls_tail)
                batch_loss_ver.append(loss_ver)
                batch_loss_refine.append(loss_refine)


                del(loss_cls)
                del(loss_ver)
                del(loss_refine)

            loss_cls = sum(batch_loss_cls)/len(batch_loss_cls)
            loss_cls_tail = sum(batch_loss_cls_tail)/len(batch_loss_cls_tail)
            loss_ver = sum(batch_loss_ver)/len(batch_loss_ver)
            loss_refine = sum(batch_loss_refine)/len(batch_loss_refine)

            loss_tatal = loss_cls + loss_cls_tail + loss_ver + 2*loss_refine 
            # import pdb
            # pdb.set_trace()
            loss_tatal.backward()

            optimizer.step()
            optimizer.clear_grad()

            loss_total_list.append(loss_tatal.item())
            loss_cls_list.append(loss_cls.item())
            loss_cls_tail_list.append(loss_cls_tail.item())
            loss_ver_list.append(loss_ver.item())
            loss_refine_list.append(loss_refine.item())

            if (batch_idx % args.show_step == 0):
                log = '({epoch}/{epochs}/{batch_i}/{all_batch}) | loss_tatal: {loss1:.4f} | loss_cls: {loss2:.4f} | loss_cls_tail: {loss3:.4f} | loss_ver: {loss4:.4f} | loss_refine: {loss5:.4f}  | Lr: {lr}'.format(
                    epoch=epoch, epochs=args.train_epochs, batch_i=batch_idx, all_batch=len(train_loader), loss1=loss_tatal.item(),
                    loss2=loss_cls.item(),loss3=loss_cls_tail.item(), loss4=loss_ver.item(), loss5=loss_refine.item(),  lr=scheduler.get_lr())
                print(log)
                log_write.append([loss_tatal.item(),loss_cls.item(),loss_cls_tail.item(),loss_ver.item(),loss_refine.item(),scheduler.get_lr()])
        scheduler.step()
        # eval
        if epoch%args.start_val==0:
            model.eval()
            val_result = val(model,args.val_dir,args.val_gt_path)
            if val_result['hmean']>best_hmean:
                best_hmean = val_result['hmean']
                paddle.save(model.state_dict(),os.path.join(args.checkpoint, 'ctpn_best_model.pdparams'))
        print('--------------------------------------------------------------------------------------------------------')
        log_write.set_split(['---------','----------','--------','----------','--------','--------'])
        print(
            "epoch_loss_total:{loss1:.4f} | epoch_loss_cls:{loss2:.4f} | epoch_loss_cls_tail:{loss3:.4f} | epoch_loss_ver:{loss4:.4f} | epoch_loss_ref:{loss5:.4f} | Lr:{lr}".
            format(loss1=np.mean(loss_total_list), loss2=np.mean(loss_cls_list),loss3=np.mean(loss_cls_tail_list), loss4=np.mean(loss_ver_list),
                   loss5=np.mean(loss_refine_list), lr=scheduler.get_lr()))
        log_write.append([np.mean(loss_total_list),np.mean(loss_cls_list),np.mean(loss_cls_tail_list),np.mean(loss_ver_list),np.mean(loss_refine_list),scheduler.get_lr()])
        print('recall:'+str(val_result['recall']),'precision:'+str(val_result['precision']),'hmean:'+str(val_result['hmean']))
        print('-------------------------------------------------------------------------------------------------------')
        log_write.set_split(['val result:','-----','----->','recall:'+str(val_result['recall'])+'\t','precision:'+str(val_result['precision'])+'\t','hmean:'+str(val_result['hmean'])])
        log_write.set_split(['---------','----------','--------','----------','--------','--------'])
        if(epoch % args.epoch_save==0 and epoch!=0):
            paddle.save(model.state_dict(),os.path.join(args.checkpoint, 'ctpn_' + str(epoch) + '.pdparams'))
            paddle.save(optimizer.state_dict(),os.path.join(args.checkpoint, 'optimizer.pdparams'))
        
        
        scheduler.step()
        log_epoch.append(epoch)
        log_loss['cls'].append(np.mean(loss_cls_list))
        log_loss['cls_tail'].append(np.mean(loss_cls_tail_list))
        log_loss['loc_y'].append(np.mean(loss_ver_list))
        log_loss['loc_x'].append(np.mean(loss_refine_list))
        log_loss['total'].append(np.mean(loss_total_list))
        DrawLoss(log_loss,log_epoch)
    log_write.close()

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--optimizer', nargs='?', type=str, default='SGD')
    parser.add_argument('--val_dir', nargs='?', type=str, default='/home/aistudio/work/icdar/aistudio/work/data/icdar/test_img')
    parser.add_argument('--val_gt_path', nargs='?', type=str, default='/home/aistudio/work/icdar/aistudio/work/data/icdar/test_gt')
    parser.add_argument('--batch_size', nargs='?', type=int, default=10, help='Batch Size') 
    parser.add_argument('--restore', nargs='?', type=bool, default=False, help='restore train')
    parser.add_argument('--restore_epoch', nargs='?', type=int, default=18, help='restore train epoch')
    parser.add_argument('--size_list', nargs='?', type=list, default = [1200], help='img max Size when train') #[768,928,1088,1200,1360]
    parser.add_argument('--num_worker', nargs='?', type=int, default=0, help='num_worker to train')
    parser.add_argument('--lr', nargs='?', type=float, default=0.08, help='Learning Rate') 
    parser.add_argument('--step_size', nargs='?', type=int, default=60, help='optimizer step size') 
    parser.add_argument('--gamma', nargs='?', type=float, default=0.1, help='optimizer decay gamma') 
    parser.add_argument('--pretrain', nargs='?', type=bool, default=True, help='If use pre model')
    parser.add_argument('--train_epochs', nargs='?', type=int, default=50, help='how epoch to train')
    parser.add_argument('--start_val', nargs='?', type=int, default=1, help=' epoch to eval')
    parser.add_argument('--show_step', nargs='?', type=int, default=10, help='step to show')
    parser.add_argument('--epoch_save', nargs='?', type=int, default=1, help='how epoch to save')
    parser.add_argument('--checkpoint', default='./model_save', type=str, help='path to save model') 
    args = parser.parse_args()
    main(args)




