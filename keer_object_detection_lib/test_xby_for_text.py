import argparse
import os.path as osp
import shutil
import tempfile

import mmcv
import torch
import torch.distributed as dist
from mmcv.runner import load_checkpoint, get_dist_info
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel

from mmdet.apis import init_dist
from mmdet.core import results2json, coco_eval
from mmdet.datasets import build_dataloader, get_dataset
from mmdet.models import build_detector
import time
import sys
sys.path.append('./sl_eval/')
from sl_metric import evaluate_results
from ssd_metric import fscore
import json
import numpy as np

def get_time_str():
    return time.strftime('%Y%m%d_%H%M%S', time.localtime())

def single_gpu_test(model, data_loader, show=False, log_dir=None):
    model.eval()
    results = []
    dataset = data_loader.dataset
    if log_dir != None:
        filename = 'inference{}.log'.format(get_time_str())
        log_file = osp.join(log_dir, filename)
        f = open(log_file, 'w')
        prog_bar = mmcv.ProgressBar(len(dataset), file=f)
    else:
        prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=not show, **data)
            for idx,class_iter in enumerate(result):
                # print(result)
                if len(class_iter)>1:
                    result[idx]=np.array(sorted(list(class_iter),key=lambda y:y[-1])[-1][:8])
                    result[idx]=result[idx].reshape(1,8)
                    # print('shape?',result[idx].shape)
                else:
                    if class_iter.shape==(0,9):
                        result[idx]=np.zeros((0,8))    
                    else:
                        result[idx]=result[idx][0,:8].reshape(1,8)
                    
            # result=[item.reshape(1,8) for item in result]
#            print(str(result) + '\n')
        results.append(result)
        # print(results)
        if show:
            model.module.show_result(data, result, dataset.img_norm_cfg)

        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()

    return results








def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--log_dir', help='log the inference speed')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()



    cfg = mmcv.Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)



    def TEST(data_loader,json_path,specific):
        model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
        checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
        # old versions did not save class info in checkpoints, this walkaround is
        # for backward compatibility
        print('checkpoint meta',checkpoint['meta']['CLASSES'])
        if 'CLASSES' in checkpoint['meta']:
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            model.CLASSES = dataset.CLASSES
        # if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader, args.show, args.log_dir)
        
        # 获得inference

        infer_boxes=[output[0] for output in outputs]
        # print('看看outputs啥情况',infer_boxes[0][0].shape)

        
        # infer_inner_boxes=np.asarray(infer_inner_boxes)
        # infer_outer_boxes=np.asarray(infer_outer_boxes)

        gt_boxes=[]        
        with open(json_path,'r') as f:
            data=json.load(f)
        annotations=data['annotations']

        for anno in annotations: 
            
            points=np.array(anno['segmentation'][0])/512

            gt_boxes.append(points.reshape(1,-1))
        

        # gt_inner_boxes=np.asarray(gt_inner_boxes)
        # gt_outer_boxes=np.asarray(gt_outer_boxes)
        print('gt_boxes len',len(gt_boxes))
        print('gt_boxes [0]',gt_boxes[0])
        print('infer_boxes len',len(infer_boxes))
        print('infer_boxes [0]',infer_boxes[0])
        # print("gt_inner_boxes的shape",gt_inner_boxes.shape)
        TP,FP,FN=evaluate_results(gt_boxes,infer_boxes,(512,512))
        recall=TP/(TP+FN)
        precision=TP/(TP+FP)
        f1=fscore(precision,recall)
        print('####################  {}集情况  ####################'.format('测试' if specific=='test' else '训练'))
        print('\n##########  Inner检测框  ##########\n:precision:{}\nrecall:{}\nf1:{}'.format(precision,recall,f1))
        
        return TP,FP,FN

    #####################  训练集和验证集统一测试  ######################
    # test_json_path='/home/jackhzhou/dataset/huxifa_od_coco_aug/annotations/instances_val2017.json'
    test_json_path='/home/jackhzhou/dataset/newtext_coco0308/annotations/instances_val2017.json'
    dataset = get_dataset(cfg.data.test)
    test_dataloader = build_dataloader(
        dataset,
        imgs_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # train_json_path='/home/jackhzhou/dataset/huxifa_od_coco_aug/annotations/instances_train2017.json'
    train_json_path='/home/jackhzhou/dataset/newtext_coco0308/annotations/instances_train2017.json'
    
    TP_te,FP_te,FN_te=TEST(test_dataloader,test_json_path,'test')

    dataset = get_dataset(cfg.data.test_tr)
    train_dataloader = build_dataloader(
        dataset,
        imgs_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    TP_tr,FP_tr,FN_tr=TEST(train_dataloader,train_json_path,'train')

    print('####################  训练集测试集综合情况  ####################')
    TP=TP_te+TP_tr
    FP=FP_te+FP_tr
    FN=FN_te+FN_tr
    recall_in=TP/(TP+FN)
    precision_in=TP/(TP+FP)
    f1_in=fscore(precision_in,recall_in)
    print('\n##########  3m检测框  ##########\n:precision:{}\nrecall:{}\nf1:{}'.format(precision_in,recall_in,f1_in))



    
    ######  对测试集进行测试  ######
        


        


if __name__ == '__main__':
    main()
