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


def multi_gpu_test(model, data_loader, tmpdir=None):
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        results.append(result)

        if rank == 0:
            batch_size = data['img'][0].size(0)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    results = collect_results(results, len(dataset), tmpdir)

    return results


def collect_results(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            tmpdir = tempfile.mkdtemp()
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, 'part_{}.pkl'.format(rank)))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, 'part_{}.pkl'.format(i))
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


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
        infer_inner_boxes=[]
        infer_outer_boxes=[]
        for output in outputs:
            # 只取前8维
            # print('len(output)',len(output))
            assert len(output)==2
            infer_inner_boxes.append(output[0])
            infer_outer_boxes.append(output[1])

        
        # infer_inner_boxes=np.asarray(infer_inner_boxes)
        # infer_outer_boxes=np.asarray(infer_outer_boxes)

        gt_inner_boxes=[]
        gt_outer_boxes=[]
        # 获得groundtruth
        
        with open(json_path,'r') as f:
            data=json.load(f)
        annotations=data['annotations']

        gt_inner_boxes=[]
        gt_outer_boxes=[]
        for anno in annotations: 
            if anno['category_id']==1:
                inner_points=np.array(anno['segmentation'][0])/512
                # if len(inner_points)!=8:
                #     print('inner points长啥样',inner_points)
                gt_inner_boxes.append(inner_points.reshape(1,-1))
            if anno['category_id']==2:
                outer_points=np.array(anno['segmentation'][0])/512
                gt_outer_boxes.append(outer_points.reshape(1,-1))

        # gt_inner_boxes=np.asarray(gt_inner_boxes)
        # gt_outer_boxes=np.asarray(gt_outer_boxes)

        # print("gt_inner_boxes的shape",gt_inner_boxes.shape)
        TP,FP,FN=evaluate_results(gt_inner_boxes,infer_inner_boxes,(512,512))
        recall=TP/(TP+FN)
        precision=TP/(TP+FP)
        f1=fscore(precision,recall)
        print('####################  {}集情况  ####################'.format('测试' if specific=='test' else '训练'))
        print('\n##########  Inner检测框  ##########\n:precision:{}\nrecall:{}\nf1:{}'.format(precision,recall,f1))
        TP_1,FP_1,FN_1=evaluate_results(gt_outer_boxes,infer_outer_boxes,(512,512))
        recall=TP_1/(TP_1+FN_1)
        precision=TP_1/(TP_1+FP_1)
        f1=fscore(precision,recall)
        print('\n##########  Outer检测框  ##########\n:precision:{}\nrecall:{}\nf1:{}'.format(precision,recall,f1))

        return TP,TP_1,FP,FP_1,FN,FN_1

    ######################  训练集和验证集统一测试  ######################
    # test_json_path='/home/jackhzhou/dataset/huxifa_od_coco_aug/annotations/instances_val2017.json'
    test_json_path='/home/jackhzhou/dataset/huxifa_text_coco0308/annotations/instances_val2017.json'
    dataset = get_dataset(cfg.data.test)
    test_dataloader = build_dataloader(
        dataset,
        imgs_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # train_json_path='/home/jackhzhou/dataset/huxifa_od_coco_aug/annotations/instances_train2017.json'
    train_json_path='/home/jackhzhou/dataset/huxifa_text_coco0308/annotations/instances_train2017.json'
    
    TP_te,TP1_te,FP_te,FP1_te,FN_te,FN1_te=TEST(test_dataloader,test_json_path,'test')

    dataset = get_dataset(cfg.data.test_tr)
    train_dataloader = build_dataloader(
        dataset,
        imgs_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    TP_tr,TP1_tr,FP_tr,FP1_tr,FN_tr,FN1_tr=TEST(train_dataloader,train_json_path,'train')

    print('####################  训练集测试集综合情况  ####################')
    TP=TP_te+TP_tr
    FP=FP_te+FP_tr
    FN=FN_te+FN_tr
    recall_in=TP/(TP+FN)
    precision_in=TP/(TP+FP)
    f1_in=fscore(precision_in,recall_in)
    print('\n##########  Inner检测框  ##########\n:precision:{}\nrecall:{}\nf1:{}'.format(precision_in,recall_in,f1_in))

    TP1=TP1_te+TP1_tr
    FP1=FP1_te+FP1_tr
    FN1=FN1_te+FN1_tr
    recall_out=TP1/(TP1+FN1)
    precision_out=TP1/(TP1+FP1)
    f1_out=fscore(precision_out,recall_out)
    print('\n##########  Outer检测框  ##########\n:precision:{}\nrecall:{}\nf1:{}'.format(precision_out,recall_out,f1_out))

    print('\n####################  InnerOuter平均情况  ####################\n:precision:{}\nrecall:{}\nf1:{}'.format((precision_out+precision_in)/2,(recall_out+recall_in)/2,(f1_out+f1_in)/2))


    
    ######  对测试集进行测试  ######
        


        


if __name__ == '__main__':
    main()
