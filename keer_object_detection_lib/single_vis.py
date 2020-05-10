from mmdet.apis import init_detector, inference_detector, show_result, draw_poly_detections
import mmcv
from mmcv import Config
from mmdet.datasets import get_dataset
import cv2
import os
import numpy as np
from tqdm import tqdm
import DOTA_devkit.polyiou as polyiou
import math
import pdb
import glob
import sys

import os
import os.path as osp
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# sys.path.append('../mask_filter_service/')
# import detect_3m 

def py_cpu_nms_poly_fast_np(dets, thresh):
    obbs = dets[:, 0:-1]
    x1 = np.min(obbs[:, 0::2], axis=1)
    y1 = np.min(obbs[:, 1::2], axis=1)
    x2 = np.max(obbs[:, 0::2], axis=1)
    y2 = np.max(obbs[:, 1::2], axis=1)
    scores = dets[:, 8]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    polys = []
    for i in range(len(dets)):
        tm_polygon = polyiou.VectorDouble([dets[i][0], dets[i][1],
                                            dets[i][2], dets[i][3],
                                            dets[i][4], dets[i][5],
                                            dets[i][6], dets[i][7]])
        polys.append(tm_polygon)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        ovr = []
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        hbb_inter = w * h
        hbb_ovr = hbb_inter / (areas[i] + areas[order[1:]] - hbb_inter)
        h_inds = np.where(hbb_ovr > 0)[0]
        tmp_order = order[h_inds + 1]
        for j in range(tmp_order.size):
            iou = polyiou.iou_poly(polys[i], polys[tmp_order[j]])
            hbb_ovr[h_inds[j]] = iou

        try:
            if math.isnan(ovr[0]):
                pdb.set_trace()
        except:
            pass
        inds = np.where(hbb_ovr <= thresh)[0]
        order = order[inds + 1]
    return keep

class DetectorModel():
    def __init__(self,
                 config_file,
                 checkpoint_file):
        # init RoITransformer
        self.config_file = config_file
        self.checkpoint_file = checkpoint_file
        self.cfg = Config.fromfile(self.config_file)
        self.data_test = self.cfg.data['test']
        self.dataset = get_dataset(self.data_test)
        self.classnames = self.dataset.CLASSES
        self.model = init_detector(config_file, checkpoint_file, device='cuda:0')

    def inference_single(self, img):
        # img = mmcv.imread(imgpath)
        img=cv2.resize(img,(512,512))
        height, width, channel = img.shape
        # total_detections = [np.zeros((0, 9)) for _ in range(len(self.classnames))]
        # print(img)
        detections = inference_detector(self.model, img)
        # nms
        # print(type(self.classnames))
        # 注意如果只有一个类别的时候需要自己包装为list
        if isinstance(self.classnames,str):
            self.classnames=[self.classnames]
        print(self.classnames)
        for i in range(len(self.classnames)):
            
            # print('此时i',i)
            keep = py_cpu_nms_poly_fast_np(detections[i], 0.1)
            
            detections[i] = detections[i][keep]
        return detections


    def inference_single_vis(self, srcpath, dstpath):
        img = mmcv.imread(srcpath)

        detections = self.inference_single(img)
        img=cv2.resize(img,(512,512))
        print('{}检测到{}个'.format(os.path.basename(srcpath),len(detections)))
        print(detections)
        # print('detections是什么',len(detections))
        img = draw_poly_detections(img, detections, self.classnames, scale=1, threshold=0.3)
        cv2.imwrite(dstpath, img)


if __name__ == '__main__':
    # 呼吸阀内框和外框的检测
    # roitransformer = DetectorModel(r'configs/HUXIFA_OD/faster_rcnn_RoITrans_r50_fpn_1x_huxifa_od_aug.py',
    #               r'work_dirs/faster_rcnn_RoITrans_r50_fpn_1x_huxifa_od_aug/epoch_240.pth')
    # 3m文字部分检测 
    # roitransformer = DetectorModel(r'configs/HUXIFA_OD/faster_rcnn_RoITrans_r50_fpn_1x_huxifa_text.py',
                   # r'work_dirs/faster_rcnn_RoITrans_r50_fpn_1x_huxifa_text/epoch_90.pth')
    # gucci目标检测
    roitransformer = DetectorModel(r'configs/GUCCI_OD/gucci_od.py',
                   r'work_dirs/gucci_od/epoch_100.pth')

    # img_path='../dataset/huxifa_od_coco/images/train2017/'
    # img_glob=glob.glob(img_path+'*')
    # img_trainval=glob.glob('../dataset/huxifa_od_coco/images/train2017/*')
    # img_trainval.extend(glob.glob('../dataset/huxifa_od_coco/images/val2017/*'))

    # 呼吸阀图片
    # img_trainval=glob.glob('../dataset/huxifa_od_coco/images/val2017/*')
    large_path='../dataset/gucci pinka test/LARGE/'
    small_path='../dataset/gucci pinka test/SMALL/'

    
    def get_imgpaths(source_path):
        img_paths=[]
        for root, _, names in os.walk(source_path):
            for name in names:
                img_paths.append(osp.join(root,name))
        return img_paths
    img_trainval=get_imgpaths(large_path)
    img_trainval.extend(get_imgpaths(small_path))
    # img_trainval_strip=[os.path.basename(img) for img in img_trainval]
    # with open('hard_examples.txt','r') as f:
    #     hard_examples=[line.strip('\n') for line in f.readlines()]
    # for exp in hard_examples:
    #     idx=img_trainval_strip.index(exp)
    #     img_name_out='Detected'+exp
    #     roitransformer.inference_single_vis(img_trainval[idx],os.path.join('demo2206',img_name_out))
                    
    
    # roitransformer.inference_single_vis('/home/jackhzhou/dataset/huxifa_od_coco_aug/images/train2017/weixin_20200201160514.jpg','./configs/test.jpg')

    for img_name in img_trainval:
        print('Detecting{}'.format(img_name))
        img_name_shrt=os.path.basename(img_name)
        img_name_out='Detected'+img_name_shrt
        roitransformer.inference_single_vis(img_name,os.path.join('ftresult/gucci0324',img_name_out))
    
    # roitransformer.inference_single_vis('../dataset/huxifa_od_coco/images/train2017/weixin_20200201160550.jpg','detected_weixin_20200201160550.jpg')