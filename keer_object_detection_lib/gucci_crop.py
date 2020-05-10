import sys
import glob
import os
import os.path as osp
import matplotlib.pyplot as plt
import cv2
import numpy as np
from crnn_data import crop_words
from single_vis import DetectorModel

import copy
import csv

from math import *

def re_rank(nodes):
    re_rank = []

    c = np.mean(nodes[0::2]),np.mean(nodes[1::2])
    
    # points = list(zip(nodes[0::2], nodes[1::2]))
    
    points = np.array(nodes).reshape((-1, 2))

    for o in points:
    # left top
        if o[0] <= c[0] and o[1] <= c[1]:
            re_rank.append(o)

    for o in points:
    # right top
        if o[0] >= c[0] and o[1] <= c[1]:
            re_rank.append(o)
            
    for o in points:
    # right bottom
        if o[0] >= c[0] and o[1] >= c[1]:
            re_rank.append(o)

    for o in points:
    # left bottom
        if o[0] <= c[0] and o[1] >= c[1]:
            re_rank.append(o)

    return c, re_rank 

def DumpRotateImage(img,degree):
 
    height, width = img.shape[:2]
    heightNew = int(width * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree))))
    widthNew = int(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))
    matRotation = cv2.getRotationMatrix2D((width//2, height//2), degree, 1)
    imgRotation = cv2.warpAffine(img, matRotation,(widthNew,heightNew),borderValue=(255,255,255))
 
    return imgRotation,matRotation

def obj_inference_crop(obj_inferece, img ,resize_scale):
    
    roitransformer = obj_inferece
    det=roitransformer.inference_single(img)

    det=det[0]
        # 如果某一类中出现了多个检测，那就取其中confidence最高的那一个检测，升序的[-1]，并且取前8个值为坐标
    # print('det:{}'.format(det))
    # assert det==np.empty((0,9))
    
    if det.shape!=(0,9):
        # print(det)
        rgt_det = det[np.argmax(det[:,-1]),:-1]

        # 根据长宽比进行调整
        # xc, yc=np.mean(rgt_det[0::2]),np.mean(rgt_det[1::2])

        _, reranked_nodes = re_rank(rgt_det) # re_rank之后是shape为(-1,2)的列表
        reranked_nodes=np.array(reranked_nodes)
        near_w = np.abs(reranked_nodes[0,0]-reranked_nodes[1,0])
        near_h = np.abs(reranked_nodes[0,1]-reranked_nodes[2,1])
        
        # 调整坐标和图片保证正立或倒立，但不倾倒



        if near_h<near_w:
            pass

        else:

            imgRotation, matRotation = DumpRotateImage(img,90)
            for i in range(len(reranked_nodes)):
                points_arr=reranked_nodes[i,:]
                reranked_nodes[i,:] = np.dot(matRotation,np.array([[points_arr[0]],[points_arr[1]],[1]])).reshape(-1)
            img = imgRotation

        rgt_det = [reranked_nodes.reshape(-1)]
        
        words=crop_words(img, np.array(rgt_det)/resize_scale, height=resize_scale, grayscale=False)
        
        return words[0]
    else:
        print('???')
        # rgt_det = det
        return None



    

def obj_load_model(model_path):
    roitransformer = DetectorModel(r'configs/GUCCI_OD/gucci_od.py',model_path)
    return roitransformer


def main(gucci_datapath):

    obj_inferece=obj_load_model('work_dirs/gucci_od/epoch_100.pth')

    
    # print(os.listdir(gucci_datapath))
    # 从文件夹中获取图片
    def get_imgpaths(source_path):
        img_paths=[]
        for root, _, names in os.walk(source_path):
            for name in names:
                img_paths.append(osp.join(root,name))
        return img_paths
    img_trainval=get_imgpaths(gucci_datapath)

    # 从txt中获取图片
    # img_trainval = []
    # with open('missing_files_02945694.txt', 'r+') as f:
    #     for line in f.readlines():
    #         img_trainval += [line.strip()]

    w=0
    det=0
    missing_files = []
    for idx,img_name in enumerate(img_trainval):
        
        print('Detecting{}'.format(img_name))
        img = cv2.imread(img_name)
        img = cv2.resize(img,(512,512))
        os.remove(img_name)
        word = obj_inference_crop(obj_inferece,img, 512)
        
        if isinstance(word,np.ndarray):
            w_ = word.shape[1]
            if w<w_:
                w=w_
            
            cv2.imwrite((img_name),word)
            det+=1
            print('检测到{}/{}'.format(det,idx+1))
            print('最长边是{}'.format(w))
            # missing_files += [img_name]
        else:
            missing_files += [img_name]
    print('仍然失误{}张'.format(len(missing_files)))
    print(missing_files)
    with open('missing_files_02945694.txt','w+') as f:
        for file in missing_files:
            f.write(file+'\n')

    # print(w)

gucci_datapath='../gucci_dataset/cropped_guccilogo0404/02945694_new_crop/13659164'
# gucci_datapath='bishetest'
main(gucci_datapath)