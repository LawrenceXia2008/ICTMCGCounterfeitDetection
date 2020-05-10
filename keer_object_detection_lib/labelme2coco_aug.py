import os
import json
import numpy as np
import glob
import shutil
from sklearn.model_selection import train_test_split
from cv2 import imread,imwrite,resize
import cv2
import matplotlib.pyplot as plt
from math import *
from functools import partial
from PIL import Image
from pathlib import Path
np.random.seed(41)

#0为背景
classname_to_id = {"3": 1,'m':2,'9':3}

class Lableme2CoCo:

    def __init__(self):
        self.images = []
        self.annotations = []
        self.categories = []
        self.img_id = 0
        self.ann_id = 0

    def save_coco_json(self, instance, save_path):
        json.dump(instance, open(save_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)  # indent=2 更加美观显示

    # 由json文件构建COCO
    def to_coco(self, json_path_list,resize_scale):
        self._init_categories()
        for json_path in json_path_list:
            obj = self.read_jsonfile(json_path)
            img_scale = (obj['imageHeight'],obj['imageWidth'])
            self.images.append(self._image(obj, json_path,resize_scale))
            shapes = obj['shapes']
            for shape in shapes:
                annotation = self._annotation(shape,img_scale,resize_scale)
                self.annotations.append(annotation)
                self.ann_id += 1
            self.img_id += 1
        instance = {}
        instance['info'] = 'spytensor created'
        instance['license'] = ['license']
        instance['images'] = self.images
        instance['annotations'] = self.annotations
        instance['categories'] = self.categories
        return instance

    def to_coco_FlipTransform(self, json_path_list,resize_scale):
        def append_ann(shapes,FlipTransform=None):
            self.images.append(self._image(obj, json_path,resize_scale,FlipTransform=FlipTransform))
            for shape in shapes:
                annotation = self._annotation(shape,img_scale,resize_scale,FlipTransform=FlipTransform)
                if annotation == None:
                    continue
                self.annotations.append(annotation)
                self.ann_id += 1
            self.img_id += 1
            
        
        self._init_categories()
        for json_path in json_path_list:
            obj = self.read_jsonfile(json_path)
            img_scale = (obj['imageHeight'],obj['imageWidth'])
            shapes = obj['shapes']
            append_ann(shapes)
            append_ann(shapes,90)
            append_ann(shapes,180)
            append_ann(shapes,270)
            
            
        instance = {}
        instance['info'] = 'spytensor created'
        instance['license'] = ['license']
        instance['images'] = self.images
        instance['annotations'] = self.annotations
        instance['categories'] = self.categories
        return instance

    # 构建类别
    def _init_categories(self):
        for k, v in classname_to_id.items():
            category = {}
            category['id'] = v
            category['name'] = k
            self.categories.append(category)

    # 构建COCO的image字段
    def _image(self, obj, path,resize_scale,FlipTransform=None):
        image = {}
        image['height'] = resize_scale
        image['width'] = resize_scale
        image['id'] = self.img_id
        if FlipTransform==None:
            image['file_name'] = os.path.basename(path).replace(".json", ".jpg")
        else:
            image['file_name'] = os.path.splitext(os.path.basename(path))[0]+'_r{}.jpg'.format(FlipTransform)
        return image

    # 构建COCO的annotation字段
    def _annotation(self, shape, img_scale, resize_scale, FlipTransform=None):
        h,w=img_scale
        label = shape['label']
        points = shape['points'].copy()
        points_arr=np.asarray(points)
        
        # 5月7日加:points的筛选策略
        if h>w:
            for i in range(len(points_arr)):
                if points_arr[i,1]<(h-w)/2 or points_arr[i,1]>(h+w)/2:
                    return None 
                else:
                    pass
        elif h<w:
            for i in range(len(points_arr)):
                if points_arr[i,0]<(w-h)/2 or points_arr[i,0]>(h+w)/2:
                    return None 
                else:
                    pass
        


        # points的调整策略
        if h>w:
            for i in range(len(points_arr)):
                points_arr[i,1]-=(h-w)/2
        elif h<w:
            for i in range(len(points_arr)):
                points_arr[i,0]-=(w-h)/2
        else: #h=w
            pass

        # 对所有方形图resize到512×512,标注框也随即被缩放到512×512
        for i in range(len(points_arr)):
            points_arr[i,0]=points_arr[i,0]/min(h,w)*resize_scale
            points_arr[i,1]=points_arr[i,1]/min(h,w)*resize_scale

            if FlipTransform==None:
                pass
            else:
                matRotation = cv2.getRotationMatrix2D(( resize_scale// 2, resize_scale // 2), FlipTransform, 1)
                points_arr[i]=np.dot(matRotation,np.array([[points_arr[i,0]],[points_arr[i,1]],[1]])).reshape(-1)
            
        print('points看一下debug',points_arr)
        annotation = {}
        annotation['id'] = self.ann_id
        annotation['image_id'] = self.img_id

        annotation['category_id'] = int(classname_to_id[label])
        annotation['segmentation'] = [np.asarray(points_arr).flatten().tolist()]
        annotation['bbox'] = self._get_box(points_arr)
        annotation['iscrowd'] = 0
        annotation['area'] = 1.0
        return annotation

    # 读取json文件，返回一个json对象
    def read_jsonfile(self, path):
        with open(path, "r", encoding='gbk') as f:
            data = f.read().encode(encoding='utf-8')
            result = json.loads(data)
            # return json.load(f)
            return result

    # COCO的格式： [x1,y1,w,h] 对应COCO的bbox格式
    def _get_box(self, points):
        min_x = min_y = np.inf
        max_x = max_y = 0
        for x, y in points:
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
        return [min_x, min_y, max_x - min_x, max_y - min_y]

def vis_anno(img_dir,coco_jsonpath,val_img_dir,val_coco_jsonpath,img_name):

    # 读取json
    try:
        with open(coco_jsonpath, "r",encoding='utf-8') as f:
            result=json.load(f)
        annotations=result['annotations']
        images=result['images']
        imgnames_json=[image['file_name'] for image in images]
        print(img_name)
        idx=imgnames_json.index(img_name)
    except ValueError:
        with open(val_coco_jsonpath, "r",encoding='utf-8') as f:
            result=json.load(f)
        annotations=result['annotations']
        images=result['images']
        imgnames_json=[image['file_name'] for image in images]
        print(img_name)
        idx=imgnames_json.index(img_name)
        img_dir=val_img_dir
    
    requested_anno=[anno for anno in annotations if anno['image_id']==idx]

    # 5月7日加:查看指定类别的标注
    # check_class = '指定要查看类的名字'
    # category_id = classname_to_id[check_class]  
    # requested_anno = [ anno for anno in requested_anno if anno["category_id"] == category_id]

    boxes=[]
    for anno in requested_anno: 
        boxes.append(np.array(anno['segmentation'][0]))
    img_name=result['images'][idx]['file_name']

    img_path=os.path.join(img_dir,img_name)
    print('此处的image path',img_path)
    img = cv2.imread(img_path)

    # 在所有边框内画点
    point_color=[(0, 0, 255),(0, 255, 255),(255,0,0),(255,255,0)]
    for i,box in enumerate(boxes):
        box_xy=np.round(box.reshape(-1,1,2))
        # print('box_xy',box_xy)
        box_xy=box_xy.astype(np.int32)
        font=cv2.FONT_HERSHEY_SIMPLEX
        for j in range(4):
            box_xy_j=tuple(box_xy[j].reshape(-1))
            cv2.circle(img, box_xy_j, 4,  point_color[(i+2)%4], 8)
            cv2.putText(img, str(j), box_xy_j, font, 1, point_color[i%4], 4)


    # img = img[:, :, (2, 1, 0)]
    # img /= 255
    # img = cv2.resize(img, (512, 512))
    path_annoimg = Path('D:/Myfile_ms/Research/CVPRJ/mask0126/mask0415/mask_od0419/标注数据可视化')/img_path
    if not os.path.exists(path_annoimg):
        os.makedirs(path_annoimg)
    img = Image.fromarray(img) 
    img.save(str(path_annoimg))
    # fig=plt.figure(figsize=(80,100),dpi=90)
    # plt.imshow(img)
    # plt.savefig(os.path.join(path_annoimg,img_name))

def dumpRotateImage(img, degree):
    height, width = img.shape[:2]
    heightNew = int(width * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree))))
    widthNew = int(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))
    matRotation = cv2.getRotationMatrix2D((width // 2, height // 2), degree, 1)
    imgRotation = cv2.warpAffine(img, matRotation, (widthNew, heightNew), borderValue=(255, 255, 255))
    return imgRotation


if __name__ == '__main__':
    # src_gt_path='D:/Myfile_ms/Research/CVPRJ/mask0126/mask0415/mask_od0419/new_total_train/gt'
    # src_image_path='D:/Myfile_ms/Research/CVPRJ/mask0126/mask0415/mask_od0419/new_total_train/image'

    # saved_coco_path = "D:/Myfile_ms/Research/CVPRJ/mask0126/mask0415/mask_od0419/"
    
    # FlipTransform_ornot=True
    # resize_scale=512

    # # 创建文件
    
    # if not os.path.exists("%scoco/annotations/"%saved_coco_path):
    #     os.makedirs("%scoco/annotations/"%saved_coco_path)
    # if not os.path.exists("%scoco/images/train2017/"%saved_coco_path):
    #     os.makedirs("%scoco/images/train2017"%saved_coco_path)
    # if not os.path.exists("%scoco/images/val2017/"%saved_coco_path):
    #     os.makedirs("%scoco/images/val2017"%saved_coco_path)
    # # 获取images目录下所有的joson文件列表
    # json_list_path = glob.glob(src_gt_path + "/*.json")
    # # 数据划分,这里没有区分val2017和tran2017目录，所有图片都放在images目录下
    # train_path, val_path = train_test_split(json_list_path, test_size=0.0,random_state=0)
    # print("train_n:", len(train_path), 'val_n:', len(val_path))

    
    
    # if FlipTransform_ornot==False:
    #     l2c_train = Lableme2CoCo()
    #     train_instance = l2c_train.to_coco(train_path,resize_scale)
    #     l2c_val = Lableme2CoCo()
    #     val_instance = l2c_val.to_coco(val_path,resize_scale)

    #     l2c_val.save_coco_json(val_instance, '%scoco/annotations/instances_val2017.json'%saved_coco_path)
    #     l2c_train.save_coco_json(train_instance, '%scoco/annotations/instances_train2017.json'%saved_coco_path)
    
    #     for file in train_path:
    #         # 使用resize
    #         # 不使用resize
    #         img=imread(os.path.join(src_image_path,os.path.basename(file).replace("json","jpg")))
    #         img=resize(img,(resize_scale,resize_scale))
    #         imwrite(os.path.join("%scoco/images/train2017/"%saved_coco_path,os.path.basename(file).replace("json","jpg")),img)
    #         # shutil.copy(os.path.join(src_image_path,os.path.basename(file).replace("json","jpg")),"%scoco/images/train2017/"%saved_coco_path)
    #     for file in val_path:
    #         img=imread(os.path.join(src_image_path,os.path.basename(file).replace("json","jpg")))
    #         img=resize(img,(resize_scale,resize_scale))
    #         imwrite(os.path.join("%scoco/images/val2017/"%saved_coco_path,os.path.basename(file).replace("json","jpg")),img)
    #         # shutil.copy(os.path.join(src_image_path,os.path.basename(file).replace("json","jpg")),"%scoco/images/val2017/"%saved_coco_path)

    # else:   # 做翻转变换
    #     l2c_train = Lableme2CoCo()
    #     train_instance = l2c_train.to_coco_FlipTransform(train_path,resize_scale)
    #     l2c_val = Lableme2CoCo()
    #     val_instance = l2c_val.to_coco_FlipTransform(val_path,resize_scale)

    #     l2c_val.save_coco_json(val_instance, '%scoco/annotations/instances_val2017.json'%saved_coco_path)
    #     l2c_train.save_coco_json(train_instance, '%scoco/annotations/instances_train2017.json'%saved_coco_path)
    
    #     for file in train_path:
    #         # 使用resize
    #         # 不使用resize
    #         img=imread(os.path.join(src_image_path,os.path.basename(file).replace("json","jpg")))
            
    #         # 5月7日13:50加上的
    #         h,w = img.shape[:2]
    #         if h>w:
    #             forehead=round(h / 2 - w / 2)
    #             img=img[forehead:forehead+w,:,:]
    #         elif h<w:
    #             forehead=round(w / 2 - h / 2)
    #             img = img[:,forehead:forehead+h,:]
    #         else:
    #             pass
            
    #         img=resize(img,(resize_scale,resize_scale))
    #         img_r90=dumpRotateImage(img, 90)
    #         img_r180=dumpRotateImage(img, 180)
    #         img_r270=dumpRotateImage(img, 270)
    #         imwrite(os.path.join("%scoco/images/train2017/"%saved_coco_path,os.path.basename(file).replace("json","jpg")),img)
    #         imwrite(os.path.join("%scoco/images/train2017/"%saved_coco_path,os.path.splitext(os.path.basename(file))[0]+'_r90.jpg'),img_r90)
    #         imwrite(os.path.join("%scoco/images/train2017/"%saved_coco_path,os.path.splitext(os.path.basename(file))[0]+'_r180.jpg'),img_r180)
    #         imwrite(os.path.join("%scoco/images/train2017/"%saved_coco_path,os.path.splitext(os.path.basename(file))[0]+'_r270.jpg'),img_r270)
    #         ## shutil.copy(os.path.join(src_image_path,os.path.basename(file).replace("json","jpg")),"%scoco/images/train2017/"%saved_coco_path)
    #     for file in val_path:
    #         img=imread(os.path.join(src_image_path,os.path.basename(file).replace("json","jpg")))

    #         # 5月7日13:50加上的
    #         h,w = img.shape[:2]
    #         if h>w:
    #             forehead=round(h / 2 - w / 2)
    #             img = img[forehead:forehead+w,:,:]
    #         elif h<w:
    #             forehead=round(w / 2 - h / 2)
    #             img = img[:,forehead:forehead+h,:]
    #         else:
    #             pass

    #         img=resize(img,(resize_scale,resize_scale))
    #         img_r90=dumpRotateImage(img, 90)
    #         img_r180=dumpRotateImage(img, 180)
    #         img_r270=dumpRotateImage(img, 270)
    #         imwrite(os.path.join("%scoco/images/val2017/"%saved_coco_path,os.path.basename(file).replace("json","jpg")),img)
    #         imwrite(os.path.join("%scoco/images/val2017/"%saved_coco_path,os.path.splitext(os.path.basename(file))[0]+'_r90.jpg'),img_r90)
    #         imwrite(os.path.join("%scoco/images/val2017/"%saved_coco_path,os.path.splitext(os.path.basename(file))[0]+'_r180.jpg'),img_r180)
    #         imwrite(os.path.join("%scoco/images/val2017/"%saved_coco_path,os.path.splitext(os.path.basename(file))[0]+'_r270.jpg'),img_r270)
            ## shutil.copy(os.path.join(src_image_path,os.path.basename(file).replace("json","jpg")),"%scoco/images/val2017/"%saved_coco_path)

    # 可视化
    root_path='D:/Myfile_ms/Research/CVPRJ/mask0126/mask0415/mask_od0419/mask_train_coco_0419'
    coco_tr_jsonpath=os.path.join(root_path,'annotations/instances_train2017.json')
    coco_tr_imgpath=os.path.join(root_path,'images/train2017')
    coco_val_jsonpath=os.path.join(root_path,'annotations/instances_val2017.json')
    coco_val_imgpath=os.path.join(root_path,'images/val2017')

    vis_anno_by_imgname=partial(vis_anno,coco_tr_imgpath,coco_tr_jsonpath,coco_val_imgpath,coco_val_jsonpath)
    vis_anno_by_imgname('1_r270.jpg')
    
    
