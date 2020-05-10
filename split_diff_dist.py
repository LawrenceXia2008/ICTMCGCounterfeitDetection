# _*_coding:utf-8_*_
import numpy as np
from sklearn.model_selection import train_test_split
import os
import sys
from shutil import copyfile
import os.path as osp
import glob


# 得到按照样品id内样本量为key排序的列表
def get_cls_statics(src_path):
    path_gen=osp.join(src_path,'GEN')
    path_fake=osp.join(src_path,'FAKE')
    path_gen_smp_ids=glob.glob(osp.join(path_gen,'*'))
    path_fake_smp_ids=glob.glob(osp.join(path_fake,'*'))

    fake_num_samples_in_id=[(id_, len(os.listdir(id_))) for id_ in path_fake_smp_ids]

    gen_num_samples_in_id=[(id_, len(os.listdir(id_))) for id_ in path_gen_smp_ids]

    # 对每一个类别的(id名称,样本个数)的元组进行排序

    fake_num_samples_in_id.sort(key=lambda x:x[1])
    gen_num_samples_in_id.sort(key=lambda x:x[1])

    return gen_num_samples_in_id, fake_num_samples_in_id

def split_train_test_in_class(cls_num_samples_in_id, fuzzy_test_ratio):

    in_o_ratios=[]

    dside_nums_o=0

    # 1. 保持in和out的总数永远相等
    # 2. 到中间或者in/out=

    if len(cls_num_samples_in_id)==1:
        # raise ValueError('只有1个样本id')
        train_smp_id=cls_num_samples_in_id[0]
        test_smp_id=cls_num_samples_in_id[1]

    elif len(cls_num_samples_in_id)==2:
        train_smp_id=cls_num_samples_in_id[1] #取大的那一个
        test_smp_id=cls_num_samples_in_id[0] #取小的那一个
    else:
        for idx in range(len(cls_num_samples_in_id)):
            

            if idx==len(cls_num_samples_in_id)//2:
                cls_num_samples_in_id_in = cls_num_samples_in_id[idx+1:len(cls_num_samples_in_id)-1-idx]
                for i in range(len(cls_num_samples_in_id_in)):
                    dside_nums_in+=cls_num_samples_in_id_in[i][1]
                break

            else:
                dside_nums_in=0
                dside_num_o=cls_num_samples_in_id[idx][1]+cls_num_samples_in_id[len(cls_num_samples_in_id)-1-idx][1]
                dside_nums_o+=dside_num_o

                cls_num_samples_in_id_in = cls_num_samples_in_id[idx+1:len(cls_num_samples_in_id)-1-idx]
                
                for i in range(len(cls_num_samples_in_id_in)):
                    dside_nums_in+=cls_num_samples_in_id_in[i][1]
                # print('idx等于{}的时候'.format(idx))
                # # print(cls_num_samples_in_id_in)
                


                # print('in',dside_nums_in)
                # print('out',dside_nums_o)

            in_o_ratios.append(dside_nums_in/dside_nums_o)

    # 0.15:0.85==0.176
    # 0.4:0.6==0.67

    # 寻找最优的分割比
    # 指定的最优分割比计算
    fuzzy_train_ratio = 1 - fuzzy_test_ratio

    fuzzy_te_by_tr = fuzzy_test_ratio/fuzzy_train_ratio

    in_o_ratios=np.array(in_o_ratios)
    # mask=np.where(in_o_ratios<0.67,1,0)
    mask=np.where(in_o_ratios>=fuzzy_te_by_tr,1,0)
    

    # 用if else的方式让其接近fuzzy_test_ratio：fuzzy_train_ratio

    if mask.any():
        index_=np.max(np.where(mask==1))  # 在范围内的情况下，希望in也就是测试集尽可能占比小
        # print(in_o_ratios[index_])
        

    else: # 没有大于0.25的，只有小于fuzzy_test_ratio的
        index_=np.argmax(np.array(in_o_ratios))


    test_smp_id=[]
    train_smp_id=[]

    for idx in range(index_+1): # 在0-index_内对其收集测试集样本
        train_smp_id.append(cls_num_samples_in_id[idx])  # name
        train_smp_id.append(cls_num_samples_in_id[len(cls_num_samples_in_id)-1-idx])  # name

    test_smp_id=list(set(cls_num_samples_in_id)-set(train_smp_id))


    test_smp_id=[id_[0] for id_ in test_smp_id]
    train_smp_id=[id_[0] for id_ in train_smp_id]

    return train_smp_id, test_smp_id


# 用于从一个样品id的列表中按照选择好的训练测试比将图片提取出来
def extract_samples_for_cls(cls_smp_id_list):
    cls_samples=[]
    for smp_id in cls_smp_id_list:
        cls_samples+=glob.glob(osp.join(smp_id,'*'))
    return cls_samples

def main(src_path, fuzzy_test_ratio):
    gen_num_samples_in_id, fake_num_samples_in_id=get_cls_statics(src_path)

    # 得到最优分割比的训练测试集划分
    fake_train, fake_test = split_train_test_in_class(fake_num_samples_in_id, fuzzy_test_ratio)
    gen_train, gen_test = split_train_test_in_class(gen_num_samples_in_id, fuzzy_test_ratio)
    # 按照划分提取图片
    fake_train=extract_samples_for_cls(fake_train)
    fake_test=extract_samples_for_cls(fake_test)
    gen_train=extract_samples_for_cls(gen_train)
    gen_test=extract_samples_for_cls(gen_test)

    
    print('#####   该2级鉴定点共有数据 {} 个,其中GEN有 {} 个, 假有 {} 个, 按照{}的模糊测试占比划分测试集后划分结果：   #####\n\
    ...Training集中有GEN {} 个，FAKE {} 个，真假比为 {:0.3f} \n\
    ...Test集中有GEN {} 个，FAKE {} 个，真假比为 {:0.3f} \n'.format(len(gen_train+fake_train+gen_test+fake_test),len(gen_train+gen_test),len(fake_train+fake_test),\
        fuzzy_test_ratio, len(gen_train), len(fake_train), len(gen_train)/len(fake_train), len(gen_test), len(fake_test), len(gen_test)/len(fake_test)))


    return gen_train, gen_test, fake_train, fake_test


if __name__=='__main__':
    '''共有2个参数'''
    '''
    params: src_path: 传入的二级鉴定点文件夹, 其下属文件夹划分必须为'GEN', 'FAKE', GEN和FAKE内有各种样品id
            fuzzy_test_ratio: 使用的模糊测试集占比,程序会根据给出的模糊测试集占比寻找最优的唯一样本id划分,尽量保持接近这个比例
                比如, 如果想要训练测试分别占比 0.8, 0.2, 就填入fuzzy_test_ratio, 但最终程序划分的GEN内或者FAKE内的样本id划分依旧由实际样本个数决定,
                只能保证给出一种接近这个比例的划分.
    '''

    src_path='../dataset/20474737/58105815/'
    gen_train, gen_test, fake_train, fake_test = main(src_path=src_path, fuzzy_test_ratio = 0.1)

    # print(len(fake_test),len(fake_train))
    # print(len(gen_test),len(gen_train))


# print(test_)

# print(b)
    # print(index_)
    # print(in_o_ratios[index_])
        # print(i)

# print(in_o_ratios)