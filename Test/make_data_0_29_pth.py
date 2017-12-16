#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from shutil import copyfile, rmtree
import config1
from PIL import Image
import os
#import json
import numpy as np
import glob
import pandas as pd
from sklearn.utils import shuffle
#import cv2
import torch as t


def get_class_weight_train(d):
    white_list_formats = {'png', 'jpg', 'jpeg', 'bmp','JPG'}
    class_number = dict()
    dirs = sorted([o for o in os.listdir(d) if os.path.isdir(os.path.join(d, o))],key=lambda x:int(x[:]))
    #只需要加上这样的一句话则实现自然排序:key=lambda x:int(x[:])
    k = 0
    image_ids=[]
    image_labels=[]
    for class_name in dirs:
        class_number[k] = 0
        iglob_iter = glob.iglob(os.path.join(d, class_name, '*.*'))
#        for i in iglob_iter:
        for ii_1,i in enumerate(iglob_iter):
#            print ii_1,i
#            i=u(i)
#                ii=
#            if ii_1 % 3==0:#不所有的图片去训练
            i=unicode(i,"utf-8")#uincode和utf-8之间的编码互换,uincode和utf-8之间的编码互换,
            #参考链接:http://blog.csdn.net/u012448083/article/details/51918681
#            print i
            img_name, ext = os.path.splitext(i)#这个地方没有打乱顺序可能对后面的结果会有影响
#            print './'+img_name[-1-1]+'/'+i.split('/')[-1]
            image_ids.append('./'+img_name.split('/')[-1-1]+'/'+i.split('/')[-1])#图片名字
#            image_ids=img_name[-1-1]+'/'+image_ids
#            print image_ids
            image_labels.append(int(class_name)-1)#图片label)
#            print img_name
#            print i
            if ext[1:] in white_list_formats:
                class_number[k] += 1
            else:
                print img_name
        k += 1
    ids=image_ids

    labels=image_labels
#    print ids
#    print "*************************"
#    print labels
    print class_number.values()
    id2ix = {id:ix for ix,id in enumerate(image_ids)}
    train = dict(ids=ids,labels = labels,id2ix=id2ix)
#    print train
    total = np.sum(class_number.values())
    print class_number
    print 'total number of train_images:',total
    return train

def get_class_weight_valid(d):
    white_list_formats = {'png', 'jpg', 'jpeg', 'bmp','JPG'}
    class_number = dict()
    dirs = sorted([o for o in os.listdir(d) if os.path.isdir(os.path.join(d, o))],key=lambda x:int(x[:]))
    #只需要加上这样的一句话则实现自然排序:key=lambda x:int(x[:])
    k = 0
    image_ids=[]
    image_labels=[]
    for class_name in dirs:
        class_number[k] = 0
        iglob_iter = glob.iglob(os.path.join(d, class_name, '*.*'))

        for ii_1,i in enumerate(iglob_iter):
#            i=u(i)
            i=unicode(i,"utf-8")#uincode和utf-8之间的编码互换,uincode和utf-8之间的编码互换,
            #参考链接:http://blog.csdn.net/u012448083/article/details/51918681
            img_name, ext = os.path.splitext(i)#这个地方没有打乱顺序可能对后面的结果会有影响
            image_ids.append('./'+img_name.split('/')[-1-1]+'/'+i.split('/')[-1])
#            image_ids.append(i.split('/')[-1])#图片名字
            image_labels.append(int(class_name)-1)#图片label
#            print img_name
#            print i
            if ext[1:] in white_list_formats:
                class_number[k] += 1
        k += 1
    ids=image_ids
    labels=image_labels
#    print ids
#    print "*************************"
#    print labels
    print class_number.values()
    id2ix = {id:ix for ix,id in enumerate(image_ids)}
    valid = dict(ids=ids,labels = labels,id2ix=id2ix)
    total = np.sum(class_number.values())
#    print valid
    print class_number
    print 'total number of valid_images:',total
    return valid

def get_class_weight_test(d):
    white_list_formats = {'png', 'jpg', 'jpeg', 'bmp','JPG'}
    class_number = dict()
    dirs = d
    #只需要加上这样的一句话则实现自然排序:key=lambda x:int(x[:])
    k = 0
    image_ids=[]
    image_labels=[]
    iglob_iter = glob.iglob(os.path.join(dirs, '*.*'))
    class_number[k] = 0
    for i in iglob_iter:
#            i=u(i)
#        print i,"*******************"
        i=unicode(i,"utf-8")#uincode和utf-8之间的编码互换,uincode和utf-8之间的编码互换,
        #参考链接:http://blog.csdn.net/u012448083/article/details/51918681
        img_name, ext = os.path.splitext(i)#这个地方没有打乱顺序可能对后面的结果会有影响
        image_ids.append(i.split('/')[-1])
#            image_ids.append(i.split('/')[-1])#图片名字
        image_labels.append(1)#图片label
#            print img_name
#            print i
        if ext[1:] in white_list_formats:
            class_number[k] += 1

    ids=image_ids
    labels=image_labels
#    print ids
#    print "*************************"
#    print labels
    print class_number.values()
    id2ix = {id:ix for ix,id in enumerate(image_ids)}
    test= dict(ids=ids,labels = labels,id2ix=id2ix)
    total = np.sum(class_number.values())
#    print test
    print class_number
    print 'total number of test_images:',total
    return test
if __name__ == '__main__':
    #set_names = ['scene_train_20170904', 'scene_validation_20170908']

#    clas_file = './data/ai_challenger_scene_test_a_20170922/scene_classes.csv'
#    test_dir = './data/pig/test_A/'
#    val_ann_file = '/home/hutao/hutao/scene-baseline-master/data/image/ai_cha/scene/ai_challenger_scene_validation_20170908/scene_validation_annotations_20170908.json'
    label_ids=[]
    label_zh=[]
    label_en=[]
    for j in range(30):
        label_ids.append(j)
        label_en.append(str(j+1))
        label_zh.append(str(j+1))
#    print label_ids
#    print label_en
#    print label_zh
#    a=pd.read_csv(clas_file,header=None)
#    label_ids,label_zh,label_en = a[0],a[1],a[2]
    id2label = {k:(v1.decode('utf8'),v2) for k,v1,v2 in zip(label_ids,label_zh,label_en)}
#    print id2label
    cwd = os.path.dirname(os.path.realpath(__file__))
    
    train = get_class_weight_train(config1.train_dir)
    valid = get_class_weight_valid(config1.valid_dir)
    test = get_class_weight_test(config1.test_dir)
#    print train
#    with open(val_ann_file) as f:
#        datas = json.load(f)
    
#    ids = [ii['image_id'] for ii in datas]
#    labels = [int(ii['label_id']) for ii in datas]
#    id2ix = {id:ix for ix,id in enumerate(ids)}
#    val = dict(ids=ids,labels = labels,id2ix=id2ix)



#    val,labels = get_class_weight_valid(config1.valid_dir)
    
#    ids = os.listdir(test_dir)
#    id2ix = {id:ix for ix,id in enumerate(ids)}
## test = SceneData(ids,None,id2ix)
#    test = dict(ids=ids,labels = labels,id2ix=id2ix)
#    
#    id2label = {k:(v1.decode('utf8'),v2) for k,v1,v2 in zip(label_ids,label_zh,label_en)}
    all = dict(train=train,test1=test,val=valid,id2label=id2label)
#    all = dict(train=train,val=valid)
    print "saveing"
#    print all
    t.save(all,'./data/pig/pig_0_29.pth')
#        

