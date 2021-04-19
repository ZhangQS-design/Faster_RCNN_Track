#!/usr/bin/env python
# coding: utf-8

# # Object Detection Demo
# Welcome to the object detection inference walkthrough!  This notebook will walk you step by step through the process of using a pre-trained model to detect objects in an image. Make sure to follow the [installation instructions](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md) before you start.

# In[2]:


import time
import numpy as np
import matplotlib.pyplot as plt
#import tensorflow as tf
import cv2
#from server.data_lab import *
#print(cv2.__file__)
#from utils import label_map_util
#from utils import visualization_utils as vis_util
#model_path = 'colab_luck/frozen_inference_graph.pb'
#config_path = 'colab_luck/detect_person.pbtxt'
model_path = 'model/Faster_Fcnn.pb'
config_path = 'model/Faster_Fcnn.pbtxt'
net = cv2.dnn.readNetFromTensorflow(model_path, config_path)
count_p = 0
a = []

def cvdetect_person(image_np):

    global count_p
    count_p += 1
    rows = image_np.shape[0]
    cols = image_np.shape[1]
    start = time.time()
    net.setInput(cv2.dnn.blobFromImage(image_np,size=(300, 300), swapRB=True, crop=False))
    cvOut = net.forward()
    end = time.time()
    #print(cvOut)
    return cvOut[0, 0, :, :]
    # for detection in cvOut[0, 0, :, :]:
    #
    #     label = detection[1]
    #     #print("---------%d-" % detection[1])
    #
    #     score = float(detection[2])
    #     #if score > 0.7 and label == 1:
    #     if score > 0.7 and label == 0:
    #     #if score > 0.7:
    #         #print("******%d**" % detection[1])
    #         #print(detection[2])
    #         left = detection[3] * cols
    #         top = detection[4] * rows
    #         right = detection[5] * cols
    #         bottom = detection[6] * rows
    #         x_center = (left + right) / 2  # 计算每个帧各个对象的中心坐标
    #         y_center = (top + bottom) / 2
    #
    #         if 200 < x_center < 400:
    #             if 200 < y_center < 400:
    #                 print("盲区有行人经过")
    #         #绘制
    #         cv2.rectangle(image_np, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), thickness=2)
    #         cv2.putText(image_np, "person", (int(left), int(top - 10)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
    # cv2.rectangle(image_np, (200, 200), (400, 400), (60, 20, 220), thickness=2)
    #
    #
    #         # print("----------------")
    #         # print("盲区有行人经过，注意减速！！！！！")
    #         # print("----------------")
    #
    # # print("Execution Time: ", end - start )
    # # a.append(end - start)
    # # if count_p == 200:
    # #     print(a)
    # #     data_write(delay_path, a)
    #
    #
    #
    # return image_np

def plt_photo(a):
    global count_p
    count_p += 1
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    # matplotlib画图中中文显示会有问题，需要这两行设置默认字体

    plt.xlabel('图片帧数')
    plt.ylabel('时延 /ms')
    plt.title("处理单元的推断时延")
    plt.xlim(xmax=200, xmin=0)
    plt.ylim(ymax=1000, ymin=0)
    # 画两条（0-9）的坐标轴并设置轴标签x，y
    x1 = 1
    for i in a:
        i = i-0.1
        y1 = i*1000*10
        colors1 = '#00CED1'  # 点的颜色
        colors2 = '#DC143C'
        area = np.pi * 4 ** 2  # 点面积
        # 画散点图
        plt.scatter(x1, y1, s=area, c=colors1, alpha=0.4, label='类别A')
        x1 = x1+1

    plt.grid(linestyle='-.')
    plt.show()