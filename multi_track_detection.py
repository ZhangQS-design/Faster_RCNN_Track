'''
先使用Faster-RCNN检测行人并标记
再用多目标跟踪器进行跟踪
'''
import numpy as np
import cv2
import os
#import tensorflow as tf
import matplotlib
from cvdetect_person import *

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
# from object_detection.utils import label_map_util
# from object_detection.utils import visualization_utils as vis_util

# 设置matplotlib正常显示中文和负号
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 系统环境设置
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 指定要使用模型的名字(此处使用FasterRcnn)
MODEL_NAME = 'faster_rcnn_inception_v2_coco_2018_01_28'
# 指定模型路径
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
# 数据集对应的label
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
NUM_CLASSES = 90
# 检测视频窗口
WINDOW_NAME = 'Pedestrian'
# 反向投影视频窗口
WINDOW_NAME2 = "Hue histogram back projection"
# 定义7种可用跟踪器
OPENCV_OBJECT_TRACKERS = {
    "csrt": cv2.TrackerCSRT_create,
    "kcf": cv2.TrackerKCF_create,
    "boosting": cv2.TrackerBoosting_create,
    "mil": cv2.TrackerMIL_create,
    "tld": cv2.TrackerTLD_create,
    "medianflow": cv2.TrackerMedianFlow_create,
    "mosse": cv2.TrackerMOSSE_create
}
# 行人检测选择区域
BORDER = [[142, 171], [101, 339], [283, 339], [296, 171]]
# BORDER = [[0, 820], [177, 1072], [1100, 744], [784, 686]]



# 返回所选择矩形框的中心点
def center(box):
    (x, y, w, h) = box
    center_x = int(x + w / 2.0)
    center_y = int(y + h / 2.0)
    return (center_x, center_y)

# 求取向量叉乘
def get_vector_cross_product(position0, position1, position):

    product_value = (position1[0]-position0[0]) * (position[1]-position0[1]) -       (position1[1]-position0[1])*(position[0]-position0[0])

    return product_value

# 判断该点是否在四边形内部
def isPosition(center_position):

    directions = []
    isPosition = True
    for i in range(0, len(BORDER)):
        direction = get_vector_cross_product(BORDER[i], BORDER[(i+1)%len(BORDER)], center_position)
        directions.append(direction)

    for i in range(0, len(directions)-1):
        if directions[i]*directions[i+1] < 0:
            isPosition = False
            break
    
    return isPosition

# 绘制直方图和折线图（每次检测到所经过选择区域的行人数）
def histograms_line(peoples):
    plt.subplots_adjust(hspace=0.45)
    plt.subplot(2, 1, 1)
    x = [i for i in range(1, len(peoples) + 1)]
    plt.bar(x, peoples)
    plt.xlabel("检测区间数")
    plt.ylabel("行人数")
    plt.title("行人检测数分布柱状图")
    plt.subplot(2, 1, 2)
    plt.scatter(x, peoples, s=100)  # 散点图
    plt.plot(x, peoples, linewidth=2)
    plt.xlabel("检测区间数")
    plt.ylabel("行人数")
    plt.title("行人检测数分布折线图")
    plt.show()


# 使用跟踪器对标记到的目标进行跟踪
def track_objects(video, object_tracker, detection_time):
    # 初始化视频流
    cap = cv2.VideoCapture(video)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)  # 帧速率
    print("视频帧速率：", frame_rate)
    frame_counts = cap.get(cv2.CAP_PROP_FRAME_COUNT)  # 总帧长
    print("视频总帧长：", frame_counts)
    video_time = frame_counts / frame_rate  # 视频总时间
    print("视频总时间：{}s".format(video_time))
    nums = 0  # 计算视频播放帧数
    times = int(1000/frame_rate)  # 每隔多少ms播放帧

    detection_frames = int(frame_rate * detection_time) # 检测的时间间隔s
    detection_nums = -1  # 检测次数
    peoples = []  # 经过选择区域的行人数
    # 在视频帧上绘制分界线进行计数
    pts = np.array(BORDER, np.int32)
    pts = pts.reshape((-1, 1, 2))
    # 循环帧并开始多目标跟踪
    flag = True
    while True:
        # 获取当前视频的帧
        ret, frame = cap.read()
        rows = frame.shape[0]
        cols = frame.shape[1]
        # 检查视频帧流是否结束
        if frame is None:
            break
        # 将当前帧重置 (加快处理速度)
        # frame = imutils.resize(frame, width=600)
        # 每隔100帧重新检测帧图像上的行人
        if nums % detection_frames == 0:
            peoples.append(0)
            detection_nums += 1
            flag = True
        # 对于每一个被跟踪的对象矩形框进行更新
        if not flag:
            (success, boxes) = trackers.update(frame)
        if flag:
            # 重新初始化多对象跟踪器
            trackers = cv2.MultiTracker_create()
            # 绘制检测识别文字
            font = cv2.FONT_ITALIC
            h, w, c = frame.shape
            cv2.putText(frame, 'Pedestrian Detection...', (int(w * 0.4), int(h * 0.85)), font, 1, (255, 255, 0), 2)
            cv2.polylines(frame, [pts], True, (255, 0, 0), 1, cv2.LINE_AA)
            cv2.imshow(WINDOW_NAME, frame)
            cv2.waitKey(1)
            # 检测该帧图像固定区域上的行人
            # selected_frame=frame[154:426,85:300]
            #_, boxes = process_image(frame)

            boxes = cvdetect_person(frame)

            # 将所有标记对象先加入到多对象跟踪器中
            for box in boxes:
                box = tuple(box[3:7])
                # 创建一个新的对象跟踪器为新的边界框并将它添加到多对象跟踪器里
                tracker = OPENCV_OBJECT_TRACKERS[object_tracker]()
                trackers.add(tracker, frame, box)
            # 展示最开始的帧检测图
            # vis_util.plt.imshow(frame)
            # vis_util.plt.show()
            flag = False

        cv2.polylines(frame, [pts], True, (255, 0, 0), 1, cv2.LINE_AA)
        # 检查边界框并在帧上进行绘制
        for detection in boxes:
            #box = tuple(box[3:7])
            left = detection[0] * cols
            top = detection[1] * rows
            right = detection[2] * cols
            bottom = detection[3] * rows
            x_center = (left + right) / 2  # 计算每个帧各个对象的中心坐标
            y_center = (top + bottom) / 2
            center_position = center(box)
            center_mod = (int(x_center), int(y_center))
            # 绘制矩形框
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
            # 绘制矩形框的质心
            cv2.circle(frame, center_mod, 2, (0, 0, 255), 2)
            # 计算进出选择区域的人数
            if nums % detection_frames == 0 and isPosition(center_mod):
                peoples[detection_nums] += 1

        # 显示框架以及选择要跟踪的对象
        cv2.imshow(WINDOW_NAME, frame)
        nums += 1
        key = cv2.waitKey(times) & 0xFF
        if key == ord("q"):
            break

    # 关闭所有窗口
    cv2.destroyAllWindows()
    cap.release()
    return peoples, video_time


if __name__ == '__main__':
    video = "./test_videos/street.mp4"
    object_tracker = "kcf"
    detection_time = 4 
    peoples, times = track_objects(video, object_tracker, detection_time)
    print(peoples)
    total_peoples = 0
    for people in peoples:
        total_peoples += people
    print("总人数：", total_peoples)
    per_peoples = total_peoples / (times / 60.0)
    print("行人密度（每分钟走过的人数）：", per_peoples)
    # 直方图折线图显示
    histograms_line(peoples)
