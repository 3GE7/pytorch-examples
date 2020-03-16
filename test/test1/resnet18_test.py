from fastai.vision import models, URLs, ImageDataBunch, cnn_learner, untar_data, accuracy, get_transforms, load_learner, open_image
import os, sys
from cv2 import cv2
import shutil
import time, datetime

def list_dir(file_dir, file_list):
    dir_list = os.listdir(file_dir)
    for cur_file in dir_list:
        path = os.path.join(file_dir, cur_file)
        if os.path.isfile(path):
            if os.path.splitext(path)[1] == '.bmp':  
                file_list.append(path)
        else:
            if os.path.isdir(path):
                list_dir(path, file_list)

def main():
    #清理分类结果文件夹
    shutil.rmtree('E:\\TestAll', True)
    time.sleep(1)
    os.mkdir('E:\\TestAll')
    #读取分类样本和网络
    file_dir = 'E:\\Data\\valid'
    net_file = 'E:\\Data'
    learn = load_learner(net_file, 'net.onnx')
    file_list = []
    result_name = []
    nCount = 1
    ntime_count = 0
    result_dir = 'E:\\TestAll'
    list_dir(file_dir, file_list)
    #逐一分类与储存
    for img_file in file_list:
        img = open_image(img_file)
        cv_image = cv2.imread(img_file)
        time_start = datetime.datetime.now()
        pred_class, pred_index, pred_rate = learn.predict(img) #预测图片
        time_end = datetime.datetime.now()
        time_use = (time_end - time_start).microseconds / 1000
        ntime_count = ntime_count + time_use
        result_name = result_dir + '\\' + str(pred_class) + '\\' + str(nCount) + '.bmp'
        dir_path = result_dir + '\\' + str(pred_class)
        if not(os.path.isdir(dir_path)):
            os.mkdir(dir_path)
        if not(cv_image is None):
            cv2.imwrite(result_name, cv_image)
        nCount = nCount + 1
    time_averg = ntime_count / nCount
    print("end")

if __name__ == '__main__':
    main()