from fastai.vision import models, URLs, ImageDataBunch, cnn_learner, untar_data, accuracy, get_transforms, load_learner, open_image
import os, sys

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
    print("end")

def main():
    file_dir = 'E:\\Git\\pytorch-examples\\test\\test1\\Data\\valid'
    net_file = 'E:\\Git\\pytorch-examples\\test\\test1'
    learn = load_learner(net_file, 'net.pkl')
    file_list = []
    list_dir(file_dir, file_list)
    for img_file in file_list:
        img = open_image(img_file)
        pred_class,pred_idx,outputs = learn.predict(img) #预测图片
        print(pred_class) #输出类别
        print(outputs) #输出每个类的概率

if __name__ == '__main__':
    main()