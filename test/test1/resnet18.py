from fastai.vision import models, URLs, ImageDataBunch, cnn_learner, untar_data, accuracy, get_transforms, load_learner
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
    
    #path = 'E:\\Git\\pytorch-examples\\test\\test1\\Data'
    #tfms = get_transforms(do_flip=False)
    #data = ImageDataBunch.from_folder(path, ds_tfms=tfms, size=64)
    #learn = cnn_learner(data, models.resnet18, metrics=accuracy)  # 构建cnn模型，使用resnet18预训练模型
    #learn.fit(1)
    #learn.export('E:\\Git\\pytorch-examples\\test\\test1\\net.pkl')
    file_dir = 'E:\\Git\\pytorch-examples\\test\\test1\\Data\\valid'
    net_file = 'E:\\Git\\pytorch-examples\\test\\test1'
    learn1 = load_learner(net_file)
    file_list = []
    list_dir(file_dir, file_list)
    print('end')
 
if __name__ == '__main__':
    main()