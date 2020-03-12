from fastai.vision import models, URLs, ImageDataBunch, cnn_learner, untar_data, accuracy, get_transforms, load_learner
import os, sys

def main():
    
    path = 'E:\\Data'
    tfms = get_transforms(do_flip=False)
    data = ImageDataBunch.from_folder(path, ds_tfms=tfms, size=64)
    net_file = 'E:\\Data'
    if os.path.isfile('E:\\Data\\net.pkl'):
        learn = load_learner(net_file, 'net.pkl')
        learn.data = data
    else:
        learn = cnn_learner(data, models.resnet18, metrics=accuracy)  # 构建cnn模型，使用resnet18预训练模型
    learn.fit(20)
    learn.export('E:\\Data\\net.pkl')

 
if __name__ == '__main__':
    main()