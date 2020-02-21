from fastai.vision import models, URLs, ImageDataBunch, cnn_learner, untar_data, accuracy
import torch
 
def main():
    
    path = untar_data(URLs.MNIST_SAMPLE)  # 下载数据集，这里只是MNIST的子集，只包含3和7的图像,会下载并解压（untar的命名原因）到/root/.fastai/data/mnist_sample（如果你是root用户）下，包含训练数据，测试数据，包含label的csv文件
    data = ImageDataBunch.from_folder(path)  # 利用ImageDataBunch读取文件夹，返回一个ImageDataBunch对象
    learn = cnn_learner(data, models.resnet18, metrics=accuracy)  # 构建cnn模型，使用resnet18预训练模型
    learn.fit(10)  # 训练一轮
 
if __name__ == '__main__':
    main()