from fastai.vision import models, ImageDataBunch, cnn_learner, untar_data, accuracy, get_transforms, load_learner
import os, sys
import torchvision.models as model
import torch

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
    #learn.fit(1)
    
    #learn.export('E:\\Data\\net.onnx')
    #learn.save('E:\\Data\\net1')
    dummy_input = torch.randn(1, 3, 100, 100, device='cuda')
    onnx_file_name = 'E://Data//net1.onnx'
    torch.onnx.export(learn.model, dummy_input, onnx_file_name)
    #torch.onnx.export(model, dummy_input, onnx_file_name)

 
if __name__ == '__main__':
    main()