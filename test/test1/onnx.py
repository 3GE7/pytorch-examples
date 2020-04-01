from torch.autograd import Variable
import torch.onnx
import torchvision
import torchvision.models as models
import torch
import torch.nn as nn

def main():
    #net_file = 'E://Data//net1.pth'
    #models = torch.load(net_file)
    #models.state_dict()
    #model = torchvision.models.resnet18(pretrained=True).cuda()
    model = torchvision.models.resnet18(pretrained=True)
    #model
    #onnx_file_name = 'E://Data//net1.onnx'
    #dummy_input = torch.randn(1, 3, 224, 224, device='cuda')
    #torch.onnx.export(model, dummy_input, onnx_file_name)
    
    #net_file = 'E://Data//net.pkl'
    #models = torch.load(net_file)
    img_size = 224
    example = torch.rand(1, 3, img_size, img_size)
    traced_script_module = torch.jit.trace(model, example)
    traced_script_module.save("E://Data//331.pt")

if __name__ == '__main__':
    main()