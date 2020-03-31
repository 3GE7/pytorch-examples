from torch.autograd import Variable
import torch.onnx
import torchvision
import torchvision.models as models
import torch

def main():
    #net_file = 'E://Data//net1.pth'
    #models = torch.load(net_file)
    #models.state_dict()
    model = torchvision.models.resnet18(pretrained=True).cuda()
    model
    onnx_file_name = 'E://Data//net1.onnx'
    dummy_input = torch.randn(1, 3, 224, 224, device='cuda')
    torch.onnx.export(model, dummy_input, onnx_file_name)

if __name__ == '__main__':
    main()