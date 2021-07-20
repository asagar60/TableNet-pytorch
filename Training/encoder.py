import torch
import torch.nn as nn
import torchvision
from efficientnet_pytorch import EfficientNet

class VGG19(nn.Module):
    def __init__(self, pretrained = True, requires_grad = True):
        super(VGG19, self).__init__()
        _vgg = torchvision.models.vgg19(pretrained=pretrained).features
        self.vgg_pool3 = torch.nn.Sequential()
        self.vgg_pool4 = torch.nn.Sequential()
        self.vgg_pool5 = torch.nn.Sequential()

        for x in range(19):
            self.vgg_pool3.add_module(str(x), _vgg[x])
        for x in range(19, 28):
            self.vgg_pool4.add_module(str(x), _vgg[x])
        for x in range(28, 37):
            self.vgg_pool5.add_module(str(x), _vgg[x])

        
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
    
    def forward(self, x): 
        pool_3_out = self.vgg_pool3(x) #torch.Size([1, 256, 128, 128])
        pool_4_out = self.vgg_pool4(pool_3_out) #torch.Size([1, 512, 64, 64])
        pool_5_out = self.vgg_pool5(pool_4_out) #torch.Size([1, 512, 32, 32])
        return (pool_3_out, pool_4_out, pool_5_out)

class ResNet(nn.Module):
    def __init__(self, pretrained = True, requires_grad = True):
        super(ResNet, self).__init__()
        resnet18 = torchvision.models.resnet34(pretrained=True)

        self.layer_1 = nn.Sequential(
            resnet18.conv1,
            resnet18.bn1,
            resnet18.relu,
            resnet18.maxpool,
            resnet18.layer1
        )
        self.layer_2 = resnet18.layer2
        self.layer_3 = resnet18.layer3
        self.layer_4 = resnet18.layer4

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        
        out_1 = self.layer_2(self.layer_1(x)) #torch.Size([1, 128, 128, 128])
        out_2 = self.layer_3(out_1) #torch.Size([1, 256, 64, 64])
        out_3 = self.layer_4(out_2) #torch.Size([1, 512, 32, 32])
        return out_1, out_2, out_3


class DenseNet(nn.Module):
    def __init__(self, pretrained = True, requires_grad = True):
        super(DenseNet, self).__init__()
        denseNet = torchvision.models.densenet121(pretrained=True).features
        self.densenet_out_1 = torch.nn.Sequential()
        self.densenet_out_2 = torch.nn.Sequential()
        self.densenet_out_3 = torch.nn.Sequential()

        for x in range(8):
            self.densenet_out_1.add_module(str(x), denseNet[x])
        for x in range(8,10):
            self.densenet_out_2.add_module(str(x), denseNet[x])
        
        self.densenet_out_3.add_module(str(10), denseNet[10])
        
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        
        out_1 = self.densenet_out_1(x) #torch.Size([1, 256, 64, 64])
        out_2 = self.densenet_out_2(out_1) #torch.Size([1, 512, 32, 32])
        out_3 = self.densenet_out_3(out_2) #torch.Size([1, 1024, 32, 32])
        return out_1, out_2, out_3

class efficientNet_B0(nn.Module):
    def __init__(self, pretrained = True, requires_grad = True):
        super(efficientNet_B0, self).__init__()
        eNet = EfficientNet.from_pretrained('efficientnet-b0')
        
        self.eNet_out_1 = torch.nn.Sequential()
        self.eNet_out_2 = torch.nn.Sequential()
        self.eNet_out_3 = torch.nn.Sequential()

        blocks = eNet._blocks

        self.eNet_out_1.add_module('_conv_stem', eNet._conv_stem)
        self.eNet_out_1.add_module('_bn0', eNet._bn0)

        for x in range(14):
            self.eNet_out_1.add_module(str(x), blocks[x])
        
        self.eNet_out_2.add_module(str(14), blocks[14])
        self.eNet_out_3.add_module(str(15), blocks[15])


    def forward(self, x):
        out_1 = self.eNet_out_1(x) #torch.Size([1, 192, 32, 32])
        out_2 = self.eNet_out_2(out_1) #torch.Size([1, 192, 32, 32])
        out_3 = self.eNet_out_3(out_2) #torch.Size([1, 320, 32, 32])
        return out_1, out_2, out_3

class efficientNet(nn.Module):
    def __init__(self, model_type = 'efficientnet-b0',  pretrained = True, requires_grad = True):
        super(efficientNet, self).__init__()
        eNet = EfficientNet.from_pretrained(model_type)

        self.eNet_out_1 = torch.nn.Sequential()
        self.eNet_out_2 = torch.nn.Sequential()
        self.eNet_out_3 = torch.nn.Sequential()

        blocks = eNet._blocks

        self.eNet_out_1.add_module('_conv_stem', eNet._conv_stem)
        self.eNet_out_1.add_module('_bn0', eNet._bn0)

        for x in range(len(blocks)-3):
            self.eNet_out_1.add_module(str(x), blocks[x])
        
        self.eNet_out_2.add_module(str(len(blocks)-2), blocks[len(blocks)-2])
        self.eNet_out_3.add_module(str(len(blocks)-1), blocks[len(blocks)-1])


    def forward(self, x):
        out_1 = self.eNet_out_1(x) #torch.Size([1, 192, 32, 32])
        out_2 = self.eNet_out_2(out_1) #torch.Size([1, 192, 32, 32])
        out_3 = self.eNet_out_3(out_2) #torch.Size([1, 320, 32, 32])
        

        """
        shapes of b1
        torch.Size([1, 192, 32, 32])
        torch.Size([1, 320, 32, 32])
        torch.Size([1, 320, 32, 32])

        shapes of b2
        torch.Size([1, 208, 32, 32])
        torch.Size([1, 352, 32, 32])
        torch.Size([1, 352, 32, 32])
        """

        return out_1, out_2, out_3



if __name__ == '__main__':
    model = efficientNet()
    x = torch.randn(1,3,1024,1024)
    model(x)