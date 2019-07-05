import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size = 3, stride = stride,
        padding = 1, bias = False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size = 1, stride = stride, bias = False)

class BasicBlock(nn.Module):

    def __init__(self,inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace = True)
        self.conv1 = conv3x3(inplanes, planes, stride = stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.conv1x1 = conv1x1(inplanes, planes, stride = stride)

    def forward(self, x):
        identity = self.conv1x1(x)

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out += identity
        return out

class ResNet(nn.Module):

    def __init__(self):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size = 5, stride = 2, padding = 2, bias = False)
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.layer1 = BasicBlock(32, 32, 2)
        self.layer2 = BasicBlock(32, 64, 2)
        self.layer3 = BasicBlock(64, 128, 2)
        self.fc = nn.Linear(2688, 1)
        self.dropout = nn.Dropout(p = 0.5)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode= 'fan_out', nonlinearity = 'relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = x.view(x.size(0), -1)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc(x)

        return x