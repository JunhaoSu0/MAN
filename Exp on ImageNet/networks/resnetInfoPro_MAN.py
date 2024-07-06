import pdb

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import copy

from .configs import Layer
from .InfoProAux import Decoder,AuxClassifier

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, groups=groups, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers,arch,local_module_num,num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, norm_layer=None, net_layer_dict=None,momentum = 0.995):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.inplanes = 64
        self.groups = groups
        self.layers = layers
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.local_module_num = local_module_num
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, norm_layer=norm_layer)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, norm_layer=norm_layer)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.bce_loss = nn.BCELoss()
        self.momentum = momentum
        self.fc = nn.Linear(2048, num_classes)
        self.wide = [256,256,512,1024,2048]
        self.config = Layer[arch][local_module_num]

        self.Encoder_Net = self._make_Encoder_Net()

        self.LB = nn.ModuleList([])
        self.EMA_Net = nn.ModuleList([])

        for net in self.Encoder_Net:
            net = net.cuda()

        for item in self.config[:-1]:
            module_index, layer_index = item
            exec('self.decoder' + str(module_index) + '_' + str(layer_index) +
                 '= Decoder(self.wide[module_index])')
            exec('self.aux' + str(module_index) + '_' + str(layer_index) +
                 '= AuxClassifier(self.wide[module_index])')

        for item in self.config[:-1]:
            module_index, layer_index = item
            if layer_index == len(eval('self.layer' + str(module_index))) - 1:
                mo, la = module_index + 1, 0
            else:
                mo, la = module_index, layer_index + 1
            self.LB.append(copy.deepcopy(eval('self.layer' + str(mo))[la]))
            self.EMA_Net.append(copy.deepcopy(eval('self.layer' + str(mo))[la]))

        for i in range(len(self.EMA_Net)):
            for param in self.EMA_Net[i].parameters():
                param.requires_grad = False

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # self.net_layer_dict = net_layer_dict
        # for layer_index in range(self.net_layer_dict['layer_num']):
        #     exec('self.fc' + str(layer_index)
        #          + " = nn.Linear(self.net_layer_dict['feature_num_list'][layer_index], num_classes)")

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, norm_layer=None):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, norm_layer=norm_layer))

        return nn.Sequential(*layers)
    def _make_Encoder_Net(self):
        Encoder_Net = nn.ModuleList([])

        Encoder_temp = nn.ModuleList([])

        local_block_index = 0

        # Build Encoder_Net
        for blocks in range(len(self.layers)):
            for layers in range(self.layers[blocks]):
                Encoder_temp.append(eval('self.layer' + str(blocks + 1))[layers])
                if blocks + 1 == self.config[local_block_index][0] \
                        and layers == self.config[local_block_index][1]:
                    Encoder_Net.append(nn.Sequential(*Encoder_temp))

                    Encoder_temp = nn.ModuleList([])
                    local_block_index += 1
        return Encoder_Net



    def forward(self, x, initial_image=None, target=None, criterion=None, layer_wise=-1, ixx_r=0, ixy_r=1):
        if self.training:
            if self.local_module_num == 1:
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu(x)
                x = self.maxpool(x)

                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)

                output = self.avgpool(x)
                output = output.view(x.size(0), -1)
                output = self.fc(output)

                loss = self.bce_loss(output,target)
                loss.backward()

                return output,loss

            else:
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu(x)
                x = self.maxpool(x)


                for i in range(len(self.Encoder_Net) - 1):
                    x = self.Encoder_Net[i](x)
                    y = self.LB[i](x) + self.EMA_Net[i](x)

                    module_index, layer_index = self.config[i]

                    y1 = eval('self.decoder' + str(module_index) + '_' + str(layer_index))(y)
                    y2 = eval('self.aux' + str(module_index) + '_' + str(layer_index))(y)


                    loss_ixy1 = self.bce_loss(y1,initial_image)
                    loss_ixy2 = criterion(y2, target)
                    loss = loss_ixy1 * 5 + loss_ixy2 * 0.75
                    loss.backward()

                    x = x.detach()

                    module_index, layer_index = self.config[i]
                    if layer_index == len(eval('self.layer' + str(module_index))) - 1:
                        mo, la = module_index + 1, 0
                    else:
                        mo, la = module_index, layer_index + 1

                    for paramEncoder, paramEMA in zip(eval('self.layer' + str(mo))[la].parameters(), self.EMA_Net[i].parameters()):
                        paramEMA.data = paramEMA.data * self.momentum + paramEncoder.data * (1 - self.momentum)



                x = self.Encoder_Net[-1](x)
                x = self.avgpool(x)
                x = x.view(x.size(0),-1)

                logits = self.fc(x)
                loss = criterion(logits,target)
                loss.backward()

            return logits,loss

        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            output = self.avgpool(x)
            output = output.view(x.size(0), -1)
            output = self.fc(output)

            return output


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2],arch = 'resnet18', **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3],arch = 'resnet34', **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3],arch = 'resnet34', **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))

    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3],arch = 'resnet101', **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3],arch = 'resnet152', **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model


def resnext50_32x4d(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3],arch = 'resnext50_32x4d', groups=32, width_per_group=4, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnext50_32x4d']))
    return model


def resnext101_32x8d(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3],arch = 'resnext101_32x8d', groups=32, width_per_group=8, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnext101_32x8d']))
    return model

# if __name__ == "__main__":
#     net = resnet101(local_module_num = 4)
#     net = net.cuda()
#     x = torch.ones(4,3,224,224).cuda()
#     target = torch.zeros(4).long().cuda()
#     print(net(x, target))