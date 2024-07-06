import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self, inplanes):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(inplanes,192,kernel_size=1,stride=1,padding=0),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.Upsample(size=[28, 28], mode='bilinear', align_corners=True),
            nn.Conv2d(192, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Upsample(size=[56, 56], mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Upsample(size=[112, 112], mode='bilinear', align_corners=True),
            nn.Conv2d(32, 12, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Upsample(size=[224, 224], mode='bilinear', align_corners=True),
            nn.Conv2d(12, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, features):
        assert features.size(3) == 14
        return self.decoder(features)



class AuxClassifier(nn.Module):
    def __init__(self, inplanes):
        super(AuxClassifier, self).__init__()
        self.head = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 2048, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(2048),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(2048, 1000)
        )

    def forward(self,features):
        return self.head(features)
