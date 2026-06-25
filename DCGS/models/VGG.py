import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.registry import register_model

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

class ClassifierHead(nn.Module):
    def __init__(self, in_features, num_classes, drop_rate=0.0):
        super(ClassifierHead, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1) 
        self.drop = nn.Dropout(drop_rate)          
        self.fc = nn.Linear(in_features, num_classes) 

    def forward(self, x):
        x = self.global_pool(x).view(x.size(0), -1) 
        x = self.drop(x)
        return self.fc(x)

    
class ConvMlp(nn.Module):
    def __init__(self, in_features=512, out_features=4096, kernel_size=7, mlp_ratio=1.0, drop_rate=0.5):
        super(ConvMlp, self).__init__()
        mid_features = int(out_features * mlp_ratio)
        self.fc1 = nn.Conv2d(in_features, mid_features, kernel_size=kernel_size, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.drop1 = nn.Dropout(drop_rate)
        self.fc2 = nn.Conv2d(mid_features, out_features, kernel_size=1, bias=True)
        self.act2 = nn.ReLU(inplace=True)
        self.drop2 = nn.Dropout(drop_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.drop2(x)
        return x


class VGG(nn.Module):
    def __init__(self, cfg_name, num_classes=1000, dropout=0.5, use_bn=True):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[cfg_name], use_bn)
        self.pre_logits = ConvMlp(512, 4096, kernel_size=7, drop_rate=dropout)
        self.head = ClassifierHead(4096, num_classes)
        self._initialize_weights()

    def _make_layers(self, cfg, use_bn):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(nn.Conv2d(in_channels, v, kernel_size=3, padding=1))
                if use_bn:
                    layers.append(nn.BatchNorm2d(v))
                layers.append(nn.ReLU(inplace=True))
                in_channels = v
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.pre_logits(x)
        x = self.head(x)
        return x



def vgg16(num_classes=1000):
    return VGG('VGG16',num_classes=num_classes)

def vgg16_bn(num_classes=1000):
    return VGG('VGG16',num_classes=num_classes, use_bn=True)

def vgg19(num_classes=1000):
    return VGG('VGG19',num_classes=num_classes)

def vgg19_bn(num_classes=1000):
    return VGG('VGG19',num_classes=num_classes, use_bn=True)
