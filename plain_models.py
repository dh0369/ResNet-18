import torch
import torch.nn as nn

class Plain(nn.Module):
    def __init__(self, num_blocks, num_classes=10):
        super(Plain, self).__init__()
        self.in_channels = 64

        self.base = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer1 = self._make_layer(64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(512, num_blocks[3], stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
    
    def _make_layer(self, out_channels, num_blocks, stride):
        layers = []
        for _ in range(num_blocks):
            layers.append(nn.Conv2d(self.in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.base(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)

        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out

def Plain18():
    return Plain([2, 2, 2, 2])

def Plain34():
    return Plain([3, 4, 6, 3])