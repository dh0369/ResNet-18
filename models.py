import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    extension = 1
    def __init__(self, in_channels, out_channels, stride=1, downsample = None):
        super(ResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv_block = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(self.out_channels)
        )

        self.downsample = downsample
    
    def forward(self, x):
        identity = x
        out = self.conv_block(x)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out += identity
        out = F.relu(out)
        return out

class BottleNeck(nn.Module):
    extension = 4
    def __init__(self, in_channels, out_channels, stride=1, downsample = None):
        super(BottleNeck, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv_block = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(),
            nn.Conv2d(self.out_channels, self.out_channels * 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.out_channels * 4)
        )

        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv_block(x)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out += identity
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.base = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.conv2_x = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.conv3_x = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.conv4_x = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.conv5_x = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1)) 
        # self.avg_pool = nn.AvgPool2d(4)
        self.fc = nn.Linear(512 * block.extension, num_classes)
    
    def _make_layer(self, block, out_channels, num_blocks, stride):
        downsample = None
        layers = []

        if stride != 1 or self.in_channels != out_channels * block.extension:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.extension, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.extension)
            )

        layers.append(block(self.in_channels, out_channels, stride=stride, downsample=downsample))
        self.in_channels = out_channels * block.extension

        for _ in range(num_blocks-1):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.base(x)
        out = self.conv2_x(out)
        out = self.conv3_x(out)
        out = self.conv4_x(out)
        out = self.conv5_x(out)
        out = self.avg_pool(out)

        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out

def ResNet18():
    return ResNet(ResidualBlock, [2, 2, 2, 2])

def ResNet34():
    return ResNet(ResidualBlock, [3, 4, 6, 3])

def ResNet50():
    return ResNet(BottleNeck, [3, 4, 6, 3])

def ResNet101():
    return ResNet(BottleNeck, [3, 4, 23, 3])

def ResNet152():
    return ResNet(BottleNeck, [3, 8, 36, 3])