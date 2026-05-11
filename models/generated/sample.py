import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.residual_add = nn.quantized.FloatFunctional()
        
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.ReLU(inplace=True)
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        residual = self.downsample(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.residual_add.add(x, residual)
        x = self.relu2(x)
        return x

class AttentionBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.mul = nn.quantized.FloatFunctional()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y)
        return self.mul.mul(x, y)

class ImprovedCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2)
        
        self.res1 = ResidualBlock(16, 16)
        self.att1 = AttentionBlock(16)
        self.pool2 = nn.MaxPool2d(2)
        
        self.res2 = ResidualBlock(16, 32, stride=2)
        self.att2 = AttentionBlock(32)
        self.pool3 = nn.MaxPool2d(2)
        
        self.res3 = ResidualBlock(32, 32)
        self.att3 = AttentionBlock(32)
        self.pool4 = nn.AdaptiveAvgPool2d(1)
        
        self.fc = nn.Linear(32, 10)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.pool1(x)
        x = self.res1(x)
        x = self.att1(x)
        x = self.pool2(x)
        x = self.res2(x)
        x = self.att2(x)
        x = self.pool3(x)
        x = self.res3(x)
        x = self.att3(x)
        x = self.pool4(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x