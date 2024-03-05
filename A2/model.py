import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        # Layer 1: Convolutional layer with 8 filters applying 7x7 filters on the input image.
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=7, stride=1, padding=0, bias=False)
        self.norm1 = nn.LayerNorm((8, 106, 106))
        self.leaky_relu1 = nn.LeakyReLU(0.01)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Layer 2: Depthwise+pointwise convolution layer --> depthwise separable convolution
        self.depthwise_conv1 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=7, stride=2, padding=0, groups=8, bias=False)
        self.norm2 = nn.LayerNorm((8, 24, 24))
        self.leaky_relu2 = nn.LeakyReLU(0.01)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pointwise_conv1 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=1, stride=1, padding=0, bias=False)

        # Layer 4: Depthwise+pointwise convolution layer --> depthwise separable convolution
        self.depthwise_conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=7, stride=1, padding=0, groups=16, bias=False)
        self.norm3 = nn.LayerNorm((16, 6, 6))
        self.leaky_relu3 = nn.LeakyReLU(0.01)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pointwise_conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=1, stride=1, padding=0, bias=False)

        # Layer 6: Fully connected layer implemented as convolutional layer
        self.fc = nn.Conv2d(in_channels=32, out_channels=num_classes, kernel_size=3, stride=1, padding=0, bias=True)

        # Initialize weights
        self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.leaky_relu1(x)
        x = self.maxpool1(x)

        x = self.depthwise_conv1(x)
        x = self.norm2(x)
        x = self.leaky_relu2(x)
        x = self.maxpool2(x)
        x = self.pointwise_conv1(x)

        x = self.depthwise_conv2(x)
        x = self.norm3(x)
        x = self.leaky_relu3(x)
        x = self.maxpool3(x)
        x = self.pointwise_conv2(x)
        print(x.shape)

        out = self.fc(x)
        
        return out
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)