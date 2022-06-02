import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3), padding=1):
        super(ConvBlock,self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.batchnorm = nn.BatchNorm2d(out_channels, eps=1e-4)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        return x
        
class StackEncoder(nn.Module):
    def __init__(self, channel1, channel2, kernel_size=(3,3), padding=1):
        super(StackEncoder, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block = nn.Sequential(
            ConvBlock(channel1, channel2, kernel_size, padding),
            ConvBlock(channel2, channel2, kernel_size, padding),     
        )
        
    def forward(self,x):
        big_out = self.block(x)
        poolout = self.maxpool(big_out)
        return big_out, poolout
     
class StackDecoder(nn.Module):
    def __init__(self, big_channel, channel1, channel2, kernel_size=(3,3), padding=1):
        super(StackDecoder,self).__init__()
        self.block = nn.Sequential(
            ConvBlock(channel1+big_channel, channel2, kernel_size, padding),
            ConvBlock(channel2, channel2, kernel_size, padding),
            ConvBlock(channel2, channel2, kernel_size, padding),
        )
        
    def forward(self, x, down_tensor):
            _, channels, height, width = down_tensor.size()  
            x = F.interpolate(x, size=(height, width), mode='bilinear')
            x = torch.cat([x, down_tensor], dim=1)  #combining channels of  input from encoder and upsampling input
            x = self.block(x)
            return x

class Unet(nn.Module):
    def __init__(self, in_channel = 3, scale = 'tiny'):
        super(Unet, self).__init__()
        if scale == 'tiny':
            self.num_kernels = [12, 24, 46, 64, 128]
        elif scale == 'large':
            self.num_kernels = [64, 128, 256, 512, 1024]
        self.down1 = StackEncoder(in_channel, self.num_kernels[0], kernel_size=(3,3))  #256
        self.down2 = StackEncoder(self.num_kernels[0], self.num_kernels[1], kernel_size=(3,3))  # 128
        self.down3 = StackEncoder(self.num_kernels[1], self.num_kernels[2], kernel_size=(3,3))  # 64
        self.down4 = StackEncoder(self.num_kernels[2], self.num_kernels[3], kernel_size=(3,3))  # 32
        self.down5 = StackEncoder(self.num_kernels[3], self.num_kernels[4], kernel_size=(3,3))  #16
        
        self.center = ConvBlock(self.num_kernels[4], self.num_kernels[4], kernel_size=(3,3), padding=1) #16
        
        self.up5 = StackDecoder(self.num_kernels[4], self.num_kernels[4], self.num_kernels[3], kernel_size=(3,3))  #32
        self.up4 = StackDecoder(self.num_kernels[3], self.num_kernels[3], self.num_kernels[2], kernel_size=(3,3)) #64
        self.up3 = StackDecoder(self.num_kernels[2], self.num_kernels[2], self.num_kernels[1], kernel_size=(3,3))
        self.up2 = StackDecoder(self.num_kernels[1], self.num_kernels[1], self.num_kernels[0], kernel_size=(3,3))
        self.up1 = StackDecoder(self.num_kernels[0], self.num_kernels[0], self.num_kernels[0], kernel_size=(3,3))
        self.conv = nn.Conv2d(self.num_kernels[0], 1, kernel_size=(1,1), bias=True)
        
    def forward(self,x):
        down1, out = self.down1(x)  
        down2, out = self.down2(out)  
        down3, out = self.down3(out)
        down4, out = self.down4(out)
        down5, out = self.down5(out)
        
        out = self.center(out)
        
        up5 = self.up5(out, down5)
        up4 = self.up4(up5, down4)
        up3 = self.up3(up4, down3)
        up2 = self.up2(up3, down2)
        up1 = self.up1(up2, down1)
        
        out = self.conv(up1)
        return out

def test():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = Unet(in_channel=3).to(device)
    summary(model, (3, 256, 256))

if __name__ == '__main__':
    test()