import torch
from torch import nn

#卷积进行了小改，增添padding，使得图像大小不变，方便concat
class convBlock(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(convBlock, self).__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class up_conv(nn.Module):
    def __init__(self,in_channel,out_channel,bilinear=True):
        super(up_conv, self).__init__()
        if bilinear:
            self.up=nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=3,stride=1,padding=1)
            )
        else:
            self.up=nn.ConvTranspose2d(in_channels=in_channel,out_channels=out_channel,kernel_size=2,
                                       stride=2,padding=0)

    def forward(self, x):
        return self.up(x)

class U_Net(nn.Module):
    def __init__(self,in_channel,out_channel,bilinear=True):
        super(U_Net, self).__init__()

        self.max_pool=nn.MaxPool2d(kernel_size=2,stride=2)

        self.conv1=convBlock(in_channel,64)
        self.conv2=convBlock(64,128)
        self.conv3=convBlock(128,256)
        self.conv4=convBlock(256,512)
        self.conv5=convBlock(512,1024)

        self.up1=up_conv(1024,512,bilinear)
        self.upconv1=convBlock(1024,512)
        self.up2 = up_conv(512, 256,bilinear)
        self.upconv2 = convBlock(512, 256)
        self.up3 = up_conv(256, 128,bilinear)
        self.upconv3 = convBlock(256, 128)
        self.up4 = up_conv(128, 64,bilinear)
        self.upconv4 = convBlock(128, 64)

        self.final_conv=nn.Conv2d(in_channels=64,out_channels=out_channel,kernel_size=1,stride=1,padding=0)

    def forward(self, x):
        x1=self.conv1(x)

        x2=self.max_pool(x1)
        x2=self.conv2(x2)

        x3=self.max_pool(x2)
        x3=self.conv3(x3)

        x4=self.max_pool(x3)
        x4=self.conv4(x4)

        x5=self.max_pool(x4)
        x5=self.conv5(x5)

        d1=self.up1(x5)
        d1=torch.cat((x4,d1),dim=1)
        d1=self.upconv1(d1)

        d2=self.up2(d1)
        d2=torch.cat((x3,d2),dim=1)
        d2=self.upconv2(d2)

        d3=self.up3(d2)
        d3=torch.cat((x2,d3),dim=1)
        d3=self.upconv3(d3)

        d4=self.up4(d3)
        d4=torch.cat((x1,d4),dim=1)
        d4=self.upconv4(d4)

        d5=self.final_conv(d4)
        return d5

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m,nn.ConvTranspose2d):
                torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0, 0.01)
                m.bias.data.zero_()


