import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

########################### GUNet ######################
class GUNet(nn.Module):
    def __init__(self, in_ch, out_ch, channels, skip_channels,
                 use_sigmoid=True, use_norm=True):
        super(GUNet, self).__init__()
        assert (len(channels) == len(skip_channels))
        self.scales = len(channels)
        self.use_sigmoid = use_sigmoid
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        self.inc = InBlock(in_ch, channels[0],use_norm=use_norm)
        for i in range(1, self.scales):
            self.down.append(DownBlock(in_ch=channels[i - 1],
                                       out_ch=channels[i],
                                       use_norm=use_norm))
        for i in range(1, self.scales):
            self.up.append(UUpBlock(in_ch=channels[-i],
                                   out_ch=channels[-i - 1],
                                   skip_ch=skip_channels[-i],
                                   use_norm=use_norm))
        self.outc = OutBlock(in_ch=channels[0],
                             out_ch=out_ch)
        # self.gf = GuidedFilter(r=2, eps=1e-4)

    def forward(self, x0, xref):
        xsup=[]
        xs = [self.inc(x0), ]
        for i in range(self.scales - 1):
            Img = F.upsample(xref, size=(xs[-1].shape[2], xs[-1].shape[3]), mode='bilinear')
            x = self.gf(Img,xs[-1])
            xs.append(self.down[i](x))

        x = xs[-1]
        Img_1 = F.upsample(xref, size=(x.shape[2], x.shape[3]), mode='bilinear')
        x = self.gf(Img_1,x)

        for i in range(self.scales - 1):
            x = self.up[i](x, xs[-2 - i])
            xsup.append(x)

        x = self.outc(x)
        x1 = torch.tanh(x[:,0:1])
        x3 = torch.sigmoid(x[:,2:3])
        x2 = torch.sigmoid(x[:,1:2])
        return torch.cat([x1, x2, x3],1)

########################### UNet ######################

class UNet(nn.Module):
    def __init__(self, in_ch, out_ch, channels, skip_channels,
                 use_sigmoid=True, use_norm=True):
        super(UNet, self).__init__()
        assert (len(channels) == len(skip_channels))
        self.scales = len(channels)
        self.use_sigmoid = use_sigmoid
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        self.inc = InBlock(in_ch, channels[0],use_norm=use_norm)
        for i in range(1, self.scales):
            self.down.append(DownBlock(in_ch=channels[i - 1],
                                       out_ch=channels[i],
                                       use_norm=use_norm))
        for i in range(1, self.scales):
            self.up.append(UUpBlock(in_ch=channels[-i],
                                   out_ch=channels[-i - 1],
                                   skip_ch=skip_channels[-i],
                                   use_norm=use_norm))
        # self.outc = OutBlock(in_ch=channels[0],
        #                      out_ch=out_ch)
        self.outc1 = nn.Linear(channels[0], 1)
        self.outc2 = nn.Linear(channels[0], 1)
        self.outc3 = nn.Linear(channels[0], 1)

    def forward(self, x0):
        xsup=[]
        xs = [self.inc(x0), ]
        for i in range(self.scales - 1):
            xs.append(self.down[i](xs[-1]))
        x = xs[-1]
        for i in range(self.scales - 1):
            x = self.up[i](x, xs[-2 - i])
            xsup.append(x)

        x1 = torch.tanh(self.outc1(x.permute(0,2,3,1)).permute(0,3,1,2))
        # x2 = torch.sigmoid(self.outc2(x.permute(0,2,3,1)).permute(0,3,1,2))
        # x3 = torch.sigmoid(self.outc3(x.permute(0,2,3,1)).permute(0,3,1,2))


        # x = self.outc(x)
        # x1 = torch.tanh(x[:,0:1])
        # x3 = torch.sigmoid(x[:,2:3])
        # x2 = torch.sigmoid(x[:,1:2])
        # return torch.cat(x1, x2, x3],1)
        return x1
########################### UNet-multitask ######################
# class UNet_multask(nn.Module):
#     def __init__(self, in_ch, out_ch, channels, skip_channels,
#                  use_sigmoid=True, use_norm=True):
#         super(UNet_multask, self).__init__()
#         assert (len(channels) == len(skip_channels))
#         self.scales = len(channels)
#         self.use_sigmoid = use_sigmoid
#         self.down = nn.ModuleList()
#         self.up = nn.ModuleList()
#         self.up2 = nn.ModuleList()
#         self.up3 = nn.ModuleList()
#         self.inc = InBlock(in_ch, channels[0],use_norm=use_norm)
#         for i in range(1, self.scales):
#             self.down.append(DownBlock(in_ch=channels[i - 1],
#                                        out_ch=channels[i],
#                                        use_norm=use_norm))
#         for i in range(1, self.scales):
#             self.up.append(AUpBlock(in_ch=channels[-i],
#                                    out_ch=channels[-i - 1],
#                                    skip_ch=skip_channels[-i],
#                                    use_norm=use_norm))
#         self.outc = OutBlock(in_ch=channels[0],
#                              out_ch=out_ch)
#
#         for i in range(1, self.scales):
#             self.up2.append(AUpBlock(in_ch=channels[-i],
#                                    out_ch=channels[-i - 1],
#                                    skip_ch=skip_channels[-i],
#                                    use_norm=use_norm))
#         self.outc2 = OutBlock(in_ch=channels[0],
#                              out_ch=out_ch)
#
#         for i in range(1, self.scales):
#             self.up3.append(AUpBlock(in_ch=channels[-i],
#                                    out_ch=channels[-i - 1],
#                                    skip_ch=skip_channels[-i],
#                                    use_norm=use_norm))
#         self.outc3 = OutBlock(in_ch=channels[0],
#                              out_ch=out_ch)
#
#     def forward(self, x0):
#         xs = [self.inc(x0), ]
#         for i in range(self.scales - 1):
#             xs.append(self.down[i](xs[-1]))
#
#         x = xs[-1]
#         for i in range(self.scales - 1):
#             x = self.up[i](x, xs[-2 - i])
#         x1 = torch.tanh(self.outc(x))
#
#         x = xs[-1]
#         for i in range(self.scales - 1):
#             x = self.up2[i](x, xs[-2 - i])
#         x2 = torch.sigmoid(self.outc2(x))
#
#         x = xs[-1]
#         for i in range(self.scales - 1):
#             x = self.up3[i](x, xs[-2 - i])
#         x3 = torch.sigmoid(self.outc3(x))
#
#         x = torch.cat([x1,x2,x3],1)
#         return x


class UNet_multask(nn.Module):
    def __init__(self, in_ch, out_ch, channels, skip_channels,
                 use_sigmoid=True, use_norm=True):
        super(UNet_multask, self).__init__()
        assert (len(channels) == len(skip_channels))
        self.scales = len(channels)
        self.use_sigmoid = use_sigmoid
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        self.up2 = nn.ModuleList()
        self.up3 = nn.ModuleList()
        self.inc = InBlock(in_ch, channels[0], use_norm=use_norm)
        for i in range(1, self.scales):
            self.down.append(DownBlock(in_ch=channels[i - 1],
                                       out_ch=channels[i],
                                       use_norm=use_norm))
        for i in range(1, self.scales):
            self.up.append(UUpBlock(in_ch=channels[-i],
                                    out_ch=channels[-i - 1],
                                    skip_ch=skip_channels[-i],
                                    use_norm=use_norm))
        self.outc = OutBlock(in_ch=channels[0],
                             out_ch=out_ch+2)

        skip_channels = [1,1,1,1]
        for i in range(1, self.scales):
            self.up2.append(UpBlock(in_ch=channels[-i],
                                     out_ch=channels[-i - 1],
                                     skip_ch=skip_channels[-i],
                                     use_norm=use_norm))
        self.outc2 = OutBlock(in_ch=channels[0],
                              out_ch=out_ch)


    def forward(self, x0):
        xs = [self.inc(x0), ]
        for i in range(self.scales - 1):
            xs.append(self.down[i](xs[-1]))

        x = xs[-1]
        for i in range(self.scales - 1):
            x = self.up[i](x, xs[-2 - i])
        x = self.outc(x)
        x1 = torch.tanh(x[:,0:1])
        x2 = torch.sigmoid(x[:,1:2])
        x3 = torch.sigmoid(x[:,2:3])

        x = xs[-1]
        for i in range(self.scales - 1):
            x = self.up2[i](x, xs[-2 - i])
        x4 = self.outc2(x)
        x4 = torch.sigmoid(x4)

        x = torch.cat([x1, x2, x3, x4], 1)

        return x


########################### Attention UNet ######################
class AUNet(nn.Module):
    def __init__(self, in_ch, out_ch, channels, skip_channels,
                 use_sigmoid=True, use_norm=True):
        super(AUNet, self).__init__()
        assert (len(channels) == len(skip_channels))
        self.scales = len(channels)
        self.use_sigmoid = use_sigmoid
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        self.inc = InBlock(in_ch, channels[0], use_norm=use_norm)
        for i in range(1, self.scales):
            self.down.append(DownBlock(in_ch=channels[i - 1],
                                       out_ch=channels[i],
                                       use_norm=use_norm))
        for i in range(1, self.scales):
            self.up.append(AUpBlock(in_ch=channels[-i],
                                    out_ch=channels[-i - 1],
                                    skip_ch=skip_channels[-i],
                                    use_norm=use_norm))
        self.outc = OutBlock(in_ch=channels[0],
                             out_ch=out_ch)

    def forward(self, x0):
        xs = [self.inc(x0), ]
        for i in range(self.scales - 1):
            xs.append(self.down[i](xs[-1]))
        x = xs[-1]
        # import pdb;pdb.set_trace()
        for i in range(self.scales - 1):
            x = self.up[i](x, xs[-2 - i])

        x = self.outc(x)
        x1 = torch.tanh(x[:, 0:1])
        x3 = torch.sigmoid(x[:, 2:3])
        x2 = torch.sigmoid(x[:, 1:2])
        return torch.cat([x1, x2, x3], 1)


#################################################################################################
class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=(3,3), use_norm=True):
        super(DownBlock, self).__init__()
        to_pad = (int((kernel_size[1] - 1) / 2), int((kernel_size[1] - 1) / 2), int((kernel_size[0] - 1) / 2), int((kernel_size[0] - 1) / 2))
        if use_norm:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size,
                          stride=2),
                # nn.ZeroPad2d(to_pad),  
                nn.ReflectionPad2d(to_pad),                 
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True),
                # nn.ReLU(),
                nn.Conv2d(out_ch, out_ch, kernel_size,
                          stride=1),
                # nn.ZeroPad2d(to_pad), 
                nn.ReflectionPad2d(to_pad),                
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True)
                # nn.ReLU()
                )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size,
                          stride=2),
                # nn.ZeroPad2d(to_pad),  
                nn.ReflectionPad2d(to_pad),                
                nn.LeakyReLU(0.2, inplace=True),
                # nn.ReLU(),
                nn.Conv2d(out_ch, out_ch, kernel_size,
                          stride=1),
                # nn.ZeroPad2d(to_pad), 
                nn.ReflectionPad2d(to_pad),                
                nn.LeakyReLU(0.2, inplace=True)
                # nn.ReLU()
                )

    def forward(self, x):
        x = self.conv(x)
        return x


class InBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=(3,3), use_norm=True):
        super(InBlock, self).__init__()
        to_pad = (int((kernel_size[1] - 1) / 2), int((kernel_size[1] - 1) / 2), int((kernel_size[0] - 1) / 2), int((kernel_size[0] - 1) / 2))
        if use_norm:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size,
                          stride=1),
                nn.ReflectionPad2d(to_pad),                
                nn.BatchNorm2d(out_ch),            
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size,
                          stride=1),
                nn.ReflectionPad2d(to_pad),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True)
                )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size,
                          stride=1),
                nn.ReflectionPad2d(to_pad),                
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size,
                          stride=1),
                nn.ReflectionPad2d(to_pad),
                nn.LeakyReLU(0.2, inplace=True)
                )

    def forward(self, x):
        x = self.conv(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, skip_ch=4, kernel_size=(3,3), use_norm=True):
        super(UpBlock, self).__init__()
        to_pad = (int((kernel_size[1] - 1) / 2), int((kernel_size[1] - 1) / 2), int((kernel_size[0] - 1) / 2), int((kernel_size[0] - 1) / 2))
        self.skip = skip_ch > 0
        if skip_ch == 0:
            skip_ch = 1
        if use_norm:
            self.conv = nn.Sequential(
                nn.BatchNorm2d(in_ch + skip_ch),
                nn.Conv2d(in_ch + skip_ch, out_ch, kernel_size, stride=1),
                # nn.ZeroPad2d(to_pad),
                nn.ReflectionPad2d(to_pad),                
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True),
                # nn.ReLU(),
                nn.Conv2d(out_ch, out_ch, kernel_size,
                          stride=1),
                # nn.ZeroPad2d(to_pad),  
                nn.ReflectionPad2d(to_pad),                
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True)
                # nn.ReLU()
                )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch + skip_ch, out_ch, kernel_size, stride=1),
                # nn.ZeroPad2d(to_pad),  
                nn.ReflectionPad2d(to_pad),                
                nn.LeakyReLU(0.2, inplace=True),
                # nn.ReLU(),
                nn.Conv2d(out_ch, out_ch, kernel_size,
                          stride=1),
                # nn.ZeroPad2d(to_pad), 
                nn.ReflectionPad2d(to_pad),                
                nn.LeakyReLU(0.2, inplace=True)
                # nn.ReLU()
                )

        if use_norm:
            self.skip_conv = nn.Sequential(
                nn.Conv2d(out_ch, skip_ch, kernel_size=1, stride=1),
                nn.BatchNorm2d(skip_ch),
                nn.LeakyReLU(0.2, inplace=True)
                # nn.ReLU()
                )
        else:
            self.skip_conv = nn.Sequential(
                nn.Conv2d(out_ch, skip_ch, kernel_size=1, stride=1),
                nn.LeakyReLU(0.2, inplace=True)
                # nn.ReLU()
                )

        self.up = nn.Upsample(scale_factor=2, mode='bicubic',
                              align_corners=True)
        self.concat = Concat()

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x2 = self.skip_conv(x2)
        if not self.skip:
            x2 = x2 * 0
        x = self.concat(x1, x2)
        x = self.conv(x)
        return x


class Concat(nn.Module):
    def __init__(self):
        super(Concat, self).__init__()

    def forward(self, *inputs):
        inputs_shapes2 = [x.shape[2] for x in inputs]
        inputs_shapes3 = [x.shape[3] for x in inputs]

        if (np.all(np.array(inputs_shapes2) == min(inputs_shapes2)) and
                np.all(np.array(inputs_shapes3) == min(inputs_shapes3))):
            inputs_ = inputs
        else:
            target_shape2 = min(inputs_shapes2)
            target_shape3 = min(inputs_shapes3)

            inputs_ = []
            for inp in inputs:
                diff2 = (inp.size(2) - target_shape2) // 2
                diff3 = (inp.size(3) - target_shape3) // 2
                inputs_.append(inp[:, :, diff2: diff2 + target_shape2,
                                   diff3:diff3 + target_shape3])
        return torch.cat(inputs_, dim=1)


class OutBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=(1,1)):
        super(OutBlock, self).__init__()
        # to_pad = (int((kernel_size[1] - 1) / 2), int((kernel_size[1] - 1) / 2), int((kernel_size[0] - 1) / 2), int((kernel_size[0] - 1) / 2))
        # self.conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size, stride=1),
        #                           nn.ReflectionPad2d(to_pad))
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=(1,1), stride=1)

    def forward(self, x):
        x = self.conv(x)         
        return x

    def __len__(self):
        return len(self._modules)

class UUpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, skip_ch=4, kernel_size=(3,3), use_norm=True):
        super(UUpBlock, self).__init__()
        to_pad = (int((kernel_size[1] - 1) / 2), int((kernel_size[1] - 1) / 2), int((kernel_size[0] - 1) / 2), int((kernel_size[0] - 1) / 2))
        self.skip = skip_ch > 0
        if skip_ch == 0:
            skip_ch = 1
        if use_norm:
            self.conv = nn.Sequential(
                nn.BatchNorm2d(in_ch + skip_ch),
                nn.Conv2d(in_ch + skip_ch, out_ch, kernel_size, stride=1),
                # nn.ZeroPad2d(to_pad),
                nn.ReflectionPad2d(to_pad),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True),
                # nn.ReLU(),
                nn.Conv2d(out_ch, out_ch, kernel_size,
                          stride=1),
                # nn.ZeroPad2d(to_pad),
                nn.ReflectionPad2d(to_pad),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True)
                # nn.ReLU()
                )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch + skip_ch, out_ch, kernel_size, stride=1),
                # nn.ZeroPad2d(to_pad),
                nn.ReflectionPad2d(to_pad),
                nn.LeakyReLU(0.2, inplace=True),
                # nn.ReLU(),
                nn.Conv2d(out_ch, out_ch, kernel_size,
                          stride=1),
                # nn.ZeroPad2d(to_pad),
                nn.ReflectionPad2d(to_pad),
                nn.LeakyReLU(0.2, inplace=True)
                # nn.ReLU()
                )

        self.up = nn.Upsample(scale_factor=2, mode='bicubic',
                              align_corners=True)
        self.concat = Concat()

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = self.concat(x1, x2)
        x = self.conv(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 定义全局平均池化
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # 定义全局最大池化
        # 定义CBAM中的通道依赖关系学习层，注意这里是使用1x1的卷积实现的，而不是全连接层
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // reduction_ratio, 1, bias=False),
                                nn.LeakyReLU(0.2, inplace=True),
                                nn.Conv2d(in_planes // reduction_ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))  # 实现全局平均池化
        max_out = self.fc(self.max_pool(x))  # 实现全局最大池化
        out = avg_out + max_out  # 两种信息融合
        # 最后利用sigmoid进行赋权
        return self.sigmoid(out) * x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        # 定义7*7的空间依赖关系学习层
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 实现channel维度的平均池化
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 实现channel维度的最大池化
        x1 = torch.cat([avg_out, max_out], dim=1)  # 拼接上述两种操作的到的两个特征图
        x1 = self.conv1(x1)  # 学习空间上的依赖关系
        # 对空间元素进行赋权
        return self.sigmoid(x1) * x


class AUpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, skip_ch=4, kernel_size=(3, 3), use_norm=True):
        super(AUpBlock, self).__init__()
        to_pad = (int((kernel_size[1] - 1) / 2), int((kernel_size[1] - 1) / 2), int((kernel_size[0] - 1) / 2),
                  int((kernel_size[0] - 1) / 2))
        self.skip = skip_ch > 0
        if skip_ch == 0:
            skip_ch = 1
        if use_norm:
            self.conv = nn.Sequential(
                nn.BatchNorm2d(in_ch + skip_ch),
                nn.Conv2d(in_ch + skip_ch, out_ch, kernel_size, stride=1),
                nn.ReflectionPad2d(to_pad),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size,
                          stride=1),
                nn.ReflectionPad2d(to_pad),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch + skip_ch, out_ch, kernel_size, stride=1),
                nn.ReflectionPad2d(to_pad),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size,
                          stride=1),
                nn.ReflectionPad2d(to_pad),
                nn.LeakyReLU(0.2, inplace=True)
            )

        # self.skip_conv = CBAM(out_ch, reduction_ratio=8)
        self.skip_conv = ChannelAttention(out_ch, reduction_ratio=8)
        self.skip_conv1 = SpatialAttention()

        self.up = nn.Upsample(scale_factor=2, mode='bicubic',
                              align_corners=True)
        self.concat = Concat()

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x2 = self.skip_conv(x2)
        x2 = self.skip_conv1(x2)
        x = self.concat(x1, x2)
        x = self.conv(x)
        return x

############################################################# commen CNN #######################################################################

class Net_CNN(nn.Module):
    def __init__(self, n_filters=16, kernel_size=(3,3)):
        super(Net_CNN, self).__init__()
        # net for petrophysical parameters
        self.net = nn.Sequential(
            nn.Conv2d(3, n_filters, kernel_size),
            nn.ReflectionPad2d((1,1,1,1)),
            nn.ReLU(),
            nn.Conv2d(n_filters, n_filters, kernel_size),
            nn.ReflectionPad2d((1,1,1,1)),
            nn.ReLU(),            
            nn.Conv2d(n_filters, n_filters * 2, kernel_size),
            nn.ReflectionPad2d((1,1,1,1)),
            nn.ReLU(),
            nn.Conv2d(n_filters*2, n_filters * 2, kernel_size),
            nn.ReflectionPad2d((1,1,1,1)),
            nn.ReLU(),            
            nn.Conv2d(n_filters * 2, n_filters *4 , kernel_size),
            nn.ReflectionPad2d((1,1,1,1)),
            nn.ReLU(),
            nn.Conv2d(n_filters * 4, n_filters *4 , kernel_size),
            nn.ReflectionPad2d((1,1,1,1)),
            nn.ReLU(),            
            nn.Conv2d(n_filters * 4, n_filters *2 , kernel_size),
            nn.ReflectionPad2d((1,1,1,1)),
            nn.Conv2d(n_filters * 2, n_filters *2 , kernel_size),
            nn.ReflectionPad2d((1,1,1,1)),            
            nn.ReLU(),            
            nn.Conv2d(n_filters * 2, n_filters , kernel_size),
            nn.ReflectionPad2d((1,1,1,1)),
            nn.ReLU(), 
            nn.Conv2d(n_filters, n_filters , kernel_size),
            nn.ReflectionPad2d((1,1,1,1)),
            nn.ReLU(),                        
            nn.Conv2d(n_filters, 3, kernel_size),
            nn.ReflectionPad2d((1,1,1,1)),
            # nn.Sigmoid()
        )


    def forward(self, x):


        x = self.net(x)
        x1 = torch.tanh(x[:,0:1])
        x3 = torch.sigmoid(x[:,2:3])
        x2 = torch.sigmoid(x[:,1:2])
        x = torch.cat([x1, x2, x3], 1)

        return x


class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.ak = nn.Parameter(torch.FloatTensor([-0.63]), requires_grad=True)
        self.bk = nn.Parameter(torch.FloatTensor([0.55]), requires_grad=True)
        self.ck = nn.Parameter(torch.FloatTensor([10]), requires_grad=True)
        # self.ak.data.clamp_(-1.0, -0.3)
        # self.bk.data.clamp_(0.1, 1)

    def forward(self, x):
        y = self.ck*x**2 + self.ak * x + self.bk
        return y
# mymodel = MyModule().cuda()  # 注释掉以支持CPU运行


class forward_model(nn.Module):
    def __init__(self,resolution_ratio=1,nonlinearity="tanh"):
        super(forward_model, self).__init__()
        self.resolution_ratio = resolution_ratio
        self.activation =  nn.ReLU() if nonlinearity=="relu" else nn.Tanh()
        self.cnn = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(3,1), padding='same'),
                                 self.activation,
                                 nn.Conv2d(in_channels=4, out_channels=4,kernel_size=(3,1), padding='same'),
                                 self.activation,
                                 nn.Conv2d(in_channels=4, out_channels=4,kernel_size=(3,1), padding='same'))


        self.wavelet = nn.Conv2d(in_channels=1,
                             out_channels=1,
                             stride=self.resolution_ratio,
                             kernel_size=(50,1),
                             padding='same')

        self.weights = nn.Parameter(torch.randn(1, 1, 50, 1), requires_grad=True)
        self.tanh = nn.Tanh()
        self.outc = nn.Linear(4, 1)

    def forward(self, y, x0):
        x = self.cnn(x0)
        x = self.outc(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) + x0
        # x = self.wavelet(x)
        # 确保x在正确的设备上
        x = x.to(y.device)
        y = F.conv2d(y, torch.flip(x, dims=[2]), stride=self.resolution_ratio, padding='same')
        return y, x






