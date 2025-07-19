# cython:language_level=39
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

########################### Guo ######################
class make_conv_bn_relu(nn.Module):
    # 
    def __init__(self, input_dim, output_dim, ks, stride, drate):
        super(make_conv_bn_relu, self).__init__()
        
        self.npadw  = (ks[0]//2)*drate                                      #  奇数可以
        self.npadh  = (ks[1]//2)*drate                                      #  奇数可以
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size = ks, \
                      stride=stride, padding = [self.npadw, self.npadh], dilation = drate, padding_mode = 'replicate'),
            nn.BatchNorm2d(output_dim, momentum=0.01),
            nn.LeakyReLU(negative_slope=0.1)         
            )

    def forward(self, inputs):        
        return self.conv_block(inputs)

class ASPP_block(nn.Module):
    # 
    def __init__(self, input_dim, output_dim):
        super(ASPP_block, self).__init__()
        
        self.conv_initial = make_conv_bn_relu(input_dim, output_dim, (1, 1), 1, 1 ) 
        
        self.conv2    = make_conv_bn_relu(output_dim, output_dim//4, (1,1), 1, 1)   
        
        self.conv31   = make_conv_bn_relu(output_dim//4, output_dim//4, (1,3), 1, 1)
        self.conv32   = make_conv_bn_relu(output_dim//4, output_dim//4, (1,3), 1, 2)
        self.conv33   = make_conv_bn_relu(output_dim//4, output_dim//4, (1,3), 1, 5)
        self.conv34   = make_conv_bn_relu(output_dim//4, output_dim//4, (1,3), 1, 8)
        
        self.conv4    = nn.Conv2d(5*output_dim//4, output_dim, kernel_size=(1, 1), \
                                  stride = 1, dilation = 1)  # no padding        
        self.relu     = nn.ReLU()         
        
    def forward(self, inputs):
        shape_x = inputs.size()
        
        out1       = self.conv_initial (inputs)
        out2       = self.conv2 (out1) 
        
        out31      = self.conv31(out2)
        out32      = self.conv32(out2)
        out33      = self.conv33(out2)
        out34      = self.conv34(out2)
        
        out312     = out31 + out32  
        out3123    = out312 + out33   
        out31234   = out3123 + out34        
        
        globalpool = torch.mean(out2, dim = (2, 3), keepdims = True)                  
        globalpool = torch.tile(globalpool, (1, 1, shape_x[2],shape_x[3]))                
                    
        out3       = torch.cat([out31, out312, out3123, out31234, globalpool], dim=1)                
        out4       = self.conv4(out3)       
        # print(out1.size())
        # print(out4.size())
        out        = self.relu(out1+out4)
        return out      
    
class GRU_block(nn.Module):
    # 
    def __init__(self, input_dim, output_dim):
        super(GRU_block, self).__init__()
        
        self.conv_initial = make_conv_bn_relu(input_dim, output_dim, (1, 1), 1, 1 )     
        self.gru_op1      = nn.GRU(output_dim, output_dim, num_layers=2, bias=True, dropout=0.1,  bidirectional=True)
        self.gru_op2      = nn.GRU(output_dim*2, output_dim, num_layers=2, bias=True, dropout=0.1,  bidirectional=True)
                
    def forward(self, inputs):        
        in_node           = self.conv_initial(inputs)                         # B, f, H, W
        
        B, f, H, W        = in_node.size()        
        in_node_rs        = in_node.reshape(f,H,W).permute(2, 1, 0)           # W,  H, 2*f 
        
        out1,_            = self.gru_op1(in_node_rs)
        out2,h            = self.gru_op2(out1)
        
        output            = out2.permute(2, 1, 0).unsqueeze(0)
        
        return output, h      
    
class DCONV_block(nn.Module):
    # 
    def __init__(self, input_dim, output_dim):
        super(DCONV_block, self).__init__()       
        
        self.conv_initial = make_conv_bn_relu(input_dim, output_dim, (1, 1), 1, 1 )     
        self.upconv1      = nn.Sequential(nn.ConvTranspose2d(output_dim, output_dim, 
                                         kernel_size = (1, 5), stride=1, padding = (0, 2), output_padding=0),   # 3-0-1
                                    nn.BatchNorm2d(output_dim, momentum=0.01),
                                    nn.LeakyReLU(negative_slope=0.1)  
                                    )       
        self.upconv2      = nn.Sequential(nn.ConvTranspose2d(output_dim, output_dim//2, 
                                         kernel_size = (1, 5), stride=1, padding = (0, 2), output_padding=0),   # 3-0-1
                                    nn.BatchNorm2d(output_dim//2, momentum=0.01),
                                    nn.LeakyReLU(negative_slope=0.1)  
                                    )          
        self.pred         = nn.Conv2d(output_dim//2, 1, kernel_size=(1, 5), padding = (0, 2))  # no padding 
        
    def forward(self, inputs):
        
        in_node           = self.conv_initial(inputs)                         # B, f, H, W        
        out1              = self.upconv1(in_node)
        out2              = self.upconv2(out1)
        pred              = self.pred(out2)
        
        return pred    

class CG_net(nn.Module):    
    #
    def __init__(self, input_dim, ufeatures):
        super(CG_net, self).__init__()
        #self.channel_size = channel_size
        self.cnn_encoder   = ASPP_block(input_dim, ufeatures)
        self.gru_encoder   = GRU_block(input_dim, ufeatures)
        self.dconv_decoder = DCONV_block(3*ufeatures, ufeatures)
        self.sp            = nn.Tanh()
       
    def forward(self, inputs):
#         size_x        = inputs.size()

        inputs_rs          = inputs.permute(0,1,3,2)
        cnn_out            = self.cnn_encoder(inputs_rs)
        gru_out,h          = self.gru_encoder(inputs_rs)
        
        cat_out            = torch.cat([cnn_out, gru_out], dim=1)
        
        pred               = self.dconv_decoder(cat_out)
        pred               = 1.1*self.sp(pred)
        pred               = pred.permute(0,1,3,2)
        return   pred

def tv_loss(x): # TV约束，可以在噪音干扰情况下改善反演结果
    dh = torch.abs(x[..., :, 1:] - x[..., :, :-1])
    dw = torch.abs(x[..., 1:, :] - x[..., :-1, :])
    return torch.mean(2*dh[..., :-1, :] + dw[..., :, :-1]) #空间方程乘以2，是为了使空间平滑，根据测试情况，可以删除

class DIPLoss(nn.Module):
    '''计算匹配损失'''
    def __init__(self, device, hw):
        super(DIPLoss, self).__init__()
        self.hw           = hw
        self.pool_initial = nn.AvgPool2d(kernel_size =(3, 3), stride = 1, padding=0)
        self.sobel        = torch.tensor([1.0/12.0,   -8.0/12.0,   0.0,   8.0/12.0,  -1.0/12.0], dtype=torch.float32).to(device) 
        
        W_operator        = torch.tensor(np.exp(-4.5*np.square(np.linspace(-hw, hw, 2*hw+1))/hw/hw), dtype=torch.float32).to(device) 
        self.W_operator   = torch.reshape(W_operator,(1,1,1,2*hw+1))    
        
        G_operator        = torch.tensor([[[[0.05]],[[0.2]], [[1.0]], [[0.2]],[[0.05]]]], dtype=torch.float32).to(device) 
        self.G_operator   = torch.reshape(G_operator,(1,1,5,1))    
        
    def forward(self, x, pred):          #   (BS, 1, nx, H ) 
        ''' 初平滑 '''

        x_rs              = x.permute(0,1,3,2)
        pred              = pred.permute(0,1,3,2)
        inputs_e          = F.pad(x_rs, (1,1,1,1), mode= 'replicate')
        in_node           = self.pool_initial(inputs_e)   

        '''先求梯度  sobel算子组合实现'''                                
        sobel_xe          = torch.reshape( self.sobel  ,(1,1,5,1))   # W
        sobel_ye          = torch.reshape( self.sobel  ,(1,1,1,5))   # H
        
        in_node_ex        = F.pad(in_node, (0,0,2,2), mode= 'replicate')
        in_node_ey        = F.pad(in_node, (2,2,0,0), mode= 'replicate')
    
        x_gx              = F.conv2d(in_node_ex, sobel_xe, stride =1, padding = [0,0])
        x_gy              = F.conv2d(in_node_ey, sobel_ye, stride =1, padding = [0,0])                                  
        #   高斯平滑获得张量  
        #   H 方向 t方向
    
        x_gx_e            = F.pad(x_gx, (self.hw,self.hw,0,0), mode= 'replicate')
        x_gy_e            = F.pad(x_gy, (self.hw,self.hw,0,0), mode= 'replicate')        
            
        x_sxx = F.conv2d(x_gx_e*x_gx_e, self.W_operator, stride=1, padding= [0, 0])    
        x_syy = F.conv2d(x_gy_e*x_gy_e, self.W_operator, stride=1, padding= [0, 0])
        x_sxy = F.conv2d(x_gx_e*x_gy_e, self.W_operator, stride=1, padding= [0, 0])
        
        #   W 方向 offset 方向  
        x_sxx_e           = F.pad(x_sxx, (0,0,2,2), mode= 'replicate')
        x_syy_e           = F.pad(x_syy, (0,0,2,2), mode= 'replicate')
        x_sxy_e           = F.pad(x_sxy, (0,0,2,2), mode= 'replicate')
    
        x_gxx    = F.conv2d(x_sxx_e, self.G_operator, stride =1, padding=[0,0])  
        x_gyy    = F.conv2d(x_syy_e, self.G_operator, stride =1, padding=[0,0])  
        x_gxy    = F.conv2d(x_sxy_e, self.G_operator, stride =1, padding=[0,0])             
        # 结构张量
        gxx,  gxy,  gyy   = x_gxx, x_gxy, x_gyy                        

        d       = 0.5*(gxx+gyy)                                                   # lamda1+lamda2
        e       = 0.5*torch.sqrt(torch.square(gxx-gyy)+4.0*torch.square(gxy))     # lamda1-lamda2
        
        C_an     = torch.divide(e,d)
    
        lamda1, lamda2 = d+e, d-e
        uxe, uye = -gxy, gxx-lamda1
        vxe, vye = -gxy, gxx-lamda2
        
        slope    = -uxe/(uye+1e-6)
        #
        pred_ex        = F.pad(pred, (0,0,2,2), mode= 'replicate')
        pred_ey        = F.pad(pred, (2,2,0,0), mode= 'replicate')
    
        p_gx           = F.conv2d(pred_ex, sobel_xe, stride =1, padding = [0,0])
        p_gy           = F.conv2d(pred_ey, sobel_ye, stride =1, padding = [0,0])   
        
        norm           = torch.sqrt(p_gx*p_gx+p_gy*p_gy)
        
        p_gx           = torch.div(p_gx, norm+1e-6)
        p_gy           = torch.div(p_gy, norm+1e-6)
            
        sina      = torch.abs( p_gx * uye  - p_gy* uxe)
        sina_loss = sina.mean()  
        
        return sina_loss, slope

class MAELoss(nn.Module):
    # 重建损失
    def __init__(self):
        super(MAELoss, self).__init__()         
        
    def forward(self, vseis, vrestore, masks = None):
        # loss_restore = torch.abs(torch.sign(vseis)*torch.sqrt(torch.abs(vseis)+0.00001)-torch.sign(vrestore)*torch.sqrt(torch.abs(vrestore)+0.00001)).mean()
        loss_restore = torch.abs(vseis-vrestore).mean()
        #print(loss_restore)
        return loss_restore


########################### Chen #####################
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

        x1 = 1.2* torch.tanh(self.outc1(x.permute(0,2,3,1)).permute(0,3,1,2))
        # x2 = torch.sigmoid(self.outc2(x.permute(0,2,3,1)).permute(0,3,1,2))
        # x3 = torch.sigmoid(self.outc3(x.permute(0,2,3,1)).permute(0,3,1,2))


        # x = self.outc(x)
        # x1 = torch.tanh(x[:,0:1])
        # x3 = torch.sigmoid(x[:,2:3])
        # x2 = torch.sigmoid(x[:,1:2])
        # return torch.cat(x1, x2, x3],1)
        return x1

class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=(9,3), use_norm=True):
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


class forward_model(nn.Module):
    def __init__(self, resolution_ratio=1, nonlinearity="tanh"):
        super(forward_model, self).__init__()
        self.resolution_ratio = resolution_ratio
        self.activation =  nn.ReLU() if nonlinearity=="relu" else nn.Tanh()
        self.cnn = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(5,1), padding='same'),
                                 self.activation,
                                 nn.Conv2d(in_channels=4, out_channels=4, kernel_size=(5,1), padding='same'),
                                 self.activation,
                                 nn.Conv2d(in_channels=4, out_channels=1, kernel_size=(5,1), padding='same'))

    def forward(self, y, x0):
        x = self.cnn(x0)+x0
        #x = self.outc(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) + x0
        # x = self.wavelet(x)
        y = F.conv2d(y, torch.flip(x, dims=[2]), stride=self.resolution_ratio, padding='same')
        return y, x






