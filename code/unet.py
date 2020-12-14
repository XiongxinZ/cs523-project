import os
import math as mt
import numpy as np
import PIL.Image as Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

# macros
IMG_WDTH = 512
IMG_HGHT = 512

class TwoDUNet(nn.Module):
    def __init__(self, in_channel=1, nf=16):
        super(TwoDUNet, self).__init__()
        # 1st set of downsampling conv layers
        self.convD_1_1 = nn.Conv2d(in_channel,nf,3, padding=1)
        self.convD_1_2 = nn.Conv2d(nf,nf,3, padding=1)
        #self.BN_1 = nn.BatchNorm2d(nf)

        # 2nd set of downsampling conv layers
        self.convD_2_1 = nn.Conv2d(nf,2*nf,3, padding=1)
        self.convD_2_2 = nn.Conv2d(2*nf,2*nf,3, padding=1)
        #self.BN_2 = nn.BatchNorm2d(2*nf)

        # 3rd set of downsampling conv layers
        self.convD_3_1 = nn.Conv2d(2*nf,4*nf,3, padding=1)
        self.convD_3_2 = nn.Conv2d(4*nf,4*nf,3, padding=1)
        #self.BN_3 = nn.BatchNorm2d(4*nf)

        # 4th set of downsampling conv layers
        self.convD_4_1 = nn.Conv2d(4*nf,8*nf,3, padding=1)
        self.convD_4_2 = nn.Conv2d(8*nf,8*nf,3, padding=1)
        #self.BN_4 = nn.BatchNorm2d(8*nf)
        
        # the bottom set of conv layers
        self.convB_1 = nn.Conv2d(8*nf,16*nf,3, padding=1)
        self.dropout_B = nn.Dropout2d(p=0.3)
        self.convB_2 = nn.Conv2d(16*nf,16*nf,3, padding=1)
        self.convB_3 = nn.Conv2d(16*nf,8*nf,1)
        #self.BN_5 = nn.BatchNorm2d(8*nf)
        
        # 1st set of upsampling conv layers
        #self.BN_6 = nn.BatchNorm2d(256)
        self.convU_1_1 = nn.Conv2d(16*nf,8*nf,3, padding=1)
        self.convU_1_2 = nn.Conv2d(8*nf,8*nf,3, padding=1)
        self.convU_1_3 = nn.Conv2d(8*nf,4*nf,1)
        #self.BN_6_out = nn.BatchNorm2d(64)
        
        # 2nd set of upsampling conv layers
        #self.BN_7 = nn.BatchNorm2d(128)
        self.convU_2_1 = nn.Conv2d(8*nf,4*nf,3, padding=1)
        self.convU_2_2 = nn.Conv2d(4*nf,4*nf,3, padding=1)
        self.convU_2_3 = nn.Conv2d(4*nf,2*nf,1)
        #self.BN_7_out = nn.BatchNorm2d(32)

        # 3rd set of upsampling conv layers
        #self.BN_8 = nn.BatchNorm2d(64)
        self.convU_3_1 = nn.Conv2d(4*nf,2*nf,3, padding=1)
        self.convU_3_2 = nn.Conv2d(2*nf,2*nf,3, padding=1)
        self.convU_3_3 = nn.Conv2d(2*nf,nf,1)
        #self.BN_8_out = nn.BatchNorm2d(16)

        # 4th set of upsampling conv layers
        #self.BN_9 = nn.BatchNorm2d(32)
        self.convU_4_1 = nn.Conv2d(2*nf,nf,3, padding=1)
        self.convU_4_2 = nn.Conv2d(nf,nf,3, padding=1)

        # the last conv layer
        self.conv_final = nn.Conv2d(nf,3,1)
        
        # the sharing layers
        self.dropout = nn.Dropout2d(p=0.2)
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

        self.weights_init()

    def forward(self, x):
        x.data = x.data.type(torch.DoubleTensor)

        x1 = F.relu(self.convD_1_2(self.dropout(F.relu(self.convD_1_1(x)))))
        x2_in = self.maxpool(x1)
        #print(x2_in.shape)

        #x2 = F.relu(self.BN_2(self.convD_2_2(self.dropout(F.relu(self.convD_2_1(x2_in))))))
        x2 = F.relu(self.convD_2_2(self.dropout(F.relu(self.convD_2_1(x2_in)))))
        x3_in = self.maxpool(x2)
        #print(x3_in.shape)

        #x3 = F.relu(self.BN_3(self.convD_3_2(self.dropout(F.relu(self.convD_3_1(x3_in))))))
        x3 = F.relu(self.convD_3_2(self.dropout(F.relu(self.convD_3_1(x3_in)))))
        x4_in = self.maxpool(x3)
        #print(x4_in.shape)

        #x4 = F.relu(self.BN_4(self.convD_4_2(self.dropout(F.relu(self.convD_4_1(x4_in))))))
        x4 = F.relu(self.convD_4_2(self.dropout(F.relu(self.convD_4_1(x4_in)))))
        x5_in = self.maxpool(x4)
        #print(x5_in.shape)

        x5 = F.relu(self.convB_2(self.dropout_B(F.relu(self.convB_1(x5_in)))))
        x6_in = self.upsample(x5)
        #x6_in = self.BN_5(self.convB_3(x6_in))
        x6_in = self.convB_3(x6_in)
        #print(x6_in.shape)
        # crop x4 to fit x6
        start = (x4.size()[2] - x6_in.size()[2]) // 2
        x4 = x4[:, :, start:(start+x6_in.size()[2]), start:(start+x6_in.size()[2])]
        # print 'x4 sizes: '
        # print x4.shape
        # print 'x6 sizes: '
        # print x6_in.shape
        x6_in = torch.cat((x4,x6_in), 1)
        #print(x6_in.shape)
        #x6_in = self.BN_6(x6_in)

        x6 = F.relu(self.convU_1_2(self.dropout(F.relu(self.convU_1_1(x6_in)))))
        #print(x6.shape)
        #x7_in = self.BN_6_out(self.convU_1_3(self.upsample(x6)))
        x7_in = self.convU_1_3(self.upsample(x6))
        #print(x7_in.shape)
        start = (x3.size()[2] - x7_in.size()[2]) // 2
        x3 = x3[:, :, start:(start+x7_in.size()[2]), start:(start+x7_in.size()[2])]
        #x7_in = self.BN_7(torch.cat((x3,x7_in), 1))
        x7_in = torch.cat((x3,x7_in), 1)
        #print(x7_in.shape)

        x7 = F.relu(self.convU_2_2(self.dropout(F.relu(self.convU_2_1(x7_in)))))
        #print(x7.shape)
        #x8_in = self.BN_7_out(self.convU_2_3(self.upsample(x7)))
        x8_in = self.convU_2_3(self.upsample(x7))
        #print(x8_in.shape)
        start = (x2.size()[2] - x8_in.size()[2]) // 2
        x2 = x2[:, :, start:(start+x8_in.size()[2]), start:(start+x8_in.size()[2])]
        #x8_in = self.BN_8(torch.cat((x2,x8_in), 1))
        x8_in = torch.cat((x2,x8_in), 1)
        #print(x8_in.shape)

        x8 = F.relu(self.convU_3_2(self.dropout(F.relu(self.convU_3_1(x8_in)))))
        #print(x8.shape)
        #x9_in = self.BN_8_out(self.convU_3_3(self.upsample(x8)))
        x9_in = self.convU_3_3(self.upsample(x8))
        #print(x9_in.shape)
        start = (x1.size()[2] - x9_in.size()[2]) // 2
        x1 = x1[:, :, start:(start+x9_in.size()[2]), start:(start+x9_in.size()[2])]
        #x9_in = self.BN_9(torch.cat((x1,x9_in), 1))
        x9_in = torch.cat((x1,x9_in), 1)
        #print(x9_in.shape)

        x9 = F.relu(self.convU_4_2(self.dropout(F.relu(self.convU_4_1(x9_in)))))
        #print(x9.shape)
        x9 = self.conv_final(x9)
        #print(x9.shape)
        xout = F.sigmoid(x9)
        #print(xout.shape)

        return xout

    def weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, mt.sqrt(2. / n))
                m.weight.data = m.weight.data.type(torch.DoubleTensor)
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.data = m.bias.data.type(torch.DoubleTensor)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                m.weight.data = m.weight.data.type(torch.DoubleTensor)
                m.bias.data = m.bias.data.type(torch.DoubleTensor)

def test_model():
    unet_ = TwoDUNet(in_channel=1, nf=16)#.cuda()
    #print(unet_)
    input = Variable(torch.zeros((8,1, 512, 512)))#.cuda()
    out = unet_(input)
    print('out shape:')
    print(out.size())

if __name__ == '__main__':
    test_model()
