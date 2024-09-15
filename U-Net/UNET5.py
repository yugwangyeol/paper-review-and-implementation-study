import numpy as numpy
import torch
import torch.nn as nn

class UNET(nn.Module):
    def __init__(self):
        super(UNET,self).__init__()

        def CBR2d(in_ch,out_ch,kernel_size=3, stride=1, padding=0, bias=True): #No padding이 맞지 않을까?
            layers = []
            layers += [nn.Conv2d(in_ch,out_ch,kernel_size=kernel_size,stride=stride,padding=padding,bias=bias)]
            layers += [nn.BatchNorm2d(num_features=out_ch)] # 이거는 왜 튀어 나온 걸까?
            layers += [nn.ReLU()]

            cbr = nn.Sequential(*layers)
            return cbr
        
        # Contracting path
        self.cont1 = CBR2d(in_ch=1,out_ch=64)
        self.cont2 = CBR2d(in_ch=64,out_ch=64)
        self.pool64 = nn.MaxPool2d(kernel_size=2)

        self.cont3 = CBR2d(in_ch=64,out_ch=128)
        self.cont4 = CBR2d(in_ch=128,out_ch=128)
        self.pool128 = nn.MaxPool2d(kernel_size=2)

        self.cont5 = CBR2d(in_ch=128,out_ch=256)
        self.cont6 = CBR2d(in_ch=256,out_ch=256)
        self.pool256 = nn.MaxPool2d(kernel_size=2)

        self.cont7 = CBR2d(in_ch=256,out_ch=512)
        self.cont8 = CBR2d(in_ch=512,out_ch=512)
        self.pool512 = nn.MaxPool2d(kernel_size=2)

        self.cont9 = CBR2d(in_ch=512,out_ch=1024)

        # Expanding path
        self.cont10 = CBR2d(in_ch=1024,out_ch=512)

        self.up512 = nn.ConvTranspose2d(in_channels=512,out_channels=512,kernel_size=2,stride=2,padding=0,bias=True) 
        # kernel_size가 2인 이유는 maxpool이 2여서 그런가?
        # Upsampling 및 전치 합성곱 연산 수행 -> 입력을 확장

        self.exp1 = CBR2d(in_ch=512*2,out_ch=512)
        self.exp2 = CBR2d(in_ch=512,out_ch=256)
        self.up256 = nn.ConvTranspose2d(in_channels=256,out_channels=256,kernel_size=2,stride=2,padding=0,bias=True)

        self.exp3 = CBR2d(in_ch=256*2,out_ch=256) 
        self.exp4 = CBR2d(in_ch=256,out_ch=128)
        self.up128 = nn.ConvTranspose2d(in_channels=128,out_channels=128,kernel_size=2,stride=2,padding=0,bias=True)

        self.exp5 = CBR2d(in_ch=128*2,out_ch=128) 
        self.exp6 = CBR2d(in_ch=128,out_ch=64)
        self.up64 = nn.ConvTranspose2d(in_channels=64,out_channels=64,kernel_size=2,stride=2,padding=0,bias=True)

        self.exp7 = CBR2d(in_ch=64*2,out_ch=64) 
        self.exp8 = CBR2d(in_ch=64,out_ch=64)
        
        self.fc = nn.Conv2d(in_channels=64,out_channels=1,kernel_size=1,stride=1,padding=0,bias=True) # out)ch가 2-> 1인게 0,1 이여서 그런가?

    def center_crop(self,image,output_size):
        h, w = image.size()[2], image.size()[3]  # 변경된 부분
        th, tw = output_size
        i = int(round((h - th) / 2.))
        j = int(round((w - tw) / 2.))
        return image[:, :, i:i + th, j:j + tw]

    def forward(self,x):
        #print('input x.shape : ', x.shape)
        con_x1 = self.cont1(x)
        #print('con_x1 shape:',con_x1.shape)
        con_x2 = self.cont2(con_x1)
        #print('con_x2 shape:',con_x2.shape)
        con_pool_1 = self.pool64(con_x2)
        #print('con_pool_1 shape:',con_pool_1.shape)

        con_x3 = self.cont3(con_pool_1)
        #print('con_x3 shape:',con_x3.shape)
        con_x4 = self.cont4(con_x3)
        #print('con_x4 shape:',con_x4.shape)
        con_pool_2 = self.pool128(con_x4)
        #print('con_pool2 shape:',con_pool_2.shape)

        con_x5 = self.cont5(con_pool_2)
        con_x6 = self.cont6(con_x5)
        con_pool_3 = self.pool256(con_x6)

        con_x7 = self.cont7(con_pool_3)
        con_x8 = self.cont8(con_x7)
        #print('con_x8 shape:',con_x8.shape)
        con_pool_4 = self.pool512(con_x8)

        con_x9 = self.cont9(con_pool_4)
        #print('con_x9 shape:',con_x9.shape)
        con_10 = self.cont10(con_x9)
        #print('con_x10 shape:',con_10.shape)
        exp_up_1 = self.up512(con_10)
        #print('exp_up_1 shape:',exp_up_1.shape)

        n_con_x8 = self.center_crop(con_x8,exp_up_1.shape[2:])
        #print('n_con_x8 :',n_con_x8.shape)
        cat_1 = torch.cat((exp_up_1,n_con_x8),dim=1)
        #print(cat_1.shape)
        exp_1 = self.exp1(cat_1)
        #print(exp_1.shape)
        exp_2 = self.exp2(exp_1)
        #print(exp_2.shape)
        exp_up_2 = self.up256(exp_2)
        #print(exp_up_2.shape)

        n_con_x6 = self.center_crop(con_x6,exp_up_2.shape[2:])
        cat_2 = torch.cat((exp_up_2,n_con_x6),dim=1)
        exp_3 = self.exp3(cat_2)
        exp_4 = self.exp4(exp_3)
        exp_up_3 = self.up128(exp_4)

        n_con_x4 = self.center_crop(con_x4,exp_up_3.shape[2:])
        cat_3 = torch.cat((exp_up_3,n_con_x4),dim=1)
        exp_5 = self.exp5(cat_3)
        exp_6 = self.exp6(exp_5)
        exp_up_4 = self.up64(exp_6)

        n_con_x2 = self.center_crop(con_x2,exp_up_4.shape[2:])
        cat_4 = torch.cat((exp_up_4,n_con_x2),dim=1)
        exp_7 = self.exp7(cat_4)
        exp_8 = self.exp8(exp_7)

        x = self.fc(exp_8)
        print('output x shape :', x.shape)

        return x
