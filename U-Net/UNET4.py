import numpy as numpy
import torch
import torch.nn as nn

class UNET(nn.Module):
    def __init__(self):
        super(UNET,self).__init__()

        def CBR2d(in_ch,out_ch,kernel_size=3, stride=1, padding=0, bias=True):
            layers = []
            layers += [nn.Conv2d(in_ch,out_ch,kernel_size=kernel_size,stride=stride,padding=padding,bias=bias)]
            layers += [nn.BatchNorm2d(num_features=out_ch)]
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
        
        self.fc = nn.Conv2d(in_channels=64,out_channels=1,kernel_size=1,stride=1,padding=0,bias=True)

        self.patch_size = 256
        self.overlap = 128

    def extract_patches(self, image, patch_size, overlap):
        patches = []
        step_size = patch_size - overlap
        padding_size = patch_size // 2 - overlap # Padding 추가
        image = torch.nn.functional.pad(image, (padding_size, padding_size, padding_size, padding_size), 'reflect')

        for i in range(0, image.size(2) - patch_size + 1, step_size):
            for j in range(0, image.size(3) - patch_size + 1, step_size):
                patch = image[:, :, i:i+patch_size, j:j+patch_size]
                patches.append(patch)

        return patches
    
    def combine_patches(self, patches, height, width, patch_size, overlap):
        step_size = patch_size - overlap
        rows = (height - patch_size) // step_size + 1
        cols = (width - patch_size) // step_size + 1
        combined_output = torch.zeros((patches[0].size(0), 1, height, width), device=patches[0].device)

        index = 0
        for i in range(0, rows):
            for j in range(0, cols):
                # Calculate the valid size of the current patch
                valid_height = min(patch_size, height - i * step_size)
                valid_width = min(patch_size, width - j * step_size)

                combined_output[:, :, i*step_size:i*step_size+valid_height, j*step_size:j*step_size+valid_width] += patches[index][:, :, :valid_height, :valid_width][:, :, :valid_height, :valid_width]
                index += 1

        return combined_output

    def center_crop(self, image, output_size):
        h, w = image.size()[2], image.size()[3]
        th, tw = output_size
        i = int(round((h - th) / 2.))
        j = int(round((w - tw) / 2.))
        return image[:, :, i:i + th, j:j + tw]

    def forward(self, x):
        print('input x.shape : ', x.shape)
        
        patches = self.extract_patches(x, self.patch_size, self.overlap)
        print(len(patches))
        outputs = []

        for patch in patches:
            print('patch size : ', patch.shape)
            con_x1 = self.cont1(patch)
            con_x2 = self.cont2(con_x1)
            con_pool_1 = self.pool64(con_x2)

            con_x3 = self.cont3(con_pool_1)
            con_x4 = self.cont4(con_x3)
            con_pool_2 = self.pool128(con_x4)

            con_x5 = self.cont5(con_pool_2)
            con_x6 = self.cont6(con_x5)
            con_pool_3 = self.pool256(con_x6)

            con_x7 = self.cont7(con_pool_3)
            con_x8 = self.cont8(con_x7)
            con_pool_4 = self.pool512(con_x8)

            con_x9 = self.cont9(con_pool_4)
            con_10 = self.cont10(con_x9)
            exp_up_1 = self.up512(con_10)

            n_con_x8 = self.center_crop(con_x8, exp_up_1.shape[2:])
            cat_1 = torch.cat((exp_up_1, n_con_x8), dim=1)
            exp_1 = self.exp1(cat_1)
            exp_2 = self.exp2(exp_1)
            exp_up_2 = self.up256(exp_2)

            n_con_x6 = self.center_crop(con_x6, exp_up_2.shape[2:])
            cat_2 = torch.cat((exp_up_2, n_con_x6), dim=1)
            exp_3 = self.exp3(cat_2)
            exp_4 = self.exp4(exp_3)
            exp_up_3 = self.up128(exp_4)

            n_con_x4 = self.center_crop(con_x4, exp_up_3.shape[2:])
            cat_3 = torch.cat((exp_up_3, n_con_x4), dim=1)
            exp_5 = self.exp5(cat_3)
            exp_6 = self.exp6(exp_5)
            exp_up_4 = self.up64(exp_6)

            n_con_x2 = self.center_crop(con_x2, exp_up_4.shape[2:])
            cat_4 = torch.cat((exp_up_4, n_con_x2), dim=1)
            exp_7 = self.exp7(cat_4)
            exp_8 = self.exp8(exp_7)

            output_patch = self.fc(exp_8)
            outputs.append(output_patch)
            print('output_patch.shape : ', output_patch.shape)
        
        combined_output = self.combine_patches(outputs, x.size(2), x.size(3), self.patch_size, self.overlap)
        print('combined_output.shape : ', combined_output.shape)
        return combined_output
