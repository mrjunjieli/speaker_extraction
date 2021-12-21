from torch.utils.data import dataloader
from Modelutils import * 
from Normolization import *

import torch
import torch.nn as nn
import cv2
from torchsummary import summary
import math
import torch.optim as optim 


class concatNet(nn.Module):

    def __init__(self,num_repeats,num_blocks,in_channels=256,
            out_channels=256,kernel_size=3,norm='gln',causal=False):
        super(concatNet, self).__init__()
        self.liner = Conv1D(556,in_channels,kernel_size=1)
        self.TCN = self._Sequential_repeat(num_repeats, num_blocks, 
                in_channels=in_channels, out_channels=out_channels,
                kernel_size=kernel_size,norm=norm, causal=causal)        
        self.relu = nn.ReLU(True)


    def forward(self, x):
        out = self.liner(x)
        out = self.TCN(out)
        out = self.relu(out)

        return out 

    def _Sequential_repeat(self, num_repeats, num_blocks, **kwargs):
        repeat_lists = [self._Sequential_block(
            num_blocks, **kwargs) for i in range(num_repeats)]
        return nn.Sequential(*repeat_lists)

    def _Sequential_block(self, num_blocks, **kwargs):
        '''
        Sequential 1-D Conv Block
        input:
            num_blocks:times the block appears
            **block_kwargs
        '''
        Conv1D_Block_lists = [Conv1D_Block(
            **kwargs, dilation=(2**i)) for i in range(num_blocks)]
        return nn.Sequential(*Conv1D_Block_lists)

    


class Encoder(nn.Module):
    '''
    Encoder of the TasNet
    '''
    def __init__(self,kernel_size,stride,outputDim=256):
        super(Encoder,self).__init__()
        self.encoder = nn.Conv1d(1,outputDim,kernel_size,stride=stride)
        self.relu = nn.ReLU(True)
    def forward(self,x):
        out = self.encoder(x)
        out = self.relu(out)
        return out 

class Decoder(nn.Module):
    '''
    Decoder of the TasNet
    '''
    def __init__(self,kernel_size,stride,inputDim=256):
        super(Decoder,self).__init__()
        self.decoder = nn.ConvTranspose1d(inputDim,1,kernel_size,stride)
    
    def forward(self,x):
        out = self.decoder(x)
        return out 


class TCN(nn.Module):
    '''
    in_channels:the encoder out_channels

    '''

    def __init__(self, out_channels, num_repeats, num_blocks,
                 kernel_size,norm='gln', causal=False):
        super(TCN, self).__init__()

        self.TCN = self._Sequential_repeat(num_repeats, num_blocks, in_channels=256, out_channels=out_channels,
         kernel_size=kernel_size,norm=norm, causal=causal)


    def forward(self, x):
        c = self.TCN(x)
        return c  #shape [-1,1,256]

    def _Sequential_repeat(self, num_repeats, num_blocks, **kwargs):
        repeat_lists = [self._Sequential_block(
            num_blocks, **kwargs) for i in range(num_repeats)]
        return nn.Sequential(*repeat_lists)

    def _Sequential_block(self, num_blocks, **kwargs):
        '''
        Sequential 1-D Conv Block
        input:
            num_blocks:times the block appears
            **block_kwargs
        '''
        Conv1D_Block_lists = [Conv1D_Block(
            **kwargs, dilation=(2**i)) for i in range(num_blocks)]
        return nn.Sequential(*Conv1D_Block_lists)


class AVModel(nn.Module):
    
    def __init__(self,):
        super(AVModel,self).__init__()

        self.audio_model_encoder = Encoder(kernel_size=40,stride=20)
        self.audio_model_TCN1 = TCN(out_channels=256, num_repeats=1, num_blocks=8, kernel_size=3)
        self.audio_model_TCN2 = TCN(out_channels=256, num_repeats=1, num_blocks=8, kernel_size=3)

        self.concat_model = concatNet(in_channels=256,out_channels=256,num_repeats=3,num_blocks=8)
        self.decoder = Decoder(kernel_size=40, stride=20, inputDim=256)
        
        self.speakerembedding = nn.Conv1d(in_channels=256,out_channels=1,kernel_size=3,stride=8)
        self.transToSpeakerEmb = nn.Linear(300, 5033)
        self.batch = nn.BatchNorm1d(256)

    def forward(self,data):
        audio_mix,audio_ref = data
        
        # print('audio_mix:',audio_mix.shape)
        encoder_output = self.audio_model_encoder(audio_mix)
    
        TCN_output = self.audio_model_TCN1(encoder_output)
        length = TCN_output.shape[2]

        encoder_output_aux = self.audio_model_encoder(audio_ref)
        TCN_output_aux = self.audio_model_TCN2(encoder_output_aux)
        TCN_output_aux = self.batch(TCN_output_aux)
        # print(TCN_output_aux.shape)4,256,2399
        speakeremb = self.speakerembedding(TCN_output_aux)  #shape[batch,1,300]

        speakeremb_new = self.transToSpeakerEmb(speakeremb)
        speakeremb_new = torch.squeeze(speakeremb_new,dim=1)
        # print(speakeremb_new)

        speakerembs = speakeremb.repeat(1,length,1)
        speakerembs = speakerembs.permute(0,2,1)

        concat_input = torch.cat((TCN_output, speakerembs), dim=1)
        # concat_input = speakerembs
        # print('tcn_output:',TCN_output.shape)
        # print('concat:',concat_input.shape)
        # print("TcN",TCN_output.shape)
        # print('xx',concat_input.shape)(,556,2399)
        concat_output = self.concat_model(concat_input)

        decoder_input = encoder_output*concat_output
        output = self.decoder(decoder_input) #output [B,C,lenght]

        # speakeremb_new = 0

        return output,speakeremb_new
        # return speakeremb_new



def count_parameters(named_parameters):
    # Count total parameters
    total_params = 0
    part_params = {}
    for name, p in sorted(list(named_parameters)):
        n_params = p.numel()
        total_params += n_params
        part_name = name.split('.')[0]
        if part_name in part_params:
            part_params[part_name] += n_params
        else:
            part_params[part_name] = n_params

    for name, n_params in part_params.items():
        print('%s #params: %.2f M' % (name, n_params/1000000))
    print("Total %.2f M parameters" % (total_params / 1000000))
    print('Estimated Total Size (MB): %0.2f' %
          (total_params * 4. / (1024 ** 2)))







if __name__ == '__main__':


    print('start')
    model = AVModel()

    count_parameters(model.named_parameters())


    # from DataLoader_M2Met import AudioDataset,AudioDataLoader
    # audio_data_trainval = AudioDataset(s1_scp='./data/M2Met/dev.scp')
    # dataLoader = AudioDataLoader(audio_data_trainval,aux_scp = './data/M2Met/dev_aux.scp',batch_size=2,shuffle=True)
    
    # for data in dataLoader:
    #     mix = data['mix'].float().cuda()
    #     s1 = data['s1'].cuda()
    #     s1_ref = data['s1_ref'].float().cuda()
    #     s1_id = data['s1_id']

    #     est_ = model([mix,s1_ref])
    #     break


    # net = VisualNet()
    # Videopath = '/Work19/2020/lijunjie/lips/test/0Fi83BHQsMA/00002-0001.jpg'
    # image = cv2.imread(Videopath,cv2.IMREAD_GRAYSCALE)
    # image = torch.from_numpy(image.astype(float))
    # image = image.unsqueeze(0)
    # image = image.unsqueeze(0)
    # image = image.unsqueeze(0)
    # print(image.shape)

    # image = image.float()
    # x = net(image)

    # net_input = torch.ones(32, 3, 10, 224, 224)
    # net_input = net_input.int()
    # # With square kernels and equal stride | 所有维度同一个参数配置
    # conv = nn.Conv3d(3, 64, kernel_size=3, stride=2, padding=1)
    # net_output = conv(net_input)
    # print(net_output.shape)  # shape=[32, 64, 5, 112, 112] | 相当于每一个维度上的卷积核大小都是3，步长都是2，pad都是1

    # # non-square kernels and unequal stride and with padding | 每一维度不同参数配置
    # conv = nn.Conv3d(3, 64, (2, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
    # net_output = conv(net_input)
    # print(net_output.shape) # shape=[32, 64, 9, 112, 112]

    # net = VisualNet().to('cuda')
    # summary(net,(1,75,120,120))
    # net = TCN(256,2,2,3).to('cuda')
    # summary(net,(1,48000))
    # net = Encoder(40,20).to('cuda')
    # summary(net,(1,48000))
    

    # net = concatNet(2).to('cuda')
    # summary(net,(2,256))
    # for name,param in net.named_parameters():
    #     print(name,'        ',param.size())

    # model = AVModel().to('cuda')
    # summary(model,)
    # for n,m in model.named_parameters():
    #     print(n,type(m))
    # optimizer = optim.Adam([{'params':model.parameters()}],lr=0.001)
    # print(optimizer.state_dict())
