from turtle import forward
import torch.nn as nn
import torch.nn.functional as F
import torch

class DC_CNN(nn.Module):
    def __init__(self,p_dropout=0.1):
        super().__init__()
        self.conv1=ConvBlock(64,32,p_dropout)
        self.cn1=Intra_BCL(out_channel=32,p_dropout=p_dropout)
        self.conv2=ConvBlock(32,32,p_dropout)
        self.cn2=Intra_BCL(out_channel=32,p_dropout=p_dropout)
        self.conv3=ConvBlock(32,32,p_dropout)
        self.cn3=Intra_BCL(out_channel=32,p_dropout=p_dropout)
        self.conv4=ConvBlock(32,16,p_dropout)
        self.cn4=Intra_BCL(out_channel=16,p_dropout=p_dropout)
        self.conv5=ConvBlock(16,8,p_dropout)
        self.cn5=Intra_BCL(out_channel=8,p_dropout=p_dropout)
        self.relu=nn.ReLU()
        
    def forward(self,batch):
        batch=torch.permute(batch,(0,2,1)) # [32,64,500]
        conv1=self.conv1(batch)
        cn1=self.cn1(conv1,batch)  # [32,32,500]

        conv2=self.conv2(cn1)
        cn2=self.cn2(conv2,batch,cn1)

        conv3=self.conv3(cn2)  # [32,32,500]
        cn3=self.cn3(conv3,batch,cn1,cn2)  # [32,16,500]

        conv4=self.conv4(cn3)
        cn4=self.cn4(conv4,batch,cn1,cn2,cn3)

        conv5=self.conv5(cn4)
        cn5=self.cn5(conv5,batch,cn1,cn2,cn3,cn4)  # [32,8,500]
        
        out=self.relu(cn5)
        out=torch.permute(out,(0,2,1))  # [32,500,8]
        return out

class MLP(nn.Module):
    def __init__(self,p_dropout=0.1,vocab=26):
        super().__init__()
        self.flat=nn.Flatten(1,2)
        self.vocab=vocab
        self.fc1=nn.Linear(4000,500) # 8000=500*16, 4000=500*8
        self.fc2=nn.Linear(500,500*self.vocab)
        self.relu=nn.ReLU()
        self.dropout=nn.Dropout(p_dropout)
        self.bn=nn.BatchNorm1d(500)    
    def forward(self,batch):
        flat=self.flat(batch)  # [32,500,16]->flatten [32,8000]
        fc1=self.fc1(flat)
        fc1=self.bn(fc1)
        fc1=self.relu(fc1)
        fc1=self.dropout(fc1)
        fc2=self.fc2(fc1)
        out=torch.reshape(fc2,(len(batch),500,self.vocab))
        out=F.log_softmax(out,dim=-1)
        return out

class Intra_BCL(nn.Module):
    def __init__(self,out_channel,p_dropout=0.1):
        super().__init__()
        self.out_channel=out_channel
        self.relu = nn.ReLU(inplace=True)
        self.dropout=nn.Dropout(p_dropout)
        self.bn=nn.BatchNorm1d(self.out_channel)

    def forward(self, *inputs):
        # input is a list of tensors, putting it together
        self.input_length=len(inputs)
        in_channels=0
        for i in inputs:
            in_channels+=i.size(1)
        kernel_size=3
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=self.out_channel, kernel_size=kernel_size,padding=(kernel_size - 1) // 2).cuda()
        combined_input = torch.cat(inputs, dim=1) # [32,96,500]
        out=self.conv.forward(combined_input)
        out=self.bn(out)
        out = self.relu(out)
        out=self.dropout(out)
        return out

class ConvBlock(nn.Module):
    def __init__(self,in_channels=500,out_channels=500,p_dropout=0.1):
        super().__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.p_dropout=p_dropout
        self.kernel_size=3
        self.conv=nn.Conv1d(in_channels=self.in_channels,out_channels=self.out_channels,kernel_size=self.kernel_size,padding=(self.kernel_size - 1) // 2)
        self.relu=nn.ReLU()
        self.dropout=nn.Dropout(self.p_dropout)
        self.bn=nn.BatchNorm1d(self.out_channels)
    def forward(self,batch):
        conv=self.conv(batch)
        conv=self.bn(conv)
        out=self.relu(conv)
        out=self.dropout(out)
        return out